"""Champion bundle export (agent + `kopt export`) and workload ingestion from traces (requires gcc)."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kopt_agent import cli
from kopt_agent.agent import AgentConfig, OptimizationAgent
from kopt_agent.backends import get_backend
from kopt_agent.bundle import BUNDLE_FORMAT, BundleContext, export_bundle
from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import Evaluator
from kopt_agent.history import History
from kopt_agent.spec import WorkloadProfile
from kopt_agent.workload import ingest_trace, parse_shape, save_profile
from ops import build_operator

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


@pytest.fixture(scope="module")
def backend():
    return get_backend("cpu_c")


# ---- 4. integration bundle ------------------------------------------------------------------

def _aligned_array(values: np.ndarray, alignment: int = 64) -> np.ndarray:
    """Copy `values` into a buffer whose data pointer is `alignment`-byte aligned (the bundle's contract)."""
    raw = np.empty(values.nbytes + alignment, dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    aligned = raw[offset:offset + values.nbytes].view(values.dtype).reshape(values.shape)
    aligned[...] = values
    return aligned


def test_export_bundle_is_self_contained_and_builds(backend, tmp_path):
    bundle = build_operator("matmul", (24, 16, 8))
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    history = History(tmp_path / "matmul", spec=bundle.spec, compile_command=backend.portable_compile_command())
    baseline = Candidate(source=bundle.baseline_source, origin="baseline")
    baseline_result = evaluator.evaluate(baseline)
    assert history.record(baseline, baseline_result), baseline_result.message  # first valid trial is always the champion
    candidate = bundle.select_template("blocked").render({"MB": 8, "NB": 64, "KB": 8, "THREADS": 1, "SCHEDULE": "static", "ALIGNED": 1})
    result = evaluator.evaluate(candidate)
    # Correctness is deterministic; whether a blocked template out-times the naive loop on a 24x16x8
    # problem is not. The export is therefore exercised on the candidate itself instead of being gated
    # on it dethroning the baseline - export_bundle takes an explicit (candidate, result) pair anyway.
    assert result.is_valid and result.is_benchmark_grade, result.message
    history.record(candidate, result)

    context = BundleContext(backend_name=backend.name, launch_abi=backend.launch_abi.to_dict(), hardware="test box", baseline=history.baseline)
    directory = export_bundle(history, candidate, result, context)
    assert directory == tmp_path / "matmul" / "bundle"
    assert {path.name for path in directory.iterdir()} == {"kernel.c", "kernel.h", "manifest.json", "build.sh", "parity_test.py", "README.md"}

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == BUNDLE_FORMAT
    assert manifest["operator"]["c_signature"] == bundle.spec.c_signature
    assert manifest["precision"] == {
        "label": "fp32", "input_dtype": "fp32", "output_dtype": "fp32", "accumulate_dtype": "fp32", "mixed": False,
        "tensor_dtypes": {"A": "fp32", "B": "fp32", "C": "fp32"},
    }
    assert manifest["launch_abi"]["pointer_space"] == "host" and manifest["launch_abi"]["entry_kind"] == "kernel"
    assert manifest["candidate"]["params"] == candidate.params and manifest["candidate"]["trial_id"] == result.trial_id
    assert manifest["fast_path"]["predicate"] == "N % 16 == 0" and "primary" in manifest["fast_path"]["activated_on"]
    assert manifest["numerics"]["grade"] == result.numeric_grade
    assert manifest["performance"]["speedup_vs_baseline"] > 0 and manifest["performance"]["timing_source"] == "host_wall"
    assert "{source}" in manifest["build"]["compile_command"] and "{artifact}" in manifest["build"]["compile_command"]

    header = (directory / "kernel.h").read_text()
    assert f"{bundle.spec.c_signature};" in header and "extern int kopt_fast_path_active;" in header
    readme = (directory / "README.md").read_text()
    assert "## Contract" in readme and "N % 16 == 0" in readme and "parity_test.py" in readme
    assert (directory / "kernel.c").read_text() == candidate.source

    # Independent build: ship the directory somewhere unrelated to the repository and to the run,
    # compile it there with nothing but build.sh, then call the kernel through the documented ABI.
    shipped = tmp_path / "downstream" / "vendor" / "matmul_kernel"
    shutil.copytree(directory, shipped)
    clean_env = {key: value for key, value in os.environ.items() if key not in ("KOPT_REPO", "PYTHONPATH")}
    built = subprocess.run(["sh", str(shipped / "build.sh")], capture_output=True, text=True, env=clean_env, cwd=str(tmp_path / "downstream"))
    assert built.returncode == 0, built.stdout + built.stderr
    library_path = shipped / "libkernel.so"
    assert library_path.exists() and "built" in built.stdout

    library = ctypes.CDLL(str(library_path))
    kernel = getattr(library, bundle.spec.symbol)
    kernel.restype = None
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    rng = np.random.default_rng(7)
    for m_dim, n_dim, k_dim in ((24, 16, 8), (7, 13, 5), (33, 48, 17)):
        a_matrix = _aligned_array(rng.standard_normal((m_dim, k_dim), dtype=np.float32))
        b_matrix = _aligned_array(rng.standard_normal((k_dim, n_dim), dtype=np.float32))
        c_matrix = _aligned_array(np.full((m_dim, n_dim), np.nan, dtype=np.float32))
        kernel(a_matrix.ctypes.data, b_matrix.ctypes.data, c_matrix.ctypes.data, m_dim, n_dim, k_dim)
        expected = (a_matrix.astype(np.float64) @ b_matrix.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(c_matrix, expected, rtol=1e-5, atol=1e-5, err_msg=f"shape {m_dim}x{n_dim}x{k_dim}")
        fast_path_flag = ctypes.c_int.in_dll(library, manifest["fast_path"]["flag_symbol"]).value
        assert fast_path_flag == int(n_dim % 16 == 0), f"fast-path flag mismatch for N={n_dim}"

    # The shipped parity test recompiles kernel.c itself and re-grades it against the reference.
    env = {**os.environ, "KOPT_REPO": str(REPO)}
    completed = subprocess.run([sys.executable, str(shipped / "parity_test.py")], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "parity OK" in completed.stdout


def test_agent_run_exports_bundle_and_cli_export_rebuilds_it(backend, tmp_path, capsys):
    bundle = build_operator("matmul", (16, 16, 16))
    config = AgentConfig(autotune_budget=2, warm_start=0, roofline=False, fusion_report=False, workers=1, repeats=2, warmup=1, output_dir=tmp_path)
    agent = OptimizationAgent(bundle, backend, config)
    agent.run()
    assert agent.bundle_dir is not None and (agent.bundle_dir / "manifest.json").exists()
    summary = json.loads((tmp_path / "matmul" / "summary.json").read_text())
    assert summary["best_candidate"]["fingerprint"] == agent.history.best[0].fingerprint
    assert summary["launch_abi"]["pointer_space"] == "host" and summary["precision"] == "fp32"

    shutil.rmtree(agent.bundle_dir)
    assert cli.main(["export", "--op", "matmul", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "bundle written to" in output
    manifest = json.loads((tmp_path / "matmul" / "bundle" / "manifest.json").read_text())
    assert manifest["candidate"]["fingerprint"] == agent.history.best[0].fingerprint
    assert manifest["performance"]["speedup_vs_baseline"] is not None  # baseline recovered from trials.jsonl
    assert (tmp_path / "matmul" / "bundle" / "kernel.c").read_text() == agent.history.best[0].source


def test_cli_export_without_a_run_fails_cleanly(tmp_path, capsys):
    assert cli.main(["export", "--op", "matmul", "--out", str(tmp_path)]) == 1
    assert "no finished run" in capsys.readouterr().err


# ---- 5. workload ingestion --------------------------------------------------------------------

def test_parse_shape_accepts_common_spellings():
    assert parse_shape("512x512x512") == (512, 512, 512)
    assert parse_shape("[64, 512, 512]") == (64, 512, 512)
    assert parse_shape("7;13;5") == (7, 13, 5)
    assert parse_shape([1, 2]) == (1, 2)
    with pytest.raises(ValueError):
        parse_shape("")


def test_ingest_csv_with_dimension_columns_filters_and_aggregates(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "op,M,N,K,count\n"
        "matmul,512,512,512,100\n"
        "matmul,64,512,512,\n"  # empty count -> one call
        "matmul,64,512,512,899\n"
        "softmax,4096,1024,,\n"  # other operator
        "matmul,0,512,512,5\n"  # invalid dimension -> skipped
        "matmul,abc,512,512,5\n"  # unparsable -> skipped
    )
    report = ingest_trace(trace, operator="matmul", dims=["M", "N", "K"])
    assert report.rows_seen == 6 and report.rows_used == 3 and report.rows_filtered_out == 1 and report.rows_skipped == 2
    assert report.profile is not None
    counts = {entry.shape: entry.count for entry in report.profile.entries}
    assert counts == {(512, 512, 512): 100, (64, 512, 512): 900}
    assert "2 distinct shapes, 1000 calls" in report.describe()

    profile_path = save_profile(report.profile, tmp_path / "out" / "profile.json")
    loaded = WorkloadProfile.load(profile_path)
    assert {entry.shape: entry.count for entry in loaded.entries} == counts


def test_ingest_jsonl_and_shape_column(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"kernel": "matmul", "shape": [512, 512, 512]}) + "\n"
        + json.dumps({"kernel": "matmul", "shape": "512x512x512", "calls": 3}) + "\n"
        + json.dumps({"kernel": "matmul", "shape": [8, 8, 8], "calls": 0}) + "\n"
        + "\n"
    )
    report = ingest_trace(trace, operator="matmul")
    assert report.rows_used == 2 and report.rows_skipped == 1
    assert {entry.shape: entry.count for entry in report.profile.entries} == {(512, 512, 512): 4}

    without_shape = tmp_path / "bad.csv"
    without_shape.write_text("op,count\nmatmul,1\n")
    report = ingest_trace(without_shape, operator="matmul")
    assert report.profile is None and report.rows_skipped == 1 and "no shape column" in report.problems[0]


def test_cli_workload_command_writes_profile_usable_by_run(tmp_path, capsys):
    trace = tmp_path / "trace.csv"
    trace.write_text("operator,shape,calls\nmatmul,32x24x16,7\nmatmul,16x16x16,2\nbias_relu,32x24,9\n")
    profile = tmp_path / "profile.json"
    assert cli.main(["workload", "--trace", str(trace), "--op", "matmul", "--profile", str(profile)]) == 0
    out = capsys.readouterr().out
    assert "2 distinct shapes, 9 calls" in out and "profile written to" in out
    bundle = build_operator("matmul", None, workload=WorkloadProfile.load(profile))
    assert bundle.spec.primary_shape == (32, 24, 16)

    empty = tmp_path / "empty.csv"
    empty.write_text("op,shape\nsoftmax,4x4\n")
    assert cli.main(["workload", "--trace", str(empty), "--op", "matmul", "--profile", str(tmp_path / "none.json")]) == 1
