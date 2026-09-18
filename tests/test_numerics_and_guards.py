"""Numeric grading, precision gating, anti-fake-success checks, fast-path contracts,
shape coverage, workload objective, A/B timing and dtype support (requires gcc)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from kopt_agent.backends import get_backend
from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import get_dtype
from kopt_agent.evaluator import Evaluator, NumericGrade, TrialStatus, robust_latency
from kopt_agent.history import History
from kopt_agent.spec import WorkloadEntry, WorkloadProfile, evaluate_fast_path_predicate
from ops import build_operator

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


@pytest.fixture(scope="module")
def backend():
    return get_backend("cpu_c")


@pytest.fixture(scope="module")
def matmul(backend):
    bundle = build_operator("matmul", (48, 40, 32))
    return bundle, Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)


def _kernel(bundle, body: str, **kwargs) -> Candidate:
    return Candidate(source=f"#include <stddef.h>\n#include <math.h>\n{bundle.spec.c_signature} {{ {body} }}\n", origin="test", **kwargs)


# ---- 2. dtypes + grading -------------------------------------------------------------------

def test_bf16_encode_decode_round_trips_with_rne():
    bf16 = get_dtype("bf16")
    values = np.array([1.0, 1.00390625, -2.5, 3.0e38, 1e-40, np.nan], dtype=np.float64)
    bits = bf16.encode(values)
    assert bits.dtype == np.uint16
    decoded = bf16.decode(bits)
    assert decoded[0] == 1.0 and decoded[1] == 1.0  # tie rounds to even
    assert decoded[2] == -2.5
    assert np.isnan(decoded[5])
    assert bf16.spacing_at(1.0) == 2.0 ** -7


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_16bit_matmul_baseline_and_template_are_graded(backend, dtype):
    bundle = build_operator("matmul", (40, 48, 32), dtype=dtype)
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    baseline = evaluator.evaluate(Candidate(source=bundle.baseline_source, origin="baseline"))
    assert baseline.status is TrialStatus.OK, baseline.message
    assert baseline.numeric_grade in {NumericGrade.BITWISE.value, NumericGrade.TIGHT.value}
    tuned = evaluator.evaluate(bundle.select_template(None).default_candidate())
    assert tuned.status is TrialStatus.OK, tuned.message
    assert tuned.scaled_ulp_error is not None and tuned.bitwise_match_rate is not None


def test_faster_but_sloppy_kernel_is_graded_reduced_precision(matmul):
    bundle, evaluator = matmul
    # A systematic 2e-4 offset stays inside the fp32 acceptance tolerance (atol 1e-3) but is
    # ~100 ULP at the output scale: accepted, yet graded reduced-precision, never silently best.
    sloppy = _kernel(
        bundle,
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc + 2e-4f; }",
    )
    result = evaluator.evaluate(sloppy)
    assert result.status is TrialStatus.OK, result.message
    assert result.numeric_grade == NumericGrade.REDUCED.value
    assert result.scaled_ulp_error > bundle.spec.policy.tight_ulp


# ---- 3. precision gating -------------------------------------------------------------------

def test_precision_sensitive_operator_never_silently_picks_reduced_precision(tmp_path):
    bundle = build_operator("softmax", (16, 32))
    spec = bundle.spec
    assert spec.precision_sensitive
    from kopt_agent.evaluator import TrialResult

    def result(trial_id, ms, grade, origin="autotune"):
        return TrialResult(trial_id, f"c{trial_id}", origin, f"fp{trial_id}", TrialStatus.OK, latency_ms_median=ms, latency_ms_min=ms,
                           latency_ms_stdev=0.0, gflops=1.0, gbps=1.0, objective_ms=ms, numeric_grade=grade, scaled_ulp_error=1.0,
                           bitwise_match_rate=0.5, max_abs_error=1e-6, per_shape={"primary": {"shape": [16, 32], "weight": 1, "latency_ms": ms}})

    gated = History(tmp_path / "gated", spec=spec)
    gated.record(Candidate("b", "baseline"), result(1, 10.0, NumericGrade.TIGHT.value, origin="baseline"))
    gated.record(Candidate("fast", "autotune"), result(2, 1.0, NumericGrade.REDUCED.value))
    gated.record(Candidate("exact", "autotune"), result(3, 4.0, NumericGrade.TIGHT.value))
    assert gated.best[1].trial_id == 3  # fastest among equal-precision candidates
    assert gated.best_reduced[1].trial_id == 2
    assert "precision <-> speed" in gated.leaderboard()

    permissive = History(tmp_path / "open", spec=spec, allow_reduced_precision=True)
    permissive.record(Candidate("b", "baseline"), result(1, 10.0, NumericGrade.TIGHT.value, origin="baseline"))
    permissive.record(Candidate("fast", "autotune"), result(2, 1.0, NumericGrade.REDUCED.value))
    assert permissive.best[1].trial_id == 2


# ---- 6. anti fake success, fast path, shape whitelist --------------------------------------

def test_kernel_that_copies_its_input_is_rejected(backend):
    bundle = build_operator("softmax", (16, 32))
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    copy = _kernel(bundle, "for (size_t i = 0; i < (size_t)M*N; i++) Y[i] = X[i];")
    result = evaluator.evaluate(copy)
    assert result.status is TrialStatus.INCORRECT
    assert "copy of input" in result.message


def test_kernel_that_leaves_cells_unwritten_is_rejected(matmul):
    bundle, evaluator = matmul
    half = _kernel(
        bundle,
        "for (int i = 0; i < M; i += 2) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; }",
    )
    result = evaluator.evaluate(half)
    assert result.status is TrialStatus.INCORRECT
    assert "non-finite" in result.message or "poison" in result.message


def test_kernel_that_accumulates_into_the_output_is_rejected(matmul):
    bundle, evaluator = matmul
    # Only correct if the output buffer happens to hold zeros: detected by the poison prefill.
    accumulate = _kernel(
        bundle,
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { if (C[(size_t)i*N+j] != C[(size_t)i*N+j]) C[(size_t)i*N+j] = 0.0f; "
        "for (int k = 0; k < K; k++) C[(size_t)i*N+j] += A[(size_t)i*K+k] * B[(size_t)k*N+j]; }",
    )
    result = evaluator.evaluate(accumulate)
    assert result.status is TrialStatus.INCORRECT
    assert any(fragment in result.message for fragment in ("previous buffer contents", "poison prefill", "outside tolerance"))


def test_shape_specialized_kernel_is_called_out(matmul):
    bundle, evaluator = matmul
    rows, cols, depth = bundle.spec.primary_shape
    specialized = _kernel(
        bundle,
        f"if (M != {rows} || N != {cols} || K != {depth}) return; "
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; }",
    )
    result = evaluator.evaluate(specialized)
    assert result.status is TrialStatus.INCORRECT
    assert result.shape_specialized
    assert "shape-specialized" in result.message
    assert 0 < result.shape_coverage < 1


def test_fast_path_predicate_evaluation():
    assert evaluate_fast_path_predicate("M % 8 == 0 && N >= 16", ("M", "N"), (16, 32))
    assert not evaluate_fast_path_predicate("M % 8 == 0 and N >= 16", ("M", "N"), (12, 32))
    with pytest.raises(ValueError):
        evaluate_fast_path_predicate("__import__('os')", ("M",), (1,))
    with pytest.raises(ValueError):
        evaluate_fast_path_predicate("Q > 1", ("M",), (1,))


def test_declared_fast_path_must_actually_activate(matmul):
    bundle, evaluator = matmul
    body = (
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; }"
    )
    # Declares a fast path but never exports the flag -> contract violation.
    no_flag = _kernel(bundle, body, fast_path_predicate="N % 8 == 0")
    assert "kopt_fast_path_active" in evaluator.evaluate(no_flag).message

    # Exports the flag but never sets it -> "declared active but stayed 0" on matching shapes.
    never = Candidate(
        source=f"#include <stddef.h>\nint kopt_fast_path_active = 0;\n{bundle.spec.c_signature} {{ {body} }}\n",
        origin="test", fast_path_predicate="N % 8 == 0",
    )
    result = evaluator.evaluate(never)
    assert result.status is TrialStatus.INCORRECT
    assert "stayed 0" in result.message

    # Honest kernel: flag mirrors the predicate, both paths correct.
    honest = Candidate(
        source=(
            f"#include <stddef.h>\nint kopt_fast_path_active = 0;\n{bundle.spec.c_signature} {{ kopt_fast_path_active = (N % 8 == 0); {body} }}\n"
        ),
        origin="test", fast_path_predicate="N % 8 == 0",
    )
    result = evaluator.evaluate(honest)
    assert result.status is TrialStatus.OK, result.message
    assert "primary" in result.fast_path["activated_on"]
    assert result.fast_path["missing_on"] == []


def test_blocked_template_aligned_fast_path_passes_contract(backend):
    bundle = build_operator("matmul", (32, 64, 16))
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    candidate = bundle.select_template("blocked").render({"MB": 8, "NB": 64, "KB": 16, "THREADS": 1, "SCHEDULE": "static", "ALIGNED": 1})
    assert candidate.fast_path_predicate == "N % 16 == 0"
    result = evaluator.evaluate(candidate)
    assert result.status is TrialStatus.OK, result.message
    assert "primary" in result.fast_path["activated_on"]


# ---- 1. workload objective + 8. A/B timing ---------------------------------------------------

def test_workload_profile_weights_the_objective(backend, tmp_path):
    profile_path = tmp_path / "wl.json"
    profile_path.write_text(json.dumps({"shapes": [{"shape": [16, 16, 16], "count": 5}, {"shape": [32, 24, 16], "count": 2}]}))
    workload = WorkloadProfile.load(profile_path)
    bundle = build_operator("matmul", None, workload=workload)
    assert bundle.spec.primary_shape == (32, 24, 16)  # dominant by count x flops
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=3, run_timeout_seconds=10)
    assert len(evaluator.timing_cases) == 2
    baseline = Candidate(source=bundle.baseline_source, origin="baseline")
    result = evaluator.evaluate(baseline)
    assert result.status is TrialStatus.OK
    assert set(result.per_shape) == {"primary", "wl(16, 16, 16)"}
    expected = sum(entry["weight"] * entry["latency_ms"] for entry in result.per_shape.values())
    assert result.objective_ms == pytest.approx(expected)

    evaluator.reference_artifact = backend.compile(baseline, bundle.spec).artifact
    tuned = evaluator.evaluate(bundle.select_template(None).default_candidate())
    assert tuned.status is TrialStatus.OK
    assert tuned.ab_speedup is not None and tuned.ab_speedup > 0
    assert all("ab_speedup" in entry for entry in tuned.per_shape.values())
    assert tuned.cpu_wall_ratio is not None and tuned.thread_utilization is not None


def test_workload_profile_validation():
    with pytest.raises(ValueError):
        WorkloadProfile([WorkloadEntry((0, 1), 1)])
    merged = WorkloadProfile([WorkloadEntry((8, 8), 2), WorkloadEntry((8, 8), 3)])
    assert merged.total_calls == 5 and len(merged.entries) == 1


def test_robust_latency_ignores_positive_noise_bursts():
    assert robust_latency([1.0, 1.1, 0.9]) == (1.0, False)
    latency, noisy = robust_latency([1.0, 5.0, 1.05, 5.1, 0.98, 5.2, 1.02])
    assert noisy and latency == pytest.approx(1.01)


# ---- 9. parity test emission ---------------------------------------------------------------

def test_history_emits_parity_test_for_new_best(backend, tmp_path):
    bundle = build_operator("matmul", (24, 16, 8))
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    history = History(tmp_path / "res" / "matmul", spec=bundle.spec, compile_command=backend.portable_compile_command())
    candidate = Candidate(source=bundle.baseline_source, origin="baseline")
    assert history.record(candidate, evaluator.evaluate(candidate))
    parity = tmp_path / "res" / "matmul" / "parity_test.py"
    assert parity.exists()
    text = parity.read_text()
    assert '"matmul"' in text and "special" in text and "POISON" in text
    assert (tmp_path / "res" / "matmul" / "best.c").exists()
