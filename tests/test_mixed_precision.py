"""Per-tensor dtypes + explicit accumulation dtype: mixed-precision operators are one spec and
are graded with the same numeric grades (requires gcc)."""

from __future__ import annotations

import shutil

import pytest

from kopt_agent.backends import get_backend
from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import default_accumulate_dtype, get_dtype, widest
from kopt_agent.evaluator import Evaluator, NumericGrade, TrialStatus
from ops import build_operator

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


@pytest.fixture(scope="module")
def backend():
    return get_backend("cpu_c")


def test_accumulate_dtype_defaults():
    fp32, fp16, bf16, fp64 = (get_dtype(name) for name in ("fp32", "fp16", "bf16", "fp64"))
    assert widest(fp16, bf16, fp32) is fp32
    assert default_accumulate_dtype(bf16, bf16) is fp32
    assert default_accumulate_dtype(fp16, fp32) is fp32
    assert default_accumulate_dtype(fp64, fp32) is fp64


def test_spec_exposes_per_tensor_and_accumulate_dtypes():
    uniform = build_operator("matmul", (8, 8, 8)).spec
    assert not uniform.mixed_precision and uniform.precision_label() == "fp32"
    assert uniform.acc_dtype.name == "fp32" and uniform.tensor_dtypes() == {"A": "fp32", "B": "fp32", "C": "fp32"}

    mixed = build_operator("matmul", (8, 8, 8), dtype="bf16", output_dtype="fp32", accumulate_dtype="fp32").spec
    assert mixed.mixed_precision and mixed.precision_label() == "bf16 -> fp32 (acc fp32)"
    assert mixed.tensor_dtypes() == {"A": "bf16", "B": "bf16", "C": "fp32"}
    assert "const __bf16* A" in mixed.c_signature and "float* C" in mixed.c_signature
    assert mixed.policy is get_dtype("fp32").policy  # tolerance follows what is stored
    assert {dtype.name for dtype in mixed.dtypes_used()} == {"bf16", "fp32"}

    wide_acc = build_operator("matmul", (8, 8, 8), accumulate_dtype="fp64").spec
    assert wide_acc.mixed_precision and wide_acc.acc_dtype.name == "fp64"
    assert {dtype.name for dtype in wide_acc.dtypes_used()} == {"fp32", "fp64"}


def test_fp64_accumulation_is_specified_and_more_precise_than_float(backend):
    bundle = build_operator("matmul", (32, 24, 96), accumulate_dtype="fp64")
    assert "typedef double acc_t;" in bundle.baseline_source
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    baseline = evaluator.evaluate(Candidate(source=bundle.baseline_source, origin="baseline"))
    assert baseline.status is TrialStatus.OK, baseline.message
    packed = evaluator.evaluate(bundle.select_template("packed").default_candidate())
    assert packed.status is TrialStatus.OK, packed.message
    assert "acc_t acc[MR][NR]" in bundle.select_template("packed").default_candidate().source

    # The same arithmetic with a float accumulator violates the declared precision: measurably
    # worse scaled-ULP error than the fp64-accumulating kernels.
    float_acc = Candidate(
        source=(
            f"#include <stddef.h>\n{bundle.spec.c_signature} {{ for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {{ float acc = 0; "
            "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; } }\n"
        ),
        origin="test",
    )
    sloppy = evaluator.evaluate(float_acc)
    assert sloppy.status is TrialStatus.OK
    assert baseline.scaled_ulp_error < sloppy.scaled_ulp_error
    assert packed.scaled_ulp_error < sloppy.scaled_ulp_error


def test_bf16_inputs_fp32_output_matmul_and_fused_variant(backend):
    if not backend.supports_dtype(get_dtype("bf16")):
        pytest.skip("this compiler does not support __bf16")
    bundle = build_operator("matmul", (40, 48, 64), dtype="bf16", output_dtype="fp32")
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    baseline = evaluator.evaluate(Candidate(source=bundle.baseline_source, origin="baseline"))
    assert baseline.status is TrialStatus.OK, baseline.message
    assert baseline.numeric_grade in {NumericGrade.BITWISE.value, NumericGrade.TIGHT.value}
    packed = evaluator.evaluate(bundle.select_template("packed").default_candidate())
    assert packed.status is TrialStatus.OK, packed.message
    assert "blocked" not in bundle.templates  # the fp32-only template is not offered for mixed precision

    fused = build_operator("matmul_bias_relu", (24, 32, 40), dtype="bf16", output_dtype="fp32")
    assert fused.spec.tensor_dtypes() == {"A": "bf16", "B": "bf16", "bias": "fp32", "C": "fp32"}
    assert [part.spec.precision_label() for part in fused.fusion_parts] == ["bf16 -> fp32 (acc fp32)", "fp32"]
    fused_evaluator = Evaluator(fused.spec, backend, warmup=1, repeats=2, run_timeout_seconds=10)
    result = fused_evaluator.evaluate(fused.select_template(None).default_candidate())
    assert result.status is TrialStatus.OK, result.message


def test_softmax_and_bias_relu_accept_precision_arguments(backend):
    softmax = build_operator("softmax", (8, 16), accumulate_dtype="fp64")
    assert "#define EXP(x) exp(x)" in softmax.baseline_source
    result = Evaluator(softmax.spec, backend, warmup=1, repeats=1, run_timeout_seconds=10).evaluate(Candidate(softmax.baseline_source, "baseline"))
    assert result.status is TrialStatus.OK, result.message

    bias_relu = build_operator("bias_relu", (8, 16), output_dtype="fp64")
    assert bias_relu.spec.tensor_dtypes() == {"X": "fp32", "bias": "fp64", "Y": "fp64"}
    result = Evaluator(bias_relu.spec, backend, warmup=1, repeats=1, run_timeout_seconds=10).evaluate(Candidate(bias_relu.baseline_source, "baseline"))
    assert result.status is TrialStatus.OK, result.message


def test_agent_checks_every_dtype_the_kernel_must_spell(backend, tmp_path, monkeypatch):
    from kopt_agent.agent import AgentConfig, OptimizationAgent

    monkeypatch.setattr(backend, "supports_dtype", lambda dtype: dtype.name == "fp32")
    bundle = build_operator("matmul", (8, 8, 8), accumulate_dtype="fp64")
    with pytest.raises(RuntimeError, match="cannot compile fp64"):
        OptimizationAgent(bundle, backend, AgentConfig(roofline=False, output_dir=tmp_path))
