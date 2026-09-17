"""End-to-end checks of the evaluation loop on the CPU backend (requires gcc)."""

import shutil

import pytest

from kopt_agent.backends import get_backend
from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import Evaluator, TrialStatus
from ops import build_operator

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


@pytest.fixture(scope="module")
def matmul_evaluator():
    bundle = build_operator("matmul", (48, 40, 32))
    return bundle, Evaluator(bundle.spec, get_backend("cpu_c"), warmup=1, repeats=2, run_timeout_seconds=10)


def test_baseline_and_template_are_correct(matmul_evaluator):
    bundle, evaluator = matmul_evaluator
    baseline = evaluator.evaluate(Candidate(source=bundle.baseline_source, origin="baseline"))
    assert baseline.status is TrialStatus.OK and baseline.latency_ms_median > 0
    tuned = evaluator.evaluate(bundle.template.default_candidate())
    assert tuned.status is TrialStatus.OK


@pytest.mark.parametrize(
    "body,expected_status,expected_fragment",
    [
        ("int x = 1 }", TrialStatus.COMPILE_ERROR, "error"),
        ("float* p = 0; p[123456789] = 1.0f;", TrialStatus.RUNTIME_ERROR, "SIGSEGV"),
        ("C[0] = 0.0f;", TrialStatus.INCORRECT, "non-finite"),
        ("for (size_t i = 0; i < (size_t)M*N; i++) C[i] = 1.0f;", TrialStatus.INCORRECT, "outside tolerance"),
        ("for (size_t i = 0; i <= (size_t)M*N; i++) C[i] = 0.0f;", TrialStatus.INCORRECT, "wrote outside the output"),
        ("(void)C; ((float*)A)[0] = 0.0f;", TrialStatus.INCORRECT, "modified const input"),
    ],
)
def test_failure_modes_are_classified(matmul_evaluator, body, expected_status, expected_fragment):
    bundle, evaluator = matmul_evaluator
    source = f"#include <stddef.h>\n{bundle.spec.c_signature} {{ {body} }}\n"
    result = evaluator.evaluate(Candidate(source=source, origin="test"))
    assert result.status is expected_status, result.message
    assert expected_fragment in result.message


def test_hang_is_reported_as_timeout():
    bundle = build_operator("matmul", (8, 8, 8))
    evaluator = Evaluator(bundle.spec, get_backend("cpu_c"), warmup=0, repeats=1, run_timeout_seconds=2)
    source = f"{bundle.spec.c_signature} {{ volatile int spin = 1; while (spin) {{}} }}\n"
    result = evaluator.evaluate(Candidate(source=source, origin="test"))
    assert result.status is TrialStatus.TIMEOUT


def test_softmax_template_variants_are_correct():
    bundle = build_operator("softmax", (64, 96))
    evaluator = Evaluator(bundle.spec, get_backend("cpu_c"), warmup=1, repeats=2, run_timeout_seconds=10)
    for online in (0, 1):
        for fastmath in (0, 1):
            result = evaluator.evaluate(bundle.template.render({"ONLINE": online, "FASTMATH": fastmath}))
            assert result.status is TrialStatus.OK, (online, fastmath, result.message)
