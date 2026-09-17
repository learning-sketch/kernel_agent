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
    tuned = evaluator.evaluate(bundle.select_template(None).default_candidate())
    assert tuned.status is TrialStatus.OK


@pytest.mark.parametrize(
    "body,expected_status,expected_fragment",
    [
        ("int x = 1 }", TrialStatus.COMPILE_ERROR, "error"),
        ("float* p = 0; p[123456789] = 1.0f;", TrialStatus.RUNTIME_ERROR, "SIGSEGV"),
        # Writes correct values only when the output is a single cell, leaving larger outputs NaN-prefilled.
        (
            "if (M * N == 1) { float acc = 0.0f; for (int k = 0; k < K; k++) acc += A[k] * B[k]; C[0] = acc; }",
            TrialStatus.INCORRECT,
            "non-finite",
        ),
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
            result = evaluator.evaluate(bundle.select_template(None).render({"ONLINE": online, "FASTMATH": fastmath}))
            assert result.status is TrialStatus.OK, (online, fastmath, result.message)


def test_batch_evaluation_is_tiered_and_ordered():
    bundle = build_operator("matmul", (96, 80, 64))
    evaluator = Evaluator(bundle.spec, get_backend("cpu_c"), warmup=1, repeats=3, run_timeout_seconds=20, workers=2, quick_repeats=2)
    template = bundle.select_template("blocked")
    broken = Candidate(source=f"#include <stddef.h>\n{bundle.spec.c_signature} {{ int x = 1 }}\n", origin="test")
    candidates = [
        Candidate(source=bundle.baseline_source, origin="baseline"),
        broken,
        template.render({"MB": 8, "NB": 64, "KB": 32, "THREADS": 1, "SCHEDULE": "static"}),
        template.render({"MB": 16, "NB": 64, "KB": 64, "THREADS": 2, "SCHEDULE": "static"}),
        template.render({"MB": 32, "NB": 128, "KB": 64, "THREADS": 2, "SCHEDULE": "dynamic"}),
    ]
    results = evaluator.evaluate_batch(candidates, top_k=2)
    assert len(results) == len(candidates)
    assert [result.fingerprint for result in results] == [candidate.fingerprint for candidate in candidates]
    assert results[1].status is TrialStatus.COMPILE_ERROR
    valid = [result for result in results if result.is_valid]
    assert len(valid) == 4
    assert sum(result.timing_tier == "full" for result in valid) == 2
    assert sum(result.timing_tier == "quick" for result in valid) == 2
    # The two fully-timed results are the two fastest according to screening.
    assert all(result.compiler_notes for result in valid)
