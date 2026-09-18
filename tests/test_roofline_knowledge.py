"""Roofline verdicts, dispatch-overhead modelling, negative knowledge and structured profiles
(pure Python: no compiler needed)."""

from pathlib import Path

from kopt_agent.backends.base import ProfileReport
from kopt_agent.evaluator import TrialResult, TrialStatus
from kopt_agent.generators.llm import FOCUS_BY_BOUND, LLMConfig, LLMGenerator
from kopt_agent.knowledge import KnowledgeBase, direction_keys, root_cause
from kopt_agent.roofline import MachinePeaks, analyze, verdict_for
from ops import build_operator

PEAKS = MachinePeaks(compute_gflops=1000.0, bandwidth_gbps=100.0, dispatch_overhead_ms=0.002, call_overhead_ms=0.0005, source="test")


def _ok(trial_id, ms, params=None, origin="evolve", **extra):
    return TrialResult(
        trial_id, f"c{trial_id}", origin, f"fp{trial_id}", TrialStatus.OK, latency_ms_median=ms, latency_ms_min=ms, latency_ms_stdev=0.0,
        gflops=1.0, gbps=1.0, objective_ms=ms, params=params or {}, numeric_grade="within-N-ULP", **extra,
    )


# ---- 4. roofline verdicts ------------------------------------------------------------------

def test_roofline_classifies_compute_memory_and_overhead_bound():
    gemm = analyze(PEAKS, flops=2 * 512**3, bytes_moved=3 * 512 * 512 * 4, measured_ms=1.0)
    assert gemm.bound == "compute" and gemm.attainable_ms > gemm.overhead_ms
    stream = analyze(PEAKS, flops=1_000_000, bytes_moved=8 * 4_000_000, measured_ms=1.0)
    assert stream.bound == "memory"
    tiny = analyze(PEAKS, flops=2 * 1024, bytes_moved=8 * 1024, measured_ms=0.003)
    assert tiny.bound == "overhead"
    assert "Fuse" in tiny.guidance
    assert tiny.attainable_ms == tiny.compute_time_ms + PEAKS.call_overhead_ms or tiny.attainable_ms == tiny.memory_time_ms + PEAKS.call_overhead_ms


def test_verdict_declares_ceiling_and_guides():
    report = analyze(PEAKS, flops=2 * 512**3, bytes_moved=3 * 512 * 512 * 4, measured_ms=0.29)
    verdict = verdict_for(report, best_ms=0.29, ceiling_fraction=0.85)
    assert verdict.at_ceiling and "AT CEILING" in verdict.describe()
    far = verdict_for(analyze(PEAKS, flops=2 * 512**3, bytes_moved=3 * 512 * 512 * 4, measured_ms=3.0), best_ms=3.0, ceiling_fraction=0.85)
    assert not far.at_ceiling and "headroom" in far.describe()
    assert far.bound in FOCUS_BY_BOUND


# ---- 7. negative knowledge -----------------------------------------------------------------

def test_root_cause_tags_failures_and_regressions():
    compile_error = TrialResult(1, "c", "autotune", "f", TrialStatus.COMPILE_ERROR, message="error: expected ';'")
    assert root_cause(compile_error).startswith("compile error")
    specialized = TrialResult(2, "c", "llm", "f", TrialStatus.INCORRECT, message="x", shape_specialized=True)
    assert "shape-specialized" in root_cause(specialized)
    parent = _ok(3, 1.0, {"NC": 256}, profile={"missed_loops": []}, compiler_notes=[])
    child = _ok(4, 1.5, {"NC": 512}, profile={"missed_loops": ["line 9: not vectorized: x  // for (...NC=512"]}, compiler_notes=[])
    tag = root_cause(child, parent)
    assert tag.startswith("regression +50%") and "lost vectorization" in tag
    assert root_cause(_ok(5, 1.02, {"NC": 512}), parent) is None
    assert direction_keys({"NC": 256, "MR": 4}, {"NC": 512, "MR": 4}) == ("NC:256->512",)


def test_knowledge_base_prunes_repeated_dead_ends_and_lists_them(tmp_path):
    knowledge = KnowledgeBase(tmp_path / "k.jsonl")
    parent = _ok(1, 1.0, {"NC": 256, "MR": 4})
    for trial_id in (2, 3):
        dead_end = knowledge.record_outcome((512, 512, 512), "hw", _ok(trial_id, 1.6, {"NC": 512, "MR": 4}), parent=parent)
        assert dead_end is not None and dead_end.keys == ("NC:256->512",)
    assert knowledge.is_pruned("hw", ("NC:256->512",))
    assert not knowledge.is_pruned("other-hw", ("NC:256->512",))
    assert not knowledge.is_pruned("hw", ("NC:256->512", "MR:4->6"))  # untested knob combination
    # A direction that once improved is never pruned.
    assert knowledge.record_outcome((512, 512, 512), "hw", _ok(4, 0.8, {"NC": 256, "MR": 6}), parent=parent) is None
    knowledge.record_outcome((512, 512, 512), "hw", _ok(5, 2.0, {"NC": 256, "MR": 6}), parent=parent)
    knowledge.record_outcome((512, 512, 512), "hw", _ok(6, 2.0, {"NC": 256, "MR": 6}), parent=parent)
    assert not knowledge.is_pruned("hw", ("MR:4->6",))
    # LLM dead end with its strategy as the direction.
    llm_fail = TrialResult(7, "llm", "llm", "f7", TrialStatus.INCORRECT, message="3 elements outside tolerance")
    knowledge.record_outcome((512, 512, 512), "hw", llm_fail, direction="use fp16 accumulators")
    summary = knowledge.dead_end_summary((512, 512, 512), "hw")
    assert any("NC 256->512" in line and "seen 2x" in line for line in summary)
    assert any("use fp16 accumulators" in line for line in summary)
    # Persisted and reloaded.
    reloaded = KnowledgeBase(tmp_path / "k.jsonl")
    assert reloaded.is_pruned("hw", ("NC:256->512",))
    assert len(reloaded.dead_end_summary((512, 512, 512), "hw")) == len(summary)


# ---- 10. structured profile + prompt --------------------------------------------------------

def test_profile_report_renders_only_filled_fields():
    empty = ProfileReport()
    assert empty.is_empty() and "no profiler" in empty.render()
    report = ProfileReport(vector_width_bits=512, missed_loops=["line 6: not vectorized: control flow"], achieved_bandwidth_fraction=0.42)
    text = report.render()
    assert "512-bit" in text and "42%" in text and "NOT vectorized" in text and "occupancy" not in text


def test_prompt_contains_verdict_dead_ends_workload_and_profile():
    bundle = build_operator("softmax", (64, 128))
    generator = LLMGenerator(LLMConfig(model="m", base_url="http://x", api_key="k"))
    report = analyze(PEAKS, flops=5 * 64 * 128, bytes_moved=8 * 64 * 128, measured_ms=0.05)
    verdict = verdict_for(report, 0.05, 0.85)
    best = _ok(
        1, 0.05, origin="autotune", roofline=report.to_dict(), scaled_ulp_error=1.0, max_abs_error=1e-7, ab_speedup=3.0,
        cpu_wall_ratio=1.0, thread_utilization=0.25, host_bound=True,
        profile=ProfileReport(vector_width_bits=256, missed_loops=["line 3: not vectorized: call"]).to_dict(),
        per_shape={"primary": {"shape": [64, 128], "weight": 1, "latency_ms": 0.05}},
    )
    prompt = generator._build_prompt(
        bundle.spec, "cpu", "rules", "// src", best, None, round_index=2, sample_index=1, peaks_summary="peaks",
        verdict=verdict, dead_ends=["CHUNK 64->1 -> regression +698%"],
    )
    assert "# Roofline verdict" in prompt and verdict.bound in prompt
    assert "Known dead ends" in prompt and "CHUNK 64->1" in prompt
    assert "PRECISION-SENSITIVE" in prompt
    assert "256-bit" in prompt and "NOT vectorized" in prompt
    assert "Interleaved A/B" in prompt and "host/serial-section bound" in prompt
    assert FOCUS_BY_BOUND[verdict.bound][1] in prompt


def test_fused_spec_composes_stage_references():
    bundle = build_operator("matmul_bias_relu", (8, 6, 4))
    assert bundle.spec.fused_stages == ("matmul", "bias_relu")
    assert [part.spec.name for part in bundle.fusion_parts] == ["matmul", "bias_relu"]
    import numpy as np

    rng = np.random.default_rng(0)
    a, b, bias = rng.standard_normal((8, 4)), rng.standard_normal((4, 6)), rng.standard_normal(6)
    expected = np.maximum(a @ b + bias, 0.0)
    assert np.allclose(bundle.spec.reference((a, b, bias), (8, 6, 4)), expected)
