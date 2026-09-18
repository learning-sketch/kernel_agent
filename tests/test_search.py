import json
import random

from kopt_agent.evaluator import TrialResult, TrialStatus
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.knowledge import KnowledgeBase
from kopt_agent.roofline import MachinePeaks, analyze


def _generator() -> TemplateGenerator:
    return TemplateGenerator(
        "int v = $A * $B + $C;",
        space={"A": [8, 16, 32, 64], "B": [1, 2, 4], "C": ["x", "y"]},
        default_params={"A": 16, "B": 2, "C": "x"},
        constraint=lambda params: int(params["A"]) * int(params["B"]) <= 128,
    )


def test_mutate_moves_to_unseen_neighbour_and_respects_constraint():
    generator = _generator()
    rng = random.Random(0)
    parent = {"A": 16, "B": 2, "C": "x"}
    seen = {generator.signature(parent)}
    children = []
    for _ in range(20):
        child = generator.mutate(parent, rng, seen)
        if child is None:
            break
        seen.add(generator.signature(child))
        children.append(child)
    assert children, "expected at least one neighbour"
    assert all(int(child["A"]) * int(child["B"]) <= 128 for child in children)
    assert len({generator.signature(child) for child in children}) == len(children)
    # A neighbour differs from the parent in at most two knobs.
    assert all(sum(child[key] != parent[key] for key in parent) <= 2 for child in children)


def test_mutate_returns_none_when_neighbourhood_is_exhausted():
    generator = TemplateGenerator("int v = $A;", space={"A": [1, 2]}, default_params={"A": 1})
    seen = {generator.signature({"A": 1}), generator.signature({"A": 2})}
    assert generator.mutate({"A": 1}, random.Random(0), seen) is None


def _ok_result(params: dict, latency_ms: float, gflops: float, origin: str = "autotune", tier: str = "full") -> TrialResult:
    return TrialResult(
        trial_id=1, candidate_label="x", origin=origin, fingerprint="f", status=TrialStatus.OK,
        latency_ms_median=latency_ms, gflops=gflops, gbps=1.0, params=params, timing_tier=tier,
    )


def test_knowledge_base_prefers_similar_shape_and_same_hardware(tmp_path):
    knowledge = KnowledgeBase(tmp_path / "knowledge.jsonl")
    knowledge.append((256, 256, 256), "hw-a", _ok_result({"MB": 8}, 1.0, 100.0))
    knowledge.append((256, 256, 256), "hw-a", _ok_result({"MB": 16}, 0.5, 200.0))
    knowledge.append((4096, 4096, 4096), "hw-a", _ok_result({"MB": 64}, 9.0, 900.0))
    knowledge.append((512, 512, 512), "hw-b", _ok_result({"MB": 32}, 1.0, 300.0))
    knowledge.append((512, 512, 512), "hw-a", _ok_result({"MB": 128}, 1.0, 50.0, origin="llm"))  # ignored: not a template
    knowledge.append((512, 512, 512), "hw-a", _ok_result({"MB": 4}, 1.0, 999.0, tier="quick"))  # ignored: screening tier

    reloaded = KnowledgeBase(tmp_path / "knowledge.jsonl")
    params = reloaded.warm_start_params((512, 512, 512), "hw-a", limit=3)
    assert params == [{"MB": 16}, {"MB": 8}, {"MB": 64}]  # same hardware first, closest shape, then fastest
    assert reloaded.warm_start_params((512, 512, 512), "hw-a", limit=4)[3] == {"MB": 32}  # other hardware last
    assert {"MB": 128} not in params and {"MB": 4} not in params
    assert reloaded.warm_start_params((512, 512), "hw-a", limit=3) == []  # rank mismatch
    assert reloaded.warm_start_params((512, 512, 512), "hw-a", limit=0) == []


def test_knowledge_base_survives_corrupt_lines(tmp_path):
    path = tmp_path / "knowledge.jsonl"
    path.write_text('not json\n{"shape": [1], "status": "ok"}\n' + json.dumps(
        {"shape": [64, 64, 64], "hardware": "hw", "params": {"A": 1}, "status": "ok", "timing_tier": "full", "latency_ms": 1.0, "gflops": 2.0}
    ) + "\n", encoding="utf-8")
    knowledge = KnowledgeBase(path)
    assert len(knowledge.entries) == 1


def test_roofline_analysis_classifies_bound():
    peaks = MachinePeaks(compute_gflops=1000.0, bandwidth_gbps=100.0, dispatch_overhead_ms=0.001, call_overhead_ms=0.0002, source="test")
    flops, bytes_moved = 2 * 512**3, 3 * 512 * 512 * 4
    compute_bound = analyze(peaks, flops=flops, bytes_moved=bytes_moved, measured_ms=2 * (flops / 1000.0 / 1e6 + 0.0002))
    assert compute_bound.bound == "compute"
    assert abs(compute_bound.fraction_of_attainable - 0.5) < 1e-9
    memory_bound = analyze(peaks, flops=5 * 4096 * 1024, bytes_moved=8 * 4096 * 1024, measured_ms=1.0)
    assert memory_bound.bound == "memory"
    assert memory_bound.memory_time_ms > memory_bound.compute_time_ms
    assert memory_bound.headroom_speedup > 1.0
