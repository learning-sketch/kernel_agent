"""Cross-run knowledge base: every evaluated template configuration is appended to
`results/<op>/knowledge.jsonl`, and later runs (other shapes, same machine or not) start
from the configurations that worked best on the most similar shape instead of from scratch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from kopt_agent.evaluator import TrialResult

TEMPLATE_ORIGINS = ("template-default", "autotune", "evolve", "warm-start")


@dataclass(frozen=True)
class KnowledgeEntry:
    shape: tuple[int, ...]
    hardware: str
    params: dict
    gflops: float
    latency_ms: float


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[KnowledgeEntry] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("status") != "ok" or not record.get("params") or record.get("timing_tier") == "quick":
                        continue
                    self.entries.append(
                        KnowledgeEntry(
                            shape=tuple(int(dim) for dim in record["shape"]),
                            hardware=str(record.get("hardware", "")),
                            params=dict(record["params"]),
                            gflops=float(record["gflops"]),
                            latency_ms=float(record["latency_ms"]),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue  # a corrupt line must not poison the whole knowledge base

    def append(self, shape: tuple[int, ...], hardware: str, result: TrialResult) -> None:
        if result.origin not in TEMPLATE_ORIGINS or not result.params:
            return
        record = {
            "shape": list(shape),
            "hardware": hardware,
            "params": result.params,
            "origin": result.origin,
            "status": result.status.value,
            "timing_tier": result.timing_tier,
            "latency_ms": result.latency_ms_median,
            "gflops": result.gflops,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if result.is_benchmark_grade:
            self.entries.append(KnowledgeEntry(tuple(shape), hardware, dict(result.params), result.gflops, result.latency_ms_median))

    def warm_start_params(self, shape: tuple[int, ...], hardware: str, limit: int) -> list[dict]:
        """Best distinct configurations from the most similar (shape, hardware) seen before."""
        if limit <= 0 or not self.entries:
            return []

        def shape_distance(entry: KnowledgeEntry) -> float:
            if len(entry.shape) != len(shape):
                return math.inf
            return sum(abs(math.log2(max(a, 1)) - math.log2(max(b, 1))) for a, b in zip(entry.shape, shape))

        # Same hardware first (tile sizes track cache sizes), then closest shape, then fastest.
        ranked = sorted(self.entries, key=lambda entry: (entry.hardware != hardware, shape_distance(entry), -entry.gflops))
        chosen: list[dict] = []
        seen: set[str] = set()
        for entry in ranked:
            if shape_distance(entry) == math.inf:
                continue
            signature = json.dumps(entry.params, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            chosen.append(entry.params)
            if len(chosen) >= limit:
                break
        return chosen
