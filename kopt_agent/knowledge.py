"""Cross-run knowledge base stored in `results/<op>/knowledge.jsonl`.

Positive knowledge: every benchmark-grade template configuration, so later runs (other
shapes, same machine or not) warm-start from what worked on the most similar shape.

Negative knowledge: *directions* that failed or regressed, with a root-cause tag - e.g.
"NC: 256->512" -> "regression -23% (loop at line 41 no longer vectorized)" or an LLM strategy
-> "incorrect: out-of-bounds write". Dead ends are handed to the LLM as a "known not to work"
list and to the mutation search as a pruning set, so the same wall is not hit twice.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kopt_agent.evaluator import TrialResult, TrialStatus

TEMPLATE_ORIGINS = ("template-default", "autotune", "evolve", "warm-start")
REGRESSION_THRESHOLD = 0.10  # a child slower than its parent by more than this is a regression
PRUNE_MIN_OBSERVATIONS = 2  # prune a direction only after it failed this many times without ever helping


@dataclass(frozen=True)
class KnowledgeEntry:
    shape: tuple[int, ...]
    hardware: str
    params: dict
    gflops: float
    latency_ms: float


@dataclass
class DeadEnd:
    shape: tuple[int, ...]
    hardware: str
    origin: str
    direction: str  # what was changed / attempted
    cause: str  # root-cause tag
    keys: tuple[str, ...] = ()  # normalized direction keys ("NC:256->512") for pruning


@dataclass
class DirectionStats:
    failures: int = 0
    improvements: int = 0
    causes: Counter = field(default_factory=Counter)


def direction_keys(parent_params: dict, child_params: dict) -> tuple[str, ...]:
    """One key per knob that differs between parent and child, e.g. ('NC:256->512',)."""
    keys = []
    for name in sorted(set(parent_params) | set(child_params)):
        before, after = parent_params.get(name), child_params.get(name)
        if before != after:
            keys.append(f"{name}:{before}->{after}")
    return tuple(keys)


def root_cause(result: TrialResult, parent: TrialResult | None = None) -> str | None:
    """Root-cause tag for a failed or regressed trial; None when the trial is fine."""
    head = (result.message or "").splitlines()[0][:160] if result.message else ""
    if result.status is TrialStatus.COMPILE_ERROR:
        return "compile error: " + (head or "see log")
    if result.status is TrialStatus.TIMEOUT:
        return "timeout / pathological slowness"
    if result.status is TrialStatus.RUNTIME_ERROR:
        return "crash: " + head
    if result.status is TrialStatus.INCORRECT:
        if result.shape_specialized:
            return "shape-specialized: correct only on the benchmark shape"
        return "incorrect: " + head
    if parent is None or not parent.is_valid or not result.is_valid:
        return None
    change = result.objective_ms / parent.objective_ms - 1.0
    if change <= REGRESSION_THRESHOLD:
        return None
    tag = f"regression {change * 100:+.0f}% vs parent"
    # Compare "line N: reason" only (the appended source snippet differs whenever a knob value does),
    # and only trust the diff when neither report was truncated.
    def normalize(entry: str) -> str:
        return entry.split("  //")[0].strip()

    parent_missed = {normalize(line) for line in parent.profile.get("missed_loops", [])}
    truncated = any(note.startswith("...") for note in result.compiler_notes + parent.compiler_notes)
    lost_vectorization = [] if truncated else [line for line in result.profile.get("missed_loops", []) if normalize(line) not in parent_missed]
    if lost_vectorization:
        tag += f" (lost vectorization: {lost_vectorization[0][:80]})"
    elif result.host_bound and not parent.host_bound:
        tag += " (threads idle: parallel efficiency dropped)"
    elif result.roofline and parent.roofline and result.roofline["fraction_of_bandwidth_peak"] > parent.roofline["fraction_of_bandwidth_peak"] * 1.2:
        tag += " (more memory traffic per FLOP: tile no longer fits cache)"
    return tag


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[KnowledgeEntry] = []
        self.dead_ends: list[DeadEnd] = []
        self._directions: dict[tuple[str, str], DirectionStats] = {}  # (hardware, key) -> stats
        if path.exists():
            self._load()

    # ---- persistence --------------------------------------------------------------------

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    kind = record.get("kind", "config")
                    if kind == "dead_end":
                        self._add_dead_end(
                            DeadEnd(
                                shape=tuple(int(dim) for dim in record["shape"]),
                                hardware=str(record.get("hardware", "")),
                                origin=str(record.get("origin", "")),
                                direction=str(record["direction"]),
                                cause=str(record["cause"]),
                                keys=tuple(record.get("keys", [])),
                            )
                        )
                    elif kind == "improvement":
                        for key in record.get("keys", []):
                            self._stats(record.get("hardware", ""), key).improvements += 1
                    else:
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

    def _write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _stats(self, hardware: str, key: str) -> DirectionStats:
        return self._directions.setdefault((hardware, key), DirectionStats())

    def _add_dead_end(self, dead_end: DeadEnd) -> None:
        self.dead_ends.append(dead_end)
        for key in dead_end.keys:
            stats = self._stats(dead_end.hardware, key)
            stats.failures += 1
            stats.causes[dead_end.cause.split(" (")[0]] += 1

    # ---- recording ----------------------------------------------------------------------

    def append(self, shape: tuple[int, ...], hardware: str, result: TrialResult) -> None:
        """Positive knowledge: a template configuration and how it did."""
        if result.origin not in TEMPLATE_ORIGINS or not result.params:
            return
        self._write(
            {
                "kind": "config",
                "shape": list(shape),
                "hardware": hardware,
                "params": result.params,
                "origin": result.origin,
                "status": result.status.value,
                "timing_tier": result.timing_tier,
                "latency_ms": result.latency_ms_median,
                "objective_ms": result.objective_ms,
                "gflops": result.gflops,
                "numeric_grade": result.numeric_grade,
            }
        )
        if result.is_benchmark_grade:
            self.entries.append(KnowledgeEntry(tuple(shape), hardware, dict(result.params), result.gflops, result.latency_ms_median))

    def record_outcome(
        self,
        shape: tuple[int, ...],
        hardware: str,
        result: TrialResult,
        parent: TrialResult | None = None,
        direction: str | None = None,
    ) -> DeadEnd | None:
        """Negative knowledge: classify a trial relative to its parent (if any). Returns the
        DeadEnd that was recorded, or None when the trial was not a dead end."""
        keys = direction_keys(parent.params, result.params) if (parent is not None and parent.params and result.params) else ()
        if direction is None:
            direction = ", ".join(key.replace(":", " ") for key in keys) if keys else (result.candidate_label or result.origin)
        cause = root_cause(result, parent)
        if cause is None:
            if parent is not None and parent.is_valid and result.is_valid and result.objective_ms < parent.objective_ms * (1 - 0.02) and keys:
                self._write({"kind": "improvement", "shape": list(shape), "hardware": hardware, "keys": list(keys)})
                for key in keys:
                    self._stats(hardware, key).improvements += 1
            return None
        dead_end = DeadEnd(tuple(shape), hardware, result.origin, direction, cause, keys)
        self._add_dead_end(dead_end)
        self._write(
            {
                "kind": "dead_end",
                "shape": list(shape),
                "hardware": hardware,
                "origin": result.origin,
                "direction": direction,
                "cause": cause,
                "keys": list(keys),
            }
        )
        return dead_end

    # ---- queries ------------------------------------------------------------------------

    def is_pruned(self, hardware: str, keys: tuple[str, ...]) -> bool:
        """True when every knob change in `keys` is a known dead end on this hardware."""
        if not keys:
            return False
        for key in keys:
            stats = self._directions.get((hardware, key))
            if stats is None or stats.improvements > 0 or stats.failures < PRUNE_MIN_OBSERVATIONS:
                return False
        return True

    def dead_end_summary(self, shape: tuple[int, ...], hardware: str, limit: int = 12) -> list[str]:
        """Human-readable "known not to work" list, most relevant first (same hardware,
        closest shape, most frequently observed)."""
        if not self.dead_ends:
            return []
        grouped: dict[tuple[str, str], list[DeadEnd]] = {}
        for dead_end in self.dead_ends:
            grouped.setdefault((dead_end.direction, dead_end.cause.split(" (")[0]), []).append(dead_end)

        def relevance(item: tuple[tuple[str, str], list[DeadEnd]]) -> tuple:
            group = item[1]
            same_hardware = any(entry.hardware == hardware for entry in group)
            distance = min(_shape_distance(entry.shape, shape) for entry in group)
            return (not same_hardware, distance, -len(group))

        lines = []
        for (direction, _), group in sorted(grouped.items(), key=relevance)[:limit]:
            cause = group[-1].cause
            times = f" (seen {len(group)}x)" if len(group) > 1 else ""
            lines.append(f"{direction} -> {cause}{times}")
        return lines

    def warm_start_params(self, shape: tuple[int, ...], hardware: str, limit: int) -> list[dict]:
        """Best distinct configurations from the most similar (shape, hardware) seen before."""
        if limit <= 0 or not self.entries:
            return []
        # Same hardware first (tile sizes track cache sizes), then closest shape, then fastest.
        ranked = sorted(self.entries, key=lambda entry: (entry.hardware != hardware, _shape_distance(entry.shape, shape), -entry.gflops))
        chosen: list[dict] = []
        seen: set[str] = set()
        for entry in ranked:
            if _shape_distance(entry.shape, shape) == math.inf:
                continue
            signature = json.dumps(entry.params, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            chosen.append(entry.params)
            if len(chosen) >= limit:
                break
        return chosen


def _shape_distance(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if len(a) != len(b):
        return math.inf
    return sum(abs(math.log2(max(x, 1)) - math.log2(max(y, 1))) for x, y in zip(a, b))
