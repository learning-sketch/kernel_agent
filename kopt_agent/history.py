"""Trial bookkeeping: append-only JSONL log, best-so-far tracking, leaderboard rendering."""

from __future__ import annotations

import json
from pathlib import Path

from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import TrialResult, TrialStatus


class History:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.output_dir / "trials.jsonl"
        self.trials_path.write_text("", encoding="utf-8")
        self.records: list[tuple[Candidate, TrialResult]] = []
        self.fingerprints: set[str] = set()
        self.best: tuple[Candidate, TrialResult] | None = None

    def record(self, candidate: Candidate, result: TrialResult) -> bool:
        """Store the trial. Returns True when this trial became the new best."""
        self.records.append((candidate, result))
        self.fingerprints.add(candidate.fingerprint)
        with self.trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**result.to_dict(), "note": candidate.note}) + "\n")
        (self.output_dir / "candidates").mkdir(exist_ok=True)
        (self.output_dir / "candidates" / f"trial_{result.trial_id:03d}_{result.status.value}.c").write_text(
            candidate.source, encoding="utf-8"
        )

        if not result.is_valid:
            return False
        if self.best is None or result.latency_ms_median < self.best[1].latency_ms_median:
            self.best = (candidate, result)
            (self.output_dir / "best.c").write_text(candidate.source, encoding="utf-8")
            return True
        return False

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in TrialStatus}
        for _, result in self.records:
            counts[result.status.value] += 1
        return counts

    def leaderboard(self, limit: int = 10) -> str:
        valid = sorted((result for _, result in self.records if result.is_valid), key=lambda r: r.latency_ms_median)
        if not valid:
            return "no correct kernel found"
        baseline = next((result for _, result in self.records if result.is_valid and result.origin == "baseline"), None)
        header = f"{'#':>3} {'trial':>5} {'latency ms':>11} {'GFLOP/s':>9} {'GB/s':>8} {'speedup':>8}  candidate"
        lines = [header, "-" * len(header)]
        for rank, result in enumerate(valid[:limit], start=1):
            speedup = baseline.latency_ms_median / result.latency_ms_median if baseline else float("nan")
            lines.append(
                f"{rank:>3} {result.trial_id:>5} {result.latency_ms_median:>11.4f} {result.gflops:>9.1f} "
                f"{result.gbps:>8.1f} {speedup:>7.2f}x  {result.candidate_label}"
            )
        return "\n".join(lines)

    def write_summary(self, extra: dict) -> Path:
        best_result = self.best[1].to_dict() if self.best else None
        summary = {
            "best": best_result,
            "best_note": self.best[0].note if self.best else None,
            "trials": len(self.records),
            "status_counts": self.status_counts(),
            **extra,
        }
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path
