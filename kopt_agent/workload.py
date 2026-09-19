"""Build a WorkloadProfile from a framework/profiler trace.

Accepted inputs (detected from the file extension, `.csv` / `.jsonl` / `.json`):

  CSV with a header row. Recognised columns (case-insensitive):
    - shape:            "512x512x512", "512,512,512" or "[512, 512, 512]"
      or dimension columns named after the operator's scalars (e.g. M,N,K), passed via `dims`
    - count | calls | num_calls | hits:  number of calls for that row (default 1: one row = one call)
    - op | operator | kernel | name:  used to keep only the rows of the requested operator
  JSON Lines: one object per line with the same keys (shape may be a list).
  JSON: an existing profile ({"shapes": [{"shape": [...], "count": n}, ...]}) is loaded as is.

Rows with a non-positive count or a non-positive dimension are skipped and reported.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from kopt_agent.spec import WorkloadEntry, WorkloadProfile

SHAPE_KEYS = ("shape", "shapes", "dims", "sizes")
# Deliberately no single-letter aliases: "N" is a GEMM dimension, not a call count.
COUNT_KEYS = ("count", "calls", "num_calls", "hits", "occurrences")
OP_KEYS = ("op", "operator", "kernel", "name")
SHAPE_SPLIT = re.compile(r"[x×,;\s]+")


@dataclass
class IngestReport:
    profile: WorkloadProfile | None
    rows_seen: int = 0
    rows_used: int = 0
    rows_skipped: int = 0
    rows_filtered_out: int = 0
    problems: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"rows: {self.rows_seen} seen, {self.rows_used} used, {self.rows_filtered_out} other operators, {self.rows_skipped} skipped",
        ]
        if self.profile is not None:
            lines.append(f"profile: {len(self.profile.entries)} distinct shapes, {self.profile.total_calls} calls")
            for entry in sorted(self.profile.entries, key=lambda e: -e.count)[:10]:
                lines.append(f"  {'x'.join(map(str, entry.shape)):<24} x{entry.count}")
        if self.problems:
            lines.append("problems (first 10):")
            lines.extend(f"  {problem}" for problem in self.problems[:10])
        return "\n".join(lines)


def parse_shape(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, (list, tuple)):
        return tuple(int(value) for value in text)
    cleaned = str(text).strip().strip("[]()")
    parts = [part for part in SHAPE_SPLIT.split(cleaned) if part]
    if not parts:
        raise ValueError(f"empty shape '{text}'")
    return tuple(int(part) for part in parts)


def _lookup(row: dict, keys: Iterable[str]):
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _row_shape(row: dict, dims: Sequence[str] | None) -> tuple[int, ...]:
    if dims:
        lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
        missing = [dim for dim in dims if dim.lower() not in lowered or lowered[dim.lower()] in (None, "")]
        if not missing:
            return tuple(int(float(lowered[dim.lower()])) for dim in dims)
        raw = _lookup(row, SHAPE_KEYS)
        if raw is None:
            raise ValueError(f"missing dimension column(s) {missing}")
        return parse_shape(raw)
    raw = _lookup(row, SHAPE_KEYS)
    if raw is None:
        raise ValueError("no shape column (expected one of shape/dims or dimension columns via --dims)")
    return parse_shape(raw)


def _row_count(row: dict, dims: Sequence[str] | None) -> int:
    reserved = {dim.lower() for dim in dims or ()}
    raw = _lookup(row, [key for key in COUNT_KEYS if key not in reserved])
    return 1 if raw is None else int(float(raw))


def _iter_rows(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["shapes"] if isinstance(data, dict) and "shapes" in data else data
        for raw in rows:
            if isinstance(raw, dict):
                yield raw
            else:
                shape, count = raw
                yield {"shape": shape, "count": count}
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def ingest_trace(path: Path, operator: str | None = None, dims: Sequence[str] | None = None) -> IngestReport:
    """Aggregate a trace into (shape, count) entries, optionally keeping only rows of `operator`."""
    report = IngestReport(profile=None)
    counts: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(_iter_rows(Path(path)), start=1):
        report.rows_seen += 1
        if operator is not None:
            row_operator = _lookup(row, OP_KEYS)
            if row_operator is not None and str(row_operator).strip().lower() != operator.lower():
                report.rows_filtered_out += 1
                continue
        try:
            shape = _row_shape(row, dims)
            count = _row_count(row, dims)
        except (ValueError, TypeError) as error:
            report.rows_skipped += 1
            report.problems.append(f"row {index}: {error}")
            continue
        if count <= 0 or any(dim <= 0 for dim in shape):
            report.rows_skipped += 1
            report.problems.append(f"row {index}: non-positive shape/count {shape} x{count}")
            continue
        counts[shape] = counts.get(shape, 0) + count
        report.rows_used += 1
    if counts:
        report.profile = WorkloadProfile([WorkloadEntry(shape, count) for shape, count in counts.items()])
    return report


def save_profile(profile: WorkloadProfile, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"shapes": [{"shape": list(entry.shape), "count": entry.count} for entry in profile.entries]}, indent=2),
        encoding="utf-8",
    )
    return path
