"""Trial bookkeeping: append-only JSONL log, best-so-far selection (with precision gating),
leaderboards (per shape and workload-weighted) and a portable parity test for the winner."""

from __future__ import annotations

import json
from pathlib import Path

from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import TrialResult, TrialStatus
from kopt_agent.spec import OperatorSpec


def candidate_record(candidate: Candidate) -> dict:
    """JSON view of a candidate minus its source (which is stored as a .c file)."""
    return {
        "origin": candidate.origin,
        "params": dict(candidate.params),
        "extra_compile_flags": list(candidate.extra_compile_flags),
        "fast_path_predicate": candidate.fast_path_predicate,
        "note": candidate.note,
        "fingerprint": candidate.fingerprint,
    }


class History:
    def __init__(
        self,
        output_dir: Path,
        spec: OperatorSpec | None = None,
        allow_reduced_precision: bool = False,
        compile_command: list[str] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.allow_reduced_precision = allow_reduced_precision
        self.compile_command = compile_command or []
        self.trials_path = self.output_dir / "trials.jsonl"
        self.trials_path.write_text("", encoding="utf-8")
        self.records: list[tuple[Candidate, TrialResult]] = []
        self.fingerprints: set[str] = set()
        self.best: tuple[Candidate, TrialResult] | None = None
        # Fastest benchmark-grade candidate that was *excluded* from best because of its
        # numeric grade; surfaced as the precision <-> speed trade-off.
        self.best_reduced: tuple[Candidate, TrialResult] | None = None
        self.baseline: TrialResult | None = None

    @property
    def precision_gated(self) -> bool:
        return bool(self.spec is not None and self.spec.precision_sensitive and not self.allow_reduced_precision)

    def eligible_for_best(self, result: TrialResult) -> bool:
        if not result.is_benchmark_grade:
            return False
        if result.is_reduced_precision and self.precision_gated:
            return False
        return True

    def record(self, candidate: Candidate, result: TrialResult) -> bool:
        """Store the trial. Returns True when this trial became the new best.

        Only benchmark-grade (full timing) results can become the best: quick screening
        timings are too noisy to rank the winner. For precision-sensitive operators a
        reduced-precision result is tracked separately and never silently wins.
        """
        self.records.append((candidate, result))
        self.fingerprints.add(candidate.fingerprint)
        with self.trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**result.to_dict(), "note": candidate.note}) + "\n")
        (self.output_dir / "candidates").mkdir(exist_ok=True)
        (self.output_dir / "candidates" / f"trial_{result.trial_id:03d}_{result.status.value}.c").write_text(
            candidate.source, encoding="utf-8"
        )
        if result.origin == "baseline" and result.is_valid and self.baseline is None:
            self.baseline = result

        if not result.is_benchmark_grade:
            return False
        if not self.eligible_for_best(result):
            if self.best_reduced is None or result.objective_ms < self.best_reduced[1].objective_ms:
                self.best_reduced = (candidate, result)
            return False
        if self.best is None or result.objective_ms < self.best[1].objective_ms:
            self.best = (candidate, result)
            (self.output_dir / "best.c").write_text(candidate.source, encoding="utf-8")
            self._write_parity_test(candidate, result)
            return True
        return False

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in TrialStatus}
        for _, result in self.records:
            counts[result.status.value] += 1
        return counts

    def valid_results(self, origins: tuple[str, ...] | None = None) -> list[TrialResult]:
        results = [result for _, result in self.records if result.is_valid and (origins is None or result.origin in origins)]
        return sorted(results, key=lambda result: result.objective_ms)

    # ---- reporting ----------------------------------------------------------------------

    def leaderboard(self, limit: int = 10) -> str:
        valid = self.valid_results()
        if not valid:
            return "no correct kernel found"
        baseline = self.baseline
        weighted = self.spec is not None and self.spec.workload is not None
        objective_header = "weighted ms" if weighted else "latency ms"
        header = (
            f"{'#':>3} {'trial':>5} {objective_header:>12} {'primary ms':>10} {'GFLOP/s':>8} {'speedup':>8} "
            f"{'A/B':>6} {'roof%':>6} {'grade':<16} candidate"
        )
        lines = [header, "-" * len(header)]
        for rank, result in enumerate(valid[:limit], start=1):
            speedup = baseline.objective_ms / result.objective_ms if baseline else float("nan")
            roof = f"{result.roofline['fraction_of_attainable'] * 100:5.0f}%" if result.roofline else "   n/a"
            ab = f"{result.ab_speedup:5.2f}x" if result.ab_speedup else "     -"
            tier = "" if result.timing_tier == "full" else " (quick)"
            excluded = " [excluded: reduced precision]" if (result.is_reduced_precision and self.precision_gated) else ""
            lines.append(
                f"{rank:>3} {result.trial_id:>5} {result.objective_ms:>12.4f} {result.latency_ms_median:>10.4f} {result.gflops:>8.1f} "
                f"{speedup:>7.2f}x {ab} {roof} {result.numeric_grade or '?':<16} {result.candidate_label}{tier}{excluded}"
            )
        if weighted and self.best is not None:
            lines.append("")
            lines.append(self.per_shape_table(self.best[1]))
        if self.best_reduced is not None and self.best is not None and self.best_reduced[1].objective_ms < self.best[1].objective_ms:
            reduced_candidate, reduced_result = self.best_reduced
            gain = self.best[1].objective_ms / reduced_result.objective_ms
            lines.append("")
            lines.append(
                f"precision <-> speed: trial {reduced_result.trial_id} ({reduced_candidate.short_label()}) is {gain:.2f}x faster "
                f"than the best but graded {reduced_result.numeric_grade} (scaled ULP error {reduced_result.scaled_ulp_error:.1f}, "
                f"bitwise match {reduced_result.bitwise_match_rate * 100:.1f}%). Re-run with --allow-reduced-precision to accept it."
            )
        return "\n".join(lines)

    def per_shape_table(self, result: TrialResult) -> str:
        """Per-shape latency and speedup of one result against the baseline, plus the weighted total."""
        baseline = self.baseline
        header = f"{'shape':<20} {'calls':>7} {'latency ms':>11} {'baseline ms':>12} {'speedup':>8} {'weighted ms':>12}  roofline"
        lines = [f"per-shape breakdown of trial {result.trial_id}:", header, "-" * len(header)]
        for label, entry in result.per_shape.items():
            base_entry = baseline.per_shape.get(label) if baseline else None
            base_ms = base_entry["latency_ms"] if base_entry else None
            speedup = f"{base_ms / entry['latency_ms']:7.2f}x" if base_ms else "       -"
            roof = entry.get("roofline")
            roof_text = f"{roof['bound']}-bound, {roof['fraction_of_attainable'] * 100:.0f}% of ceiling" if roof else ""
            lines.append(
                f"{'x'.join(map(str, entry['shape'])):<20} {entry['weight']:>7} {entry['latency_ms']:>11.4f} "
                f"{(base_ms if base_ms else float('nan')):>12.4f} {speedup} {entry['weight'] * entry['latency_ms']:>12.4f}  {roof_text}"
            )
        if baseline:
            lines.append(
                f"{'weighted total':<20} {sum(e['weight'] for e in result.per_shape.values()):>7} {'':>11} {baseline.objective_ms:>12.4f} "
                f"{baseline.objective_ms / result.objective_ms:7.2f}x {result.objective_ms:>12.4f}"
            )
        return "\n".join(lines)

    def write_summary(self, extra: dict) -> Path:
        summary = {
            "best": self.best[1].to_dict() if self.best else None,
            "best_note": self.best[0].note if self.best else None,
            "best_candidate": candidate_record(self.best[0]) if self.best else None,
            "best_reduced_precision": self.best_reduced[1].to_dict() if self.best_reduced else None,
            "trials": len(self.records),
            "status_counts": self.status_counts(),
            **extra,
        }
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    # ---- parity test --------------------------------------------------------------------

    def _write_parity_test(self, candidate: Candidate, result: TrialResult) -> None:
        if self.spec is None:
            return
        self.write_parity_test(candidate, result, self.output_dir / "parity_test.py", kernel_filename="best.c")

    def parity_compile_command(self, candidate: Candidate) -> list[str]:
        """The portable compile command with the candidate's own extra flags spliced in."""
        compile_command = self.compile_command or ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "{source}", "-lm", "-o", "{artifact}"]
        command_with_flags = []
        for token in compile_command:
            command_with_flags.append(token)
            if token == "-fPIC":
                command_with_flags.extend(candidate.extra_compile_flags)
        if candidate.extra_compile_flags and "-fPIC" not in compile_command:
            index = command_with_flags.index("{source}") if "{source}" in command_with_flags else len(command_with_flags)
            command_with_flags[index:index] = list(candidate.extra_compile_flags)
        return command_with_flags

    def write_parity_test(self, candidate: Candidate, result: TrialResult, path: Path, kernel_filename: str) -> Path:
        """Emit the standalone parity test next to `kernel_filename` (the kernel source in the
        same directory as the test)."""
        spec = self.spec
        if spec is None:
            raise ValueError("a spec is required to write a parity test")
        shapes = [list(spec.primary_shape)] + [list(shape) for shape, _ in spec.timing_shapes()] + [list(shape) for shape in spec.edge_shapes]
        unique_shapes = []
        for shape in shapes:
            if shape not in unique_shapes:
                unique_shapes.append(shape)
        script = PARITY_TEMPLATE.format(
            operator=spec.name,
            dtype=spec.dtype.name,
            output_dtype=repr(spec.out_dtype.name if spec.output_dtype is not None else None),
            accumulate_dtype=repr(spec.acc_dtype.name if spec.accumulate_dtype is not None else None),
            precision=spec.precision_label(),
            symbol=spec.symbol,
            kernel_file=kernel_filename,
            scalar_names=json.dumps(list(spec.scalar_names)),
            shapes=json.dumps(unique_shapes),
            compile_command=json.dumps(self.parity_compile_command(candidate)),
            atol=repr(spec.policy.atol),
            rtol=repr(spec.policy.rtol),
            fast_path=repr(candidate.fast_path_predicate),
            trial_id=result.trial_id,
            grade=result.numeric_grade,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8")
        return path


PARITY_TEMPLATE = '''"""Parity / regression test for the winning kernel of operator `{operator}` ({precision}), trial {trial_id}.

Standalone apart from numpy, a C compiler and the `ops` / `kopt_agent` packages of the
kernel-opt-agent repository for the reference implementation (either `pip install` that repo or
point KOPT_REPO at a checkout). Run with `pytest parity_test.py` (or `python parity_test.py`)
after integrating {kernel_file} somewhere else to make sure it still matches the reference on the
benchmark, workload and edge shapes, on special values (NaN / Inf / signed zero / denormals), and
that it neither modifies its const inputs nor depends on the prior contents of the output buffer.
Numeric grade at the time of writing: {grade}.
"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _locate_repo():
    override = os.environ.get("KOPT_REPO")
    if override:
        return Path(override)
    for ancestor in [HERE, *HERE.parents]:
        if (ancestor / "ops" / "__init__.py").exists() and (ancestor / "kopt_agent").is_dir():
            return ancestor
    return None


REPO = _locate_repo()
if REPO is not None:
    sys.path.insert(0, str(REPO))

from ops import build_operator  # noqa: E402  (falls back to the installed package)

OPERATOR = "{operator}"
DTYPE = "{dtype}"
OUTPUT_DTYPE = {output_dtype}
ACCUMULATE_DTYPE = {accumulate_dtype}
SYMBOL = "{symbol}"
KERNEL_FILE = "{kernel_file}"
SCALAR_NAMES = {scalar_names}
SHAPES = {shapes}
COMPILE_COMMAND = {compile_command}
ATOL, RTOL = {atol}, {rtol}
FAST_PATH_PREDICATE = {fast_path}
POISON = 0x5C
ALIGNMENT = 64


def _compile():
    artifact = Path(tempfile.mkdtemp()) / "kernel.so"
    command = [token.format(source=str(HERE / KERNEL_FILE), artifact=str(artifact)) for token in COMPILE_COMMAND]
    subprocess.run(command, check=True)
    return ctypes.CDLL(str(artifact))


LIBRARY = _compile()
FUNCTION = getattr(LIBRARY, SYMBOL)
FUNCTION.restype = None


def _aligned_like(shape, dtype, alignment=ALIGNMENT):
    """The kernel contract promises 64-byte aligned, contiguous tensors (aligned vector loads
    may be used), so every buffer handed to it is allocated that way."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize if len(shape) else dtype.itemsize
    raw = np.empty(nbytes + alignment, dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


def _aligned_copy(array):
    aligned = _aligned_like(array.shape, array.dtype)
    aligned[...] = array
    return aligned


def _run(bundle, shape, inputs_storage, prefill):
    spec = bundle.spec
    case = spec.make_case(tuple(shape), spec.dtype)
    output = _aligned_like(case.output.shape, case.output.dtype.storage)
    output.view(np.uint8).fill(prefill)
    pointers = [a.ctypes.data_as(ctypes.c_void_p) for a in inputs_storage] + [output.ctypes.data_as(ctypes.c_void_p)]
    FUNCTION.argtypes = [ctypes.c_void_p] * len(pointers) + [ctypes.c_int] * len(case.scalars)
    FUNCTION(*pointers, *[ctypes.c_int(s) for s in case.scalars])
    return case, output


def _check(shape, make_values, strict_special=True):
    bundle = build_operator(OPERATOR, tuple(shape), dtype=DTYPE, output_dtype=OUTPUT_DTYPE, accumulate_dtype=ACCUMULATE_DTYPE)
    spec = bundle.spec
    case = spec.make_case(tuple(shape), spec.dtype)
    rng = np.random.default_rng(1234)
    inputs_storage = []
    for tensor in case.inputs:
        values = make_values(tensor, rng)
        inputs_storage.append(_aligned_copy(tensor.dtype.encode(np.asarray(values, dtype=np.float64))))
    snapshots = [a.copy() for a in inputs_storage]
    decoded = [t.dtype.decode(a) for t, a in zip(case.inputs, inputs_storage)]
    expected = case.output.dtype.decode(case.output.dtype.encode(np.asarray(spec.reference(decoded, case.scalars), dtype=np.float64)))

    _, out_nan = _run(bundle, shape, inputs_storage, 0xFF)
    _, out_poison = _run(bundle, shape, inputs_storage, POISON)
    for tensor, before, after in zip(case.inputs, snapshots, inputs_storage):
        assert np.array_equal(before.view(np.uint8), after.view(np.uint8)), f"kernel modified const input {{tensor.name}}"
    finite = np.isfinite(expected)
    for label, out in (("nan-prefill", out_nan), ("poison-prefill", out_poison)):
        got = case.output.dtype.decode(out)
        same_special = np.array_equal(np.isnan(got), np.isnan(expected)) and np.array_equal(np.isinf(got), np.isinf(expected))
        if strict_special:
            assert same_special, f"{{label}}: NaN/Inf pattern differs from the reference on shape {{shape}}"
        elif not same_special:
            warnings.warn(f"{{label}}: NaN/Inf propagation differs from the reference on shape {{shape}} (fast-math kernel?)")
        assert np.isfinite(got[finite]).all(), f"{{label}}: non-finite output where the reference is finite"
        err = np.abs(got[finite] - expected[finite])
        tol = ATOL + RTOL * np.abs(expected[finite])
        assert (err <= tol).all(), f"{{label}}: {{int((err > tol).sum())}} elements outside tolerance on shape {{shape}}"


def _normal(tensor, rng):
    return rng.standard_normal(tensor.shape)


def _special(tensor, rng):
    values = rng.standard_normal(tensor.shape)
    flat = values.reshape(-1)
    if flat.size >= 4:
        flat[0] = np.nan
        flat[1] = np.inf
        flat[2] = -0.0
        flat[3] = 1e-40  # denormal in fp32 (flushes to zero in 16-bit formats)
    return values


def test_parity_on_all_shapes():
    for shape in SHAPES:
        _check(shape, _normal)


def test_parity_with_special_values():
    # Finite elements must still match exactly to tolerance; NaN/Inf propagation differences
    # are reported as warnings because fast-math kernels legitimately differ there.
    for shape in SHAPES[:2]:
        _check(shape, _special, strict_special=False)


if __name__ == "__main__":
    test_parity_on_all_shapes()
    test_parity_with_special_values()
    print("parity OK for", OPERATOR, DTYPE, "on", len(SHAPES), "shapes")
'''
