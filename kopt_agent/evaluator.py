"""Turns Candidates into TrialResults.

Single candidate: compile -> verify on every test case -> full timing.

Batch (tiered, cheaper): compile + verify every candidate in parallel (they are subprocess
bound, so threads are enough), then time the survivors coarsely one at a time, and only
re-time the top-k with the full repeat count. Slow or broken candidates never get the
expensive treatment.
"""

from __future__ import annotations

import enum
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kopt_agent.backends.base import Backend, CompileResult
from kopt_agent.candidate import Candidate
from kopt_agent.roofline import MachinePeaks, analyze
from kopt_agent.spec import OperatorSpec, TestCase


class TrialStatus(str, enum.Enum):
    OK = "ok"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    INCORRECT = "incorrect"


@dataclass
class TrialResult:
    trial_id: int
    candidate_label: str
    origin: str
    fingerprint: str
    status: TrialStatus
    message: str = ""
    latency_ms_median: float | None = None
    latency_ms_min: float | None = None
    latency_ms_stdev: float | None = None
    gflops: float | None = None
    gbps: float | None = None
    max_abs_error: float | None = None
    compile_seconds: float = 0.0
    params: dict = field(default_factory=dict)
    # "full" = benchmark-grade timing; "quick" = coarse screening timing (not eligible for best)
    timing_tier: str = "full"
    roofline: dict | None = None
    compiler_notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status is TrialStatus.OK and self.latency_ms_median is not None

    @property
    def is_benchmark_grade(self) -> bool:
        return self.is_valid and self.timing_tier == "full"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class VerificationOutcome:
    ok: bool
    status: TrialStatus
    message: str = ""
    max_abs_error: float = 0.0


@dataclass
class _Verified:
    """A candidate that compiled and passed every test case, ready for timing."""

    candidate: Candidate
    base: dict
    artifact: Path
    compiled: CompileResult
    max_abs_error: float


class Evaluator:
    def __init__(
        self,
        spec: OperatorSpec,
        backend: Backend,
        warmup: int = 3,
        repeats: int = 15,
        run_timeout_seconds: float = 60.0,
        seed: int = 0,
        peaks: MachinePeaks | None = None,
        workers: int = 1,
        quick_repeats: int = 3,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.warmup = warmup
        self.repeats = repeats
        self.quick_repeats = max(1, quick_repeats)
        self.run_timeout_seconds = run_timeout_seconds
        self.peaks = peaks
        self.workers = max(1, workers)
        self.cases = spec.all_cases()
        rng = np.random.default_rng(seed)
        # Inputs and reference outputs are generated once and shared by every trial.
        self.case_inputs: dict[str, list[np.ndarray]] = {}
        self.case_expected: dict[str, np.ndarray] = {}
        for case in self.cases:
            inputs = [spec.generate_input(tensor, rng) for tensor in case.inputs]
            self.case_inputs[case.label] = inputs
            self.case_expected[case.label] = np.asarray(spec.reference(inputs, case.scalars), dtype=case.output.dtype)
        self.primary_case = next(case for case in self.cases if case.label == "primary")
        self._trial_counter = 0
        self._counter_lock = threading.Lock()

    # ---- public API ---------------------------------------------------------------------

    def evaluate(self, candidate: Candidate) -> TrialResult:
        """Full-grade evaluation of one candidate."""
        outcome = self._compile_and_verify(candidate)
        if isinstance(outcome, TrialResult):
            return outcome
        return self._time(outcome, warmup=self.warmup, repeats=self.repeats, tier="full")

    def evaluate_batch(self, candidates: list[Candidate], top_k: int) -> list[TrialResult]:
        """Tiered evaluation. Results are returned in input order."""
        if not candidates:
            return []
        top_k = max(1, top_k)

        if self.workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                outcomes = list(pool.map(self._compile_and_verify, candidates))
        else:
            outcomes = [self._compile_and_verify(candidate) for candidate in candidates]

        results: list[TrialResult | None] = [outcome if isinstance(outcome, TrialResult) else None for outcome in outcomes]
        survivors = [(index, outcome) for index, outcome in enumerate(outcomes) if isinstance(outcome, _Verified)]

        # Coarse screening: fewer than top_k survivors means everyone gets full timing directly.
        if len(survivors) <= top_k:
            for index, verified in survivors:
                results[index] = self._time(verified, warmup=self.warmup, repeats=self.repeats, tier="full")
            return [result for result in results if result is not None]

        quick: list[tuple[int, _Verified, TrialResult]] = []
        for index, verified in survivors:
            quick_result = self._time(verified, warmup=1, repeats=self.quick_repeats, tier="quick")
            results[index] = quick_result
            if quick_result.is_valid:
                quick.append((index, verified, quick_result))

        quick.sort(key=lambda item: item[2].latency_ms_median)
        for index, verified, _ in quick[:top_k]:
            results[index] = self._time(verified, warmup=self.warmup, repeats=self.repeats, tier="full")
        return [result for result in results if result is not None]

    # ---- stages -------------------------------------------------------------------------

    def _next_trial_id(self) -> int:
        with self._counter_lock:
            self._trial_counter += 1
            return self._trial_counter

    def _compile_and_verify(self, candidate: Candidate) -> TrialResult | _Verified:
        base = dict(
            trial_id=self._next_trial_id(),
            candidate_label=candidate.short_label(),
            origin=candidate.origin,
            fingerprint=candidate.fingerprint,
            params=dict(candidate.params),
        )
        compiled = self.backend.compile(candidate, self.spec)
        if not compiled.ok or compiled.artifact is None:
            return TrialResult(status=TrialStatus.COMPILE_ERROR, message=compiled.log[-2000:], compile_seconds=compiled.compile_seconds, **base)

        worst_error = 0.0
        # Cheap edge shapes first so an out-of-bounds tile loop fails before the big run.
        for case in sorted(self.cases, key=lambda c: c.output.numel):
            run = self.backend.run(
                compiled.artifact, self.spec, case, self.case_inputs[case.label],
                warmup=0, repeats=0, timeout_seconds=self.run_timeout_seconds,
            )
            if not run.ok or run.output is None:
                status = {"timeout": TrialStatus.TIMEOUT, "memory": TrialStatus.INCORRECT}.get(run.error_kind, TrialStatus.RUNTIME_ERROR)
                return TrialResult(
                    status=status, message=f"[{case.label}] {run.error}", compile_seconds=compiled.compile_seconds,
                    compiler_notes=compiled.optimization_report, **base,
                )
            verification = self._verify(case, run.output)
            worst_error = max(worst_error, verification.max_abs_error)
            if not verification.ok:
                return TrialResult(
                    status=verification.status, message=f"[{case.label}] {verification.message}", max_abs_error=worst_error,
                    compile_seconds=compiled.compile_seconds, compiler_notes=compiled.optimization_report, **base,
                )
        return _Verified(candidate, base, compiled.artifact, compiled, worst_error)

    def _time(self, verified: _Verified, warmup: int, repeats: int, tier: str) -> TrialResult:
        run = self.backend.run(
            verified.artifact, self.spec, self.primary_case, self.case_inputs["primary"],
            warmup=warmup, repeats=repeats, timeout_seconds=self.run_timeout_seconds,
        )
        common = dict(
            compile_seconds=verified.compiled.compile_seconds,
            compiler_notes=verified.compiled.optimization_report,
            max_abs_error=verified.max_abs_error,
            **verified.base,
        )
        if not run.ok or not run.timings_ms:
            status = TrialStatus.TIMEOUT if run.error_kind == "timeout" else TrialStatus.RUNTIME_ERROR
            return TrialResult(status=status, message=f"[timing] {run.error or 'no timings collected'}", **common)

        # Screening uses the minimum: with only a few repeats, one burst of hypervisor/OS noise
        # would otherwise corrupt the median and wrongly eliminate a good schedule. The full tier
        # has enough repeats for the median to be the honest number.
        median_ms = min(run.timings_ms) if tier == "quick" else statistics.median(run.timings_ms)
        seconds = max(median_ms / 1e3, 1e-12)
        gflops = self.spec.flops(self.spec.primary_shape) / seconds / 1e9
        gbps = self.spec.bytes_moved(self.spec.primary_shape) / seconds / 1e9
        roofline = None
        if self.peaks is not None:
            roofline = analyze(self.peaks, self.spec.flops(self.spec.primary_shape), self.spec.bytes_moved(self.spec.primary_shape), gflops, gbps).to_dict()
        return TrialResult(
            status=TrialStatus.OK,
            latency_ms_median=median_ms,
            latency_ms_min=min(run.timings_ms),
            latency_ms_stdev=statistics.pstdev(run.timings_ms) if len(run.timings_ms) > 1 else 0.0,
            gflops=gflops,
            gbps=gbps,
            timing_tier=tier,
            roofline=roofline,
            **common,
        )

    def _verify(self, case: TestCase, output: np.ndarray) -> VerificationOutcome:
        expected = self.case_expected[case.label]
        if output.shape != expected.shape:
            return VerificationOutcome(False, TrialStatus.INCORRECT, f"output shape {output.shape} != {expected.shape}")
        non_finite = ~np.isfinite(output)
        if non_finite.any():
            count = int(non_finite.sum())
            return VerificationOutcome(
                False,
                TrialStatus.INCORRECT,
                f"{count} non-finite output elements (NaN/Inf) - likely unwritten output cells or overflow",
                max_abs_error=float("inf"),
            )
        abs_error = np.abs(output.astype(np.float64) - expected.astype(np.float64))
        tolerance = self.spec.atol + self.spec.rtol * np.abs(expected.astype(np.float64))
        max_abs_error = float(abs_error.max()) if abs_error.size else 0.0
        violations = abs_error > tolerance
        if violations.any():
            worst_index = np.unravel_index(int(np.argmax(abs_error - tolerance)), abs_error.shape)
            return VerificationOutcome(
                False,
                TrialStatus.INCORRECT,
                (
                    f"{int(violations.sum())}/{violations.size} elements outside tolerance "
                    f"(atol={self.spec.atol}, rtol={self.spec.rtol}); worst at index {tuple(int(i) for i in worst_index)}: "
                    f"got {output[worst_index]:.6g}, expected {expected[worst_index]:.6g}"
                ),
                max_abs_error=max_abs_error,
            )
        return VerificationOutcome(True, TrialStatus.OK, max_abs_error=max_abs_error)
