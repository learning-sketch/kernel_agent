"""Turns a Candidate into a TrialResult: compile, verify on every test case, then benchmark."""

from __future__ import annotations

import enum
import statistics
from dataclasses import asdict, dataclass, field

import numpy as np

from kopt_agent.backends.base import Backend
from kopt_agent.candidate import Candidate
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

    @property
    def is_valid(self) -> bool:
        return self.status is TrialStatus.OK and self.latency_ms_median is not None

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


class Evaluator:
    def __init__(
        self,
        spec: OperatorSpec,
        backend: Backend,
        warmup: int = 3,
        repeats: int = 15,
        run_timeout_seconds: float = 60.0,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.warmup = warmup
        self.repeats = repeats
        self.run_timeout_seconds = run_timeout_seconds
        self.cases = spec.all_cases()
        rng = np.random.default_rng(seed)
        # Inputs and reference outputs are generated once and shared by every trial, so a
        # trial costs one compile + one process per case instead of re-running numpy each time.
        self.case_inputs: dict[str, list[np.ndarray]] = {}
        self.case_expected: dict[str, np.ndarray] = {}
        for case in self.cases:
            inputs = [spec.generate_input(tensor, rng) for tensor in case.inputs]
            self.case_inputs[case.label] = inputs
            self.case_expected[case.label] = np.asarray(spec.reference(inputs, case.scalars), dtype=case.output.dtype)
        self._trial_counter = 0

    def evaluate(self, candidate: Candidate) -> TrialResult:
        self._trial_counter += 1
        base = dict(
            trial_id=self._trial_counter,
            candidate_label=candidate.short_label(),
            origin=candidate.origin,
            fingerprint=candidate.fingerprint,
            params=dict(candidate.params),
        )

        compiled = self.backend.compile(candidate, self.spec)
        if not compiled.ok or compiled.artifact is None:
            return TrialResult(status=TrialStatus.COMPILE_ERROR, message=compiled.log[-2000:], compile_seconds=compiled.compile_seconds, **base)

        worst_error = 0.0
        primary_timings: list[float] = []
        for case in self.cases:
            is_primary = case.label == "primary"
            run = self.backend.run(
                compiled.artifact,
                self.spec,
                case,
                self.case_inputs[case.label],
                warmup=self.warmup if is_primary else 0,
                repeats=self.repeats if is_primary else 0,
                timeout_seconds=self.run_timeout_seconds,
            )
            if not run.ok or run.output is None:
                status = {"timeout": TrialStatus.TIMEOUT, "memory": TrialStatus.INCORRECT}.get(run.error_kind, TrialStatus.RUNTIME_ERROR)
                return TrialResult(status=status, message=f"[{case.label}] {run.error}", compile_seconds=compiled.compile_seconds, **base)

            verification = self._verify(case, run.output)
            worst_error = max(worst_error, verification.max_abs_error)
            if not verification.ok:
                return TrialResult(
                    status=verification.status,
                    message=f"[{case.label}] {verification.message}",
                    max_abs_error=worst_error,
                    compile_seconds=compiled.compile_seconds,
                    **base,
                )
            if is_primary:
                primary_timings = run.timings_ms

        if not primary_timings:
            return TrialResult(status=TrialStatus.RUNTIME_ERROR, message="no timings collected", compile_seconds=compiled.compile_seconds, **base)

        median_ms = statistics.median(primary_timings)
        stdev_ms = statistics.pstdev(primary_timings) if len(primary_timings) > 1 else 0.0
        seconds = max(median_ms / 1e3, 1e-12)
        return TrialResult(
            status=TrialStatus.OK,
            latency_ms_median=median_ms,
            latency_ms_min=min(primary_timings),
            latency_ms_stdev=stdev_ms,
            gflops=self.spec.flops(self.spec.primary_shape) / seconds / 1e9,
            gbps=self.spec.bytes_moved(self.spec.primary_shape) / seconds / 1e9,
            max_abs_error=worst_error,
            compile_seconds=compiled.compile_seconds,
            **base,
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
