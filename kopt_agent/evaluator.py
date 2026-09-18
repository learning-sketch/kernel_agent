"""Turns Candidates into TrialResults.

Single candidate: compile -> verify on every test case -> benchmark-grade timing on every
workload shape (interleaved A/B against the baseline when available).

Batch (tiered, cheaper): compile + verify every candidate in parallel, time the survivors
coarsely on the primary shape one at a time, and only re-time the top-k fully.

Verification is deliberately paranoid (see runner.py): NaN and poison prefills, canary pages,
input snapshots, fast-path activation assertions, numeric grading against the fp64 reference
rounded to the target dtype, and shape coverage so a kernel that only works on the benchmark
shape is called out as shape-specialized instead of quietly failing.
"""

from __future__ import annotations

import enum
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kopt_agent.backends.base import Backend, CompileResult, ProfileReport, RunResult
from kopt_agent.candidate import Candidate
from kopt_agent.roofline import MachinePeaks, analyze
from kopt_agent.spec import OperatorSpec, TestCase, evaluate_fast_path_predicate


class TrialStatus(str, enum.Enum):
    OK = "ok"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    INCORRECT = "incorrect"


class NumericGrade(str, enum.Enum):
    BITWISE = "bitwise-equal"
    TIGHT = "within-N-ULP"
    REDUCED = "reduced-precision"


GRADE_ORDER = {NumericGrade.BITWISE: 0, NumericGrade.TIGHT: 1, NumericGrade.REDUCED: 2}
# A candidate is "reduced-precision" when its scaled-ULP error exceeds the policy threshold AND
# this multiple of the baseline's own error (the baseline defines what the dtype can achieve).
PRECISION_ANCHOR_FACTOR = 2.0
# Timing samples more than this factor above the fastest one are treated as external noise
# (preempted vCPU, hypervisor stall) rather than kernel behaviour.
NOISE_FACTOR = 1.5


def robust_latency(timings_ms: list[float]) -> tuple[float, bool]:
    """Latency estimate that survives positive-only noise bursts.

    Returns (latency, noisy). Samples within NOISE_FACTOR of the fastest form the "fast mode";
    when more than a quarter of the samples fall outside it the distribution is bimodal
    (kernel time plus preempted samples): the median of the fast mode is used and the run is
    flagged noisy. Otherwise the plain median is returned."""
    if not timings_ms:
        raise ValueError("no timings")
    fastest = min(timings_ms)
    fast_mode = [t for t in timings_ms if t <= fastest * NOISE_FACTOR]
    if len(fast_mode) >= 0.75 * len(timings_ms):
        return statistics.median(timings_ms), False
    return statistics.median(fast_mode), True


@dataclass
class TrialResult:
    trial_id: int
    candidate_label: str
    origin: str
    fingerprint: str
    status: TrialStatus
    message: str = ""
    # Primary-shape numbers (display) ...
    latency_ms_median: float | None = None
    latency_ms_min: float | None = None
    latency_ms_stdev: float | None = None
    gflops: float | None = None
    gbps: float | None = None
    # ... and the search objective: count-weighted total time over the workload shapes
    # (equals the primary latency when no workload profile is given).
    objective_ms: float | None = None
    per_shape: dict[str, dict] = field(default_factory=dict)
    max_abs_error: float | None = None
    numeric_grade: str | None = None
    scaled_ulp_error: float | None = None
    bitwise_match_rate: float | None = None
    shape_coverage: float | None = None  # fraction of correctness cases passed
    shape_specialized: bool = False  # passed the benchmark shape, failed other shapes
    fast_path: dict = field(default_factory=dict)
    ab_speedup: float | None = None  # interleaved A/B vs baseline, workload-weighted
    ab_reference_ms: float | None = None
    cpu_wall_ratio: float | None = None
    thread_utilization: float | None = None
    host_bound: bool = False
    compile_seconds: float = 0.0
    params: dict = field(default_factory=dict)
    # "full" = benchmark-grade timing; "quick" = coarse screening timing (not eligible for best)
    timing_tier: str = "full"
    roofline: dict | None = None
    compiler_notes: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status is TrialStatus.OK and self.objective_ms is not None

    @property
    def is_benchmark_grade(self) -> bool:
        return self.is_valid and self.timing_tier == "full"

    @property
    def is_reduced_precision(self) -> bool:
        return self.numeric_grade == NumericGrade.REDUCED.value

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
    grade: NumericGrade = NumericGrade.BITWISE
    scaled_ulp_error: float = 0.0
    bitwise_match_rate: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class _Verified:
    """A candidate that compiled and passed every test case, ready for timing."""

    candidate: Candidate
    base: dict
    artifact: Path
    compiled: CompileResult
    max_abs_error: float
    grade: NumericGrade
    scaled_ulp_error: float
    bitwise_match_rate: float
    fast_path: dict
    notes: list[str]


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
        self.reference_artifact: Path | None = None  # baseline .so for interleaved A/B timing
        self.cases = spec.all_cases()
        self.timing_cases = [case for case in self.cases if case.weight > 0]
        self.primary_case = next(case for case in self.cases if case.label == "primary")
        rng = np.random.default_rng(seed)
        # Inputs (storage dtype) and rounded fp64 references are generated once per case.
        self.case_inputs: dict[str, list[np.ndarray]] = {}
        self.case_expected: dict[str, np.ndarray] = {}  # decoded float64, already rounded to the dtype
        self.case_expected_storage: dict[str, np.ndarray] = {}
        dtype = spec.dtype
        for case in self.cases:
            inputs = [spec.generate_input(tensor, rng) for tensor in case.inputs]
            decoded_inputs = [tensor.dtype.decode(array) for tensor, array in zip(case.inputs, inputs)]
            reference = np.asarray(spec.reference(decoded_inputs, case.scalars), dtype=np.float64)
            storage = case.output.dtype.encode(reference)
            self.case_inputs[case.label] = inputs
            self.case_expected_storage[case.label] = storage
            self.case_expected[case.label] = case.output.dtype.decode(storage)
        self._trial_counter = 0
        self._counter_lock = threading.Lock()
        self._dtype = dtype
        # Scaled-ULP error of the baseline kernel; set by `calibrate_precision` after the baseline ran.
        self.precision_anchor_ulp: float | None = None

    # ---- public API ---------------------------------------------------------------------

    def tight_ulp_threshold(self) -> float:
        threshold = float(self.spec.policy.tight_ulp)
        if self.precision_anchor_ulp is not None:
            threshold = max(threshold, PRECISION_ANCHOR_FACTOR * self.precision_anchor_ulp)
        return threshold

    def calibrate_precision(self, baseline: TrialResult) -> None:
        """Anchor the within-N-ULP threshold on the baseline: a naive fp32 (or fp16/bf16) kernel
        already differs from the fp64 reference by summation order, and that level of error is
        by definition acceptable. The baseline's own grade is re-derived accordingly."""
        if not baseline.is_valid or baseline.scaled_ulp_error is None:
            return
        self.precision_anchor_ulp = baseline.scaled_ulp_error
        if baseline.numeric_grade == NumericGrade.REDUCED.value:
            baseline.numeric_grade = NumericGrade.TIGHT.value

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

        quick.sort(key=lambda item: item[2].objective_ms)
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
        common = dict(compile_seconds=compiled.compile_seconds, compiler_notes=compiled.optimization_report, profile=compiled.profile.to_dict(), **base)

        worst_error = 0.0
        worst_grade = NumericGrade.BITWISE
        worst_ulp = 0.0
        min_match_rate = 1.0
        notes: list[str] = []
        failures: list[str] = []
        passed_labels: list[str] = []
        fast_path = {"declared": candidate.fast_path_predicate is not None, "activated_on": [], "missing_on": [], "unexpected_on": []}

        # Cheap edge shapes first so an out-of-bounds tile loop fails before the big run.
        for case in sorted(self.cases, key=lambda c: c.output.numel):
            run = self.backend.run(
                compiled.artifact, self.spec, case, self.case_inputs[case.label],
                warmup=0, repeats=0, timeout_seconds=self.run_timeout_seconds, verify=True,
            )
            if not run.ok or run.output is None:
                if run.error_kind == "memory":
                    failures.append(f"[{case.label}] {run.error}")
                    continue
                status = TrialStatus.TIMEOUT if run.error_kind == "timeout" else TrialStatus.RUNTIME_ERROR
                return TrialResult(status=status, message=f"[{case.label}] {run.error}", shape_coverage=len(passed_labels) / len(self.cases), **common)

            fast_path_problem = self._check_fast_path(candidate, case, run, fast_path)
            if fast_path_problem:
                failures.append(f"[{case.label}] {fast_path_problem}")
                continue

            verification = self._verify(case, run)
            worst_error = max(worst_error, verification.max_abs_error)
            notes.extend(f"[{case.label}] {note}" for note in verification.notes)
            if not verification.ok:
                failures.append(f"[{case.label}] {verification.message}")
                continue
            passed_labels.append(case.label)
            if GRADE_ORDER[verification.grade] > GRADE_ORDER[worst_grade]:
                worst_grade = verification.grade
            worst_ulp = max(worst_ulp, verification.scaled_ulp_error)
            min_match_rate = min(min_match_rate, verification.bitwise_match_rate)

        if failures:
            coverage = len(passed_labels) / len(self.cases)
            specialized = "primary" in passed_labels
            headline = failures[0]
            if specialized:
                headline = (
                    f"shape-specialized kernel: passes the benchmark shape but fails {len(failures)} other shape(s) "
                    f"(shape whitelists are not accepted). First failure: {failures[0]}"
                )
            return TrialResult(
                status=TrialStatus.INCORRECT, message=headline + ("" if len(failures) == 1 else f" (+{len(failures) - 1} more)"),
                max_abs_error=worst_error, shape_coverage=coverage, shape_specialized=specialized, fast_path=fast_path,
                notes=notes + failures[1:], **common,
            )

        if fast_path["declared"] and not fast_path["activated_on"]:
            notes.append("declared fast path never activated on any test shape (predicate false everywhere or flag never set)")

        return _Verified(candidate, base, compiled.artifact, compiled, worst_error, worst_grade, worst_ulp, min_match_rate, fast_path, notes)

    def _check_fast_path(self, candidate: Candidate, case: TestCase, run: RunResult, fast_path: dict) -> str | None:
        if candidate.fast_path_predicate is None:
            return None
        if run.fast_path_active is None:
            return "candidate declares a fast path but does not export `int kopt_fast_path_active`"
        try:
            expected_active = evaluate_fast_path_predicate(candidate.fast_path_predicate, self.spec.scalar_names, case.scalars)
        except (ValueError, SyntaxError) as error:
            return f"invalid fast-path predicate '{candidate.fast_path_predicate}': {error}"
        if expected_active and not run.fast_path_active:
            fast_path["missing_on"].append(case.label)
            return f"fast path declared active for this shape ('{candidate.fast_path_predicate}') but kopt_fast_path_active stayed 0"
        if run.fast_path_active and not expected_active:
            fast_path["unexpected_on"].append(case.label)
        if run.fast_path_active:
            fast_path["activated_on"].append(case.label)
        return None

    def _time(self, verified: _Verified, warmup: int, repeats: int, tier: str) -> TrialResult:
        common = dict(
            compile_seconds=verified.compiled.compile_seconds,
            compiler_notes=verified.compiled.optimization_report,
            max_abs_error=verified.max_abs_error,
            numeric_grade=verified.grade.value,
            scaled_ulp_error=verified.scaled_ulp_error,
            bitwise_match_rate=verified.bitwise_match_rate,
            fast_path=verified.fast_path,
            shape_coverage=1.0,
            notes=list(verified.notes),
            **verified.base,
        )
        cases = self.timing_cases if tier == "full" else [self.primary_case]
        use_ab = tier == "full" and self.reference_artifact is not None and self.reference_artifact != verified.artifact
        per_shape: dict[str, dict] = {}
        weighted_total = 0.0
        weighted_reference = 0.0
        total_weight = 0
        primary_timings: list[float] = []
        primary_run: RunResult | None = None

        for case in cases:
            run = self.backend.run(
                verified.artifact, self.spec, case, self.case_inputs[case.label],
                warmup=warmup, repeats=repeats, timeout_seconds=self.run_timeout_seconds, verify=False,
                reference_artifact=self.reference_artifact if use_ab else None,
            )
            if not run.ok or not run.timings_ms:
                status = TrialStatus.TIMEOUT if run.error_kind == "timeout" else TrialStatus.RUNTIME_ERROR
                return TrialResult(status=status, message=f"[timing {case.label}] {run.error or 'no timings collected'}", **common)

            # Screening uses the minimum: one burst of hypervisor/OS noise would otherwise corrupt a
            # 3-5 sample median and wrongly eliminate a good schedule. Full timing uses the
            # noise-robust median (see robust_latency).
            if tier == "quick":
                latency_ms, noisy = min(run.timings_ms), False
            else:
                latency_ms, noisy = robust_latency(run.timings_ms)
            if noisy:
                inflated = sum(1 for t in run.timings_ms if t > min(run.timings_ms) * NOISE_FACTOR)
                common["notes"].append(f"[{case.label}] noisy timing: {inflated}/{len(run.timings_ms)} samples inflated by external stalls; fast-mode median used")
            shape = case.shape or self.spec.primary_shape
            seconds = max(latency_ms, 1e-12) / 1e3
            flops = self.spec.flops(shape)
            bytes_moved = self.spec.bytes_moved(shape)
            entry = {
                "shape": list(shape),
                "weight": case.weight,
                "latency_ms": latency_ms,
                "gflops": flops / seconds / 1e9,
                "gbps": bytes_moved / seconds / 1e9,
            }
            if self.peaks is not None:
                entry["roofline"] = analyze(self.peaks, flops, bytes_moved, latency_ms).to_dict()
            if run.reference_timings_ms:
                reference_ms, _ = robust_latency(run.reference_timings_ms)
                entry["reference_ms"] = reference_ms
                entry["ab_speedup"] = reference_ms / max(latency_ms, 1e-12)
                weighted_reference += case.weight * reference_ms
            per_shape[case.label] = entry
            weighted_total += case.weight * latency_ms
            total_weight += case.weight
            if case.label == "primary":
                primary_timings = run.timings_ms
                primary_run = run

        if not primary_timings:
            # Primary shape carries weight 0 in this workload: time it once, unweighted, for display.
            run = self.backend.run(
                verified.artifact, self.spec, self.primary_case, self.case_inputs["primary"],
                warmup=warmup, repeats=repeats, timeout_seconds=self.run_timeout_seconds, verify=False,
            )
            if not run.ok or not run.timings_ms:
                return TrialResult(status=TrialStatus.RUNTIME_ERROR, message=f"[timing primary] {run.error}", **common)
            primary_timings, primary_run = run.timings_ms, run

        primary_ms = min(primary_timings) if tier == "quick" else robust_latency(primary_timings)[0]
        primary_shape = self.spec.primary_shape
        primary_seconds = max(primary_ms, 1e-12) / 1e3
        gflops = self.spec.flops(primary_shape) / primary_seconds / 1e9
        gbps = self.spec.bytes_moved(primary_shape) / primary_seconds / 1e9
        roofline = analyze(self.peaks, self.spec.flops(primary_shape), self.spec.bytes_moved(primary_shape), primary_ms).to_dict() if self.peaks else None

        profile = ProfileReport(**verified.compiled.profile.to_dict())
        if roofline is not None:
            profile.achieved_compute_fraction = min(1.0, roofline["fraction_of_compute_peak"])
            profile.achieved_bandwidth_fraction = min(1.0, roofline["fraction_of_bandwidth_peak"])

        objective_ms = weighted_total if total_weight > 0 else primary_ms
        ab_speedup = (weighted_reference / weighted_total) if (weighted_reference > 0 and weighted_total > 0) else None
        cpu_wall_ratio = primary_run.cpu_wall_ratio if primary_run else None
        threads_available = primary_run.threads_available if primary_run else None
        thread_utilization = (cpu_wall_ratio / threads_available) if (cpu_wall_ratio is not None and threads_available) else None
        host_bound = thread_utilization is not None and thread_utilization < 0.5
        if host_bound:
            profile.notes.append(f"threads busy only {thread_utilization * 100:.0f}% of wall time: host/dispatch or serial section limited")

        return TrialResult(
            status=TrialStatus.OK,
            latency_ms_median=primary_ms,
            latency_ms_min=min(primary_timings),
            latency_ms_stdev=statistics.pstdev(primary_timings) if len(primary_timings) > 1 else 0.0,
            gflops=gflops,
            gbps=gbps,
            objective_ms=objective_ms,
            per_shape=per_shape,
            ab_speedup=ab_speedup,
            ab_reference_ms=(weighted_reference if total_weight > 0 else None) if ab_speedup else None,
            cpu_wall_ratio=cpu_wall_ratio,
            thread_utilization=thread_utilization,
            host_bound=host_bound,
            timing_tier=tier,
            roofline=roofline,
            profile=profile.to_dict(),
            **common,
        )

    # ---- numerics -----------------------------------------------------------------------

    def _verify(self, case: TestCase, run: RunResult) -> VerificationOutcome:
        outcome = self._compare(case, run.output)
        if not outcome.ok:
            return outcome
        if run.poison_left_in_output:
            return VerificationOutcome(False, TrialStatus.INCORRECT, "poison prefill survived in the output: some elements were never written", outcome.max_abs_error)
        if run.prefill_dependent:
            if run.poison_output is None:
                return VerificationOutcome(False, TrialStatus.INCORRECT, "output changed with the output prefill (reads uninitialized output memory)", outcome.max_abs_error)
            poison_outcome = self._compare(case, run.poison_output)
            if not poison_outcome.ok:
                return VerificationOutcome(
                    False, TrialStatus.INCORRECT,
                    "output depends on the previous buffer contents (accumulates into / reads uninitialized output): " + poison_outcome.message,
                    max(outcome.max_abs_error, poison_outcome.max_abs_error),
                )
            outcome.notes.append("nondeterministic output across runs (within tolerance)")
        return outcome

    def _compare(self, case: TestCase, output_storage: np.ndarray) -> VerificationOutcome:
        dtype = case.output.dtype
        expected = self.case_expected[case.label]
        expected_storage = self.case_expected_storage[case.label]
        if output_storage.shape != expected.shape:
            return VerificationOutcome(False, TrialStatus.INCORRECT, f"output shape {output_storage.shape} != {expected.shape}")
        output = dtype.decode(output_storage)
        policy = self.spec.policy

        expected_finite = np.isfinite(expected)
        # Special values: where the reference is NaN/Inf the kernel must reproduce exactly that.
        if not expected_finite.all():
            expected_nan = np.isnan(expected)
            if not np.array_equal(np.isnan(output), expected_nan):
                return VerificationOutcome(False, TrialStatus.INCORRECT, "NaN pattern differs from the reference")
            expected_inf = np.isinf(expected)
            if not np.array_equal(np.isinf(output) & (np.sign(output) == np.sign(expected)), expected_inf):
                return VerificationOutcome(False, TrialStatus.INCORRECT, "Inf pattern/sign differs from the reference")
        non_finite_extra = ~np.isfinite(output) & expected_finite
        if non_finite_extra.any():
            count = int(non_finite_extra.sum())
            return VerificationOutcome(
                False, TrialStatus.INCORRECT,
                f"{count} non-finite output elements (NaN/Inf) where the reference is finite - unwritten output cells or overflow",
                max_abs_error=float("inf"),
            )

        # Anti "fake success" heuristics with specific messages (the tolerance check would also
        # catch them, but the model learns more from the diagnosis than from "wrong values").
        if output.size and not output.any() and (np.abs(expected) > policy.atol).any():
            return VerificationOutcome(False, TrialStatus.INCORRECT, "output is all zeros: the kernel did not compute anything")
        for tensor, array in zip(case.inputs, self.case_inputs[case.label]):
            if array.shape == output_storage.shape and array.dtype == output_storage.dtype and np.array_equal(array.view(np.uint8), output_storage.view(np.uint8)):
                if not np.allclose(expected, tensor.dtype.decode(array), atol=policy.atol, rtol=policy.rtol):
                    return VerificationOutcome(False, TrialStatus.INCORRECT, f"output is a bitwise copy of input '{tensor.name}': the kernel did not compute anything")

        finite = expected_finite & np.isfinite(output)
        abs_error = np.zeros_like(expected)
        abs_error[finite] = np.abs(output[finite] - expected[finite])
        tolerance = policy.atol + policy.rtol * np.abs(expected)
        max_abs_error = float(abs_error.max()) if abs_error.size else 0.0
        violations = abs_error > tolerance
        if violations.any():
            worst_index = np.unravel_index(int(np.argmax(abs_error - tolerance)), abs_error.shape)
            return VerificationOutcome(
                False, TrialStatus.INCORRECT,
                (
                    f"{int(violations.sum())}/{violations.size} elements outside tolerance "
                    f"(atol={policy.atol}, rtol={policy.rtol}, dtype {dtype.name}); worst at index {tuple(int(i) for i in worst_index)}: "
                    f"got {output[worst_index]:.6g}, expected {expected[worst_index]:.6g}"
                ),
                max_abs_error=max_abs_error,
            )

        # Grading: ULP measured at the output's magnitude scale, plus the bitwise agreement rate.
        scale = float(np.abs(expected[expected_finite]).max()) if expected_finite.any() else 0.0
        ulp_at_scale = dtype.spacing_at(scale)
        scaled_ulp_error = max_abs_error / ulp_at_scale if ulp_at_scale > 0 else 0.0
        itemsize = np.dtype(dtype.storage).itemsize
        element_equal = (output_storage.view(np.uint8).reshape(-1, itemsize) == expected_storage.view(np.uint8).reshape(-1, itemsize)).all(axis=1)
        match_rate = float(element_equal.mean()) if element_equal.size else 1.0
        if match_rate == 1.0:
            grade = NumericGrade.BITWISE
        elif scaled_ulp_error <= self.tight_ulp_threshold():
            grade = NumericGrade.TIGHT
        else:
            grade = NumericGrade.REDUCED
        return VerificationOutcome(True, TrialStatus.OK, max_abs_error=max_abs_error, grade=grade, scaled_ulp_error=scaled_ulp_error, bitwise_match_rate=match_rate)
