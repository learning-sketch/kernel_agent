"""The optimization agent.

Phases (all through the same evaluator, so "faster" always means "faster AND correct on
every test case"):

1. baseline      - the naive kernel: correctness anchor, speedup denominator and the A/B
                   reference that is interleaved with every benchmark-grade timing.
2. autotune      - template schedules: warm start from the knowledge base, random sampling,
                   then evolutionary refinement of the top configurations. Candidates are
                   evaluated in tiered batches (parallel compile/verify, quick timing, full
                   timing only for the top-k). Directions recorded as dead ends are pruned.
3. llm refine    - best-of-N kernels per round from an LLM that sees the roofline verdict,
                   structured profiler feedback, known dead ends and the outcome of every
                   previous attempt.

After each phase the roofline verdict of the current best is checked: once it is within the
configured fraction of the attainable time the remaining phases are skipped ("at the
ceiling"). The objective is the workload-weighted total time when a workload profile is
given, otherwise the primary-shape latency. For fused operators the run ends with a fusion
gain report (fused kernel vs. the stages executed separately).
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from kopt_agent.backends.base import Backend
from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import Evaluator, TrialResult
from kopt_agent.generators.llm import LLMGenerator, LLMUnavailable, RoundFeedback
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.history import History
from kopt_agent.knowledge import KnowledgeBase, direction_keys
from kopt_agent.roofline import MachinePeaks, Verdict, analyze, measure_peaks, verdict_for
from kopt_agent.spec import OperatorSpec

logger = logging.getLogger("kopt")


@dataclass
class AgentConfig:
    autotune_budget: int = 12
    evolve_fraction: float = 0.5  # share of the autotune budget spent on mutating the top configs
    evolve_parents: int = 3
    llm_rounds: int = 0
    llm_samples: int = 1  # best-of-N candidates per LLM round
    llm_patience: int = 4
    workers: int = 1  # parallel compile/verify workers and concurrent LLM requests
    top_k: int = 3  # candidates per batch that receive benchmark-grade timing
    warm_start: int = 3  # configurations seeded from the knowledge base
    template_name: str | None = None  # which of the operator's templates to tune (default: bundle default)
    roofline: bool = True
    ceiling_fraction: float = 0.85  # declare "at the ceiling" when best >= this fraction of attainable
    stop_at_ceiling: bool = True
    allow_reduced_precision: bool = False  # let reduced-precision candidates win precision-sensitive ops
    fusion_report: bool = True
    seed: int = 0
    warmup: int = 3
    repeats: int = 15
    quick_repeats: int = 5
    run_timeout_seconds: float = 60.0
    output_dir: Path = Path("results")


@dataclass
class OperatorBundle:
    spec: OperatorSpec
    baseline_source: str
    templates: dict[str, TemplateGenerator] = field(default_factory=dict)
    default_template: str | None = None
    # For fused operators: bundles of the individual stages at the same shapes, used to
    # measure "fused vs. run separately".
    fusion_parts: list["OperatorBundle"] = field(default_factory=list)

    def select_template(self, name: str | None) -> TemplateGenerator | None:
        if not self.templates:
            return None
        chosen = name or self.default_template or next(iter(self.templates))
        if chosen not in self.templates:
            raise KeyError(f"unknown template '{chosen}' for {self.spec.name}, available: {sorted(self.templates)}")
        return self.templates[chosen]


class OptimizationAgent:
    def __init__(
        self,
        bundle: OperatorBundle,
        backend: Backend,
        config: AgentConfig,
        llm: LLMGenerator | None = None,
    ) -> None:
        self.bundle = bundle
        self.spec = bundle.spec
        self.backend = backend
        self.config = config
        self.llm = llm
        self.template = bundle.select_template(config.template_name)
        for dtype in self.spec.dtypes_used():
            if not backend.supports_dtype(dtype):
                raise RuntimeError(
                    f"backend '{backend.name}' cannot compile {dtype.name} kernels on this machine "
                    f"(the compiler rejects the C type '{dtype.c_type}'; gcc >= 12 for _Float16, >= 13 for __bf16). "
                    f"Operator precision: {self.spec.precision_label()}. Use fp32 or a newer toolchain."
                )
        self.hardware = backend.hardware_summary()
        self.peaks: MachinePeaks | None = None
        if config.roofline:
            try:
                self.peaks = measure_peaks(backend)
            except RuntimeError as error:
                logger.warning("roofline: peak probes failed, continuing without roofline (%s)", error)
        self.evaluator = self._make_evaluator(self.spec)
        operator_dir = Path(config.output_dir) / self.spec.name
        compile_command = getattr(backend, "portable_compile_command", lambda: None)()
        self.history = History(
            operator_dir, spec=self.spec, allow_reduced_precision=config.allow_reduced_precision, compile_command=compile_command
        )
        # Knowledge is per template: tile parameters of one schedule mean nothing to another.
        template_tag = config.template_name or bundle.default_template or "template"
        self.knowledge = KnowledgeBase(operator_dir / f"knowledge_{template_tag}.jsonl")
        self.rng = random.Random(config.seed)
        self._seen_signatures: set[tuple] = set()
        self._parents: dict[str, TrialResult] = {}  # child fingerprint -> parent result (for negative knowledge)
        self.verdict: Verdict | None = None
        self.stopped_at_ceiling = False
        self.fusion_gain: dict | None = None

    def _make_evaluator(self, spec: OperatorSpec) -> Evaluator:
        config = self.config
        return Evaluator(
            spec,
            self.backend,
            warmup=config.warmup,
            repeats=config.repeats,
            run_timeout_seconds=config.run_timeout_seconds,
            seed=config.seed,
            peaks=self.peaks,
            workers=config.workers,
            quick_repeats=config.quick_repeats,
        )

    def run(self) -> History:
        started = time.perf_counter()
        logger.info(
            "operator=%s precision=%s shape=%s backend=%s (%s)%s", self.spec.name, self.spec.precision_label(), self.spec.primary_shape,
            self.backend.name, self.backend.launch_abi.describe(),
            " precision-sensitive" + ("" if self.config.allow_reduced_precision else " (reduced-precision candidates cannot win)") if self.spec.precision_sensitive else "",
        )
        logger.info("hardware: %s", self.hardware)
        if self.peaks is not None:
            logger.info("roofline: %s", self.peaks.describe())
        if self.spec.workload is not None:
            logger.info(
                "workload: %d calls over %d shapes - objective is the call-weighted total time: %s",
                self.spec.workload.total_calls, len(self.spec.workload.entries),
                ", ".join(f"{'x'.join(map(str, e.shape))} x{e.count}" for e in self.spec.workload.entries),
            )
        logger.info("correctness cases: %s", ", ".join(case.label for case in self.evaluator.cases))

        self._phase_baseline()
        if not self._at_ceiling("baseline"):
            self._phase_autotune()
        if not self._at_ceiling("autotune"):
            self._phase_llm_refine()
        self._at_ceiling("final")
        self._phase_fusion_report()

        elapsed = time.perf_counter() - started
        self.history.write_summary(
            {
                "operator": self.spec.name,
                "dtype": self.spec.dtype.name,
                "shape": list(self.spec.primary_shape),
                "workload": [{"shape": list(e.shape), "count": e.count} for e in self.spec.workload.entries] if self.spec.workload else None,
                "backend": self.backend.name,
                "hardware": self.hardware,
                "peaks": (
                    {
                        "compute_gflops": self.peaks.compute_gflops, "bandwidth_gbps": self.peaks.bandwidth_gbps,
                        "dispatch_overhead_ms": self.peaks.dispatch_overhead_ms, "call_overhead_ms": self.peaks.call_overhead_ms,
                    }
                    if self.peaks else None
                ),
                "verdict": self.verdict.to_dict() if self.verdict else None,
                "stopped_at_ceiling": self.stopped_at_ceiling,
                "fusion_gain": self.fusion_gain,
                "dead_ends": self.knowledge.dead_end_summary(self.spec.primary_shape, self.hardware, limit=50),
                "wall_seconds": elapsed,
                "llm_calls": self.llm.calls if self.llm else 0,
            }
        )
        logger.info("finished in %.1fs, %d trials, statuses=%s", elapsed, len(self.history.records), self.history.status_counts())
        return self.history

    # ---- phases -----------------------------------------------------------------------

    def _phase_baseline(self) -> None:
        candidate = Candidate(source=self.bundle.baseline_source, origin="baseline", note="naive reference kernel")
        result = self.evaluator.evaluate(candidate)
        self.evaluator.calibrate_precision(result)
        result = self._record(candidate, result, phase="baseline")
        if not result.is_valid:
            raise RuntimeError(
                f"baseline kernel for '{self.spec.name}' failed ({result.status.value}: {result.message}). "
                "The baseline must be correct: it anchors speedups and seeds the LLM."
            )
        compiled = self.backend.compile(candidate, self.spec)  # cached: gives us the artifact for A/B timing
        if compiled.ok and compiled.artifact is not None:
            self.evaluator.reference_artifact = compiled.artifact

    def _phase_autotune(self) -> None:
        template = self.template
        budget = self.config.autotune_budget
        if template is None or budget <= 0:
            logger.info("autotune: skipped (%s)", "no template" if template is None else "budget 0")
            return

        evolve_budget = int(round(budget * min(max(self.config.evolve_fraction, 0.0), 1.0)))
        sample_budget = budget - evolve_budget

        # Stage 1: default + warm start + random exploration, one tiered batch.
        initial: list[Candidate] = [self._template_candidate(template, dict(template.default_params), "template-default")]
        warm_params = self.knowledge.warm_start_params(self.spec.primary_shape, self.hardware, self.config.warm_start)
        for params in warm_params:
            candidate = self._template_candidate(template, params, "warm-start")
            if candidate is not None:
                initial.append(candidate)
        random_needed = max(0, sample_budget - len(warm_params))
        for params in template.iter_configs(random_needed + len(self._seen_signatures), self.rng):
            if len(initial) >= 1 + len(warm_params) + random_needed:
                break
            candidate = self._template_candidate(template, params, "autotune")
            if candidate is not None:
                initial.append(candidate)
        logger.info(
            "autotune: stage 1 evaluating %d schedules (%d warm-start, %d random) of a %d-point space",
            len(initial), len(warm_params), len(initial) - 1 - len(warm_params), template.space_size(),
        )
        self._evaluate_batch(initial, phase="autotune")
        if self._at_ceiling("autotune stage 1"):
            return

        # Stage 2: evolve the best configurations found so far, skipping known dead-end directions.
        remaining = evolve_budget
        generation = 0
        pruned_total = 0
        while remaining > 0:
            generation += 1
            parents = self.history.valid_results(origins=("template-default", "warm-start", "autotune", "evolve"))[: self.config.evolve_parents]
            if not parents:
                logger.info("evolve: no correct template configuration to mutate, stopping")
                break
            children: list[Candidate] = []
            batch_size = min(remaining, max(self.config.workers * 2, 2))
            attempts_without_child = 0
            while len(children) < batch_size and attempts_without_child < 12:
                parent = self.rng.choice(parents)
                child_params = template.mutate(parent.params, self.rng, self._seen_signatures)
                if child_params is None:
                    attempts_without_child += 1
                    continue
                keys = direction_keys({**template.default_params, **parent.params}, {**template.default_params, **child_params})
                if self.knowledge.is_pruned(self.hardware, keys):
                    self._seen_signatures.add(template.signature(child_params))
                    pruned_total += 1
                    attempts_without_child += 1
                    continue
                candidate = self._template_candidate(template, child_params, "evolve")
                if candidate is None:
                    attempts_without_child += 1
                    continue
                self._parents[candidate.fingerprint] = parent
                children.append(candidate)
            if not children:
                logger.info("evolve: neighbourhood of the top configurations is exhausted, stopping")
                break
            logger.info("evolve: generation %d, %d children of %d parents%s", generation, len(children), len(parents), f" ({pruned_total} dead-end moves pruned so far)" if pruned_total else "")
            self._evaluate_batch(children, phase=f"evolve g{generation}")
            remaining -= len(children)
            if self._at_ceiling(f"evolve g{generation}"):
                return

    def _phase_llm_refine(self) -> None:
        if self.config.llm_rounds <= 0:
            return
        if self.llm is None:
            logger.info("llm: skipped (no endpoint configured; set KOPT_LLM_API_KEY / OPENAI_API_KEY)")
            return

        rounds_without_gain = 0
        feedback: RoundFeedback | None = None
        for round_index in range(1, self.config.llm_rounds + 1):
            best_candidate, best_result = self.history.best if self.history.best else (None, None)
            best_source = best_candidate.source if best_candidate else self.bundle.baseline_source
            dead_ends = self.knowledge.dead_end_summary(self.spec.primary_shape, self.hardware)
            try:
                candidates, problems = self.llm.propose_many(
                    self.config.llm_samples,
                    spec=self.spec,
                    hardware_summary=self.hardware,
                    language_guidance=self.backend.language_guidance(),
                    best_source=best_source,
                    best_result=best_result,
                    feedback=feedback,
                    round_index=round_index,
                    peaks_summary=self.peaks.describe() if self.peaks else "",
                    verdict=self.verdict,
                    dead_ends=dead_ends,
                )
            except LLMUnavailable as error:
                logger.warning("llm: stopping, %s", error)
                return

            for problem in problems:
                logger.warning("llm round %d: unusable reply (%s)", round_index, problem)

            fresh: list[Candidate] = []
            duplicates = 0
            batch_fingerprints: set[str] = set()
            for candidate in candidates:
                if candidate.fingerprint in self.history.fingerprints or candidate.fingerprint in batch_fingerprints:
                    duplicates += 1
                    continue
                batch_fingerprints.add(candidate.fingerprint)
                fresh.append(candidate)

            extra = ""
            if duplicates:
                logger.info("llm round %d: %d reply(ies) identical to earlier candidates, skipped", round_index, duplicates)
                extra = f"{duplicates} of your previous replies were byte-identical to kernels already evaluated. Propose genuinely different schedules."
            if problems:
                extra += (" " if extra else "") + "Some replies had no usable ```c block or missed the required symbol; reply with exactly one complete C file."

            if not fresh:
                rounds_without_gain += 1
                feedback = RoundFeedback(attempts=[], extra=extra)
                if rounds_without_gain >= self.config.llm_patience:
                    logger.info("llm: no improvement for %d rounds, stopping early", rounds_without_gain)
                    break
                continue

            best_before = self.history.best[1].objective_ms if self.history.best else math.inf
            if best_result is not None:
                for candidate in fresh:
                    self._parents[candidate.fingerprint] = best_result
            results = self._evaluate_batch(fresh, phase=f"llm r{round_index}", top_k=max(self.config.top_k, 1))
            feedback = RoundFeedback(attempts=list(zip(fresh, results)), extra=extra)
            improved = self.history.best is not None and self.history.best[1].objective_ms < best_before
            rounds_without_gain = 0 if improved else rounds_without_gain + 1
            if self._at_ceiling(f"llm r{round_index}"):
                return
            if rounds_without_gain >= self.config.llm_patience:
                logger.info("llm: no improvement for %d rounds, stopping early", rounds_without_gain)
                break

    def _phase_fusion_report(self) -> None:
        """Fused best vs. the stages run separately (best of baseline / template default per stage)."""
        if not self.bundle.fusion_parts or not self.config.fusion_report or self.history.best is None:
            return
        stage_times: list[dict] = []
        for part in self.bundle.fusion_parts:
            evaluator = self._make_evaluator(part.spec)
            candidates = [Candidate(source=part.baseline_source, origin="baseline", note="stage baseline")]
            template = part.select_template(None)
            if template is not None:
                candidates.append(template.default_candidate())
            best_ms = math.inf
            best_label = "none"
            for candidate in candidates:
                result = evaluator.evaluate(candidate)
                if result.is_valid and result.objective_ms < best_ms:
                    best_ms, best_label = result.objective_ms, f"{candidate.origin}"
            if not math.isfinite(best_ms):
                logger.warning("fusion report: no correct kernel for stage %s, skipping report", part.spec.name)
                return
            stage_times.append({"stage": part.spec.name, "shape": list(part.spec.primary_shape), "best_ms": best_ms, "kernel": best_label})
        separate_ms = sum(stage["best_ms"] for stage in stage_times)
        fused_ms = self.history.best[1].objective_ms
        self.fusion_gain = {
            "stages": stage_times,
            "separate_total_ms": separate_ms,
            "fused_best_ms": fused_ms,
            "fusion_speedup": separate_ms / fused_ms if fused_ms > 0 else None,
        }
        logger.info(
            "fusion: %s separately = %.4f ms (%s); fused best = %.4f ms -> fusion speedup %.2fx",
            " + ".join(stage["stage"] for stage in stage_times), separate_ms,
            ", ".join(f"{stage['stage']} {stage['best_ms']:.4f} ms [{stage['kernel']}]" for stage in stage_times),
            fused_ms, separate_ms / fused_ms if fused_ms > 0 else float("nan"),
        )

    # ---- roofline verdict -------------------------------------------------------------

    def _compute_verdict(self) -> Verdict | None:
        if self.peaks is None or self.history.best is None:
            return None
        best = self.history.best[1]
        if self.spec.workload is None or not best.per_shape:
            report = analyze(self.peaks, self.spec.flops(self.spec.primary_shape), self.spec.bytes_moved(self.spec.primary_shape), best.latency_ms_median)
            return verdict_for(report, best.latency_ms_median, self.config.ceiling_fraction)
        # Workload: the ceiling is the call-weighted sum of per-shape ceilings; the bound is that
        # of the shape that dominates the weighted time.
        weighted_attainable = 0.0
        dominant_entry = None
        dominant_time = -1.0
        for entry in best.per_shape.values():
            if entry["weight"] <= 0:
                continue
            report = analyze(self.peaks, self.spec.flops(tuple(entry["shape"])), self.spec.bytes_moved(tuple(entry["shape"])), entry["latency_ms"])
            weighted_attainable += entry["weight"] * report.attainable_ms
            if entry["weight"] * entry["latency_ms"] > dominant_time:
                dominant_time = entry["weight"] * entry["latency_ms"]
                dominant_entry = report
        if dominant_entry is None:
            return None
        fraction = min(1.0, weighted_attainable / max(best.objective_ms, 1e-12))
        return Verdict(
            bound=dominant_entry.bound,
            attainable_ms=weighted_attainable,
            best_ms=best.objective_ms,
            fraction_of_attainable=fraction,
            at_ceiling=fraction >= self.config.ceiling_fraction,
            ceiling_fraction=self.config.ceiling_fraction,
            guidance=dominant_entry.guidance,
        )

    def _at_ceiling(self, phase: str) -> bool:
        self.verdict = self._compute_verdict()
        if self.verdict is None:
            return False
        logger.info("verdict after %s: %s", phase, self.verdict.describe())
        if self.verdict.at_ceiling and self.config.stop_at_ceiling:
            if not self.stopped_at_ceiling:
                logger.info("at the ceiling: stopping the search, remaining phases skipped (use --no-stop-at-ceiling to continue)")
            self.stopped_at_ceiling = True
            return True
        return False

    # ---- helpers ----------------------------------------------------------------------

    def _template_candidate(self, template: TemplateGenerator, params: dict, origin: str) -> Candidate | None:
        merged = {**template.default_params, **params}
        if not template.constraint(merged):
            return None
        signature = template.signature(merged)
        if signature in self._seen_signatures:
            return None
        self._seen_signatures.add(signature)
        return template.render({key: merged[key] for key in template.space}, origin=origin)

    def _evaluate_batch(self, candidates: list[Candidate], phase: str, top_k: int | None = None) -> list[TrialResult]:
        results = self.evaluator.evaluate_batch(candidates, top_k=top_k or self.config.top_k)
        for candidate, result in zip(candidates, results):
            self._record(candidate, result, phase)
        return results

    def _record(self, candidate: Candidate, result: TrialResult, phase: str) -> TrialResult:
        became_best = self.history.record(candidate, result)
        self.knowledge.append(self.spec.primary_shape, self.hardware, result)
        parent = self._parents.pop(candidate.fingerprint, None)
        if result.origin != "baseline":
            direction = (candidate.note or None) if result.origin == "llm" else None
            dead_end = self.knowledge.record_outcome(self.spec.primary_shape, self.hardware, result, parent=parent, direction=direction)
            if dead_end is not None:
                logger.info("[%s] trial %d dead end: %s -> %s", phase, result.trial_id, dead_end.direction, dead_end.cause)
        if result.is_valid:
            marker = "  <-- new best" if became_best else ""
            if result.is_reduced_precision and self.history.precision_gated and result.is_benchmark_grade:
                marker = "  (reduced-precision: excluded from best; --allow-reduced-precision to accept)"
            tier = "" if result.timing_tier == "full" else " (quick)"
            roof = f", {result.roofline['fraction_of_attainable'] * 100:.0f}% of roofline" if result.roofline else ""
            ab = f", A/B {result.ab_speedup:.2f}x" if result.ab_speedup else ""
            objective = f", weighted {result.objective_ms:.4f} ms" if self.spec.workload is not None else ""
            logger.info(
                "[%s] trial %d %s: %.4f ms%s%s, %.1f GFLOP/s, %.1f GB/s, %s%s%s%s",
                phase, result.trial_id, candidate.short_label(), result.latency_ms_median, tier, objective, result.gflops, result.gbps,
                result.numeric_grade, roof, ab, marker,
            )
            for note in result.notes:
                logger.info("[%s] trial %d note: %s", phase, result.trial_id, note)
        else:
            first_line = result.message.strip().splitlines()[0] if result.message.strip() else ""
            logger.info("[%s] trial %d %s: %s - %s", phase, result.trial_id, candidate.short_label(), result.status.value, first_line[:200])
        return result
