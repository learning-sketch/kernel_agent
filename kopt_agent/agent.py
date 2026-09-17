"""The optimization agent.

Phases (all through the same evaluator, so "faster" always means "faster AND correct on
every test case"):

1. baseline      - the naive kernel: correctness anchor and speedup denominator.
2. autotune      - template schedules: warm start from the knowledge base, random sampling,
                   then evolutionary refinement of the top configurations. Candidates are
                   evaluated in tiered batches (parallel compile/verify, quick timing, full
                   timing only for the top-k).
3. llm refine    - best-of-N kernels per round from an LLM that sees roofline position,
                   compiler vectorizer notes and the outcome of every previous attempt.
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
from kopt_agent.knowledge import KnowledgeBase
from kopt_agent.roofline import MachinePeaks, measure_peaks
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
        self.hardware = backend.hardware_summary()
        self.peaks: MachinePeaks | None = None
        if config.roofline:
            try:
                self.peaks = measure_peaks(backend)
            except RuntimeError as error:
                logger.warning("roofline: peak probes failed, continuing without roofline (%s)", error)
        self.evaluator = Evaluator(
            self.spec,
            backend,
            warmup=config.warmup,
            repeats=config.repeats,
            run_timeout_seconds=config.run_timeout_seconds,
            seed=config.seed,
            peaks=self.peaks,
            workers=config.workers,
            quick_repeats=config.quick_repeats,
        )
        operator_dir = Path(config.output_dir) / self.spec.name
        self.history = History(operator_dir)
        # Knowledge is per template: tile parameters of one schedule mean nothing to another.
        template_tag = config.template_name or bundle.default_template or "template"
        self.knowledge = KnowledgeBase(operator_dir / f"knowledge_{template_tag}.jsonl")
        self.rng = random.Random(config.seed)
        self._seen_signatures: set[tuple] = set()

    def run(self) -> History:
        started = time.perf_counter()
        logger.info("operator=%s shape=%s backend=%s", self.spec.name, self.spec.primary_shape, self.backend.name)
        logger.info("hardware: %s", self.hardware)
        if self.peaks is not None:
            logger.info("roofline: %s", self.peaks.describe())
        logger.info("correctness cases: %s", ", ".join(case.label for case in self.evaluator.cases))

        self._phase_baseline()
        self._phase_autotune()
        self._phase_llm_refine()

        elapsed = time.perf_counter() - started
        self.history.write_summary(
            {
                "operator": self.spec.name,
                "shape": list(self.spec.primary_shape),
                "backend": self.backend.name,
                "hardware": self.hardware,
                "peaks": {"compute_gflops": self.peaks.compute_gflops, "bandwidth_gbps": self.peaks.bandwidth_gbps} if self.peaks else None,
                "wall_seconds": elapsed,
                "llm_calls": self.llm.calls if self.llm else 0,
            }
        )
        logger.info("finished in %.1fs, %d trials, statuses=%s", elapsed, len(self.history.records), self.history.status_counts())
        return self.history

    # ---- phases -----------------------------------------------------------------------

    def _phase_baseline(self) -> None:
        candidate = Candidate(source=self.bundle.baseline_source, origin="baseline", note="naive reference kernel")
        result = self._record(candidate, self.evaluator.evaluate(candidate), phase="baseline")
        if not result.is_valid:
            raise RuntimeError(
                f"baseline kernel for '{self.spec.name}' failed ({result.status.value}: {result.message}). "
                "The baseline must be correct: it anchors speedups and seeds the LLM."
            )

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

        # Stage 2: evolve the best configurations found so far.
        remaining = evolve_budget
        generation = 0
        while remaining > 0:
            generation += 1
            parents = self.history.valid_results(origins=("template-default", "warm-start", "autotune", "evolve"))[: self.config.evolve_parents]
            if not parents:
                logger.info("evolve: no correct template configuration to mutate, stopping")
                break
            children: list[Candidate] = []
            batch_size = min(remaining, max(self.config.workers * 2, 2))
            attempts_without_child = 0
            while len(children) < batch_size and attempts_without_child < 8:
                parent = self.rng.choice(parents)
                child_params = template.mutate(parent.params, self.rng, self._seen_signatures)
                if child_params is None:
                    attempts_without_child += 1
                    continue
                candidate = self._template_candidate(template, child_params, "evolve")
                if candidate is None:
                    attempts_without_child += 1
                    continue
                children.append(candidate)
            if not children:
                logger.info("evolve: neighbourhood of the top configurations is exhausted, stopping")
                break
            logger.info("evolve: generation %d, %d children of %d parents", generation, len(children), len(parents))
            self._evaluate_batch(children, phase=f"evolve g{generation}")
            remaining -= len(children)

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

            best_before = self.history.best[1].latency_ms_median if self.history.best else math.inf
            results = self._evaluate_batch(fresh, phase=f"llm r{round_index}", top_k=max(self.config.top_k, 1))
            feedback = RoundFeedback(attempts=list(zip(fresh, results)), extra=extra)
            improved = self.history.best is not None and self.history.best[1].latency_ms_median < best_before
            rounds_without_gain = 0 if improved else rounds_without_gain + 1
            if rounds_without_gain >= self.config.llm_patience:
                logger.info("llm: no improvement for %d rounds, stopping early", rounds_without_gain)
                break

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
        if result.is_valid:
            marker = "  <-- new best" if became_best else ""
            tier = "" if result.timing_tier == "full" else " (quick)"
            roof = f", {result.roofline['fraction_of_attainable'] * 100:.0f}% of roofline" if result.roofline else ""
            logger.info(
                "[%s] trial %d %s: %.4f ms%s, %.1f GFLOP/s, %.1f GB/s%s%s",
                phase, result.trial_id, candidate.short_label(), result.latency_ms_median, tier, result.gflops, result.gbps, roof, marker,
            )
        else:
            first_line = result.message.strip().splitlines()[0] if result.message.strip() else ""
            logger.info("[%s] trial %d %s: %s - %s", phase, result.trial_id, candidate.short_label(), result.status.value, first_line[:160])
        return result
