"""The optimization agent: baseline -> autotune the template schedule -> LLM refinement.

Every phase goes through the same evaluator, so "faster" always means "faster AND still
correct on every test case".
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from kopt_agent.backends.base import Backend
from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import Evaluator, TrialResult
from kopt_agent.generators.llm import LLMGenerator, LLMUnavailable
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.history import History
from kopt_agent.spec import OperatorSpec

logger = logging.getLogger("kopt")


@dataclass
class AgentConfig:
    autotune_budget: int = 12
    llm_rounds: int = 0
    llm_patience: int = 4
    seed: int = 0
    warmup: int = 3
    repeats: int = 15
    run_timeout_seconds: float = 60.0
    output_dir: Path = Path("results")


@dataclass
class OperatorBundle:
    spec: OperatorSpec
    baseline_source: str
    template: TemplateGenerator | None = None


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
        self.evaluator = Evaluator(
            self.spec,
            backend,
            warmup=config.warmup,
            repeats=config.repeats,
            run_timeout_seconds=config.run_timeout_seconds,
            seed=config.seed,
        )
        self.history = History(Path(config.output_dir) / self.spec.name)
        self.rng = random.Random(config.seed)

    def run(self) -> History:
        started = time.perf_counter()
        logger.info("operator=%s shape=%s backend=%s", self.spec.name, self.spec.primary_shape, self.backend.name)
        logger.info("hardware: %s", self.backend.hardware_summary())
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
                "hardware": self.backend.hardware_summary(),
                "wall_seconds": elapsed,
                "llm_calls": self.llm.calls if self.llm else 0,
            }
        )
        logger.info("finished in %.1fs, %d trials, statuses=%s", elapsed, len(self.history.records), self.history.status_counts())
        return self.history

    # ---- phases -----------------------------------------------------------------------

    def _phase_baseline(self) -> None:
        candidate = Candidate(source=self.bundle.baseline_source, origin="baseline", note="naive reference kernel")
        result = self._evaluate_and_record(candidate, phase="baseline")
        if not result.is_valid:
            raise RuntimeError(
                f"baseline kernel for '{self.spec.name}' failed ({result.status.value}: {result.message}). "
                "The baseline must be correct: it anchors speedups and seeds the LLM."
            )

    def _phase_autotune(self) -> None:
        template = self.bundle.template
        if template is None or self.config.autotune_budget <= 0:
            logger.info("autotune: skipped (%s)", "no template" if template is None else "budget 0")
            return
        logger.info("autotune: searching %d of %d schedules", min(self.config.autotune_budget, template.space_size()), template.space_size())
        self._evaluate_and_record(template.default_candidate(), phase="autotune")
        for params in template.iter_configs(self.config.autotune_budget, self.rng):
            self._evaluate_and_record(template.render(params), phase="autotune")

    def _phase_llm_refine(self) -> None:
        if self.config.llm_rounds <= 0:
            return
        if self.llm is None:
            logger.info("llm: skipped (no endpoint configured; set KOPT_LLM_API_KEY / OPENAI_API_KEY)")
            return

        rounds_without_gain = 0
        last_attempt: tuple[Candidate, TrialResult] | None = None
        extra_feedback = ""
        for round_index in range(1, self.config.llm_rounds + 1):
            best_candidate, best_result = self.history.best if self.history.best else (None, None)
            best_source = best_candidate.source if best_candidate else self.bundle.baseline_source
            try:
                candidate = self.llm.propose(
                    spec=self.spec,
                    hardware_summary=self.backend.hardware_summary(),
                    language_guidance=self.backend.language_guidance(),
                    best_source=best_source,
                    best_result=best_result,
                    last_attempt=last_attempt,
                    round_index=round_index,
                    extra_feedback=extra_feedback,
                )
            except LLMUnavailable as error:
                logger.warning("llm: stopping, %s", error)
                return
            except ValueError as error:
                logger.warning("llm round %d: unusable reply (%s)", round_index, error)
                rounds_without_gain += 1
                if rounds_without_gain >= self.config.llm_patience:
                    break
                continue

            if candidate.fingerprint in self.history.fingerprints:
                logger.info("llm round %d: identical to an earlier candidate, skipping evaluation", round_index)
                rounds_without_gain += 1
                last_attempt = None
                extra_feedback = "Your previous reply was byte-identical to a kernel already evaluated. Propose a genuinely different schedule."
                if rounds_without_gain >= self.config.llm_patience:
                    logger.info("llm: no improvement for %d rounds, stopping early", rounds_without_gain)
                    break
                continue

            result = self._evaluate_and_record(candidate, phase=f"llm r{round_index}")
            last_attempt = (candidate, result)
            extra_feedback = ""
            improved = self.history.best is not None and self.history.best[1].trial_id == result.trial_id
            rounds_without_gain = 0 if improved else rounds_without_gain + 1
            if rounds_without_gain >= self.config.llm_patience:
                logger.info("llm: no improvement for %d rounds, stopping early", rounds_without_gain)
                break

    # ---- helpers ----------------------------------------------------------------------

    def _evaluate_and_record(self, candidate: Candidate, phase: str) -> TrialResult:
        result = self.evaluator.evaluate(candidate)
        became_best = self.history.record(candidate, result)
        if result.is_valid:
            marker = "  <-- new best" if became_best else ""
            logger.info(
                "[%s] trial %d %s: %.4f ms, %.1f GFLOP/s, %.1f GB/s%s",
                phase, result.trial_id, candidate.short_label(), result.latency_ms_median, result.gflops, result.gbps, marker,
            )
        else:
            first_line = result.message.strip().splitlines()[0] if result.message.strip() else ""
            logger.info("[%s] trial %d %s: %s - %s", phase, result.trial_id, candidate.short_label(), result.status.value, first_line[:160])
        return result
