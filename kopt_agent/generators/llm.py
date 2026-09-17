"""LLM-driven kernel rewriting. Talks to any OpenAI-compatible chat-completions endpoint
(OpenAI, DeepSeek, Qwen/DashScope, vLLM, Ollama, ...) using only the standard library.

Each round the model sees: the operator contract, the hardware, the current best kernel with
its measured numbers, and the outcome of the previous attempt (compiler errors, wrong
elements, crash signal). It must answer with one complete C translation unit.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import TrialResult, TrialStatus
from kopt_agent.spec import OperatorSpec

OPTIMIZATION_PLAYBOOK = """Techniques worth considering (pick what the measurements justify):
- Cache/register blocking: tile so the working set fits L1/L2; keep a register tile (e.g. 4x8 or 6x16 floats) of accumulators.
- Vectorization: contiguous inner loop over the fastest-varying dimension, FMA intrinsics or `#pragma omp simd`, aligned loads.
- Loop order / interchange so the innermost loop streams unit-stride memory.
- Packing: copy tiles of the operands into contiguous, aligned scratch buffers before the hot loop.
- Parallelism: OpenMP over the outermost independent dimension; avoid false sharing; static schedule for regular work.
- Reductions: split into per-thread partials, tree-reduce, keep numerically stable (e.g. subtract row max before exp).
- Precision: keep float32 accumulation semantics close to the reference; no fast-math tricks that change results beyond tolerance.
- Remainders: every tile loop must handle sizes that are not multiples of the tile; never read/write out of bounds.
- Memory-bound ops: fuse passes so each element is read from DRAM once; prefer streaming stores only when data is not reused."""

SYSTEM_PROMPT = (
    "You are a senior high-performance kernel engineer. You write correct, fast, portable C kernels and "
    "reason from measurements. Respond with exactly one ```c code block containing a complete translation "
    "unit (includes + the required function). No main(), no prose outside the code block, except an optional "
    "one-line `// strategy: ...` comment at the top of the file."
)


class LLMUnavailable(RuntimeError):
    """Raised when the endpoint cannot be reached or is not configured."""


@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.4
    max_tokens: int = 4096
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMConfig | None":
        api_key = os.environ.get("KOPT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            model=model or os.environ.get("KOPT_LLM_MODEL", "gpt-4o"),
            base_url=os.environ.get("KOPT_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=api_key,
        )


@dataclass
class RoundFeedback:
    """Everything the model should know about the previous round."""

    attempts: list[tuple[Candidate, TrialResult]] = field(default_factory=list)
    extra: str = ""


# Each parallel sample is nudged toward a different part of the design space so best-of-N
# does not return N near-identical kernels.
SAMPLE_FOCUS = (
    "",
    "Focus on register blocking: a fixed MRxNR accumulator tile living entirely in vector registers.",
    "Focus on data layout: pack operand tiles into contiguous aligned scratch buffers before the hot loop.",
    "Focus on the parallel decomposition and load balance across threads; avoid false sharing.",
    "Focus on explicit SIMD intrinsics for the innermost loop and aligned loads/stores.",
    "Focus on minimising memory traffic: fuse passes so each element is read from DRAM once.",
)


class LLMGenerator:
    def __init__(self, config: LLMConfig, workers: int = 1) -> None:
        self.config = config
        self.workers = max(1, workers)
        self.calls = 0

    def propose(
        self,
        spec: OperatorSpec,
        hardware_summary: str,
        language_guidance: str,
        best_source: str,
        best_result: TrialResult | None,
        feedback: RoundFeedback | None,
        round_index: int,
        sample_index: int = 0,
        peaks_summary: str = "",
    ) -> Candidate:
        prompt = self._build_prompt(
            spec, hardware_summary, language_guidance, best_source, best_result, feedback, round_index, sample_index, peaks_summary
        )
        temperature = self.config.temperature if sample_index == 0 else min(1.0, self.config.temperature + 0.3)
        reply = self._chat(prompt, temperature)
        source = _extract_c_block(reply)
        if source is None:
            raise ValueError("model reply contained no ```c code block")
        if spec.symbol not in source:
            raise ValueError(f"model reply does not define the required symbol '{spec.symbol}'")
        strategy_match = re.search(r"//\s*strategy:\s*(.+)", source)
        note = strategy_match.group(1).strip() if strategy_match else ""
        return Candidate(source=source, origin="llm", params={"round": round_index, "sample": sample_index}, note=note)

    def propose_many(self, samples: int, **kwargs) -> tuple[list[Candidate], list[str]]:
        """Best-of-N: request `samples` candidates concurrently. Returns (candidates, problems).
        Endpoint failures propagate as LLMUnavailable; unusable replies are reported in `problems`."""
        samples = max(1, samples)
        candidates: list[Candidate] = []
        problems: list[str] = []

        def one(sample_index: int) -> Candidate:
            return self.propose(sample_index=sample_index, **kwargs)

        if samples == 1 or self.workers == 1:
            outcomes = []
            for sample_index in range(samples):
                try:
                    outcomes.append(one(sample_index))
                except ValueError as error:
                    outcomes.append(error)
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, samples)) as pool:
                futures = [pool.submit(one, sample_index) for sample_index in range(samples)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except ValueError as error:
                        outcomes.append(error)

        for sample_index, outcome in enumerate(outcomes):
            if isinstance(outcome, Candidate):
                candidates.append(outcome)
            else:
                problems.append(f"sample {sample_index}: {outcome}")
        return candidates, problems

    def _build_prompt(
        self,
        spec: OperatorSpec,
        hardware_summary: str,
        language_guidance: str,
        best_source: str,
        best_result: TrialResult | None,
        feedback: RoundFeedback | None,
        round_index: int,
        sample_index: int,
        peaks_summary: str,
    ) -> str:
        sections = [
            f"# Operator\n{spec.name}: {spec.description}",
            f"Required prototype (exact):\n```c\n{spec.c_signature};\n```",
            "Benchmark shape: " + "x".join(map(str, spec.primary_shape))
            + "; correctness is also checked on edge shapes: "
            + ", ".join("x".join(map(str, shape)) for shape in spec.edge_shapes)
            + f". Tolerance atol={spec.atol}, rtol={spec.rtol}. Output memory is prefilled with NaN, so every element must be written.",
            "# Hardware\n" + hardware_summary + (f"\nMeasured peaks: {peaks_summary}" if peaks_summary else ""),
            "# Language rules\n" + language_guidance,
        ]
        if spec.notes:
            sections.append("# Operator notes\n- " + "\n- ".join(spec.notes))

        if best_result is not None and best_result.is_valid:
            sections.append(
                "# Current best kernel\n"
                + _describe_measurement(best_result)
                + f"\n```c\n{best_source}\n```"
            )
        else:
            sections.append("# Current best kernel\nNone is correct yet. Here is the starting point:\n```c\n" + best_source + "\n```")

        if feedback is not None and feedback.attempts:
            sections.append(_describe_attempts(feedback.attempts, best_result, round_index))
        if feedback is not None and feedback.extra:
            sections.append("# Feedback\n" + feedback.extra)

        sections.append("# Playbook\n" + OPTIMIZATION_PLAYBOOK)
        focus = SAMPLE_FOCUS[sample_index % len(SAMPLE_FOCUS)]
        sections.append(
            f"# Task (round {round_index}, sample {sample_index})\nProduce a faster kernel than the current best that still passes all checks. "
            + (focus + " " if focus else "")
            + "Return one complete C file in a single ```c block."
        )
        return "\n\n".join(sections)

    def _chat(self, user_prompt: str, temperature: float) -> str:
        payload = {
            "model": self.config.model,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise LLMUnavailable(f"LLM endpoint returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMUnavailable(f"cannot reach LLM endpoint {self.config.base_url}: {error}") from error

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMUnavailable(f"unexpected LLM response shape: {json.dumps(body)[:500]}") from error


def _describe_measurement(result: TrialResult) -> str:
    lines = [
        f"median latency {result.latency_ms_median:.4f} ms, {result.gflops:.1f} GFLOP/s, "
        f"{result.gbps:.1f} GB/s effective bandwidth, max abs error {result.max_abs_error:.3g}."
    ]
    if result.roofline:
        roof = result.roofline
        lines.append(
            f"Roofline: {roof['bound']}-bound at this shape (arithmetic intensity {roof['arithmetic_intensity']:.1f} FLOP/B); "
            f"this kernel reaches {roof['fraction_of_attainable'] * 100:.0f}% of the attainable {roof['attainable_gflops']:.0f} GFLOP/s "
            f"({roof['fraction_of_compute_peak'] * 100:.0f}% of FMA peak, {roof['fraction_of_bandwidth_peak'] * 100:.0f}% of bandwidth peak)."
        )
    if result.compiler_notes:
        lines.append("Compiler vectorizer report (line numbers refer to the kernel below):\n- " + "\n- ".join(result.compiler_notes[:25]))
    return "\n".join(lines)


def _describe_attempts(attempts: list[tuple[Candidate, TrialResult]], best_result: TrialResult | None, round_index: int) -> str:
    def verdict(result: TrialResult) -> str:
        if result.status is TrialStatus.OK:
            if best_result is not None and best_result.is_valid and result.latency_ms_median >= best_result.latency_ms_median:
                return f"correct but not faster: {result.latency_ms_median:.4f} ms vs best {best_result.latency_ms_median:.4f} ms"
            return f"correct, {result.latency_ms_median:.4f} ms (became the new best)"
        return f"{result.status.value}: {result.message[-1200:]}"

    # Show full source for one attempt only: the fastest correct one, else the first failure.
    correct = [pair for pair in attempts if pair[1].is_valid]
    featured = min(correct, key=lambda pair: pair[1].latency_ms_median) if correct else attempts[0]
    lines = [f"# Previous round ({round_index - 1}) - {len(attempts)} attempt(s)"]
    for candidate, result in attempts:
        strategy = candidate.note or "(no strategy comment)"
        lines.append(f"- sample {candidate.params.get('sample', 0)} [{strategy}] -> {verdict(result)}")
        if result.compiler_notes and not result.is_valid:
            lines.append("  compiler notes: " + " | ".join(result.compiler_notes[:5]))
    featured_candidate, featured_result = featured
    lines.append(
        f"\nSource of the {'fastest correct' if correct else 'first failed'} attempt (sample {featured_candidate.params.get('sample', 0)}):\n"
        f"```c\n{featured_candidate.source[-6000:]}\n```"
    )
    if featured_result.is_valid and featured_result.compiler_notes:
        lines.append("Its vectorizer report:\n- " + "\n- ".join(featured_result.compiler_notes[:15]))
    lines.append("Fix the failures if any; otherwise try a materially different optimization than every attempt above.")
    return "\n".join(lines)


def _extract_c_block(text: str) -> str | None:
    fenced = re.findall(r"```(?:c|C|cpp|c\+\+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip() + "\n"
    if "#include" in text and "(" in text:
        return text.strip() + "\n"
    return None
