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
from dataclasses import dataclass

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


class LLMGenerator:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.calls = 0

    def propose(
        self,
        spec: OperatorSpec,
        hardware_summary: str,
        language_guidance: str,
        best_source: str,
        best_result: TrialResult | None,
        last_attempt: tuple[Candidate, TrialResult] | None,
        round_index: int,
        extra_feedback: str = "",
    ) -> Candidate:
        prompt = self._build_prompt(spec, hardware_summary, language_guidance, best_source, best_result, last_attempt, round_index, extra_feedback)
        reply = self._chat(prompt)
        source = _extract_c_block(reply)
        if source is None:
            raise ValueError("model reply contained no ```c code block")
        if spec.symbol not in source:
            raise ValueError(f"model reply does not define the required symbol '{spec.symbol}'")
        strategy_match = re.search(r"//\s*strategy:\s*(.+)", source)
        note = strategy_match.group(1).strip() if strategy_match else ""
        return Candidate(source=source, origin="llm", params={"round": round_index}, note=note)

    def _build_prompt(
        self,
        spec: OperatorSpec,
        hardware_summary: str,
        language_guidance: str,
        best_source: str,
        best_result: TrialResult | None,
        last_attempt: tuple[Candidate, TrialResult] | None,
        round_index: int,
        extra_feedback: str = "",
    ) -> str:
        sections = [
            f"# Operator\n{spec.name}: {spec.description}",
            f"Required prototype (exact):\n```c\n{spec.c_signature};\n```",
            "Benchmark shape: " + "x".join(map(str, spec.primary_shape))
            + "; correctness is also checked on edge shapes: "
            + ", ".join("x".join(map(str, shape)) for shape in spec.edge_shapes)
            + f". Tolerance atol={spec.atol}, rtol={spec.rtol}. Output memory is prefilled with NaN, so every element must be written.",
            "# Hardware\n" + hardware_summary,
            "# Language rules\n" + language_guidance,
        ]
        if spec.notes:
            sections.append("# Operator notes\n- " + "\n- ".join(spec.notes))

        if best_result is not None and best_result.is_valid:
            sections.append(
                "# Current best kernel\n"
                f"median latency {best_result.latency_ms_median:.4f} ms, {best_result.gflops:.1f} GFLOP/s, "
                f"{best_result.gbps:.1f} GB/s effective bandwidth, max abs error {best_result.max_abs_error:.3g}.\n"
                f"```c\n{best_source}\n```"
            )
        else:
            sections.append("# Current best kernel\nNone is correct yet. Here is the starting point:\n```c\n" + best_source + "\n```")

        if last_attempt is not None:
            candidate, result = last_attempt
            if result.status is TrialStatus.OK:
                verdict = (
                    f"correct but not faster: {result.latency_ms_median:.4f} ms vs best "
                    f"{best_result.latency_ms_median:.4f} ms" if best_result and best_result.is_valid else "correct"
                )
            else:
                verdict = f"{result.status.value}: {result.message[-1500:]}"
            sections.append(
                f"# Previous attempt (round {round_index - 1})\nOutcome: {verdict}\n"
                f"```c\n{candidate.source[-6000:]}\n```\n"
                "Fix the problem if it failed; otherwise try a materially different optimization."
            )

        if extra_feedback:
            sections.append("# Feedback\n" + extra_feedback)

        sections.append("# Playbook\n" + OPTIMIZATION_PLAYBOOK)
        sections.append(
            f"# Task (round {round_index})\nProduce a faster kernel than the current best that still passes all checks. "
            "Return one complete C file in a single ```c block."
        )
        return "\n\n".join(sections)

    def _chat(self, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
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


def _extract_c_block(text: str) -> str | None:
    fenced = re.findall(r"```(?:c|C|cpp|c\+\+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip() + "\n"
    if "#include" in text and "(" in text:
        return text.strip() + "\n"
    return None
