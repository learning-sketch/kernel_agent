"""Backend contract. A backend knows how to compile candidate source for one target and
how to execute it in isolation (so a crashing kernel cannot take down the agent)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from kopt_agent.candidate import Candidate
from kopt_agent.spec import OperatorSpec, TestCase


@dataclass
class CompileResult:
    ok: bool
    artifact: Path | None = None
    log: str = ""
    compile_seconds: float = 0.0
    # Compiler notes about which loops were (not) vectorized; fed back to the LLM.
    optimization_report: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    ok: bool
    output: np.ndarray | None = None
    timings_ms: list[float] = field(default_factory=list)
    error: str = ""
    # "timeout" | "memory" (out-of-bounds write / input mutation) | "runtime" (crash, load failure)
    error_kind: str = "runtime"


class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def compile(self, candidate: Candidate, spec: OperatorSpec) -> CompileResult: ...

    @abstractmethod
    def run(
        self,
        artifact: Path,
        spec: OperatorSpec,
        case: TestCase,
        inputs: Sequence[np.ndarray],
        warmup: int,
        repeats: int,
        timeout_seconds: float,
    ) -> RunResult: ...

    @abstractmethod
    def hardware_summary(self) -> str: ...

    @abstractmethod
    def language_guidance(self) -> str:
        """Backend-specific rules the LLM generator must follow (language, headers, ABI)."""
