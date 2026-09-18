"""Backend contract. A backend knows how to compile candidate source for one target and
how to execute it in isolation (so a crashing kernel cannot take down the agent)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import DType
from kopt_agent.spec import OperatorSpec, TestCase


@dataclass
class ProfileReport:
    """Backend-neutral profiler / compiler feedback. Every field is optional: a backend fills
    what it can measure, and the LLM prompt renders whatever is present in one fixed layout.

    CPU backends typically fill vector width and loop vectorization; GPU backends fill
    occupancy, register/spill counts and stall reasons; the evaluator fills achieved
    bandwidth/compute fractions from the roofline model."""

    vector_width_bits: int | None = None  # widest vector the compiler emitted for hot loops
    vectorized_loops: list[str] = field(default_factory=list)  # "line 38: ..." entries
    missed_loops: list[str] = field(default_factory=list)  # "line 6: not vectorized: control flow in loop"
    registers_per_thread: int | None = None
    register_spill_bytes: int | None = None
    occupancy: float | None = None  # 0..1 achieved occupancy (GPU)
    achieved_bandwidth_fraction: float | None = None  # 0..1 of peak bandwidth
    achieved_compute_fraction: float | None = None  # 0..1 of peak FMA throughput
    stall_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # anything else worth telling the model

    def is_empty(self) -> bool:
        return not any(
            [
                self.vector_width_bits, self.vectorized_loops, self.missed_loops, self.registers_per_thread,
                self.register_spill_bytes, self.occupancy is not None, self.achieved_bandwidth_fraction is not None,
                self.achieved_compute_fraction is not None, self.stall_reasons, self.notes,
            ]
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self, max_lines: int = 12) -> str:
        """Fixed-layout text block for prompts and logs."""
        lines: list[str] = []
        if self.vector_width_bits:
            lines.append(f"vector width used: {self.vector_width_bits}-bit")
        if self.achieved_compute_fraction is not None:
            lines.append(f"achieved compute: {self.achieved_compute_fraction * 100:.0f}% of FMA peak")
        if self.achieved_bandwidth_fraction is not None:
            lines.append(f"achieved bandwidth: {self.achieved_bandwidth_fraction * 100:.0f}% of streaming peak")
        if self.occupancy is not None:
            lines.append(f"occupancy: {self.occupancy * 100:.0f}%")
        if self.registers_per_thread is not None:
            lines.append(f"registers/thread: {self.registers_per_thread}")
        if self.register_spill_bytes is not None:
            lines.append(f"register spills: {self.register_spill_bytes} bytes")
        if self.stall_reasons:
            lines.append("stall reasons: " + "; ".join(self.stall_reasons[:max_lines]))
        if self.missed_loops:
            lines.append("loops NOT vectorized:\n  - " + "\n  - ".join(self.missed_loops[:max_lines]))
        if self.vectorized_loops:
            lines.append("loops vectorized:\n  - " + "\n  - ".join(self.vectorized_loops[:max_lines]))
        if self.notes:
            lines.append("notes: " + "; ".join(self.notes[:max_lines]))
        return "\n".join(lines) if lines else "(no profiler data)"


@dataclass
class CompileResult:
    ok: bool
    artifact: Path | None = None
    log: str = ""
    compile_seconds: float = 0.0
    # Raw compiler notes (kept for logs) and the structured view of them.
    optimization_report: list[str] = field(default_factory=list)
    profile: ProfileReport = field(default_factory=ProfileReport)


@dataclass
class RunResult:
    ok: bool
    output: np.ndarray | None = None
    timings_ms: list[float] = field(default_factory=list)  # device/kernel time per call
    host_timings_ms: list[float] = field(default_factory=list)  # wall time per call incl. dispatch
    reference_timings_ms: list[float] = field(default_factory=list)  # interleaved A/B reference, if requested
    cpu_wall_ratio: float | None = None
    threads_available: int | None = None
    fast_path_active: int | None = None  # value of `kopt_fast_path_active` after the verify call, if exported
    poison_left_in_output: bool = False
    prefill_dependent: bool = False
    poison_output: np.ndarray | None = None  # output of the poison-prefilled run when it differed
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
        verify: bool = True,
        reference_artifact: Path | None = None,
    ) -> RunResult: ...

    @abstractmethod
    def hardware_summary(self) -> str: ...

    def supports_dtype(self, dtype: DType) -> bool:
        """Whether this backend's toolchain can compile kernels over `dtype` (e.g. _Float16 / __bf16
        need a recent gcc/clang on x86). fp32 is always expected to work."""
        return dtype.name == "fp32"

    @abstractmethod
    def language_guidance(self) -> str:
        """Backend-specific rules the LLM generator must follow (language, headers, ABI)."""
