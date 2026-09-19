"""Backend contract. A backend knows how to compile candidate source for one target and how to
execute it, and describes the launch ABI candidates must implement.

Execution model
---------------
`Backend.run` returns a `RunResult`; the agent, evaluator and generators never look deeper.
The default `run` opens a `KernelSession` (device primitives: place buffers, launch, time with
the device clock, copy back) and drives the shared protocol in `protocol.py`, so an accelerator
backend only implements the primitives - H2D/D2H happen inside `upload`/`download` and are
reported as `h2d_ms`/`d2h_ms`, never inside `timings_ms`. The CPU backend overrides `run` to
execute the same protocol in a crash-isolated subprocess.

The `LaunchABI` tells generators and the exported bundle how a candidate is called: whether the
kernel receives host or device pointers, whether a stream/queue argument is appended, and which
symbol the harness launches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from kopt_agent.backends.protocol import (
    TIMING_HOST_WALL,
    KernelSession,
    LaunchJob,
    ProtocolError,
    run_protocol,
)
from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import DType
from kopt_agent.spec import OperatorSpec, TestCase

if TYPE_CHECKING:  # avoid an import cycle: roofline imports Backend
    from kopt_agent.roofline import MachinePeaks


@dataclass(frozen=True)
class LaunchABI:
    """How the harness calls a candidate and what the candidate must therefore expose.

    - `pointer_space`: "host" (kernel dereferences the pointers it receives) or "device"
      (pointers are device allocations; the entry point is a host *launcher* that enqueues the
      real kernel and must not dereference them on the host).
    - `stream_argument`: when True the launcher receives one extra trailing `void*` stream /
      queue handle after the int scalars and must enqueue onto it.
    - `synchronous_launch`: whether the entry point may return before the kernel finished. The
      session synchronizes itself when False; `launch` in the protocol is always synchronous
      from the protocol's point of view.
    - `fast_path_flag`: how `kopt_fast_path_active` is exposed ("global_int" = an exported int
      the harness reads via the symbol table; "none" = fast paths are not supported).
    """

    pointer_space: str = "host"
    stream_argument: bool = False
    synchronous_launch: bool = True
    fast_path_flag: str = "global_int"
    entry_kind: str = "kernel"  # "kernel" (CPU: the symbol is the kernel) | "launcher" (accelerators)

    def describe(self) -> str:
        parts = [
            f"the exported symbol is the {self.entry_kind}",
            f"pointers are {self.pointer_space} pointers",
            "an extra trailing `void* stream` argument follows the int scalars" if self.stream_argument else "no stream argument",
            "the call is synchronous" if self.synchronous_launch else "the call may return before the kernel finished (the harness synchronizes)",
        ]
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


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
    # Kernel-only time per call. On CPU backends this is host wall time around the synchronous
    # call; on accelerators it comes from the device timer and excludes H2D/D2H and launch latency.
    timings_ms: list[float] = field(default_factory=list)
    host_timings_ms: list[float] = field(default_factory=list)  # host wall time per call incl. dispatch/launch
    reference_timings_ms: list[float] = field(default_factory=list)  # interleaved A/B reference, if requested
    timing_source: str = TIMING_HOST_WALL  # "host_wall" | "device_timer"
    h2d_ms: float | None = None  # total upload time for this run (never part of timings_ms)
    d2h_ms: float | None = None  # total download time during verification (never part of timings_ms)
    cpu_wall_ratio: float | None = None
    threads_available: int | None = None
    fast_path_active: int | None = None  # value of `kopt_fast_path_active` after the verify call, if exported
    poison_left_in_output: bool = False
    prefill_dependent: bool = False
    poison_output: np.ndarray | None = None  # output of the poison-prefilled run when it differed
    error: str = ""
    # "timeout" | "memory" (out-of-bounds write / input mutation) | "runtime" (crash, load failure)
    error_kind: str = "runtime"


def launch_job_for(case: TestCase, inputs: Sequence[np.ndarray], warmup: int, repeats: int, verify: bool, reference_kernel=None) -> LaunchJob:
    """Build the protocol job for one test case (shared by every backend)."""
    return LaunchJob(
        inputs=[np.ascontiguousarray(array) for array in inputs],
        input_names=[tensor.name for tensor in case.inputs],
        output_shape=tuple(case.output.shape),
        output_dtype=np.dtype(case.output.dtype.storage),
        scalars=[int(value) for value in case.scalars],
        verify=verify,
        warmup=warmup,
        repeats=repeats,
        reference_kernel=reference_kernel,
    )


class Backend(ABC):
    name: str = "abstract"
    launch_abi: LaunchABI = LaunchABI()

    @abstractmethod
    def compile(self, candidate: Candidate, spec: OperatorSpec) -> CompileResult: ...

    def open_session(self, artifact: Path, spec: OperatorSpec, case: TestCase) -> KernelSession:
        """Create the device session used by the default `run`. Accelerator backends implement
        this (and `compile`); the CPU backend overrides `run` instead to get process isolation."""
        raise NotImplementedError(f"backend '{self.name}' implements neither run() nor open_session()")

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
    ) -> RunResult:
        """Default in-process execution through a KernelSession + the shared protocol.

        `timeout_seconds` cannot be enforced on an in-process launch; a session that supports
        it should implement its own watchdog (e.g. a device-side timeout) and raise ProtocolError.
        """
        session = self.open_session(artifact, spec, case)
        try:
            kernel = session.load_kernel(str(artifact), spec.symbol)
            reference_kernel = session.load_kernel(str(reference_artifact), spec.symbol) if (reference_artifact and repeats > 0) else None
            job = launch_job_for(case, inputs, warmup, repeats, verify, reference_kernel)
            outcome = run_protocol(session, kernel, job)
        except ProtocolError as error:
            session.close()
            return RunResult(False, error=error.message, error_kind=error.kind)
        except Exception as error:  # noqa: BLE001 - a device runtime error is a runtime failure of the candidate
            session.close()
            return RunResult(False, error=f"{type(error).__name__}: {error}", error_kind="runtime")
        return RunResult(
            True,
            output=outcome.output,
            timings_ms=outcome.timings_ms,
            host_timings_ms=outcome.host_timings_ms,
            reference_timings_ms=outcome.reference_timings_ms,
            timing_source=outcome.timing_source,
            h2d_ms=outcome.h2d_ms,
            d2h_ms=outcome.d2h_ms,
            cpu_wall_ratio=outcome.cpu_wall_ratio,
            threads_available=outcome.threads_available,
            fast_path_active=outcome.fast_path_active,
            poison_left_in_output=outcome.poison_left_in_output,
            prefill_dependent=outcome.prefill_dependent,
            poison_output=outcome.poison_output,
        )

    @abstractmethod
    def hardware_summary(self) -> str: ...

    def supports_dtype(self, dtype: DType) -> bool:
        """Whether this backend's toolchain can compile kernels over `dtype` (e.g. _Float16 / __bf16
        need a recent gcc/clang on x86). fp32 is always expected to work."""
        return dtype.name == "fp32"

    def measure_peaks(self) -> "MachinePeaks | None":
        """Device peaks for the roofline model (datasheet numbers or a backend-specific probe).
        Return None to let `roofline.measure_peaks` run its portable C probes through
        `compile`/`run` - only meaningful for backends that execute plain C on the host."""
        return None

    def portable_compile_command(self) -> list[str] | None:
        """Command template (with {source} / {artifact} placeholders) that builds a candidate
        outside the agent, used by the parity test and the exported bundle."""
        return None

    @abstractmethod
    def language_guidance(self) -> str:
        """Backend-specific rules the LLM generator must follow (language, headers, ABI)."""
