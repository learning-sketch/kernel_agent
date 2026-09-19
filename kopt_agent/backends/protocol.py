"""The verification / timing protocol every backend runs, written once against a small set of
device primitives (`KernelSession`).

Why this split exists
---------------------
The agent, evaluator and generators only ever see a `RunResult`. What differs between targets is
*where buffers live* (host memory vs. device memory), *how a kernel is launched* (direct call
vs. a host launcher taking device pointers and a stream/queue) and *which clock is authoritative*
(host wall clock vs. device timer events). Everything else - poison prefills, unwritten-cell
detection, const-input snapshots, fast-path flag readout, warmup, interleaved A/B timing - is the
same protocol on every target, so it lives here and a new backend only implements the primitives.

Contract for accelerator backends (the "fill in the blanks"):

  * `upload` / `allocate_output` decide placement. The protocol never touches device memory
    directly: it only ever holds the opaque handles a session hands back.
  * `download` is the only way the protocol reads results (D2H). It is called during
    verification only, never inside the timed block.
  * `launch` is synchronous (it must not return before the kernel finished). `timed_launch`
    returns the kernel-only duration measured with the *backend's own timer* (CUDA/HIP events,
    Level Zero timestamps, ...). Host-side transfer time is measured by the protocol around
    `upload` / `download` and reported separately - it never enters `timings_ms`.
  * `fast_path_flag` reads the kernel's `kopt_fast_path_active` (an int the launcher exposes or
    copies back); return None when the kernel does not export one.
  * `memory_violations` is the hook for guard/canary checks the session can do cheaply (the host
    session checks canary pages; a device session may check guard allocations or return []).

This module deliberately imports nothing from the rest of the package (numpy only) so it can
run inside the crash-isolated runner subprocess.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

ALIGNMENT_BYTES = 64
GUARD_BYTES = 4096
GUARD_PATTERN = 0xA5
NAN_PREFILL_BYTE = 0xFF  # all-ones bytes decode to NaN in every float format (fp32/fp16/bf16/fp64)
POISON_BYTE = 0x5C  # 0x5C5C5C5C as float32 = 2.48e17, as fp16 0x5C5C = 279; never a plausible result
FAST_PATH_SYMBOL = "kopt_fast_path_active"

TIMING_HOST_WALL = "host_wall"  # host clock around a synchronous call (CPU backends)
TIMING_DEVICE_TIMER = "device_timer"  # device events/timestamps around the kernel only


class GuardedBuffer:
    """A 64-byte aligned host tensor surrounded by canary pages. Used by host sessions and by
    device sessions that stage through pinned/host memory."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None:
        nbytes = int(np.prod(shape)) * dtype.itemsize if len(shape) else dtype.itemsize
        self._raw = np.empty(GUARD_BYTES + nbytes + GUARD_BYTES + ALIGNMENT_BYTES, dtype=np.uint8)
        start = GUARD_BYTES + ((-(self._raw.ctypes.data + GUARD_BYTES)) % ALIGNMENT_BYTES)
        self._front = self._raw[start - GUARD_BYTES : start]
        self._back = self._raw[start + nbytes : start + nbytes + GUARD_BYTES]
        self._front.fill(GUARD_PATTERN)
        self._back.fill(GUARD_PATTERN)
        self.bytes_view = self._raw[start : start + nbytes]
        self.array = self.bytes_view.view(dtype).reshape(tuple(shape))

    def intact(self) -> bool:
        return bool((self._front == GUARD_PATTERN).all() and (self._back == GUARD_PATTERN).all())

    @classmethod
    def from_array(cls, array: np.ndarray) -> "GuardedBuffer":
        buffer = cls(array.shape, array.dtype)
        buffer.array[...] = array
        return buffer


@dataclass
class LaunchJob:
    """Everything the protocol needs for one kernel on one test case."""

    inputs: list[np.ndarray]  # storage arrays, in ABI order
    input_names: list[str]
    output_shape: tuple[int, ...]
    output_dtype: np.dtype  # storage dtype of the output
    scalars: list[int]
    verify: bool = True
    warmup: int = 0
    repeats: int = 0
    reference_kernel: Any | None = None  # a loaded kernel handle for interleaved A/B, or None


class ProtocolError(Exception):
    """Raised by the protocol (memory-safety violations) or by a session (launch failures)."""

    def __init__(self, message: str, kind: str = "runtime") -> None:  # kind: "memory" | "runtime"
        super().__init__(message)
        self.message = message
        self.kind = kind


@dataclass
class ProtocolResult:
    output: np.ndarray | None = None
    poison_output: np.ndarray | None = None
    fast_path_active: int | None = None
    poison_left_in_output: bool = False
    prefill_dependent: bool = False
    timings_ms: list[float] = field(default_factory=list)
    host_timings_ms: list[float] = field(default_factory=list)
    reference_timings_ms: list[float] = field(default_factory=list)
    cpu_wall_ratio: float | None = None
    threads_available: int | None = None
    timing_source: str = TIMING_HOST_WALL
    h2d_ms: float | None = None  # total upload time (excluded from timings)
    d2h_ms: float | None = None  # total download time during verification (excluded from timings)

    def to_payload(self) -> dict:
        """JSON-safe dict (arrays are saved to files by the caller)."""
        return {
            "ok": True,
            "fast_path_active": self.fast_path_active,
            "poison_left_in_output": self.poison_left_in_output,
            "prefill_dependent": self.prefill_dependent,
            "timings_ms": self.timings_ms,
            "host_timings_ms": self.host_timings_ms,
            "reference_timings_ms": self.reference_timings_ms,
            "cpu_wall_ratio": self.cpu_wall_ratio,
            "threads_available": self.threads_available,
            "timing_source": self.timing_source,
            "h2d_ms": self.h2d_ms,
            "d2h_ms": self.d2h_ms,
        }


class KernelSession(ABC):
    """Device primitives for one (kernel artifact, test case) execution. See the module docstring
    for the contract. `timing_source` tells the evaluator which clock produced `timings_ms`."""

    timing_source: str = TIMING_DEVICE_TIMER

    @abstractmethod
    def load_kernel(self, artifact: str, symbol: str) -> Any:
        """Load the compiled artifact and return an opaque kernel handle for `launch`."""

    @abstractmethod
    def upload(self, name: str, array: np.ndarray) -> Any:
        """H2D: place an input tensor where the kernel expects it; return a buffer handle."""

    @abstractmethod
    def allocate_output(self, shape: Sequence[int], dtype: np.dtype) -> Any:
        """Allocate the output buffer on the device; contents are set by `fill_bytes` before use."""

    @abstractmethod
    def fill_bytes(self, buffer: Any, byte: int) -> None:
        """Set every byte of `buffer` to `byte` (memset on the device)."""

    @abstractmethod
    def download(self, buffer: Any) -> np.ndarray:
        """D2H: return a host copy of `buffer` as a storage-dtype array of the right shape."""

    @abstractmethod
    def launch(self, kernel: Any, inputs: Sequence[Any], output: Any, scalars: Sequence[int]) -> None:
        """Synchronous launch: return only after the kernel completed."""

    @abstractmethod
    def timed_launch(self, kernel: Any, inputs: Sequence[Any], output: Any, scalars: Sequence[int]) -> float:
        """Launch and return the kernel-only duration in milliseconds measured with the device timer."""

    def fast_path_flag(self, kernel: Any) -> int | None:
        return None

    def reset_fast_path_flag(self, kernel: Any) -> None:
        """Set the kernel's flag to 0 before a verification call (no-op when not exported)."""

    def memory_violations(self, input_names: Sequence[str], inputs: Sequence[Any], output: Any) -> list[str]:
        """Guard checks after a verification call (canary pages, guard allocations). Const-input
        mutation is detected by the protocol itself through `download`."""
        return []

    def threads_available(self) -> int | None:
        return None

    def close(self) -> None:
        """Release device resources."""


def run_protocol(session: KernelSession, kernel: Any, job: LaunchJob) -> ProtocolResult:
    """Run verification and/or timing of `kernel` on `job` through `session`.

    Raises ProtocolError for memory-safety violations; a crash inside the kernel is expected to
    kill the (isolated) process or raise from the session, which the backend maps to a
    RunResult.
    """
    result = ProtocolResult(timing_source=session.timing_source)

    h2d_started = time.perf_counter()
    input_handles = [session.upload(name, array) for name, array in zip(job.input_names, job.inputs)]
    output_handle = session.allocate_output(job.output_shape, job.output_dtype)
    result.h2d_ms = (time.perf_counter() - h2d_started) * 1e3
    d2h_total = 0.0

    def check_memory(label: str) -> None:
        nonlocal d2h_total
        violations = list(session.memory_violations(job.input_names, input_handles, output_handle))
        started = time.perf_counter()
        for name, original, handle in zip(job.input_names, job.inputs, input_handles):
            after = session.download(handle)
            if not np.array_equal(np.ascontiguousarray(after).view(np.uint8), np.ascontiguousarray(original).view(np.uint8)):
                violations.append(f"modified const input '{name}'")
        d2h_total += (time.perf_counter() - started) * 1e3
        if violations:
            raise ProtocolError(f"memory safety violation{label}: " + "; ".join(violations), kind="memory")

    if job.verify:
        session.fill_bytes(output_handle, NAN_PREFILL_BYTE)
        session.reset_fast_path_flag(kernel)
        session.launch(kernel, input_handles, output_handle, job.scalars)
        check_memory("")
        # `download` may hand back a view of the live buffer (host sessions do), so copy before
        # the buffer is reused for the poison run.
        started = time.perf_counter()
        first_output = np.array(session.download(output_handle), copy=True, order="C")
        d2h_total += (time.perf_counter() - started) * 1e3
        result.fast_path_active = session.fast_path_flag(kernel)

        session.fill_bytes(output_handle, POISON_BYTE)
        session.launch(kernel, input_handles, output_handle, job.scalars)
        check_memory(" (poison run)")
        started = time.perf_counter()
        second_output = np.array(session.download(output_handle), copy=True, order="C")
        d2h_total += (time.perf_counter() - started) * 1e3

        itemsize = np.dtype(job.output_dtype).itemsize
        element_bytes = second_output.view(np.uint8).reshape(-1, itemsize) if second_output.size else np.empty((0, itemsize), np.uint8)
        result.poison_left_in_output = bool(second_output.size and (element_bytes == POISON_BYTE).all(axis=1).any())
        result.prefill_dependent = not np.array_equal(first_output.view(np.uint8), second_output.view(np.uint8))
        result.output = first_output
        if result.prefill_dependent:
            result.poison_output = second_output
        result.d2h_ms = d2h_total

    if job.repeats > 0:
        reference = job.reference_kernel

        def candidate_call() -> None:
            session.launch(kernel, input_handles, output_handle, job.scalars)

        def reference_call() -> None:
            session.launch(reference, input_handles, output_handle, job.scalars)

        for _ in range(job.warmup):
            candidate_call()
            if reference is not None:
                reference_call()

        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        candidate_ms: list[float] = []
        host_ms: list[float] = []
        reference_ms: list[float] = []
        if reference is None:
            for _ in range(job.repeats):
                host_started = time.perf_counter_ns()
                candidate_ms.append(session.timed_launch(kernel, input_handles, output_handle, job.scalars))
                host_ms.append((time.perf_counter_ns() - host_started) / 1e6)
        else:
            # Interleaved A/B. Each timed call is preceded by an untimed call of the same kernel:
            # switching kernels changes the cache state and, on VMs, lets idle worker threads
            # sleep, so a strictly alternating pattern penalises the kernel that runs second.
            for _ in range(job.repeats):
                reference_call()
                reference_ms.append(session.timed_launch(reference, input_handles, output_handle, job.scalars))
                candidate_call()
                host_started = time.perf_counter_ns()
                candidate_ms.append(session.timed_launch(kernel, input_handles, output_handle, job.scalars))
                host_ms.append((time.perf_counter_ns() - host_started) / 1e6)
        wall_elapsed = time.perf_counter() - wall_before
        cpu_elapsed = time.process_time() - cpu_before

        result.timings_ms = candidate_ms
        result.host_timings_ms = host_ms
        result.reference_timings_ms = reference_ms
        result.cpu_wall_ratio = (cpu_elapsed / wall_elapsed) if wall_elapsed > 0 else None
        result.threads_available = session.threads_available()

    session.close()
    return result


__all__ = [
    "ALIGNMENT_BYTES", "GUARD_BYTES", "GUARD_PATTERN", "NAN_PREFILL_BYTE", "POISON_BYTE", "FAST_PATH_SYMBOL",
    "TIMING_HOST_WALL", "TIMING_DEVICE_TIMER", "GuardedBuffer", "LaunchJob", "ProtocolError", "ProtocolResult",
    "KernelSession", "run_protocol",
]
