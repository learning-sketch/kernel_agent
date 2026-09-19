"""Isolated kernel runner for host (CPU) kernels. Reads one JSON job from stdin, loads the shared
object with ctypes, runs the shared verification / timing protocol and prints one JSON line.

The protocol itself (NaN + poison prefills, canary pages, const-input snapshots, fast-path flag
readout, warmup, interleaved A/B timing, CPU/wall ratio) lives in `backends/protocol.py`; this
file only supplies the host-memory `KernelSession` and the process boundary.

Because this process deliberately runs kernels that may crash, core dumps are disabled before
anything is loaded: a few hundred crash-classified candidates would otherwise litter the
working directory with multi-hundred-MB `core` files.

Kept dependency-free on the rest of the package (numpy + protocol.py only) so a crash here is
always attributable to the generated kernel, not to the agent.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from typing import Any, Sequence

import numpy as np

from kopt_agent.backends.protocol import (
    FAST_PATH_SYMBOL,
    TIMING_HOST_WALL,
    GuardedBuffer,
    LaunchJob,
    ProtocolError,
    run_protocol,
)


def disable_core_dumps() -> None:
    """RLIMIT_CORE = 0 for this process and everything it execs. No-op on platforms without
    the resource module (Windows)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):  # pragma: no cover - hard limit already lower / not permitted
        pass


class LoadedKernel:
    def __init__(self, artifact: str, symbol: str, pointer_count: int, scalar_count: int) -> None:
        self.library = ctypes.CDLL(artifact)
        self.function = getattr(self.library, symbol)
        self.function.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * scalar_count
        self.function.restype = None
        try:
            self.fast_path_flag: ctypes.c_int | None = ctypes.c_int.in_dll(self.library, FAST_PATH_SYMBOL)
        except ValueError:
            self.fast_path_flag = None


class HostSession:
    """KernelSession over host memory: buffers are guarded numpy arrays, the launcher is the
    kernel symbol itself (host pointers, no stream), the timer is the host wall clock."""

    timing_source = TIMING_HOST_WALL

    def __init__(self, pointer_count: int, scalar_count: int) -> None:
        self.pointer_count = pointer_count
        self.scalar_count = scalar_count

    def load_kernel(self, artifact: str, symbol: str) -> LoadedKernel:
        try:
            return LoadedKernel(artifact, symbol, self.pointer_count, self.scalar_count)
        except (OSError, AttributeError) as error:
            raise ProtocolError(f"cannot load symbol '{symbol}' from {artifact}: {error}") from error

    def upload(self, name: str, array: np.ndarray) -> GuardedBuffer:
        return GuardedBuffer.from_array(array)

    def allocate_output(self, shape: Sequence[int], dtype: np.dtype) -> GuardedBuffer:
        return GuardedBuffer(shape, np.dtype(dtype))

    def fill_bytes(self, buffer: GuardedBuffer, byte: int) -> None:
        buffer.bytes_view.fill(byte)

    def download(self, buffer: GuardedBuffer) -> np.ndarray:
        return buffer.array

    @staticmethod
    def _arguments(inputs: Sequence[GuardedBuffer], output: GuardedBuffer, scalars: Sequence[int]) -> list[Any]:
        pointers = [buffer.array.ctypes.data_as(ctypes.c_void_p) for buffer in inputs] + [output.array.ctypes.data_as(ctypes.c_void_p)]
        return pointers + [ctypes.c_int(value) for value in scalars]

    def launch(self, kernel: LoadedKernel, inputs: Sequence[GuardedBuffer], output: GuardedBuffer, scalars: Sequence[int]) -> None:
        kernel.function(*self._arguments(inputs, output, scalars))

    def timed_launch(self, kernel: LoadedKernel, inputs: Sequence[GuardedBuffer], output: GuardedBuffer, scalars: Sequence[int]) -> float:
        arguments = self._arguments(inputs, output, scalars)
        started = time.perf_counter_ns()
        kernel.function(*arguments)
        return (time.perf_counter_ns() - started) / 1e6

    def fast_path_flag(self, kernel: LoadedKernel) -> int | None:
        return None if kernel.fast_path_flag is None else int(kernel.fast_path_flag.value)

    def reset_fast_path_flag(self, kernel: LoadedKernel) -> None:
        if kernel.fast_path_flag is not None:
            kernel.fast_path_flag.value = 0

    def memory_violations(self, input_names: Sequence[str], inputs: Sequence[GuardedBuffer], output: GuardedBuffer) -> list[str]:
        violations = [f"wrote outside input '{name}'" for name, buffer in zip(input_names, inputs) if not buffer.intact()]
        if not output.intact():
            violations.append("wrote outside the output buffer")
        return violations

    def threads_available(self) -> int:
        return int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))

    def close(self) -> None:
        pass


def _fail(message: str, kind: str = "runtime") -> int:
    print(json.dumps({"ok": False, "error_kind": kind, "error": message}))
    return 0


def main() -> int:
    disable_core_dumps()
    job_data = json.loads(sys.stdin.read())
    scalar_count = len(job_data["scalars"])
    pointer_count = len(job_data["input_names"]) + 1
    session = HostSession(pointer_count, scalar_count)

    try:
        kernel = session.load_kernel(job_data["artifact"], job_data["symbol"])
        reference_kernel = None
        ab = job_data.get("ab")
        if ab and int(job_data.get("repeats", 0)) > 0:
            reference_kernel = session.load_kernel(ab["artifact"], ab["symbol"])
    except ProtocolError as error:
        return _fail(error.message, error.kind)

    with np.load(job_data["inputs_path"]) as archive:
        inputs = [archive[name] for name in job_data["input_names"]]

    job = LaunchJob(
        inputs=inputs,
        input_names=list(job_data["input_names"]),
        output_shape=tuple(job_data["output_shape"]),
        output_dtype=np.dtype(job_data["output_dtype"]),
        scalars=[int(value) for value in job_data["scalars"]],
        verify=bool(job_data.get("verify", True)),
        warmup=int(job_data.get("warmup", 0)),
        repeats=int(job_data.get("repeats", 0)),
        reference_kernel=reference_kernel,
    )
    try:
        result = run_protocol(session, kernel, job)
    except ProtocolError as error:
        return _fail(error.message, error.kind)

    if result.output is not None:
        np.save(job_data["output_path"], result.output)
    if result.poison_output is not None and job_data.get("poison_output_path"):
        np.save(job_data["poison_output_path"], result.poison_output)
    print(json.dumps(result.to_payload()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
