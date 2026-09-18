"""Isolated kernel runner. Reads one JSON job from stdin, loads the shared object with ctypes,
runs the kernel and prints one JSON line with the result.

Verification protocol (when `verify` is true):
  1. output prefilled with 0xFF bytes (NaN)        -> saved as the result to compare
  2. output prefilled with a poison value          -> must be bit-identical to run 1, so the
     kernel neither depends on the previous buffer contents nor leaves poison behind
  3. canary pages around every tensor and input snapshots are checked after each run
  4. if the library exports `int kopt_fast_path_active`, its value after run 1 is reported

Timing protocol (when `repeats` > 0): warmup, then timed calls. If an `ab` artifact is given
the candidate and the reference kernel are called alternately in the same process so slow
drift (frequency, noisy neighbours) hits both equally. CPU time is sampled around the timed
block so the caller can see whether the kernel actually kept its threads busy.

Kept dependency-free on the rest of the package so a crash here is always attributable to
the generated kernel, not to the agent.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import numpy as np

ALIGNMENT_BYTES = 64
GUARD_BYTES = 4096
GUARD_PATTERN = 0xA5
POISON_BYTE = 0x5C  # 0x5C5C5C5C as float32 = 2.48e17, as fp16 0x5C5C = 279; never a plausible result
FAST_PATH_SYMBOL = "kopt_fast_path_active"


class GuardedBuffer:
    """A 64-byte aligned tensor surrounded by canary pages."""

    def __init__(self, shape: list[int] | tuple[int, ...], dtype: np.dtype) -> None:
        nbytes = int(np.prod(shape)) * dtype.itemsize if len(shape) else dtype.itemsize
        self._raw = np.empty(GUARD_BYTES + nbytes + GUARD_BYTES + ALIGNMENT_BYTES, dtype=np.uint8)
        start = GUARD_BYTES + ((-(self._raw.ctypes.data + GUARD_BYTES)) % ALIGNMENT_BYTES)
        self._front = self._raw[start - GUARD_BYTES : start]
        self._back = self._raw[start + nbytes : start + nbytes + GUARD_BYTES]
        self._front.fill(GUARD_PATTERN)
        self._back.fill(GUARD_PATTERN)
        self.bytes_view = self._raw[start : start + nbytes]
        self.array = self.bytes_view.view(dtype).reshape(shape)

    def intact(self) -> bool:
        return bool((self._front == GUARD_PATTERN).all() and (self._back == GUARD_PATTERN).all())

    @classmethod
    def from_array(cls, array: np.ndarray) -> "GuardedBuffer":
        buffer = cls(array.shape, array.dtype)
        buffer.array[...] = array
        return buffer


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


def _fail(message: str, kind: str = "runtime") -> int:
    print(json.dumps({"ok": False, "error_kind": kind, "error": message}))
    return 0


def _memory_violations(names, buffers, snapshots, output_buffer) -> list[str]:
    violations = []
    for name, buffer, snapshot in zip(names, buffers, snapshots):
        if not buffer.intact():
            violations.append(f"wrote outside input '{name}'")
        elif not np.array_equal(buffer.bytes_view, snapshot):
            violations.append(f"modified const input '{name}'")
    if not output_buffer.intact():
        violations.append("wrote outside the output buffer")
    return violations


def _timed_calls(call, repeats: int) -> list[float]:
    timings_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    return timings_ms


def main() -> int:
    job = json.loads(sys.stdin.read())
    scalar_count = len(job["scalars"])
    pointer_count = len(job["input_names"]) + 1
    try:
        kernel = LoadedKernel(job["artifact"], job["symbol"], pointer_count, scalar_count)
    except (OSError, AttributeError) as error:
        return _fail(f"cannot load symbol '{job['symbol']}': {error}")

    with np.load(job["inputs_path"]) as archive:
        input_buffers = [GuardedBuffer.from_array(archive[name]) for name in job["input_names"]]
    input_snapshots = [buffer.bytes_view.copy() for buffer in input_buffers]
    output_buffer = GuardedBuffer(job["output_shape"], np.dtype(job["output_dtype"]))
    output = output_buffer.array

    pointers = [buffer.array.ctypes.data_as(ctypes.c_void_p) for buffer in input_buffers] + [output.ctypes.data_as(ctypes.c_void_p)]
    scalars = [ctypes.c_int(value) for value in job["scalars"]]

    def call_candidate() -> None:
        kernel.function(*pointers, *scalars)

    result: dict = {"ok": True}

    if job.get("verify", True):
        # All-ones bytes decode to NaN in every float format (fp32/fp16/bf16), so unwritten cells show up as NaN.
        output_buffer.bytes_view.fill(0xFF)
        if kernel.fast_path_flag is not None:
            kernel.fast_path_flag.value = 0
        call_candidate()
        violations = _memory_violations(job["input_names"], input_buffers, input_snapshots, output_buffer)
        if violations:
            return _fail("memory safety violation: " + "; ".join(violations), kind="memory")
        first_output = output.copy()
        result["fast_path_active"] = None if kernel.fast_path_flag is None else int(kernel.fast_path_flag.value)

        output_buffer.bytes_view.fill(POISON_BYTE)
        call_candidate()
        violations = _memory_violations(job["input_names"], input_buffers, input_snapshots, output_buffer)
        if violations:
            return _fail("memory safety violation (poison run): " + "; ".join(violations), kind="memory")
        itemsize = np.dtype(job["output_dtype"]).itemsize
        element_bytes = output_buffer.bytes_view.reshape(-1, itemsize)
        result["poison_left_in_output"] = bool(output.size and (element_bytes == POISON_BYTE).all(axis=1).any())
        result["prefill_dependent"] = not np.array_equal(first_output.view(np.uint8), output.view(np.uint8))
        np.save(job["output_path"], first_output)
        if result["prefill_dependent"] and job.get("poison_output_path"):
            np.save(job["poison_output_path"], output)

    repeats = int(job.get("repeats", 0))
    if repeats > 0:
        ab = job.get("ab")
        reference_call = None
        if ab:
            try:
                reference = LoadedKernel(ab["artifact"], ab["symbol"], pointer_count, scalar_count)
            except (OSError, AttributeError) as error:
                return _fail(f"cannot load A/B reference kernel: {error}")
            reference_call = lambda: reference.function(*pointers, *scalars)  # noqa: E731

        for _ in range(int(job.get("warmup", 0))):
            call_candidate()
            if reference_call is not None:
                reference_call()

        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        if reference_call is None:
            candidate_ms = _timed_calls(call_candidate, repeats)
            reference_ms: list[float] = []
        else:
            # Interleaved A/B. Each timed call is preceded by an untimed call of the same kernel:
            # switching kernels changes the cache state and, on VMs, lets idle worker threads
            # sleep, so a strictly alternating pattern penalises the kernel that runs second.
            candidate_ms, reference_ms = [], []
            for _ in range(repeats):
                reference_call()
                reference_ms.extend(_timed_calls(reference_call, 1))
                call_candidate()
                candidate_ms.extend(_timed_calls(call_candidate, 1))
        wall_elapsed = time.perf_counter() - wall_before
        cpu_elapsed = time.process_time() - cpu_before

        result["timings_ms"] = candidate_ms
        result["reference_timings_ms"] = reference_ms
        # On a CPU backend host and device are the same clock; a GPU backend reports device
        # (event) time here and host time separately.
        result["host_timings_ms"] = candidate_ms
        result["cpu_wall_ratio"] = (cpu_elapsed / wall_elapsed) if wall_elapsed > 0 else None
        result["threads_available"] = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
