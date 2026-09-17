"""Isolated kernel runner. Reads one JSON job from stdin, loads the shared object with ctypes,
runs the kernel (warmup + timed repeats) and prints one JSON line with the result.

Kept dependency-free on the rest of the package so a crash here is always attributable to
the generated kernel, not to the agent.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time

import numpy as np

ALIGNMENT_BYTES = 64
GUARD_BYTES = 4096
GUARD_PATTERN = 0xA5


class GuardedBuffer:
    """A 64-byte aligned tensor surrounded by canary pages.

    Out-of-bounds writes that land just past a tensor (the classic "tile loop ignores the
    remainder" bug) do not always corrupt visible output, so the canaries are checked
    explicitly after the first kernel call.
    """

    def __init__(self, shape: list[int] | tuple[int, ...], dtype: np.dtype) -> None:
        nbytes = int(np.prod(shape)) * dtype.itemsize if len(shape) else dtype.itemsize
        self._raw = np.empty(GUARD_BYTES + nbytes + GUARD_BYTES + ALIGNMENT_BYTES, dtype=np.uint8)
        start = GUARD_BYTES + ((-(self._raw.ctypes.data + GUARD_BYTES)) % ALIGNMENT_BYTES)
        self._front = self._raw[start - GUARD_BYTES : start]
        self._back = self._raw[start + nbytes : start + nbytes + GUARD_BYTES]
        self._front.fill(GUARD_PATTERN)
        self._back.fill(GUARD_PATTERN)
        self.array = self._raw[start : start + nbytes].view(dtype).reshape(shape)

    def intact(self) -> bool:
        return bool((self._front == GUARD_PATTERN).all() and (self._back == GUARD_PATTERN).all())

    @classmethod
    def from_array(cls, array: np.ndarray) -> "GuardedBuffer":
        buffer = cls(array.shape, array.dtype)
        buffer.array[...] = array
        return buffer


def main() -> int:
    job = json.loads(sys.stdin.read())
    try:
        library = ctypes.CDLL(job["artifact"])
        function = getattr(library, job["symbol"])
    except (OSError, AttributeError) as error:
        print(json.dumps({"ok": False, "error": f"cannot load symbol '{job['symbol']}': {error}"}))
        return 0

    with np.load(job["inputs_path"]) as archive:
        input_buffers = [GuardedBuffer.from_array(archive[name]) for name in job["input_names"]]
        input_snapshots = [buffer.array.copy() for buffer in input_buffers]
    output_buffer = GuardedBuffer(job["output_shape"], np.dtype(job["output_dtype"]))
    output = output_buffer.array

    pointer_type = ctypes.c_void_p
    function.argtypes = [pointer_type] * (len(input_buffers) + 1) + [ctypes.c_int] * len(job["scalars"])
    function.restype = None

    pointers = [buffer.array.ctypes.data_as(pointer_type) for buffer in input_buffers] + [output.ctypes.data_as(pointer_type)]
    scalars = [ctypes.c_int(value) for value in job["scalars"]]

    # NaN prefill: a kernel that forgets to write part of the output is caught by the checker.
    output.fill(np.nan)
    function(*pointers, *scalars)

    violations = []
    for name, buffer, snapshot in zip(job["input_names"], input_buffers, input_snapshots):
        if not buffer.intact():
            violations.append(f"wrote outside input '{name}'")
        elif not np.array_equal(buffer.array, snapshot, equal_nan=True):
            violations.append(f"modified const input '{name}'")
    if not output_buffer.intact():
        violations.append("wrote outside the output buffer")
    if violations:
        print(json.dumps({"ok": False, "error_kind": "memory", "error": "memory safety violation: " + "; ".join(violations)}))
        return 0

    np.save(job["output_path"], output)

    for _ in range(int(job["warmup"])):
        function(*pointers, *scalars)

    timings_ms: list[float] = []
    for _ in range(int(job["repeats"])):
        started = time.perf_counter_ns()
        function(*pointers, *scalars)
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)

    print(json.dumps({"ok": True, "timings_ms": timings_ms}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
