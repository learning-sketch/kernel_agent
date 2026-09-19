"""Documented template for an accelerator (GPU / NPU / DSP) backend.

Copy this file, rename the classes, and fill in the bodies marked `raise NotImplementedError`.
Nothing in `agent.py`, `evaluator.py`, `history.py` or the generators has to change: they only
consume `CompileResult` / `RunResult`, and the verification + timing protocol
(`backends/protocol.py`) is driven for you by `Backend.run`.

What a candidate looks like on an accelerator
---------------------------------------------
The candidate source contains the device kernel *and* a host launcher with the operator's
`c_signature`, e.g. for matmul

    extern "C" void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K, void* stream);

where A/B/C are **device** pointers and `stream` is the queue to enqueue on (present only when
`launch_abi.stream_argument` is True). The launcher picks the grid, enqueues the kernel and
returns; the session synchronizes. Fast paths keep the same contract as on CPU: the launcher
exports `int kopt_fast_path_active` (a host-side int it sets when it chose the fast path).

Timing contract
---------------
`timings_ms` must be kernel-only device time (events / timestamps around the kernel). Host
launch latency is what `host_timings_ms` is for. H2D/D2H happen in `upload` / `download`; the
protocol measures them itself and reports `h2d_ms` / `d2h_ms` separately - never fold them into
the kernel time.

Roofline
--------
Return device peaks from `measure_peaks` (datasheet numbers or your own probe kernels). The
portable C probes in `roofline.py` only make sense for backends that run plain C on the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from kopt_agent.backends.base import Backend, CompileResult, LaunchABI
from kopt_agent.backends.protocol import TIMING_DEVICE_TIMER, KernelSession
from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import DType
from kopt_agent.spec import OperatorSpec, TestCase

if TYPE_CHECKING:
    from kopt_agent.roofline import MachinePeaks


class AcceleratorSessionTemplate(KernelSession):
    """One (artifact, test case) execution on the device. Handles returned by `upload` /
    `allocate_output` / `load_kernel` are opaque to the protocol; use whatever your runtime
    gives you (device pointers, buffer objects, module + function handles)."""

    timing_source = TIMING_DEVICE_TIMER

    def __init__(self, stream: Any, scalar_count: int, pointer_count: int) -> None:
        self.stream = stream  # queue every launch is enqueued on
        self.scalar_count = scalar_count
        self.pointer_count = pointer_count

    def load_kernel(self, artifact: str, symbol: str) -> Any:
        """Load the compiled module (shared object / cubin / hsaco / SPIR-V) and resolve the host
        launcher `symbol`. Raise ProtocolError(message, "runtime") when the symbol is missing."""
        raise NotImplementedError

    def upload(self, name: str, array: np.ndarray) -> Any:
        """Allocate device memory of `array.nbytes` (optionally with guard regions for
        `memory_violations`) and copy the host array into it (H2D). Return the device handle."""
        raise NotImplementedError

    def allocate_output(self, shape: Sequence[int], dtype: np.dtype) -> Any:
        """Allocate the output on the device (contents are set by `fill_bytes`)."""
        raise NotImplementedError

    def fill_bytes(self, buffer: Any, byte: int) -> None:
        """Device memset of the whole buffer to `byte` (0xFF = NaN prefill, 0x5C = poison)."""
        raise NotImplementedError

    def download(self, buffer: Any) -> np.ndarray:
        """D2H copy into a host array with the buffer's shape and storage dtype. Called only
        during verification, never inside the timed block."""
        raise NotImplementedError

    def launch(self, kernel: Any, inputs: Sequence[Any], output: Any, scalars: Sequence[int]) -> None:
        """Call the host launcher with (device pointers..., int scalars..., stream) and
        synchronize the stream before returning. Convert runtime errors (illegal address,
        launch failure) into ProtocolError(message, "runtime")."""
        raise NotImplementedError

    def timed_launch(self, kernel: Any, inputs: Sequence[Any], output: Any, scalars: Sequence[int]) -> float:
        """Record a start event, launch, record a stop event, synchronize, return elapsed ms
        between the events - kernel time only, no transfers."""
        raise NotImplementedError

    def fast_path_flag(self, kernel: Any) -> int | None:
        """Read the launcher's `int kopt_fast_path_active` (host global in the loaded module).
        Return None when the module does not export it."""
        return None

    def reset_fast_path_flag(self, kernel: Any) -> None:
        """Set the flag to 0 before a verification launch (no-op when not exported)."""

    def memory_violations(self, input_names: Sequence[str], inputs: Sequence[Any], output: Any) -> list[str]:
        """Optional: check guard regions around device allocations after a verification launch.
        Const-input mutation is already detected by the protocol via `download`."""
        return []

    def threads_available(self) -> int | None:
        """Return None: the CPU/wall utilisation heuristic does not apply to device kernels."""
        return None

    def close(self) -> None:
        """Free device allocations, unload the module, destroy events."""


class AcceleratorBackendTemplate(Backend):
    """Fill-in-the-blanks backend. Register it in `backends/__init__.py` under a name and select
    it with `kopt run --backend <name>`."""

    name = "accelerator_template"
    launch_abi = LaunchABI(
        pointer_space="device",
        stream_argument=True,
        synchronous_launch=False,
        fast_path_flag="global_int",
        entry_kind="launcher",
    )

    def __init__(self, device_index: int = 0, work_dir: Path | None = None) -> None:
        self.device_index = device_index
        self.work_dir = work_dir

    def compile(self, candidate: Candidate, spec: OperatorSpec) -> CompileResult:
        """Run the device compiler (nvcc / hipcc / your toolchain) on `candidate.source` with
        `candidate.extra_compile_flags`, producing a loadable module. Fill `CompileResult.profile`
        with whatever the compiler reports (registers per thread, spill bytes, occupancy) so the
        LLM sees it in the standard layout."""
        raise NotImplementedError

    def open_session(self, artifact: Path, spec: OperatorSpec, case: TestCase) -> KernelSession:
        """Create a stream/queue on `self.device_index` and return the session for this case."""
        raise NotImplementedError

    def hardware_summary(self) -> str:
        """One line for logs and LLM prompts: device name, SM/CU count, memory, clocks."""
        raise NotImplementedError

    def supports_dtype(self, dtype: DType) -> bool:
        """fp16/bf16 are usually native on accelerators; return False only for types the
        toolchain cannot spell."""
        return dtype.name in {"fp32", "fp16", "bf16"}

    def measure_peaks(self) -> "MachinePeaks | None":
        """Datasheet or probed peaks: FMA GFLOP/s, HBM/DRAM GB/s, kernel launch floor (ms).
        `dispatch_overhead_ms` is the cost of one launch, `call_overhead_ms` the floor of one
        launcher call from the host (usually the same number on accelerators)."""
        raise NotImplementedError

    def portable_compile_command(self) -> list[str] | None:
        """E.g. ["nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC", "{source}", "-o", "{artifact}"]."""
        return None

    def language_guidance(self) -> str:
        return (
            "Target: accelerator. Reply with one complete source file that contains the device kernel(s) and a host "
            "launcher with exactly the required prototype plus a trailing `void* stream` argument. Pointers are DEVICE "
            "pointers: never dereference them on the host. The launcher enqueues onto the given stream and returns. "
            "Fast paths follow the usual contract (`// fast_path:` comment + exported `int kopt_fast_path_active`). "
            + self.launch_abi.describe()
        )
