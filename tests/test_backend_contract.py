"""Backend / launch-ABI contract: the shared protocol runs unchanged over a device-style session
(backend-owned buffers, device timer, transfers excluded), the crash-test runner disables core
dumps, and the accelerator stub pins the contract (requires gcc)."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from kopt_agent.backends import get_backend
from kopt_agent.backends.base import Backend, LaunchABI
from kopt_agent.backends.cpu_c import CpuCBackend
from kopt_agent.backends.protocol import TIMING_DEVICE_TIMER, FAST_PATH_SYMBOL, KernelSession, ProtocolError
from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import Evaluator, TrialStatus
from kopt_agent.roofline import MachinePeaks, measure_peaks
from kopt_agent.spec import OperatorSpec
from kopt_agent.spec import TestCase as OperatorCase  # aliased so pytest does not try to collect it
from ops import build_operator

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


# ---- 1. no core dumps from the crash-classification subprocess ------------------------------

def test_runner_subprocess_disables_core_dumps():
    code = (
        "import resource, kopt_agent.runner as runner\n"
        "runner.disable_core_dumps()\n"
        "print(resource.getrlimit(resource.RLIMIT_CORE)[0])\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert completed.stdout.strip() == "0"


def test_crashing_kernel_leaves_no_core_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = get_backend("cpu_c")
    bundle = build_operator("matmul", (4, 4, 4))
    evaluator = Evaluator(bundle.spec, backend, warmup=1, repeats=1, run_timeout_seconds=10)
    crash = Candidate(source=f"#include <stddef.h>\n{bundle.spec.c_signature} {{ float* p = 0; p[123456789] = 1.0f; (void)C; }}\n", origin="test")
    result = evaluator.evaluate(crash)
    assert result.status is TrialStatus.RUNTIME_ERROR and "SIGSEGV" in result.message
    assert not [path for path in tmp_path.iterdir() if path.name == "core" or path.name.startswith("core.")]
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text().split()
    assert "core" in gitignore and "core.*" in gitignore


# ---- 2. device-style backend through the unchanged evaluator --------------------------------

class FakeDeviceSession(KernelSession):
    """Mimics an accelerator: 'device memory' is a private pool of arrays the kernel reads
    from; the protocol only sees opaque integer handles, H2D/D2H are explicit copies, and the
    'device timer' brackets just the kernel call."""

    timing_source = TIMING_DEVICE_TIMER

    def __init__(self, pointer_count: int, scalar_count: int) -> None:
        self.pool: dict[int, np.ndarray] = {}
        self.pointer_count = pointer_count
        self.scalar_count = scalar_count
        self.h2d_bytes = 0
        self.d2h_calls = 0
        self.launches = 0
        self.closed = False

    def load_kernel(self, artifact: str, symbol: str) -> Any:
        library = ctypes.CDLL(artifact)
        try:
            function = getattr(library, symbol)
        except AttributeError as error:
            raise ProtocolError(f"missing launcher {symbol}") from error
        function.argtypes = [ctypes.c_void_p] * self.pointer_count + [ctypes.c_int] * self.scalar_count
        function.restype = None
        try:
            flag = ctypes.c_int.in_dll(library, FAST_PATH_SYMBOL)
        except ValueError:
            flag = None
        return {"library": library, "function": function, "flag": flag}

    def upload(self, name: str, array: np.ndarray) -> int:
        handle = len(self.pool) + 1
        self.pool[handle] = np.array(array, copy=True, order="C")
        self.h2d_bytes += array.nbytes
        return handle

    def allocate_output(self, shape: Sequence[int], dtype: np.dtype) -> int:
        handle = len(self.pool) + 1
        self.pool[handle] = np.empty(tuple(shape), dtype=dtype)
        return handle

    def fill_bytes(self, buffer: int, byte: int) -> None:
        self.pool[buffer].view(np.uint8).fill(byte)

    def download(self, buffer: int) -> np.ndarray:
        self.d2h_calls += 1
        return self.pool[buffer].copy()

    def _call(self, kernel: Any, inputs: Sequence[int], output: int, scalars: Sequence[int]) -> None:
        pointers = [self.pool[handle].ctypes.data_as(ctypes.c_void_p) for handle in inputs] + [self.pool[output].ctypes.data_as(ctypes.c_void_p)]
        kernel["function"](*pointers, *[ctypes.c_int(value) for value in scalars])
        self.launches += 1

    def launch(self, kernel: Any, inputs: Sequence[int], output: int, scalars: Sequence[int]) -> None:
        self._call(kernel, inputs, output, scalars)

    def timed_launch(self, kernel: Any, inputs: Sequence[int], output: int, scalars: Sequence[int]) -> float:
        started = time.perf_counter_ns()
        self._call(kernel, inputs, output, scalars)
        return (time.perf_counter_ns() - started) / 1e6

    def fast_path_flag(self, kernel: Any) -> int | None:
        return None if kernel["flag"] is None else int(kernel["flag"].value)

    def reset_fast_path_flag(self, kernel: Any) -> None:
        if kernel["flag"] is not None:
            kernel["flag"].value = 0

    def close(self) -> None:
        self.closed = True


class FakeDeviceBackend(CpuCBackend):
    """gcc compiles the 'device' code; execution goes through Backend.run + FakeDeviceSession
    instead of the CPU subprocess runner."""

    name = "fake_device"
    launch_abi = LaunchABI(pointer_space="device", stream_argument=False, synchronous_launch=True, entry_kind="launcher")
    run = Backend.run  # the default protocol-driven implementation

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sessions: list[FakeDeviceSession] = []

    def open_session(self, artifact: Path, spec: OperatorSpec, case: OperatorCase) -> KernelSession:
        session = FakeDeviceSession(len(case.inputs) + 1, len(case.scalars))
        self.sessions.append(session)
        return session

    def measure_peaks(self) -> MachinePeaks:
        return MachinePeaks(compute_gflops=1000.0, bandwidth_gbps=500.0, dispatch_overhead_ms=0.005, call_overhead_ms=0.005, source="datasheet")


@pytest.fixture(scope="module")
def device_backend():
    return FakeDeviceBackend()


@pytest.fixture(scope="module")
def device_matmul(device_backend):
    bundle = build_operator("matmul", (24, 20, 16))
    return bundle, Evaluator(bundle.spec, device_backend, warmup=1, repeats=3, run_timeout_seconds=10)


def _kernel(bundle, body: str, **kwargs) -> Candidate:
    return Candidate(source=f"#include <stddef.h>\n{bundle.spec.c_signature} {{ {body} }}\n", origin="test", **kwargs)


def test_device_backend_reports_device_time_and_transfers_separately(device_backend, device_matmul):
    bundle, evaluator = device_matmul
    result = evaluator.evaluate(Candidate(source=bundle.baseline_source, origin="baseline"))
    assert result.status is TrialStatus.OK, result.message
    assert result.timing_source == TIMING_DEVICE_TIMER
    assert result.transfer_ms is not None and result.transfer_ms["h2d"] is not None and result.transfer_ms["d2h"] is not None
    assert result.thread_utilization is None and not result.host_bound  # CPU heuristics do not apply to device kernels
    assert result.per_shape["primary"]["latency_ms"] > 0
    session = device_backend.sessions[-1]
    assert session.closed and session.h2d_bytes > 0


def test_device_backend_keeps_every_correctness_guard(device_matmul):
    bundle, evaluator = device_matmul
    body_ok = (
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; }"
    )
    half = _kernel(bundle, body_ok.replace("i < M; i++", "i < M; i += 2"))
    assert evaluator.evaluate(half).status is TrialStatus.INCORRECT

    accumulate = _kernel(
        bundle,
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { if (C[(size_t)i*N+j] != C[(size_t)i*N+j]) C[(size_t)i*N+j] = 0.0f; "
        "for (int k = 0; k < K; k++) C[(size_t)i*N+j] += A[(size_t)i*K+k] * B[(size_t)k*N+j]; }",
    )
    result = evaluator.evaluate(accumulate)
    assert result.status is TrialStatus.INCORRECT
    assert any(fragment in result.message for fragment in ("previous buffer contents", "poison prefill", "outside tolerance"))

    mutate = _kernel(bundle, "(void)C; ((float*)A)[0] = 0.0f;")
    result = evaluator.evaluate(mutate)
    assert result.status is TrialStatus.INCORRECT and "modified const input" in result.message

    honest = Candidate(
        source=f"#include <stddef.h>\nint kopt_fast_path_active = 0;\n{bundle.spec.c_signature} {{ kopt_fast_path_active = (N % 4 == 0); {body_ok} }}\n",
        origin="test", fast_path_predicate="N % 4 == 0",
    )
    result = evaluator.evaluate(honest)
    assert result.status is TrialStatus.OK, result.message
    assert "primary" in result.fast_path["activated_on"]


def test_device_backend_interleaved_ab_and_roofline_from_backend_peaks(device_backend, device_matmul):
    bundle, evaluator = device_matmul
    baseline = Candidate(source=bundle.baseline_source, origin="baseline")
    evaluator.reference_artifact = device_backend.compile(baseline, bundle.spec).artifact
    tuned = evaluator.evaluate(bundle.select_template(None).default_candidate())
    assert tuned.status is TrialStatus.OK, tuned.message
    assert tuned.ab_speedup is not None and tuned.ab_speedup > 0

    peaks = measure_peaks(device_backend, use_cache=False)
    assert peaks.source == "datasheet" and peaks.compute_gflops == 1000.0


# ---- accelerator stub pins the contract ----------------------------------------------------

def test_accelerator_stub_documents_the_contract():
    from kopt_agent.backends.accelerator_stub import AcceleratorBackendTemplate, AcceleratorSessionTemplate

    backend = AcceleratorBackendTemplate()
    assert backend.launch_abi.pointer_space == "device" and backend.launch_abi.stream_argument
    assert "DEVICE pointers" in backend.language_guidance()
    with pytest.raises(NotImplementedError):
        backend.compile(Candidate(source="", origin="test"), build_operator("matmul", (4, 4, 4)).spec)
    session = AcceleratorSessionTemplate(stream=None, scalar_count=3, pointer_count=3)
    assert session.timing_source == TIMING_DEVICE_TIMER
    for method in ("load_kernel", "upload", "allocate_output", "fill_bytes", "download", "launch", "timed_launch"):
        assert callable(getattr(session, method))
    description = LaunchABI(pointer_space="device", stream_argument=True, synchronous_launch=False, entry_kind="launcher").describe()
    assert "device pointers" in description and "stream" in description
