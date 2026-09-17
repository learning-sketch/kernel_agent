"""Roofline model: measure the machine's peak FMA throughput and streaming bandwidth once,
then tell every trial how far it is from the attainable ceiling and which resource bounds it.

"1.7 ms" says nothing to an LLM; "compute-bound, 38% of attainable peak" does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kopt_agent.backends.base import Backend
from kopt_agent.candidate import Candidate
from kopt_agent.hardware import supports_avx512
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase

logger = logging.getLogger("kopt")

# Independent accumulator chains per thread. Too few chains -> FMA-latency bound and the peak is
# underestimated, so several widths are probed and the best one is kept.
FMA_VECTOR_OPTIONS = (8, 12, 16, 24)
FMA_LANES = 16
FMA_ITERS = 1_500_000
# gcc defaults to 256-bit vectors on many AVX-512 parts; the wider encoding can double FMA throughput.
WIDE_VECTOR_FLAGS = ("-mprefer-vector-width=512",)
BANDWIDTH_FLOATS = 16 * 1024 * 1024  # 64 MiB in + 64 MiB out per pass
BANDWIDTH_ITERS = 4

def peak_fma_source(vectors: int) -> str:
    return f"""#include <omp.h>
void peak_fma_kernel(const float* X, float* Y, int GROUPS, int ITERS) {{
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < GROUPS; g++) {{
        float acc[{vectors}][{FMA_LANES}];
        for (int v = 0; v < {vectors}; v++)
            for (int l = 0; l < {FMA_LANES}; l++)
                acc[v][l] = X[(v * {FMA_LANES} + l) % 128];
        const float a = X[0] * 0.001f + 0.999f;
        const float b = X[1] * 0.001f;
        for (int it = 0; it < ITERS; it++) {{
            #pragma GCC unroll {vectors}
            for (int v = 0; v < {vectors}; v++) {{
                #pragma omp simd
                for (int l = 0; l < {FMA_LANES}; l++) acc[v][l] = acc[v][l] * a + b;
            }}
        }}
        float sum = 0.0f;
        for (int v = 0; v < {vectors}; v++)
            for (int l = 0; l < {FMA_LANES}; l++) sum += acc[v][l];
        Y[g] = sum;
    }}
}}
"""


PEAK_BANDWIDTH_SOURCE = """#include <omp.h>
void peak_bandwidth_kernel(const float* X, float* Y, int N, int ITERS) {
    for (int it = 0; it < ITERS; it++) {
        const float shift = (float)it;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) Y[i] = X[i] + shift;
    }
}
"""


@dataclass
class MachinePeaks:
    compute_gflops: float
    bandwidth_gbps: float
    source: str  # "measured" | "cached"

    def describe(self) -> str:
        return (
            f"peak FMA throughput ~{self.compute_gflops:.0f} GFLOP/s (all threads), "
            f"streaming bandwidth ~{self.bandwidth_gbps:.1f} GB/s ({self.source})"
        )


@dataclass
class RooflineReport:
    arithmetic_intensity: float  # FLOP per byte of the operator at the benchmark shape
    attainable_gflops: float
    bound: str  # "compute" | "memory"
    fraction_of_attainable: float
    fraction_of_compute_peak: float
    fraction_of_bandwidth_peak: float

    def describe(self) -> str:
        return (
            f"{self.bound}-bound (intensity {self.arithmetic_intensity:.1f} FLOP/B); "
            f"{self.fraction_of_attainable * 100:.0f}% of attainable {self.attainable_gflops:.0f} GFLOP/s; "
            f"{self.fraction_of_compute_peak * 100:.0f}% of FMA peak, {self.fraction_of_bandwidth_peak * 100:.0f}% of bandwidth peak"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(peaks: MachinePeaks, flops: int, bytes_moved: int, attained_gflops: float, attained_gbps: float) -> RooflineReport:
    intensity = flops / max(bytes_moved, 1)
    memory_ceiling = intensity * peaks.bandwidth_gbps
    attainable = min(peaks.compute_gflops, memory_ceiling)
    return RooflineReport(
        arithmetic_intensity=intensity,
        attainable_gflops=attainable,
        bound="compute" if memory_ceiling >= peaks.compute_gflops else "memory",
        fraction_of_attainable=attained_gflops / max(attainable, 1e-9),
        fraction_of_compute_peak=attained_gflops / max(peaks.compute_gflops, 1e-9),
        fraction_of_bandwidth_peak=attained_gbps / max(peaks.bandwidth_gbps, 1e-9),
    )


def _cache_path(backend: Backend) -> Path:
    cache_root = Path(os.environ.get("KOPT_CACHE_DIR") or Path.home() / ".cache" / "kopt")
    digest = hashlib.sha1(f"{backend.name}|{backend.hardware_summary()}".encode("utf-8")).hexdigest()[:12]
    return cache_root / f"peaks_{backend.name}_{digest}.json"


def _probe_spec(name: str, symbol: str, source_text: str) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        description="hardware probe",
        c_signature=f"void {symbol}(const float* X, float* Y, int A, int B)",
        symbol=symbol,
        primary_shape=(1,),
        make_case=lambda shape: TestCase((TensorSpec("X", (1,)),), TensorSpec("Y", (1,)), (1, 1)),
        reference=lambda inputs, scalars: inputs[0],
        flops=lambda shape: 0,
        bytes_moved=lambda shape: 0,
    )


def _run_probe(
    backend: Backend,
    name: str,
    symbol: str,
    source_text: str,
    x: np.ndarray,
    y_shape: tuple[int, ...],
    scalars: tuple[int, int],
    repeats: int,
    flags: tuple[str, ...] = (),
) -> float:
    spec = _probe_spec(name, symbol, source_text)
    compiled = backend.compile(Candidate(source=source_text, origin="probe", extra_compile_flags=flags), spec)
    if not compiled.ok or compiled.artifact is None:
        raise RuntimeError(f"{name} probe failed to compile: {compiled.log[:300]}")
    case = TestCase((TensorSpec("X", x.shape),), TensorSpec("Y", y_shape), scalars, label=name)
    run = backend.run(compiled.artifact, spec, case, [x], warmup=1, repeats=repeats, timeout_seconds=120.0)
    if not run.ok or not run.timings_ms:
        raise RuntimeError(f"{name} probe failed to run: {run.error}")
    return statistics.median(run.timings_ms) / 1e3


def measure_peaks(backend: Backend, use_cache: bool = True) -> MachinePeaks:
    cache_path = _cache_path(backend)
    if use_cache and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return MachinePeaks(float(data["compute_gflops"]), float(data["bandwidth_gbps"]), "cached")
        except (OSError, KeyError, ValueError, TypeError):
            pass

    threads = getattr(backend, "threads", os.cpu_count() or 1)
    groups = threads * 4
    fma_input = np.linspace(0.5, 1.5, 128, dtype=np.float32)
    compute_gflops = 0.0
    flag_options: tuple[tuple[str, ...], ...] = ((),) + ((WIDE_VECTOR_FLAGS,) if supports_avx512() else ())
    for vectors in FMA_VECTOR_OPTIONS:
        for flags in flag_options:
            fma_seconds = _run_probe(
                backend, "peak_fma", "peak_fma_kernel", peak_fma_source(vectors), fma_input, (groups,), (groups, FMA_ITERS),
                repeats=3, flags=flags,
            )
            compute_gflops = max(compute_gflops, groups * FMA_ITERS * vectors * FMA_LANES * 2 / fma_seconds / 1e9)

    stream_input = np.ones(BANDWIDTH_FLOATS, dtype=np.float32)
    bandwidth_seconds = _run_probe(
        backend, "peak_bandwidth", "peak_bandwidth_kernel", PEAK_BANDWIDTH_SOURCE, stream_input, (BANDWIDTH_FLOATS,),
        (BANDWIDTH_FLOATS, BANDWIDTH_ITERS), repeats=3,
    )
    bandwidth_gbps = BANDWIDTH_ITERS * BANDWIDTH_FLOATS * 8 / bandwidth_seconds / 1e9

    peaks = MachinePeaks(compute_gflops, bandwidth_gbps, "measured")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"compute_gflops": compute_gflops, "bandwidth_gbps": bandwidth_gbps}), encoding="utf-8")
    except OSError as error:
        logger.debug("could not cache peaks: %s", error)
    return peaks
