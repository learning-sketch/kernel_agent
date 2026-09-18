"""Roofline model with a verdict.

Once per machine we measure peak FMA throughput, streaming bandwidth and the floor cost of
dispatching a (parallel) kernel. Every trial is then placed against the attainable time

    attainable_ms = max(flops / peak_compute, bytes / peak_bandwidth) + dispatch_overhead

and classified as compute-, memory- or dispatch-overhead-bound. The verdict tells the agent
two things a raw number cannot: whether the search has reached the ceiling (stop burning
budget) and which family of optimizations can still pay off (steer the LLM and the search).
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
DISPATCH_REPEATS = 200

GUIDANCE = {
    "compute": (
        "Compute-bound: the FMA units are the limiter. Work on register tiling (a fixed MRxNR accumulator "
        "tile in registers), more independent FMA chains (ILP), full-width vectors, packed operands so the "
        "inner loop is unit-stride, and loop unrolling. Reducing memory traffic further will not help."
    ),
    "memory": (
        "Memory-bound: bytes moved are the limiter. Fuse passes so each element is read from memory once, "
        "vectorize loads/stores, use all threads to saturate bandwidth, consider streaming stores for "
        "write-once outputs and cache blocking for reuse. More arithmetic cleverness will not help."
    ),
    "overhead": (
        "Dispatch-overhead-bound: the operator is so small that launching the kernel / spawning the "
        "parallel region costs more than the work. Do not micro-optimize the inner loop. Fuse this operator "
        "with its neighbours, batch several calls into one launch, or skip the parallel region below a "
        "size threshold."
    ),
}


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

# The cheapest possible parallel kernel: what any OpenMP kernel pays before doing work.
DISPATCH_SOURCE = """#include <omp.h>
void dispatch_kernel(const float* X, float* Y, int N, int UNUSED) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) Y[i] = X[i];
}
"""

# The cheapest possible serial kernel: the floor of one call through the harness.
CALL_SOURCE = """
void call_kernel(const float* X, float* Y, int N, int UNUSED) {
    Y[0] = X[0];
}
"""


@dataclass
class MachinePeaks:
    compute_gflops: float
    bandwidth_gbps: float
    dispatch_overhead_ms: float  # floor of a parallel (OpenMP region) kernel launch
    call_overhead_ms: float = 0.0  # floor of a serial kernel call
    source: str = "measured"  # "measured" | "cached"

    def describe(self) -> str:
        return (
            f"peak FMA throughput ~{self.compute_gflops:.0f} GFLOP/s (all threads), "
            f"streaming bandwidth ~{self.bandwidth_gbps:.1f} GB/s, "
            f"launch floors: serial call ~{self.call_overhead_ms * 1e3:.2f} us, parallel region ~{self.dispatch_overhead_ms * 1e3:.2f} us ({self.source})"
        )


@dataclass
class RooflineReport:
    arithmetic_intensity: float  # FLOP per byte of the operator at this shape
    compute_time_ms: float
    memory_time_ms: float
    overhead_ms: float
    attainable_ms: float
    attainable_gflops: float
    bound: str  # "compute" | "memory" | "overhead"
    fraction_of_attainable: float  # attainable_ms / measured_ms, capped at 1
    fraction_of_compute_peak: float
    fraction_of_bandwidth_peak: float
    headroom_speedup: float  # measured_ms / attainable_ms: how much faster the ceiling still allows

    @property
    def guidance(self) -> str:
        return GUIDANCE[self.bound]

    def describe(self) -> str:
        return (
            f"{self.bound}-bound (intensity {self.arithmetic_intensity:.1f} FLOP/B; ceiling {self.attainable_ms:.4f} ms = "
            f"max(compute {self.compute_time_ms:.4f}, memory {self.memory_time_ms:.4f}) + call floor {self.overhead_ms:.4f}); "
            f"{self.fraction_of_attainable * 100:.0f}% of attainable, headroom {self.headroom_speedup:.2f}x"
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["guidance"] = self.guidance
        return data


@dataclass
class Verdict:
    """Decision derived from the roofline of the current best kernel."""

    bound: str
    attainable_ms: float
    best_ms: float
    fraction_of_attainable: float
    at_ceiling: bool
    ceiling_fraction: float
    guidance: str

    def describe(self) -> str:
        state = (
            f"AT CEILING: best is within {(1 - self.fraction_of_attainable) * 100:.0f}% of the attainable "
            f"{self.attainable_ms:.4f} ms (threshold {self.ceiling_fraction * 100:.0f}%) - stop spending budget here"
            if self.at_ceiling
            else f"headroom {self.best_ms / max(self.attainable_ms, 1e-12):.2f}x to the attainable {self.attainable_ms:.4f} ms"
        )
        return f"{self.bound}-bound; {state}. {self.guidance}"

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(peaks: MachinePeaks, flops: int, bytes_moved: int, measured_ms: float) -> RooflineReport:
    compute_time_ms = flops / max(peaks.compute_gflops, 1e-9) / 1e6
    memory_time_ms = bytes_moved / max(peaks.bandwidth_gbps, 1e-9) / 1e6
    work_time_ms = max(compute_time_ms, memory_time_ms)
    # The unavoidable floor is one serial call; the *parallel* dispatch floor decides whether
    # spreading the work over threads can pay for itself at all.
    overhead_ms = peaks.call_overhead_ms
    attainable_ms = work_time_ms + overhead_ms
    if peaks.dispatch_overhead_ms >= work_time_ms:
        bound = "overhead"
    elif compute_time_ms >= memory_time_ms:
        bound = "compute"
    else:
        bound = "memory"
    measured_seconds = max(measured_ms, 1e-12) / 1e3
    return RooflineReport(
        arithmetic_intensity=flops / max(bytes_moved, 1),
        compute_time_ms=compute_time_ms,
        memory_time_ms=memory_time_ms,
        overhead_ms=overhead_ms,
        attainable_ms=attainable_ms,
        attainable_gflops=flops / max(attainable_ms, 1e-12) / 1e6,
        bound=bound,
        fraction_of_attainable=min(1.0, attainable_ms / max(measured_ms, 1e-12)),
        fraction_of_compute_peak=(flops / measured_seconds / 1e9) / max(peaks.compute_gflops, 1e-9),
        fraction_of_bandwidth_peak=(bytes_moved / measured_seconds / 1e9) / max(peaks.bandwidth_gbps, 1e-9),
        headroom_speedup=max(measured_ms, 1e-12) / max(attainable_ms, 1e-12),
    )


def verdict_for(report: RooflineReport, best_ms: float, ceiling_fraction: float) -> Verdict:
    return Verdict(
        bound=report.bound,
        attainable_ms=report.attainable_ms,
        best_ms=best_ms,
        fraction_of_attainable=report.fraction_of_attainable,
        at_ceiling=report.fraction_of_attainable >= ceiling_fraction,
        ceiling_fraction=ceiling_fraction,
        guidance=report.guidance,
    )


def _cache_path(backend: Backend) -> Path:
    cache_root = Path(os.environ.get("KOPT_CACHE_DIR") or Path.home() / ".cache" / "kopt")
    digest = hashlib.sha1(f"{backend.name}|{backend.hardware_summary()}|v3".encode("utf-8")).hexdigest()[:12]
    return cache_root / f"peaks_{backend.name}_{digest}.json"


def _probe_spec(name: str, symbol: str) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        description="hardware probe",
        c_signature=f"void {symbol}(const float* X, float* Y, int A, int B)",
        symbol=symbol,
        primary_shape=(1,),
        make_case=lambda shape, dtype: TestCase((TensorSpec("X", (1,)),), TensorSpec("Y", (1,)), (1, 1)),
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
    statistic=statistics.median,
) -> float:
    spec = _probe_spec(name, symbol)
    compiled = backend.compile(Candidate(source=source_text, origin="probe", extra_compile_flags=flags), spec)
    if not compiled.ok or compiled.artifact is None:
        raise RuntimeError(f"{name} probe failed to compile: {compiled.log[:300]}")
    case = TestCase((TensorSpec("X", x.shape),), TensorSpec("Y", y_shape), scalars, label=name)
    run = backend.run(compiled.artifact, spec, case, [x], warmup=2, repeats=repeats, timeout_seconds=120.0, verify=False)
    if not run.ok or not run.timings_ms:
        raise RuntimeError(f"{name} probe failed to run: {run.error}")
    return statistic(run.timings_ms) / 1e3


def measure_peaks(backend: Backend, use_cache: bool = True) -> MachinePeaks:
    cache_path = _cache_path(backend)
    if use_cache and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return MachinePeaks(
                float(data["compute_gflops"]), float(data["bandwidth_gbps"]), float(data["dispatch_overhead_ms"]),
                float(data["call_overhead_ms"]), "cached",
            )
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

    # Dispatch floor: the fastest observed call of a trivial parallel kernel on 64 elements.
    dispatch_seconds = _run_probe(
        backend, "dispatch", "dispatch_kernel", DISPATCH_SOURCE, np.ones(64, dtype=np.float32), (64,), (64, 0),
        repeats=DISPATCH_REPEATS, statistic=min,
    )

    call_seconds = _run_probe(
        backend, "call", "call_kernel", CALL_SOURCE, np.ones(64, dtype=np.float32), (64,), (64, 0),
        repeats=DISPATCH_REPEATS, statistic=min,
    )

    peaks = MachinePeaks(compute_gflops, bandwidth_gbps, dispatch_seconds * 1e3, call_seconds * 1e3, "measured")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "compute_gflops": compute_gflops, "bandwidth_gbps": bandwidth_gbps,
                    "dispatch_overhead_ms": peaks.dispatch_overhead_ms, "call_overhead_ms": peaks.call_overhead_ms,
                }
            ),
            encoding="utf-8",
        )
    except OSError as error:
        logger.debug("could not cache peaks: %s", error)
    return peaks
