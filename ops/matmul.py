"""GEMM: C[M,N] = A[M,K] @ B[K,N] in fp32, fp16 or bf16 (float accumulation)."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, get_dtype
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile
from ops.matmul_packed import build_packed_template

DEFAULT_SHAPE = (512, 512, 512)
SYMBOL = "matmul_kernel"


def signature(dtype: DType) -> str:
    c_type = dtype.c_type
    return f"void {SYMBOL}(const {c_type}* A, const {c_type}* B, {c_type}* C, int M, int N, int K)"


def baseline_source(dtype: DType) -> str:
    return f"""#include <stddef.h>

typedef {dtype.c_type} elem_t;

/* Naive triple loop. Correct, cache-hostile (B is walked with stride N). */
{signature(dtype)} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {{
                acc += (float)A[(size_t)i * K + k] * (float)B[(size_t)k * N + j];
            }}
            C[(size_t)i * N + j] = (elem_t)acc;
        }}
    }}
}}
"""


# fp32 only: partial sums live in C, which is fine for float but would lose precision for 16-bit C.
BLOCKED_TEMPLATE_SOURCE = (
    """#include <omp.h>
#include <string.h>
#include <stddef.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* Fast-path contract: when ALIGNED=1 the vector loop assumes 64-byte aligned rows (N % 16 == 0);
   any other N takes the generic loop. The harness checks the flag on every test shape. */
int kopt_fast_path_active = 0;

/* Cache-blocked GEMM. Rows blocked by MB (one OpenMP work item each), reduction by KB,
   columns by NB. The innermost loop streams a row of B and a row of C with unit stride. */
"""
    + signature(get_dtype("fp32"))
    + """ {
    const int aligned_path = $ALIGNED && (N % 16 == 0);
    kopt_fast_path_active = aligned_path;
    #pragma omp parallel for schedule($SCHEDULE) num_threads($THREADS)
    for (int i0 = 0; i0 < M; i0 += $MB) {
        const int i1 = MIN(i0 + $MB, M);
        for (int i = i0; i < i1; i++) {
            memset(C + (size_t)i * N, 0, (size_t)N * sizeof(float));
        }
        for (int k0 = 0; k0 < K; k0 += $KB) {
            const int k1 = MIN(k0 + $KB, K);
            for (int j0 = 0; j0 < N; j0 += $NB) {
                const int j1 = MIN(j0 + $NB, N);
                for (int i = i0; i < i1; i++) {
                    float* Ci = C + (size_t)i * N;
                    const float* Ai = A + (size_t)i * K;
                    for (int k = k0; k < k1; k++) {
                        const float a = Ai[k];
                        const float* Bk = B + (size_t)k * N;
                        if (aligned_path) {
                            float* Cia = (float*)__builtin_assume_aligned(Ci + j0, 64);
                            const float* Bka = (const float*)__builtin_assume_aligned(Bk + j0, 64);
                            #pragma omp simd aligned(Cia, Bka : 64)
                            for (int j = 0; j < j1 - j0; j++) {
                                Cia[j] += a * Bka[j];
                            }
                        } else {
                            #pragma omp simd
                            for (int j = j0; j < j1; j++) {
                                Ci[j] += a * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }
}
"""
)


def make_case(shape: tuple[int, ...], dtype: DType) -> TestCase:
    rows, cols, depth = shape
    return TestCase(
        inputs=(TensorSpec("A", (rows, depth), dtype), TensorSpec("B", (depth, cols), dtype)),
        output=TensorSpec("C", (rows, cols), dtype),
        scalars=(rows, cols, depth),
    )


def reference(inputs, scalars) -> np.ndarray:
    matrix_a, matrix_b = inputs
    return np.asarray(matrix_a, dtype=np.float64) @ np.asarray(matrix_b, dtype=np.float64)


def flops(shape: tuple[int, ...]) -> int:
    return 2 * shape[0] * shape[1] * shape[2]


def build(shape: tuple[int, ...], dtype: str = "fp32", workload: WorkloadProfile | None = None) -> OperatorBundle:
    rows, cols, depth = shape
    element = get_dtype(dtype)
    itemsize = element.itemsize
    spec = OperatorSpec(
        name="matmul",
        description=f"C[M,N] = A[M,K] @ B[K,N] for row-major {element.name} matrices (accumulate in float); C must be fully overwritten (not accumulated).",
        c_signature=signature(element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=make_case,
        reference=reference,
        flops=flops,
        bytes_moved=lambda s: itemsize * (s[0] * s[2] + s[2] * s[1] + s[0] * s[1]),
        scalar_names=("M", "N", "K"),
        # Prime and tiny sizes defeat every "assume multiple of tile" shortcut; 48 columns is a
        # multiple of 16 but not of 64, so an "N % 16" fast path must still handle it.
        edge_shapes=((1, 1, 1), (7, 13, 5), (33, 65, 17), (64, 1, 128), (1, 129, 3), (24, 48, 40)),
        dtype=element,
        workload=workload,
        precision_sensitive=False,
        notes=[
            "A, B and C are distinct buffers (no aliasing).",
            f"Benchmark arithmetic intensity is high ({flops(shape) / (itemsize * (rows * depth + depth * cols + rows * cols)):.0f} FLOP/byte) so the kernel is compute bound: register tiling + FMA vectorization matter most.",
        ]
        + (["Accumulate in float and round to the element type only when storing C."] if element.name != "fp32" else []),
    )

    templates = {"packed": build_packed_template(signature(element), element)}
    if element.name == "fp32":
        thread_options = sorted({1, 2, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
        templates["blocked"] = TemplateGenerator(
            template_source=BLOCKED_TEMPLATE_SOURCE,
            space={
                "MB": [8, 16, 32, 64],
                "NB": [64, 128, 256, 512],
                "KB": [32, 64, 128, 256],
                "THREADS": thread_options,
                "SCHEDULE": ["static", "dynamic"],
                "ALIGNED": [0, 1],
            },
            default_params={"MB": 32, "NB": 256, "KB": 128, "THREADS": os.cpu_count() or 1, "SCHEDULE": "static", "ALIGNED": 0},
            # A tile of B (KB x NB floats) plus a row strip of C should stay inside L2.
            constraint=lambda params: int(params["KB"]) * int(params["NB"]) * 4 <= 512 * 1024,
            fast_path=lambda params: "N % 16 == 0" if int(params["ALIGNED"]) else None,
        )
    return OperatorBundle(
        spec=spec,
        baseline_source=baseline_source(element),
        templates=templates,
        default_template="packed",
    )
