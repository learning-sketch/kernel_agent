"""float32 GEMM: C[M,N] = A[M,K] @ B[K,N]."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase

DEFAULT_SHAPE = (512, 512, 512)
SYMBOL = "matmul_kernel"
SIGNATURE = f"void {SYMBOL}(const float* A, const float* B, float* C, int M, int N, int K)"

BASELINE_SOURCE = f"""#include <stddef.h>

/* Naive triple loop. Correct, cache-hostile (B is walked with stride N). */
{SIGNATURE} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {{
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            }}
            C[(size_t)i * N + j] = acc;
        }}
    }}
}}
"""

TEMPLATE_SOURCE = (
    """#include <omp.h>
#include <string.h>
#include <stddef.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* Cache-blocked GEMM. Rows blocked by MB (one OpenMP work item each), reduction by KB,
   columns by NB. The innermost loop streams a row of B and a row of C with unit stride. */
"""
    + SIGNATURE
    + """ {
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
"""
)


def _make_case(shape: tuple[int, ...]) -> TestCase:
    rows, cols, depth = shape
    return TestCase(
        inputs=(TensorSpec("A", (rows, depth)), TensorSpec("B", (depth, cols))),
        output=TensorSpec("C", (rows, cols)),
        scalars=(rows, cols, depth),
    )


def _reference(inputs, scalars) -> np.ndarray:
    matrix_a, matrix_b = inputs
    return (matrix_a.astype(np.float64) @ matrix_b.astype(np.float64)).astype(np.float32)


def build(shape: tuple[int, ...]) -> OperatorBundle:
    rows, cols, depth = shape
    spec = OperatorSpec(
        name="matmul",
        description="C[M,N] = A[M,K] @ B[K,N] for row-major float32 matrices; C must be fully overwritten (not accumulated).",
        c_signature=SIGNATURE,
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=_make_case,
        reference=_reference,
        flops=lambda s: 2 * s[0] * s[1] * s[2],
        bytes_moved=lambda s: 4 * (s[0] * s[2] + s[2] * s[1] + s[0] * s[1]),
        # Prime and tiny sizes defeat every "assume multiple of tile" shortcut.
        edge_shapes=((1, 1, 1), (7, 13, 5), (33, 65, 17), (64, 1, 128), (1, 129, 3)),
        atol=1e-3,
        rtol=1e-4,
        notes=[
            "A, B and C are distinct buffers (no aliasing).",
            f"Benchmark arithmetic intensity is high ({2 * rows * cols * depth / (4 * (rows * depth + depth * cols + rows * cols)):.0f} FLOP/byte) so the kernel is compute bound: register tiling + FMA vectorization matter most.",
        ],
    )

    thread_options = sorted({1, 2, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
    template = TemplateGenerator(
        template_source=TEMPLATE_SOURCE,
        space={
            "MB": [8, 16, 32, 64],
            "NB": [64, 128, 256, 512],
            "KB": [32, 64, 128, 256],
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
        },
        default_params={"MB": 32, "NB": 256, "KB": 128, "THREADS": os.cpu_count() or 1, "SCHEDULE": "static"},
        # A tile of B (KB x NB floats) plus a row strip of C should stay inside L2.
        constraint=lambda params: int(params["KB"]) * int(params["NB"]) * 4 <= 512 * 1024,
    )
    return OperatorBundle(spec=spec, baseline_source=BASELINE_SOURCE, template=template)
