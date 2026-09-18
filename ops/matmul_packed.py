"""BLIS/GotoBLAS-style GEMM template: packed panels + register-tiled micro-kernel.

Schedule (per thread group):
    for jc in N by NC:            # B block of KC x NC shared by all threads
      for pc in K by KC:
        pack B[pc:pc+KC, jc:jc+NC] into NR-wide panels (cooperatively)
        for ic in M by MC (parallel):
          pack A[ic:ic+MC, pc:pc+KC] into MR-high panels (per thread)
          for each (NR panel, MR panel): micro-kernel with an MR x NR accumulator tile in registers

All remainders (M, N, K not multiples of the tiles) are handled by zero-padding the packed
panels and writing partial tiles through a scratch tile.

The template is element-type generic (float / _Float16 / __bf16: packing converts to float,
arithmetic is float, stores convert back) and takes an optional fused epilogue that is applied
exactly once, on the last K block, so element-wise tail operators (bias, activation) can be
folded into the GEMM without a second pass over C.
"""

from __future__ import annotations

import os

from kopt_agent.dtypes import DType
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.hardware import supports_avx512

L2_BUDGET_BYTES = 1 << 20  # KC x NC block of B should stay L2-resident
L1_BUDGET_BYTES = 128 << 10  # MC x KC block of A should stay close to L1/L2
MAX_ACCUMULATOR_FLOATS = 24 * 16  # 24 vector registers of 16 floats for the accumulator tile


def packed_template_source(signature: str, dtype: DType, epilogue: str = "(v)", aux_name: str = "NULL") -> str:
    """`epilogue` is a C expression over `v` (float partial result), `row`, `col` and `aux`
    (the extra `const elem_t*` operand named by `aux_name`, e.g. a bias vector)."""
    return (
        f"""#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

typedef {dtype.c_type} elem_t;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MR $MR
#define NR $NR
#define MC $MC
#define KC $KC
#define NC $NC
#define ROUND_UP(x, m) ((((x) + (m) - 1) / (m)) * (m))
/* Applied once per output element on the final K block. */
#define EPILOGUE(v, row, col, aux) ({epilogue})

/* B panel: kc rows of NR contiguous columns, zero padded past jn. */
static void pack_b_panel(const elem_t* B, int N, int k0, int kc, int j0, int jn, float* Bp) {{
    for (int k = 0; k < kc; k++) {{
        const elem_t* src = B + (size_t)(k0 + k) * N + j0;
        float* dst = Bp + (size_t)k * NR;
        int j = 0;
        for (; j < jn; j++) dst[j] = (float)src[j];
        for (; j < NR; j++) dst[j] = 0.0f;
    }}
}}

/* A panel stored k-major: Ap[k * MR + i], zero padded past im rows. */
static void pack_a_panel(const elem_t* A, int K, int i0, int im, int k0, int kc, float* Ap) {{
    for (int i = 0; i < im; i++) {{
        const elem_t* src = A + (size_t)(i0 + i) * K + k0;
        for (int k = 0; k < kc; k++) Ap[(size_t)k * MR + i] = (float)src[k];
    }}
    for (int i = im; i < MR; i++) {{
        for (int k = 0; k < kc; k++) Ap[(size_t)k * MR + i] = 0.0f;
    }}
}}

/* tile[MR x NR] = Ap * Bp over kc. The accumulator tile stays in registers. */
static inline void micro_kernel(int kc, const float* Ap, const float* Bp, float* tile) {{
    float acc[MR][NR];
    for (int i = 0; i < MR; i++) {{
        #pragma omp simd
        for (int j = 0; j < NR; j++) acc[i][j] = 0.0f;
    }}
    for (int k = 0; k < kc; k++) {{
        const float* b = Bp + (size_t)k * NR;
        const float* a = Ap + (size_t)k * MR;
        for (int i = 0; i < MR; i++) {{
            const float ai = a[i];
            #pragma omp simd
            for (int j = 0; j < NR; j++) acc[i][j] += ai * b[j];
        }}
    }}
    for (int i = 0; i < MR; i++) {{
        #pragma omp simd
        for (int j = 0; j < NR; j++) tile[i * NR + j] = acc[i][j];
    }}
}}

/* Write an im x jn tile into C: overwrite on the first K block, accumulate otherwise, and apply
   the epilogue on the last one. */
static inline void store_tile(const float* tile, elem_t* C, int ldc, int im, int jn, int row0, int col0,
                              int accumulate, int final, const elem_t* aux) {{
    for (int i = 0; i < im; i++) {{
        elem_t* c = C + (size_t)i * ldc;
        const float* t = tile + i * NR;
        if (!accumulate && final) {{
            #pragma omp simd
            for (int j = 0; j < jn; j++) c[j] = (elem_t)EPILOGUE(t[j], row0 + i, col0 + j, aux);
        }} else if (!accumulate) {{
            #pragma omp simd
            for (int j = 0; j < jn; j++) c[j] = (elem_t)t[j];
        }} else if (final) {{
            #pragma omp simd
            for (int j = 0; j < jn; j++) c[j] = (elem_t)EPILOGUE((float)c[j] + t[j], row0 + i, col0 + j, aux);
        }} else {{
            #pragma omp simd
            for (int j = 0; j < jn; j++) c[j] = (elem_t)((float)c[j] + t[j]);
        }}
    }}
}}

"""
        + signature
        + f""" {{
    const elem_t* aux = {aux_name};
    const int nc_padded = ROUND_UP(MIN(NC, N), NR);
    const int mc_padded = ROUND_UP(MIN(MC, M), MR);
    const size_t bp_bytes = ROUND_UP((size_t)KC * nc_padded * sizeof(float), 64);
    const size_t ap_bytes = ROUND_UP((size_t)KC * mc_padded * sizeof(float), 64);
    float* Bp = (float*)aligned_alloc(64, bp_bytes);
    if (Bp == NULL) return;
    int allocation_failed = 0;

    #pragma omp parallel num_threads($THREADS)
    {{
        float* Ap = (float*)aligned_alloc(64, ap_bytes);
        float tile[MR * NR] __attribute__((aligned(64)));
        if (Ap == NULL) {{
            #pragma omp atomic write
            allocation_failed = 1;
        }}
        /* Worksharing loops below must be reached by every thread or by none. */
        #pragma omp barrier
        for (int jc = 0; jc < N && !allocation_failed; jc += NC) {{
            const int nc = MIN(NC, N - jc);
            const int n_panels = (nc + NR - 1) / NR;
            for (int pc = 0; pc < K; pc += KC) {{
                const int kc = MIN(KC, K - pc);
                const int accumulate = pc != 0;
                const int final = pc + kc >= K;

                #pragma omp for schedule(static)
                for (int p = 0; p < n_panels; p++) {{
                    const int j0 = jc + p * NR;
                    pack_b_panel(B, N, pc, kc, j0, MIN(NR, jc + nc - j0), Bp + (size_t)p * kc * NR);
                }}

                #pragma omp for schedule($SCHEDULE)
                for (int ic = 0; ic < M; ic += MC) {{
                    const int mc = MIN(MC, M - ic);
                    const int m_panels = (mc + MR - 1) / MR;
                    for (int q = 0; q < m_panels; q++) {{
                        const int i0 = ic + q * MR;
                        pack_a_panel(A, K, i0, MIN(MR, ic + mc - i0), pc, kc, Ap + (size_t)q * kc * MR);
                    }}
                    for (int p = 0; p < n_panels; p++) {{
                        const int j0 = jc + p * NR;
                        const int jn = MIN(NR, jc + nc - j0);
                        const float* bp = Bp + (size_t)p * kc * NR;
                        for (int q = 0; q < m_panels; q++) {{
                            const int i0 = ic + q * MR;
                            const int im = MIN(MR, ic + mc - i0);
                            micro_kernel(kc, Ap + (size_t)q * kc * MR, bp, tile);
                            store_tile(tile, C + (size_t)i0 * N + j0, N, im, jn, i0, j0, accumulate, final, aux);
                        }}
                    }}
                }}
                /* The implicit barrier of the loop above guarantees Bp is no longer read before repacking. */
            }}
        }}
        free(Ap);
    }}
    free(Bp);
}}
"""
    )


def build_packed_template(signature: str, dtype: DType, epilogue: str = "(v)", aux_name: str = "NULL") -> TemplateGenerator:
    thread_count = os.cpu_count() or 1
    thread_options = sorted({1, max(1, thread_count // 2), thread_count})
    return TemplateGenerator(
        template_source=packed_template_source(signature, dtype, epilogue, aux_name),
        space={
            "MR": [4, 6, 8],
            "NR": [16, 32, 64],
            "MC": [32, 64, 128],
            "KC": [128, 256, 512],
            "NC": [256, 512, 1024],
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
            # gcc prefers 256-bit vectors on many AVX-512 CPUs; 1 asks for 512-bit zmm code.
            "WIDE": [0, 1] if supports_avx512() else [0],
        },
        default_params={"MR": 6, "NR": 16, "MC": 64, "KC": 256, "NC": 512, "THREADS": thread_count, "SCHEDULE": "dynamic", "WIDE": 0},
        extra_flags=lambda p: ("-mprefer-vector-width=512",) if int(p["WIDE"]) else (),
        constraint=lambda p: (
            int(p["MR"]) * int(p["NR"]) <= MAX_ACCUMULATOR_FLOATS
            and int(p["NC"]) % int(p["NR"]) == 0
            and int(p["KC"]) * int(p["NC"]) * 4 <= L2_BUDGET_BYTES
            and int(p["MC"]) * int(p["KC"]) * 4 <= L1_BUDGET_BYTES
        ),
    )
