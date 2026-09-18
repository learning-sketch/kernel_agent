"""Element-wise tail operator: Y[i, j] = max(X[i, j] + bias[j], 0). The typical candidate for
fusion into the GEMM that produces X; on its own it is memory- or dispatch-overhead-bound."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, get_dtype
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile

DEFAULT_SHAPE = (512, 512)
SYMBOL = "bias_relu_kernel"


def signature(dtype: DType) -> str:
    return f"void {SYMBOL}(const {dtype.c_type}* X, const {dtype.c_type}* bias, {dtype.c_type}* Y, int M, int N)"


def baseline_source(dtype: DType) -> str:
    return f"""#include <stddef.h>

typedef {dtype.c_type} elem_t;

{signature(dtype)} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            const float v = (float)X[(size_t)i * N + j] + (float)bias[j];
            Y[(size_t)i * N + j] = (elem_t)(v > 0.0f ? v : 0.0f);
        }}
    }}
}}
"""


def template_source(dtype: DType) -> str:
    return (
        f"""#include <omp.h>
#include <stddef.h>

typedef {dtype.c_type} elem_t;

/* SERIAL_BELOW: problems smaller than this many elements skip the parallel region entirely,
   because spawning threads costs more than the work (dispatch-overhead-bound regime). */
"""
        + signature(dtype)
        + """ {
    const size_t total = (size_t)M * N;
    if (total < (size_t)$SERIAL_BELOW) {
        for (size_t idx = 0; idx < total; idx++) {
            const float v = (float)X[idx] + (float)bias[idx % N];
            Y[idx] = (elem_t)(v > 0.0f ? v : 0.0f);
        }
        return;
    }
    #pragma omp parallel for schedule($SCHEDULE) num_threads($THREADS)
    for (int i = 0; i < M; i++) {
        const elem_t* row_in = X + (size_t)i * N;
        elem_t* row_out = Y + (size_t)i * N;
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            const float v = (float)row_in[j] + (float)bias[j];
            row_out[j] = (elem_t)(v > 0.0f ? v : 0.0f);
        }
    }
}
"""
    )


def make_case(shape: tuple[int, ...], dtype: DType) -> TestCase:
    rows, cols = shape
    return TestCase(
        inputs=(TensorSpec("X", (rows, cols), dtype), TensorSpec("bias", (cols,), dtype)),
        output=TensorSpec("Y", (rows, cols), dtype),
        scalars=(rows, cols),
    )


def reference(inputs, scalars) -> np.ndarray:
    matrix, bias = inputs
    return np.maximum(np.asarray(matrix, dtype=np.float64) + np.asarray(bias, dtype=np.float64)[None, :], 0.0)


def build(shape: tuple[int, ...], dtype: str = "fp32", workload: WorkloadProfile | None = None) -> OperatorBundle:
    element = get_dtype(dtype)
    spec = OperatorSpec(
        name="bias_relu",
        description=f"Y[i,j] = max(X[i,j] + bias[j], 0) over X[M,N] ({element.name}).",
        c_signature=signature(element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=make_case,
        reference=reference,
        flops=lambda s: 2 * s[0] * s[1],
        bytes_moved=lambda s: element.itemsize * (2 * s[0] * s[1] + s[1]),
        scalar_names=("M", "N"),
        edge_shapes=((1, 1), (3, 7), (5, 1025), (257, 1)),
        dtype=element,
        workload=workload,
        notes=["Pure streaming operator: one read of X, one write of Y; bias is tiny and stays in cache."],
    )
    thread_options = sorted({1, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
    template = TemplateGenerator(
        template_source=template_source(element),
        space={
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
            "SERIAL_BELOW": [0, 4096, 65536],
        },
        default_params={"THREADS": os.cpu_count() or 1, "SCHEDULE": "static", "SERIAL_BELOW": 4096},
    )
    return OperatorBundle(spec=spec, baseline_source=baseline_source(element), templates={"rows": template}, default_template="rows")
