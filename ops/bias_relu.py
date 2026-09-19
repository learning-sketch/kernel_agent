"""Element-wise tail operator: Y[i, j] = max(X[i, j] + bias[j], 0). The typical candidate for
fusion into the GEMM that produces X; on its own it is memory- or dispatch-overhead-bound.

Precision: X in `dtype`; bias and Y in `output_dtype` (defaults to `dtype`); the add happens in
`accumulate_dtype` (defaults to fp32)."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, default_accumulate_dtype, get_dtype
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile

DEFAULT_SHAPE = (512, 512)
SYMBOL = "bias_relu_kernel"


def signature(dtype: DType, out_dtype: DType | None = None) -> str:
    out_dtype = out_dtype or dtype
    return f"void {SYMBOL}(const {dtype.c_type}* X, const {out_dtype.c_type}* bias, {out_dtype.c_type}* Y, int M, int N)"


def _typedefs(dtype: DType, out_dtype: DType, acc_dtype: DType) -> str:
    return f"typedef {dtype.c_type} in_t;\ntypedef {out_dtype.c_type} out_t;\ntypedef {acc_dtype.c_type} acc_t;\n"


def baseline_source(dtype: DType, out_dtype: DType | None = None, acc_dtype: DType | None = None) -> str:
    out_dtype = out_dtype or dtype
    acc_dtype = acc_dtype or default_accumulate_dtype(dtype, out_dtype)
    return f"""#include <stddef.h>

{_typedefs(dtype, out_dtype, acc_dtype)}
{signature(dtype, out_dtype)} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            const acc_t v = (acc_t)X[(size_t)i * N + j] + (acc_t)bias[j];
            Y[(size_t)i * N + j] = (out_t)(v > (acc_t)0 ? v : (acc_t)0);
        }}
    }}
}}
"""


def template_source(dtype: DType, out_dtype: DType | None = None, acc_dtype: DType | None = None) -> str:
    out_dtype = out_dtype or dtype
    acc_dtype = acc_dtype or default_accumulate_dtype(dtype, out_dtype)
    return (
        f"""#include <omp.h>
#include <stddef.h>

{_typedefs(dtype, out_dtype, acc_dtype)}
/* SERIAL_BELOW: problems smaller than this many elements skip the parallel region entirely,
   because spawning threads costs more than the work (dispatch-overhead-bound regime). */
"""
        + signature(dtype, out_dtype)
        + """ {
    const size_t total = (size_t)M * N;
    if (total < (size_t)$SERIAL_BELOW) {
        for (size_t idx = 0; idx < total; idx++) {
            const acc_t v = (acc_t)X[idx] + (acc_t)bias[idx % N];
            Y[idx] = (out_t)(v > (acc_t)0 ? v : (acc_t)0);
        }
        return;
    }
    #pragma omp parallel for schedule($SCHEDULE) num_threads($THREADS)
    for (int i = 0; i < M; i++) {
        const in_t* row_in = X + (size_t)i * N;
        out_t* row_out = Y + (size_t)i * N;
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            const acc_t v = (acc_t)row_in[j] + (acc_t)bias[j];
            row_out[j] = (out_t)(v > (acc_t)0 ? v : (acc_t)0);
        }
    }
}
"""
    )


def make_case(shape: tuple[int, ...], dtype: DType, out_dtype: DType | None = None) -> TestCase:
    rows, cols = shape
    out_dtype = out_dtype or dtype
    return TestCase(
        inputs=(TensorSpec("X", (rows, cols), dtype), TensorSpec("bias", (cols,), out_dtype)),
        output=TensorSpec("Y", (rows, cols), out_dtype),
        scalars=(rows, cols),
    )


def reference(inputs, scalars) -> np.ndarray:
    matrix, bias = inputs
    return np.maximum(np.asarray(matrix, dtype=np.float64) + np.asarray(bias, dtype=np.float64)[None, :], 0.0)


def build(
    shape: tuple[int, ...],
    dtype: str = "fp32",
    workload: WorkloadProfile | None = None,
    output_dtype: str | None = None,
    accumulate_dtype: str | None = None,
) -> OperatorBundle:
    element = get_dtype(dtype)
    out_element = get_dtype(output_dtype) if output_dtype else element
    acc_element = get_dtype(accumulate_dtype) if accumulate_dtype else default_accumulate_dtype(element, out_element)
    in_size, out_size = element.itemsize, out_element.itemsize
    spec = OperatorSpec(
        name="bias_relu",
        description=f"Y[i,j] = max(X[i,j] + bias[j], 0) over X[M,N]: X in {element.name}, bias and Y in {out_element.name}, arithmetic in {acc_element.name}.",
        c_signature=signature(element, out_element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=lambda s, in_dtype: make_case(s, in_dtype, out_element),
        reference=reference,
        flops=lambda s: 2 * s[0] * s[1],
        bytes_moved=lambda s: in_size * s[0] * s[1] + out_size * (s[0] * s[1] + s[1]),
        scalar_names=("M", "N"),
        edge_shapes=((1, 1), (3, 7), (5, 1025), (257, 1)),
        dtype=element,
        output_dtype=out_element,
        accumulate_dtype=acc_element,
        workload=workload,
        notes=["Pure streaming operator: one read of X, one write of Y; bias is tiny and stays in cache."],
    )
    thread_options = sorted({1, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
    template = TemplateGenerator(
        template_source=template_source(element, out_element, acc_element),
        space={
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
            "SERIAL_BELOW": [0, 4096, 65536],
        },
        default_params={"THREADS": os.cpu_count() or 1, "SCHEDULE": "static", "SERIAL_BELOW": 4096},
    )
    return OperatorBundle(
        spec=spec, baseline_source=baseline_source(element, out_element, acc_element), templates={"rows": template}, default_template="rows"
    )
