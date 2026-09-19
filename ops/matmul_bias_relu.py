"""Fused operator: C = relu(A @ B + bias). Stages `matmul -> bias_relu` with one reference and
one kernel signature, so the agent can measure the fused kernel against running the two
stages separately (the fusion gain report).

Precision follows matmul: A, B in `dtype`, C in `output_dtype`, reduction in `accumulate_dtype`.
The bias lives on the output side (same type as C), which is the usual mixed-precision layout
(bf16 activations/weights, fp32 bias and output)."""

from __future__ import annotations

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, default_accumulate_dtype, get_dtype
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile
from ops import bias_relu, matmul
from ops.matmul_packed import build_packed_template

DEFAULT_SHAPE = (512, 512, 512)
SYMBOL = "matmul_bias_relu_kernel"
EPILOGUE = "(((v) + (acc_t)aux[col]) > (acc_t)0 ? ((v) + (acc_t)aux[col]) : (acc_t)0)"


def signature(dtype: DType, out_dtype: DType | None = None) -> str:
    out_dtype = out_dtype or dtype
    return (
        f"void {SYMBOL}(const {dtype.c_type}* A, const {dtype.c_type}* B, const {out_dtype.c_type}* bias, "
        f"{out_dtype.c_type}* C, int M, int N, int K)"
    )


def baseline_source(dtype: DType, out_dtype: DType | None = None, acc_dtype: DType | None = None) -> str:
    out_dtype = out_dtype or dtype
    acc_dtype = acc_dtype or default_accumulate_dtype(dtype, out_dtype)
    return f"""#include <stddef.h>

typedef {dtype.c_type} in_t;
typedef {out_dtype.c_type} out_t;
typedef {acc_dtype.c_type} acc_t;

/* Naive fused loop: GEMM with the bias + ReLU applied as each output element completes. */
{signature(dtype, out_dtype)} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            acc_t acc = (acc_t)0;
            for (int k = 0; k < K; k++) {{
                acc += (acc_t)A[(size_t)i * K + k] * (acc_t)B[(size_t)k * N + j];
            }}
            acc += (acc_t)bias[j];
            C[(size_t)i * N + j] = (out_t)(acc > (acc_t)0 ? acc : (acc_t)0);
        }}
    }}
}}
"""


def make_case(shape: tuple[int, ...], dtype: DType, out_dtype: DType | None = None) -> TestCase:
    rows, cols, depth = shape
    out_dtype = out_dtype or dtype
    return TestCase(
        inputs=(TensorSpec("A", (rows, depth), dtype), TensorSpec("B", (depth, cols), dtype), TensorSpec("bias", (cols,), out_dtype)),
        output=TensorSpec("C", (rows, cols), out_dtype),
        scalars=(rows, cols, depth),
    )


def reference(inputs, scalars) -> np.ndarray:
    matrix_a, matrix_b, bias = inputs
    product = matmul.reference((matrix_a, matrix_b), scalars[:3])
    return bias_relu.reference((product, bias), scalars[:2])


def build(
    shape: tuple[int, ...],
    dtype: str = "fp32",
    workload: WorkloadProfile | None = None,
    output_dtype: str | None = None,
    accumulate_dtype: str | None = None,
) -> OperatorBundle:
    rows, cols, depth = shape
    element = get_dtype(dtype)
    out_element = get_dtype(output_dtype) if output_dtype else element
    acc_element = get_dtype(accumulate_dtype) if accumulate_dtype else default_accumulate_dtype(element, out_element)
    in_size, out_size = element.itemsize, out_element.itemsize
    spec = OperatorSpec(
        name="matmul_bias_relu",
        description=(
            f"C[M,N] = relu(A[M,K] @ B[K,N] + bias[N]) for row-major matrices: A, B in {element.name}, bias and C in "
            f"{out_element.name}, accumulate in {acc_element.name}; one fused kernel call."
        ),
        c_signature=signature(element, out_element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=lambda s, in_dtype: make_case(s, in_dtype, out_element),
        reference=reference,
        flops=lambda s: 2 * s[0] * s[1] * s[2] + 2 * s[0] * s[1],
        bytes_moved=lambda s: in_size * (s[0] * s[2] + s[2] * s[1]) + out_size * (s[0] * s[1] + s[1]),
        scalar_names=("M", "N", "K"),
        edge_shapes=((1, 1, 1), (7, 13, 5), (33, 65, 17), (64, 1, 128), (1, 129, 3)),
        dtype=element,
        output_dtype=out_element,
        accumulate_dtype=acc_element,
        workload=workload,
        fused_stages=("matmul", "bias_relu"),
        notes=[
            "A, B, bias and C are distinct buffers (no aliasing).",
            "Apply bias + ReLU exactly once per output element, after the full K reduction (i.e. on the last K block).",
            f"Accumulate in {acc_element.c_type}; add the bias in {acc_element.c_type} and round to {out_element.c_type} only when storing C.",
        ],
    )
    # Unfused stages at the same shapes: GEMM on (M,N,K) producing the output type, then
    # bias_relu on (M,N) in the output type. Their best separate times are summed and compared
    # with the fused kernel.
    parts = [
        matmul.build(shape, dtype=dtype, output_dtype=out_element.name, accumulate_dtype=acc_element.name),
        bias_relu.build((rows, cols), dtype=out_element.name, accumulate_dtype=acc_element.name),
    ]
    return OperatorBundle(
        spec=spec,
        baseline_source=baseline_source(element, out_element, acc_element),
        templates={
            "packed": build_packed_template(
                signature(element, out_element), element, epilogue=EPILOGUE, aux_name="bias",
                aux_dtype=out_element, out_dtype=out_element, acc_dtype=acc_element,
            )
        },
        default_template="packed",
        fusion_parts=parts,
    )
