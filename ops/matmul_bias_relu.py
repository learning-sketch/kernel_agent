"""Fused operator: C = relu(A @ B + bias). Stages `matmul -> bias_relu` with one reference and
one kernel signature, so the agent can measure the fused kernel against running the two
stages separately (the fusion gain report)."""

from __future__ import annotations

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, get_dtype
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile
from ops import bias_relu, matmul
from ops.matmul_packed import build_packed_template

DEFAULT_SHAPE = (512, 512, 512)
SYMBOL = "matmul_bias_relu_kernel"
EPILOGUE = "(((v) + (float)aux[col]) > 0.0f ? ((v) + (float)aux[col]) : 0.0f)"


def signature(dtype: DType) -> str:
    c_type = dtype.c_type
    return f"void {SYMBOL}(const {c_type}* A, const {c_type}* B, const {c_type}* bias, {c_type}* C, int M, int N, int K)"


def baseline_source(dtype: DType) -> str:
    return f"""#include <stddef.h>

typedef {dtype.c_type} elem_t;

/* Naive fused loop: GEMM with the bias + ReLU applied as each output element completes. */
{signature(dtype)} {{
    for (int i = 0; i < M; i++) {{
        for (int j = 0; j < N; j++) {{
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {{
                acc += (float)A[(size_t)i * K + k] * (float)B[(size_t)k * N + j];
            }}
            acc += (float)bias[j];
            C[(size_t)i * N + j] = (elem_t)(acc > 0.0f ? acc : 0.0f);
        }}
    }}
}}
"""


def make_case(shape: tuple[int, ...], dtype: DType) -> TestCase:
    rows, cols, depth = shape
    return TestCase(
        inputs=(TensorSpec("A", (rows, depth), dtype), TensorSpec("B", (depth, cols), dtype), TensorSpec("bias", (cols,), dtype)),
        output=TensorSpec("C", (rows, cols), dtype),
        scalars=(rows, cols, depth),
    )


def reference(inputs, scalars) -> np.ndarray:
    matrix_a, matrix_b, bias = inputs
    product = matmul.reference((matrix_a, matrix_b), scalars[:3])
    return bias_relu.reference((product, bias), scalars[:2])


def build(shape: tuple[int, ...], dtype: str = "fp32", workload: WorkloadProfile | None = None) -> OperatorBundle:
    rows, cols, depth = shape
    element = get_dtype(dtype)
    itemsize = element.itemsize
    spec = OperatorSpec(
        name="matmul_bias_relu",
        description=f"C[M,N] = relu(A[M,K] @ B[K,N] + bias[N]) for row-major {element.name} matrices; one fused kernel call.",
        c_signature=signature(element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=make_case,
        reference=reference,
        flops=lambda s: 2 * s[0] * s[1] * s[2] + 2 * s[0] * s[1],
        bytes_moved=lambda s: itemsize * (s[0] * s[2] + s[2] * s[1] + s[0] * s[1] + s[1]),
        scalar_names=("M", "N", "K"),
        edge_shapes=((1, 1, 1), (7, 13, 5), (33, 65, 17), (64, 1, 128), (1, 129, 3)),
        dtype=element,
        workload=workload,
        fused_stages=("matmul", "bias_relu"),
        notes=[
            "A, B, bias and C are distinct buffers (no aliasing).",
            "Apply bias + ReLU exactly once per output element, after the full K reduction (i.e. on the last K block).",
        ],
    )
    # Unfused stages at the same shapes: GEMM on (M,N,K) then bias_relu on (M,N). Their best
    # separate times are summed and compared with the fused kernel.
    parts = [matmul.build(shape, dtype=dtype), bias_relu.build((rows, cols), dtype=dtype)]
    return OperatorBundle(
        spec=spec,
        baseline_source=baseline_source(element),
        templates={"packed": build_packed_template(signature(element), element, epilogue=EPILOGUE, aux_name="bias")},
        default_template="packed",
        fusion_parts=parts,
    )
