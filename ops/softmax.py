"""Row-wise softmax: Y[i, :] = exp(X[i, :] - max_i) / sum(exp(X[i, :] - max_i)), fp32 / fp16 / bf16.

Marked precision-sensitive: probabilities feed downstream losses/sampling, so a faster kernel
that is only "reduced-precision" correct must be opted into explicitly."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.dtypes import DType, NumericPolicy, get_dtype
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase, WorkloadProfile

DEFAULT_SHAPE = (4096, 1024)
SYMBOL = "softmax_kernel"


def signature(dtype: DType) -> str:
    return f"void {SYMBOL}(const {dtype.c_type}* X, {dtype.c_type}* Y, int M, int N)"


def baseline_source(dtype: DType) -> str:
    return f"""#include <math.h>
#include <stddef.h>

typedef {dtype.c_type} elem_t;

/* Naive, single-threaded, three passes per row. Numerically stable via max subtraction. */
{signature(dtype)} {{
    for (int i = 0; i < M; i++) {{
        const elem_t* row_in = X + (size_t)i * N;
        elem_t* row_out = Y + (size_t)i * N;
        float row_max = (float)row_in[0];
        for (int j = 1; j < N; j++) {{
            if ((float)row_in[j] > row_max) row_max = (float)row_in[j];
        }}
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {{
            const float e = expf((float)row_in[j] - row_max);
            row_out[j] = (elem_t)e;
            row_sum += e;
        }}
        for (int j = 0; j < N; j++) {{
            row_out[j] = (elem_t)((float)row_out[j] / row_sum);
        }}
    }}
}}
"""


def template_source(dtype: DType) -> str:
    return (
        f"""#include <math.h>
#include <omp.h>
#include <stddef.h>

typedef {dtype.c_type} elem_t;

/* Rows are independent -> parallel over rows. ONLINE=1 fuses the max and sum passes with the
   online-softmax rescaling trick (one fewer read of the row); ONLINE=0 keeps three passes.
   All arithmetic is float; only loads/stores touch the element type. */
"""
        + signature(dtype)
        + """ {
    #pragma omp parallel for schedule($SCHEDULE, $CHUNK) num_threads($THREADS)
    for (int i = 0; i < M; i++) {
        const elem_t* row_in = X + (size_t)i * N;
        elem_t* row_out = Y + (size_t)i * N;
#if $ONLINE
        float running_max = (float)row_in[0];
        float running_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            const float value = (float)row_in[j];
            if (value > running_max) {
                running_sum = running_sum * expf(running_max - value) + 1.0f;
                running_max = value;
            } else {
                running_sum += expf(value - running_max);
            }
        }
        const float inv_sum = 1.0f / running_sum;
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            row_out[j] = (elem_t)(expf((float)row_in[j] - running_max) * inv_sum);
        }
#else
        float row_max = (float)row_in[0];
        #pragma omp simd reduction(max : row_max)
        for (int j = 0; j < N; j++) {
            const float value = (float)row_in[j];
            row_max = value > row_max ? value : row_max;
        }
        float row_sum = 0.0f;
        #pragma omp simd reduction(+ : row_sum)
        for (int j = 0; j < N; j++) {
            const float e = expf((float)row_in[j] - row_max);
            row_out[j] = (elem_t)e;
            row_sum += e;
        }
        const float inv_sum = 1.0f / row_sum;
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            row_out[j] = (elem_t)((float)row_out[j] * inv_sum);
        }
#endif
    }
}
"""
    )


def make_case(shape: tuple[int, ...], dtype: DType) -> TestCase:
    rows, cols = shape
    return TestCase(inputs=(TensorSpec("X", (rows, cols), dtype),), output=TensorSpec("Y", (rows, cols), dtype), scalars=(rows, cols))


def reference(inputs, scalars) -> np.ndarray:
    (matrix,) = inputs
    values = np.asarray(matrix, dtype=np.float64)
    shifted = np.exp(values - values.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def generate_input(tensor: TensorSpec, rng: np.random.Generator) -> np.ndarray:
    # Realistic logit spread; the +-88 sentinels make exp() without max-subtraction overflow float32.
    # (A much wider spread would push most exp() results into denormals and benchmark the FPU
    # denormal path instead of the kernel.)
    scale = 8.0 if tensor.dtype.name == "fp32" else 4.0
    values = rng.standard_normal(tensor.shape) * scale
    if tensor.shape[0] > 1 and tensor.shape[1] > 1:
        sentinel = 88.0 if tensor.dtype.name != "fp16" else 60.0  # fp16 max is 65504 ~ exp(11); keep exp(x-max) representable
        values[0, 0] = sentinel
        values[-1, -1] = -sentinel
    return values


def build(shape: tuple[int, ...], dtype: str = "fp32", workload: WorkloadProfile | None = None) -> OperatorBundle:
    element = get_dtype(dtype)
    spec = OperatorSpec(
        name="softmax",
        description=f"Row-wise softmax over the last dimension of X[M,N] ({element.name}); each row of Y sums to 1.",
        c_signature=signature(element),
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=make_case,
        reference=reference,
        flops=lambda s: 5 * s[0] * s[1],
        bytes_moved=lambda s: 2 * element.itemsize * s[0] * s[1],
        scalar_names=("M", "N"),
        edge_shapes=((1, 1), (3, 7), (5, 1025), (1, 4096), (257, 1)),
        dtype=element,
        # Tighter than the generic fp32 policy: outputs are probabilities in [0, 1].
        numeric_policy=NumericPolicy(atol=1e-6, rtol=1e-4, tight_ulp=16) if element.name == "fp32" else None,
        input_generator=generate_input,
        workload=workload,
        precision_sensitive=True,
        notes=[
            "Inputs contain large-magnitude sentinels, so exp() must be computed after subtracting the row max or it overflows.",
            "This operator is memory bound: the ideal kernel reads X once and writes Y once.",
        ],
    )

    thread_options = sorted({1, 2, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
    template = TemplateGenerator(
        template_source=template_source(element),
        space={
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
            "CHUNK": [1, 16, 64],
            "ONLINE": [0, 1],
            # -ffast-math lets gcc call libmvec's vector expf; the numeric grade decides whether that is acceptable.
            "FASTMATH": [0, 1],
        },
        default_params={"THREADS": os.cpu_count() or 1, "SCHEDULE": "static", "CHUNK": 16, "ONLINE": 0, "FASTMATH": 0},
        extra_flags=lambda params: ("-ffast-math",) if int(params["FASTMATH"]) else (),
    )
    return OperatorBundle(spec=spec, baseline_source=baseline_source(element), templates={"rows": template}, default_template="rows")
