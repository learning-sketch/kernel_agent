"""Row-wise float32 softmax: Y[i, :] = exp(X[i, :] - max_i) / sum(exp(X[i, :] - max_i))."""

from __future__ import annotations

import os

import numpy as np

from kopt_agent.agent import OperatorBundle
from kopt_agent.generators.template import TemplateGenerator
from kopt_agent.spec import OperatorSpec, TensorSpec, TestCase

DEFAULT_SHAPE = (4096, 1024)
SYMBOL = "softmax_kernel"
SIGNATURE = f"void {SYMBOL}(const float* X, float* Y, int M, int N)"

BASELINE_SOURCE = f"""#include <math.h>
#include <stddef.h>

/* Naive, single-threaded, three passes per row. Numerically stable via max subtraction. */
{SIGNATURE} {{
    for (int i = 0; i < M; i++) {{
        const float* row_in = X + (size_t)i * N;
        float* row_out = Y + (size_t)i * N;
        float row_max = row_in[0];
        for (int j = 1; j < N; j++) {{
            if (row_in[j] > row_max) row_max = row_in[j];
        }}
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {{
            row_out[j] = expf(row_in[j] - row_max);
            row_sum += row_out[j];
        }}
        for (int j = 0; j < N; j++) {{
            row_out[j] /= row_sum;
        }}
    }}
}}
"""

TEMPLATE_SOURCE = (
    """#include <math.h>
#include <omp.h>
#include <stddef.h>

/* Rows are independent -> parallel over rows. ONLINE=1 fuses the max and sum passes with the
   online-softmax rescaling trick (one fewer read of the row); ONLINE=0 keeps three passes. */
"""
    + SIGNATURE
    + """ {
    #pragma omp parallel for schedule($SCHEDULE, $CHUNK) num_threads($THREADS)
    for (int i = 0; i < M; i++) {
        const float* row_in = X + (size_t)i * N;
        float* row_out = Y + (size_t)i * N;
#if $ONLINE
        float running_max = row_in[0];
        float running_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            const float value = row_in[j];
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
            row_out[j] = expf(row_in[j] - running_max) * inv_sum;
        }
#else
        float row_max = row_in[0];
        #pragma omp simd reduction(max : row_max)
        for (int j = 0; j < N; j++) {
            row_max = row_in[j] > row_max ? row_in[j] : row_max;
        }
        float row_sum = 0.0f;
        #pragma omp simd reduction(+ : row_sum)
        for (int j = 0; j < N; j++) {
            const float e = expf(row_in[j] - row_max);
            row_out[j] = e;
            row_sum += e;
        }
        const float inv_sum = 1.0f / row_sum;
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            row_out[j] *= inv_sum;
        }
#endif
    }
}
"""
)


def _make_case(shape: tuple[int, ...]) -> TestCase:
    rows, cols = shape
    return TestCase(inputs=(TensorSpec("X", (rows, cols)),), output=TensorSpec("Y", (rows, cols)), scalars=(rows, cols))


def _reference(inputs, scalars) -> np.ndarray:
    (matrix,) = inputs
    values = matrix.astype(np.float64)
    shifted = np.exp(values - values.max(axis=1, keepdims=True))
    return (shifted / shifted.sum(axis=1, keepdims=True)).astype(np.float32)


def _generate_input(tensor: TensorSpec, rng: np.random.Generator) -> np.ndarray:
    # Realistic logit spread; the +-88 sentinels make exp() without max-subtraction overflow float32.
    # (A much wider spread would push most exp() results into denormals and benchmark the FPU
    # denormal path instead of the kernel.)
    values = rng.standard_normal(tensor.shape) * 8.0
    if tensor.shape[0] > 1 and tensor.shape[1] > 1:
        values[0, 0] = 88.0
        values[-1, -1] = -88.0
    return values


def build(shape: tuple[int, ...]) -> OperatorBundle:
    spec = OperatorSpec(
        name="softmax",
        description="Row-wise softmax over the last dimension of X[M,N]; each row of Y sums to 1.",
        c_signature=SIGNATURE,
        symbol=SYMBOL,
        primary_shape=shape,
        make_case=_make_case,
        reference=_reference,
        flops=lambda s: 5 * s[0] * s[1],
        bytes_moved=lambda s: 8 * s[0] * s[1],
        edge_shapes=((1, 1), (3, 7), (5, 1025), (1, 4096), (257, 1)),
        atol=1e-6,
        rtol=1e-4,
        input_generator=_generate_input,
        notes=[
            "Inputs contain values near +-88, so exp() must be computed after subtracting the row max or it overflows.",
            "This operator is memory bound: the ideal kernel reads X once and writes Y once.",
        ],
    )

    thread_options = sorted({1, 2, max(1, (os.cpu_count() or 1) // 2), os.cpu_count() or 1})
    template = TemplateGenerator(
        template_source=TEMPLATE_SOURCE,
        space={
            "THREADS": thread_options,
            "SCHEDULE": ["static", "dynamic"],
            "CHUNK": [1, 16, 64],
            "ONLINE": [0, 1],
            # -ffast-math lets gcc call libmvec's vector expf; the tolerance check decides whether that is acceptable.
            "FASTMATH": [0, 1],
        },
        default_params={"THREADS": os.cpu_count() or 1, "SCHEDULE": "static", "CHUNK": 16, "ONLINE": 0, "FASTMATH": 0},
        extra_flags=lambda params: ("-ffast-math",) if int(params["FASTMATH"]) else (),
    )
    return OperatorBundle(spec=spec, baseline_source=BASELINE_SOURCE, templates={"rows": template}, default_template="rows")
