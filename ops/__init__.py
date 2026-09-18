"""Operator registry. Each module exposes `build(shape, dtype, workload) -> OperatorBundle` and `DEFAULT_SHAPE`."""

from __future__ import annotations

from typing import Callable

from kopt_agent.agent import OperatorBundle
from kopt_agent.spec import WorkloadProfile
from ops import bias_relu, matmul, matmul_bias_relu, softmax

Builder = Callable[..., OperatorBundle]

OP_REGISTRY: dict[str, tuple[Builder, tuple[int, ...], str]] = {
    "matmul": (matmul.build, matmul.DEFAULT_SHAPE, "C[M,N] = A[M,K] @ B[K,N]"),
    "softmax": (softmax.build, softmax.DEFAULT_SHAPE, "row-wise softmax over X[M,N] (precision-sensitive)"),
    "bias_relu": (bias_relu.build, bias_relu.DEFAULT_SHAPE, "Y = max(X[M,N] + bias[N], 0)"),
    "matmul_bias_relu": (matmul_bias_relu.build, matmul_bias_relu.DEFAULT_SHAPE, "fused: relu(A @ B + bias)"),
}


def build_operator(
    name: str,
    shape: tuple[int, ...] | None,
    dtype: str = "fp32",
    workload: WorkloadProfile | None = None,
) -> OperatorBundle:
    if name not in OP_REGISTRY:
        raise KeyError(f"unknown operator '{name}', available: {sorted(OP_REGISTRY)}")
    builder, default_shape, _ = OP_REGISTRY[name]
    if workload is not None:
        for entry in workload.entries:
            if len(entry.shape) != len(default_shape):
                raise ValueError(f"workload shape {entry.shape} has {len(entry.shape)} dims, operator '{name}' expects {len(default_shape)}")
        if shape is None:
            # Primary (display) shape = the one that consumes the most work in the trace.
            probe = builder(workload.entries[0].shape, dtype=dtype)
            shape = workload.dominant_shape(probe.spec.flops)
    chosen_shape = tuple(shape or default_shape)
    if len(chosen_shape) != len(default_shape):
        raise ValueError(f"operator '{name}' expects {len(default_shape)} dims, got {chosen_shape}")
    if any(dim <= 0 for dim in chosen_shape):
        raise ValueError(f"all dims must be positive, got {chosen_shape}")
    return builder(chosen_shape, dtype=dtype, workload=workload)
