"""Operator registry. Each module exposes `build(shape) -> OperatorBundle` and `DEFAULT_SHAPE`."""

from __future__ import annotations

from typing import Callable

from kopt_agent.agent import OperatorBundle
from ops import matmul, softmax

OP_REGISTRY: dict[str, tuple[Callable[[tuple[int, ...]], OperatorBundle], tuple[int, ...], str]] = {
    "matmul": (matmul.build, matmul.DEFAULT_SHAPE, "C[M,N] = A[M,K] @ B[K,N], float32"),
    "softmax": (softmax.build, softmax.DEFAULT_SHAPE, "row-wise softmax over X[M,N], float32"),
}


def build_operator(name: str, shape: tuple[int, ...] | None) -> OperatorBundle:
    if name not in OP_REGISTRY:
        raise KeyError(f"unknown operator '{name}', available: {sorted(OP_REGISTRY)}")
    builder, default_shape, _ = OP_REGISTRY[name]
    chosen_shape = shape or default_shape
    if len(chosen_shape) != len(default_shape):
        raise ValueError(f"operator '{name}' expects {len(default_shape)} dims, got {chosen_shape}")
    if any(dim <= 0 for dim in chosen_shape):
        raise ValueError(f"all dims must be positive, got {chosen_shape}")
    return builder(tuple(chosen_shape))
