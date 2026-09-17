"""Operator specification: what the kernel must compute, and how to check it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: type = np.float32

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class TestCase:
    """One concrete set of shapes. The kernel ABI receives pointers first, then int scalars."""

    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    scalars: tuple[int, ...]
    label: str = "primary"


@dataclass
class OperatorSpec:
    """Everything the agent needs to know about one operator.

    - `reference` is the ground truth (numpy), run on the host.
    - `make_case(shape)` builds a TestCase for a given shape so the same kernel can be
      checked on the benchmark shape and on awkward edge shapes.
    - `c_signature` is the exact function prototype every candidate must implement.
    """

    name: str
    description: str
    c_signature: str
    symbol: str
    primary_shape: tuple[int, ...]
    make_case: Callable[[tuple[int, ...]], TestCase]
    reference: Callable[[Sequence[np.ndarray], Sequence[int]], np.ndarray]
    flops: Callable[[tuple[int, ...]], int]
    bytes_moved: Callable[[tuple[int, ...]], int]
    edge_shapes: tuple[tuple[int, ...], ...] = ()
    rtol: float = 1e-4
    atol: float = 1e-4
    input_generator: Callable[[TensorSpec, np.random.Generator], np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)

    def generate_input(self, tensor: TensorSpec, rng: np.random.Generator) -> np.ndarray:
        if self.input_generator is not None:
            return np.ascontiguousarray(self.input_generator(tensor, rng), dtype=tensor.dtype)
        return np.ascontiguousarray(rng.standard_normal(tensor.shape), dtype=tensor.dtype)

    def all_cases(self) -> list[TestCase]:
        cases = [self.make_case(self.primary_shape)]
        for shape in self.edge_shapes:
            case = self.make_case(shape)
            cases.append(TestCase(case.inputs, case.output, case.scalars, label=f"edge{shape}"))
        return cases
