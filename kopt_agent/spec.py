"""Operator specification: what the kernel must compute, on which shapes, and how to check it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from kopt_agent.dtypes import DTYPES, DType, NumericPolicy


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DType = DTYPES["fp32"]

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize


@dataclass(frozen=True)
class TestCase:
    """One concrete set of shapes. The kernel ABI receives pointers first, then int scalars.

    `weight` is the number of calls this shape receives in the workload trace; 0 means the
    case is used for correctness only and never timed.
    """

    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    scalars: tuple[int, ...]
    label: str = "primary"
    weight: int = 0
    shape: tuple[int, ...] = ()  # the operator-level shape this case was built from

    @property
    def shape_label(self) -> str:
        return "x".join(str(dim) for dim in self.output.shape)


@dataclass(frozen=True)
class WorkloadEntry:
    shape: tuple[int, ...]
    count: int


@dataclass
class WorkloadProfile:
    """(shape, call count) signature of one operator in a representative end-to-end trace.

    The search objective becomes the count-weighted total time over these shapes instead of
    the latency of a single benchmark shape.
    """

    entries: list[WorkloadEntry]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("workload profile has no shapes")
        merged: dict[tuple[int, ...], int] = {}
        for entry in self.entries:
            if entry.count <= 0 or any(dim <= 0 for dim in entry.shape):
                raise ValueError(f"workload entry must have positive shape and count: {entry}")
            merged[entry.shape] = merged.get(entry.shape, 0) + entry.count
        self.entries = [WorkloadEntry(shape, count) for shape, count in merged.items()]

    @classmethod
    def load(cls, path: Path) -> "WorkloadProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_entries = data["shapes"] if isinstance(data, dict) else data
        entries = []
        for raw in raw_entries:
            if isinstance(raw, dict):
                entries.append(WorkloadEntry(tuple(int(d) for d in raw["shape"]), int(raw.get("count", 1))))
            else:
                shape, count = raw
                entries.append(WorkloadEntry(tuple(int(d) for d in shape), int(count)))
        return cls(entries)

    @property
    def total_calls(self) -> int:
        return sum(entry.count for entry in self.entries)

    def dominant_shape(self, flops: Callable[[tuple[int, ...]], int]) -> tuple[int, ...]:
        """The shape that consumes the most work (count x flops) - a sensible primary shape."""
        return max(self.entries, key=lambda entry: entry.count * max(flops(entry.shape), 1)).shape


@dataclass
class OperatorSpec:
    """Everything the agent needs to know about one operator.

    - `reference` is the ground truth, computed in float64 on the host from decoded inputs.
    - `make_case(shape, dtype)` builds a TestCase for a shape so the same kernel can be
      checked on the benchmark shape, the workload shapes and awkward edge shapes.
    - `c_signature` is the exact function prototype every candidate must implement.
    """

    name: str
    description: str
    c_signature: str
    symbol: str
    primary_shape: tuple[int, ...]
    make_case: Callable[[tuple[int, ...], DType], TestCase]
    reference: Callable[[Sequence[np.ndarray], Sequence[int]], np.ndarray]
    flops: Callable[[tuple[int, ...]], int]
    bytes_moved: Callable[[tuple[int, ...]], int]
    scalar_names: tuple[str, ...] = ()
    edge_shapes: tuple[tuple[int, ...], ...] = ()
    dtype: DType = DTYPES["fp32"]
    numeric_policy: NumericPolicy | None = None  # overrides dtype.policy when set
    input_generator: Callable[[TensorSpec, np.random.Generator], np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)
    workload: WorkloadProfile | None = None
    # Reduced-precision (but accepted) candidates may not become the best unless explicitly allowed.
    precision_sensitive: bool = False
    # For fused operators: the names of the stages that were fused (informational).
    fused_stages: tuple[str, ...] = ()

    @property
    def policy(self) -> NumericPolicy:
        return self.numeric_policy or self.dtype.policy

    @property
    def atol(self) -> float:
        return self.policy.atol

    @property
    def rtol(self) -> float:
        return self.policy.rtol

    def generate_input(self, tensor: TensorSpec, rng: np.random.Generator) -> np.ndarray:
        """Return the *storage* array handed to the kernel (already rounded to the dtype)."""
        if self.input_generator is not None:
            values = np.asarray(self.input_generator(tensor, rng), dtype=np.float64)
        else:
            values = rng.standard_normal(tensor.shape)
        return tensor.dtype.encode(values)

    def timing_shapes(self) -> list[tuple[tuple[int, ...], int]]:
        """(shape, weight) pairs that are benchmarked."""
        if self.workload is None:
            return [(self.primary_shape, 1)]
        pairs = [(entry.shape, entry.count) for entry in self.workload.entries]
        if self.primary_shape not in {shape for shape, _ in pairs}:
            pairs.insert(0, (self.primary_shape, 0))
        return pairs

    def all_cases(self) -> list[TestCase]:
        cases: list[TestCase] = []
        seen: set[tuple[int, ...]] = set()
        for shape, weight in self.timing_shapes():
            case = self.make_case(shape, self.dtype)
            label = "primary" if shape == self.primary_shape else f"wl{shape}"
            cases.append(TestCase(case.inputs, case.output, case.scalars, label=label, weight=weight, shape=tuple(shape)))
            seen.add(shape)
        for shape in self.edge_shapes:
            if shape in seen:
                continue
            seen.add(shape)
            case = self.make_case(shape, self.dtype)
            cases.append(TestCase(case.inputs, case.output, case.scalars, label=f"edge{shape}", weight=0, shape=tuple(shape)))
        return cases


def evaluate_fast_path_predicate(expression: str, scalar_names: Sequence[str], scalars: Sequence[int]) -> bool:
    """Evaluate a candidate's declared fast-path predicate (e.g. "M % 8 == 0 and N >= 64")
    over the case's integer scalars. Only arithmetic/comparison/boolean syntax is allowed."""
    import ast

    allowed = (
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load,
        ast.Constant, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    normalized = expression.replace("&&", " and ").replace("||", " or ").replace("!=", "__NE__").replace("!", " not ").replace("__NE__", "!=")
    tree = ast.parse(normalized, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported syntax in fast-path predicate: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in scalar_names:
            raise ValueError(f"unknown name '{node.id}' in fast-path predicate (allowed: {', '.join(scalar_names)})")
    return bool(eval(compile(tree, "<fast-path>", "eval"), {"__builtins__": {}}, dict(zip(scalar_names, scalars))))
