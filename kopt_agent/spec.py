"""Operator specification: what the kernel must compute, on which shapes, and how to check it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from kopt_agent.dtypes import DTYPES, DType, NumericPolicy, default_accumulate_dtype


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

    - `reference` is the ground truth, computed in float64 on the host from decoded inputs and
      rounded to each output tensor's own dtype before comparison.
    - `make_case(shape, dtype)` builds a TestCase for a shape so the same kernel can be
      checked on the benchmark shape, the workload shapes and awkward edge shapes. `dtype` is
      the input element type; the case's TensorSpecs decide the type of every tensor.
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
    # Mixed precision: `dtype` is the element type of the inputs (the "operator dtype");
    # `output_dtype` may differ (e.g. bf16 in, fp32 out) and `accumulate_dtype` pins the
    # precision of the reduction (defaults to at least fp32). Each TensorSpec carries its own
    # dtype, so make_case may also assign per-tensor types beyond these two.
    dtype: DType = DTYPES["fp32"]
    output_dtype: DType | None = None
    accumulate_dtype: DType | None = None
    numeric_policy: NumericPolicy | None = None  # overrides the output dtype's policy when set
    input_generator: Callable[[TensorSpec, np.random.Generator], np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)
    workload: WorkloadProfile | None = None
    # Reduced-precision (but accepted) candidates may not become the best unless explicitly allowed.
    precision_sensitive: bool = False
    # For fused operators: the names of the stages that were fused (informational).
    fused_stages: tuple[str, ...] = ()

    @property
    def out_dtype(self) -> DType:
        return self.output_dtype or self.dtype

    @property
    def acc_dtype(self) -> DType:
        return self.accumulate_dtype or default_accumulate_dtype(self.dtype, self.out_dtype)

    @property
    def mixed_precision(self) -> bool:
        return self.out_dtype.name != self.dtype.name or self.acc_dtype.name != default_accumulate_dtype(self.dtype, self.out_dtype).name

    def dtypes_used(self) -> list[DType]:
        """Distinct element types a kernel must be able to spell (inputs, output, accumulator)."""
        seen: dict[str, DType] = {}
        case = self.primary_case()
        for tensor in (*case.inputs, case.output):
            seen.setdefault(tensor.dtype.name, tensor.dtype)
        for dtype in (self.dtype, self.out_dtype, self.acc_dtype):
            seen.setdefault(dtype.name, dtype)
        return list(seen.values())

    def precision_label(self) -> str:
        """Short human label: "fp32" or "bf16 -> fp32 (acc fp32)"."""
        if not self.mixed_precision:
            return self.dtype.name
        return f"{self.dtype.name} -> {self.out_dtype.name} (acc {self.acc_dtype.name})"

    def tensor_dtypes(self) -> dict[str, str]:
        """Per-tensor element types of the primary case, by tensor name (inputs then output)."""
        case = self.primary_case()
        return {**{tensor.name: tensor.dtype.name for tensor in case.inputs}, case.output.name: case.output.dtype.name}

    def primary_case(self) -> TestCase:
        return self.make_case(self.primary_shape, self.dtype)

    @property
    def policy(self) -> NumericPolicy:
        # Correctness is judged on what is stored, so the output type sets the tolerance.
        return self.numeric_policy or self.out_dtype.policy

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
