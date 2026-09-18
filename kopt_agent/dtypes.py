"""Element types the agent can target, with their C spelling, numpy storage and numeric policy.

bf16 has no numpy dtype, so it is stored as uint16 and (de)coded here. The reference is
always computed in float64 and rounded to the target dtype before comparison, so "correct"
means "as good as the best possible result in that dtype", not "close to fp64".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumericPolicy:
    """Acceptance and grading thresholds for one dtype.

    - `atol`/`rtol`: acceptance bound (candidate is `incorrect` beyond it).
    - `tight_ulp`: grade `within-N-ULP` if the worst error is at most this many ULPs *at the
      output's magnitude scale* (spacing of max|expected|); errors near zero therefore do
      not blow up into false alarms the way per-element ULP would.
    - Anything accepted but looser than `tight_ulp` is graded `reduced-precision`.
    """

    atol: float
    rtol: float
    tight_ulp: float


@dataclass(frozen=True)
class DType:
    name: str  # "fp32" | "fp16" | "bf16"
    c_type: str  # spelling inside kernel prototypes
    storage: type  # numpy dtype used for buffers handed to the kernel
    itemsize: int
    mantissa_bits: int
    policy: NumericPolicy

    def encode(self, values: np.ndarray) -> np.ndarray:
        """float64/float32 -> storage array, rounding to nearest even."""
        if self.name == "bf16":
            bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
            # RNE on the low 16 bits; NaN payload preserved by forcing the quiet bit.
            rounding = ((bits >> 16) & 1) + 0x7FFF
            rounded = ((bits + rounding) >> 16).astype(np.uint16)
            nan_mask = np.isnan(values)
            rounded[nan_mask] = 0x7FC0 | ((bits[nan_mask] >> 16) & 0x8000).astype(np.uint16)
            return np.ascontiguousarray(rounded)
        return np.ascontiguousarray(values, dtype=self.storage)

    def decode(self, buffer: np.ndarray) -> np.ndarray:
        """storage array -> float64 for comparison."""
        if self.name == "bf16":
            return (buffer.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return buffer.astype(np.float64)

    def spacing_at(self, magnitude: float) -> float:
        """ULP of this dtype at |magnitude| (used for scaled-ULP grading)."""
        if magnitude == 0.0 or not np.isfinite(magnitude):
            return float(np.finfo(np.float32).tiny) if self.name == "bf16" else float(np.finfo(self.storage).tiny)
        exponent = np.floor(np.log2(magnitude))
        return float(2.0 ** (exponent - self.mantissa_bits))

    def max_finite(self) -> float:
        if self.name == "bf16":
            return float(np.finfo(np.float32).max)
        return float(np.finfo(self.storage).max)


DTYPES: dict[str, DType] = {
    "fp32": DType("fp32", "float", np.float32, 4, 23, NumericPolicy(atol=1e-3, rtol=1e-4, tight_ulp=16)),
    # 16-bit acceptance is intentionally relative: results are compared against the fp64
    # reference rounded into the same 16-bit format.
    "fp16": DType("fp16", "_Float16", np.float16, 2, 10, NumericPolicy(atol=1e-2, rtol=8e-3, tight_ulp=4)),
    "bf16": DType("bf16", "__bf16", np.uint16, 2, 7, NumericPolicy(atol=5e-2, rtol=3e-2, tight_ulp=4)),
}


def get_dtype(name: str) -> DType:
    try:
        return DTYPES[name]
    except KeyError as error:
        raise KeyError(f"unknown dtype '{name}', available: {sorted(DTYPES)}") from error
