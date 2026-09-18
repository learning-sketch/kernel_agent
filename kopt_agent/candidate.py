"""A candidate kernel implementation proposed by some generator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Candidate:
    source: str
    origin: str  # "baseline" | "autotune" | "llm"
    params: dict = field(default_factory=dict)
    extra_compile_flags: tuple[str, ...] = ()
    note: str = ""
    # Declared activation predicate of an optional fast path, over the kernel's int scalars
    # (e.g. "M % 8 == 0 and N % 16 == 0"). A kernel that declares one must export
    # `int kopt_fast_path_active` and set it to 1 exactly when the fast path ran. The
    # evaluator asserts activation on matching inputs and correctness of the fallback on
    # the others, so "fast" cannot mean "only works on the benchmark shape".
    fast_path_predicate: str | None = None

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha1()
        hasher.update(self.source.encode("utf-8"))
        hasher.update("\0".join(self.extra_compile_flags).encode("utf-8"))
        return hasher.hexdigest()[:12]

    def short_label(self) -> str:
        if self.params:
            joined = ",".join(f"{key}={value}" for key, value in sorted(self.params.items()))
            return f"{self.origin}[{joined}]"
        return f"{self.origin}[{self.fingerprint}]"
