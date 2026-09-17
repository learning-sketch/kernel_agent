"""Parametric kernel templates: a hand-written, known-correct schedule whose tuning knobs
(tile sizes, unroll factors, thread counts, ...) are searched automatically."""

from __future__ import annotations

import itertools
import random
from string import Template
from typing import Callable, Iterator

from kopt_agent.candidate import Candidate

Params = dict[str, object]


class TemplateGenerator:
    def __init__(
        self,
        template_source: str,
        space: dict[str, list],
        default_params: Params,
        constraint: Callable[[Params], bool] | None = None,
        extra_flags: Callable[[Params], tuple[str, ...]] | None = None,
    ) -> None:
        self.template = Template(template_source)
        self.space = space
        self.default_params = default_params
        self.constraint = constraint or (lambda _params: True)
        self.extra_flags = extra_flags or (lambda _params: ())
        missing = set(self.template.get_identifiers()) - set(default_params)
        if missing:
            raise ValueError(f"template placeholders without defaults: {sorted(missing)}")

    def render(self, params: Params, origin: str = "autotune") -> Candidate:
        merged = {**self.default_params, **params}
        source = self.template.substitute({key: str(value) for key, value in merged.items()})
        return Candidate(source=source, origin=origin, params=dict(params), extra_compile_flags=self.extra_flags(merged))

    def default_candidate(self) -> Candidate:
        return self.render(dict(self.default_params), origin="template-default")

    def space_size(self) -> int:
        total = 1
        for values in self.space.values():
            total *= len(values)
        return total

    def iter_configs(self, budget: int, rng: random.Random) -> Iterator[Params]:
        """Yield up to `budget` distinct configurations satisfying the constraint.

        Small spaces are enumerated exhaustively (shuffled); large spaces are sampled
        without replacement so the same schedule is never compiled twice.
        """
        if budget <= 0:
            return
        keys = list(self.space)
        if self.space_size() <= 4096:
            configs = [dict(zip(keys, values)) for values in itertools.product(*(self.space[key] for key in keys))]
            configs = [config for config in configs if self.constraint(config)]
            rng.shuffle(configs)
            yield from configs[:budget]
            return

        seen: set[tuple] = set()
        attempts = 0
        emitted = 0
        while emitted < budget and attempts < budget * 50:
            attempts += 1
            config = {key: rng.choice(self.space[key]) for key in keys}
            signature = tuple(config[key] for key in keys)
            if signature in seen or not self.constraint(config):
                continue
            seen.add(signature)
            emitted += 1
            yield config
