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
        fast_path: Callable[[Params], str | None] | None = None,
    ) -> None:
        self.template = Template(template_source)
        self.space = space
        self.default_params = default_params
        self.constraint = constraint or (lambda _params: True)
        self.extra_flags = extra_flags or (lambda _params: ())
        # Optional: the activation predicate of the fast path a configuration enables (None = no fast path).
        self.fast_path = fast_path or (lambda _params: None)
        missing = set(self.template.get_identifiers()) - set(default_params)
        if missing:
            raise ValueError(f"template placeholders without defaults: {sorted(missing)}")

    def render(self, params: Params, origin: str = "autotune") -> Candidate:
        merged = {**self.default_params, **params}
        source = self.template.substitute({key: str(value) for key, value in merged.items()})
        return Candidate(
            source=source, origin=origin, params=dict(params), extra_compile_flags=self.extra_flags(merged),
            fast_path_predicate=self.fast_path(merged),
        )

    def default_candidate(self) -> Candidate:
        return self.render(dict(self.default_params), origin="template-default")

    def space_size(self) -> int:
        total = 1
        for values in self.space.values():
            total *= len(values)
        return total

    def signature(self, params: Params) -> tuple:
        merged = {**self.default_params, **params}
        return tuple(str(merged[key]) for key in self.space)

    def mutate(self, parent: Params, rng: random.Random, seen: set[tuple], attempts: int = 32) -> Params | None:
        """Neighbourhood move: shift one or two knobs to an adjacent value in their option list
        (so tile sizes double/halve rather than jump), occasionally re-draw a knob at random.
        Returns None if no unseen, constraint-satisfying neighbour is found."""
        keys = list(self.space)
        merged_parent = {**self.default_params, **parent}
        for _ in range(attempts):
            child = dict(merged_parent)
            for key in rng.sample(keys, k=min(len(keys), rng.choice((1, 1, 2)))):
                options = self.space[key]
                if len(options) < 2:
                    continue
                if rng.random() < 0.25 or child[key] not in options:
                    child[key] = rng.choice(options)
                else:
                    index = options.index(child[key])
                    step = rng.choice((-1, 1))
                    child[key] = options[min(max(index + step, 0), len(options) - 1)]
            child = {key: child[key] for key in keys}
            signature = self.signature(child)
            if signature in seen or not self.constraint(child):
                continue
            return child
        return None

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
