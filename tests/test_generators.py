import random

import pytest

from kopt_agent.generators.llm import _extract_c_block
from kopt_agent.generators.template import TemplateGenerator


def test_template_rejects_placeholders_without_defaults():
    with pytest.raises(ValueError):
        TemplateGenerator("void k(void) { int a = $A + $B; }", space={"A": [1]}, default_params={"A": 1})


def test_template_config_sampling_is_distinct_and_respects_constraint():
    generator = TemplateGenerator(
        "int v = $A * $B;",
        space={"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]},
        default_params={"A": 1, "B": 1},
        constraint=lambda params: params["A"] * params["B"] <= 8,
    )
    configs = list(generator.iter_configs(100, random.Random(0)))
    assert len(configs) == len({tuple(sorted(config.items())) for config in configs})
    assert all(config["A"] * config["B"] <= 8 for config in configs)
    assert list(generator.iter_configs(0, random.Random(0))) == []


def test_render_substitutes_and_fingerprints_differ():
    generator = TemplateGenerator("int v = $A;", space={"A": [1, 2]}, default_params={"A": 1})
    first, second = generator.render({"A": 1}), generator.render({"A": 2})
    assert "int v = 1;" in first.source and "int v = 2;" in second.source
    assert first.fingerprint != second.fingerprint


def test_extract_c_block_prefers_longest_fenced_block():
    reply = "Sure.\n```c\nint a;\n```\nand\n```c\n#include <x.h>\nvoid k(void) {}\n```"
    assert _extract_c_block(reply).startswith("#include <x.h>")
    assert _extract_c_block("no code here") is None
