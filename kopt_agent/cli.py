"""Command line entry point: `kopt run --op matmul --shape 512 512 512 --autotune-budget 12 --llm-rounds 5`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kopt_agent.agent import AgentConfig, OptimizationAgent
from kopt_agent.backends import BACKENDS, get_backend
from kopt_agent.generators.llm import LLMConfig, LLMGenerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kopt", description="Search for the fastest correct kernel of an operator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the optimization loop for one operator")
    run.add_argument("--op", required=True, help="operator name (see `kopt list-ops`)")
    run.add_argument("--shape", type=int, nargs="+", help="benchmark shape, e.g. --shape 512 512 512")
    run.add_argument("--backend", default="cpu_c", choices=sorted(BACKENDS))
    run.add_argument("--autotune-budget", type=int, default=12, help="number of template schedules to try (0 disables)")
    run.add_argument("--llm-rounds", type=int, default=0, help="LLM refinement rounds (0 disables)")
    run.add_argument("--llm-patience", type=int, default=4, help="stop LLM phase after this many rounds without a new best")
    run.add_argument("--model", default=None, help="model name for the OpenAI-compatible endpoint (default: $KOPT_LLM_MODEL)")
    run.add_argument("--repeats", type=int, default=15, help="timed runs per candidate (median is reported)")
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--run-timeout", type=float, default=60.0, help="seconds before a candidate is declared hung")
    run.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS for the runner (default: all cores)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("-v", "--verbose", action="store_true")

    subparsers.add_parser("list-ops", help="list registered operators")

    show = subparsers.add_parser("show", help="print the best kernel found for an operator")
    show.add_argument("--op", required=True)
    show.add_argument("--out", type=Path, default=Path("results"))
    return parser


def _command_list_ops() -> int:
    from ops import OP_REGISTRY

    for name, (_, default_shape, description) in sorted(OP_REGISTRY.items()):
        print(f"{name:<10} default shape {'x'.join(map(str, default_shape)):<16} {description}")
    return 0


def _command_show(args: argparse.Namespace) -> int:
    best_path = args.out / args.op / "best.c"
    if not best_path.exists():
        print(f"no result at {best_path}; run `kopt run --op {args.op}` first", file=sys.stderr)
        return 1
    print(best_path.read_text(encoding="utf-8"))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    from ops import build_operator

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("kopt")

    if args.autotune_budget < 0 or args.llm_rounds < 0 or args.repeats < 1:
        print("budgets must be >= 0 and --repeats >= 1", file=sys.stderr)
        return 2

    try:
        bundle = build_operator(args.op, tuple(args.shape) if args.shape else None)
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    backend = get_backend(args.backend, threads=args.threads)

    llm = None
    if args.llm_rounds > 0:
        llm_config = LLMConfig.from_env(model=args.model)
        if llm_config is None:
            logger.warning("--llm-rounds given but no API key found; set KOPT_LLM_API_KEY (or OPENAI_API_KEY), KOPT_LLM_BASE_URL, KOPT_LLM_MODEL")
        else:
            llm = LLMGenerator(llm_config)
            logger.info("llm: %s @ %s", llm_config.model, llm_config.base_url)

    config = AgentConfig(
        autotune_budget=args.autotune_budget,
        llm_rounds=args.llm_rounds,
        llm_patience=args.llm_patience,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
        run_timeout_seconds=args.run_timeout,
        output_dir=args.out,
    )
    agent = OptimizationAgent(bundle, backend, config, llm=llm)
    try:
        history = agent.run()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print()
    print(history.leaderboard())
    if history.best is not None:
        best_candidate, best_result = history.best
        print(f"\nbest: trial {best_result.trial_id} ({best_candidate.short_label()}) -> {history.output_dir / 'best.c'}")
    print(f"trial log: {history.trials_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list-ops":
        return _command_list_ops()
    if args.command == "show":
        return _command_show(args)
    return _command_run(args)


if __name__ == "__main__":
    sys.exit(main())
