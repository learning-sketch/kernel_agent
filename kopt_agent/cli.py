"""Command line entry point: `kopt run --op matmul --shape 512 512 512 --autotune-budget 12 --llm-rounds 5`.

Workload-driven: `kopt run --op matmul --workload profile.json` where profile.json is
{"shapes": [{"shape": [512, 512, 512], "count": 120}, {"shape": [64, 512, 512], "count": 900}]}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kopt_agent.agent import AgentConfig, OptimizationAgent
from kopt_agent.backends import BACKENDS, get_backend
from kopt_agent.dtypes import DTYPES
from kopt_agent.generators.llm import LLMConfig, LLMGenerator
from kopt_agent.spec import WorkloadProfile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kopt", description="Search for the fastest correct kernel of an operator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the optimization loop for one operator")
    run.add_argument("--op", required=True, help="operator name (see `kopt list-ops`)")
    run.add_argument("--shape", type=int, nargs="+", help="benchmark shape, e.g. --shape 512 512 512")
    run.add_argument("--workload", type=Path, default=None, help="JSON workload profile: (shape, call count) pairs; the objective becomes the call-weighted total time")
    run.add_argument("--dtype", default="fp32", choices=sorted(DTYPES), help="element type of the inputs")
    run.add_argument("--output-dtype", default=None, choices=sorted(DTYPES), help="element type of the output (default: same as --dtype)")
    run.add_argument("--accumulate-dtype", default=None, choices=sorted(DTYPES), help="accumulation precision (default: fp32, or fp64 for fp64 I/O)")
    run.add_argument("--allow-reduced-precision", action="store_true", help="let reduced-precision candidates win precision-sensitive operators")
    run.add_argument("--ceiling-fraction", type=float, default=0.85, help="declare 'at the ceiling' when best >= this fraction of the roofline-attainable time")
    run.add_argument("--no-stop-at-ceiling", action="store_true", help="keep searching even when the roofline verdict says the ceiling is reached")
    run.add_argument("--no-fusion-report", action="store_true", help="skip the fused-vs-separate measurement for fused operators")
    run.add_argument("--backend", default="cpu_c", choices=sorted(BACKENDS))
    run.add_argument("--autotune-budget", type=int, default=12, help="number of template schedules to try (0 disables)")
    run.add_argument("--template", default=None, help="which template of the operator to tune (see `kopt list-ops`)")
    run.add_argument("--evolve-fraction", type=float, default=0.5, help="share of the autotune budget spent mutating the top configs")
    run.add_argument("--warm-start", type=int, default=3, help="configs seeded from results/<op>/knowledge.jsonl (0 disables)")
    run.add_argument("--workers", type=int, default=None, help="parallel compile/verify workers and concurrent LLM requests (default: min(4, cores))")
    run.add_argument("--top-k", type=int, default=3, help="candidates per batch that get benchmark-grade timing")
    run.add_argument("--quick-repeats", type=int, default=5, help="timed runs used for coarse screening")
    run.add_argument("--no-roofline", action="store_true", help="skip the peak FMA / bandwidth probes")
    run.add_argument("--llm-rounds", type=int, default=0, help="LLM refinement rounds (0 disables)")
    run.add_argument("--llm-samples", type=int, default=1, help="best-of-N candidates requested per LLM round")
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

    workload = subparsers.add_parser("workload", help="build a workload profile from a profiler trace (CSV / JSONL / JSON)")
    workload.add_argument("--trace", type=Path, required=True, help="trace file; one row per call (or with a count column)")
    workload.add_argument("--op", default=None, help="keep only rows whose op/operator/kernel column equals this name")
    workload.add_argument("--dims", nargs="+", default=None, help="dimension column names in operator order, e.g. --dims M N K")
    workload.add_argument("--profile", type=Path, required=True, help="where to write the profile JSON (input for `kopt run --workload`)")

    export = subparsers.add_parser("export", help="(re)build the integration bundle of a finished run from results/<op>/")
    export.add_argument("--op", required=True)
    export.add_argument("--out", type=Path, default=Path("results"))
    export.add_argument("--bundle-dir", type=Path, default=None, help="destination (default: results/<op>/bundle)")
    return parser


def _command_workload(args: argparse.Namespace) -> int:
    from kopt_agent.workload import ingest_trace, save_profile

    try:
        report = ingest_trace(args.trace, operator=args.op, dims=args.dims)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: cannot read trace {args.trace}: {error}", file=sys.stderr)
        return 2
    print(report.describe())
    if report.profile is None:
        print("error: no usable rows in the trace", file=sys.stderr)
        return 1
    path = save_profile(report.profile, args.profile)
    print(f"profile written to {path}")
    return 0


def _command_export(args: argparse.Namespace) -> int:
    from kopt_agent.backends import get_backend
    from kopt_agent.bundle import BundleContext, export_bundle, load_bundle_inputs
    from kopt_agent.candidate import Candidate
    from kopt_agent.evaluator import TrialResult, TrialStatus
    from kopt_agent.history import History
    from kopt_agent.spec import WorkloadEntry, WorkloadProfile
    from ops import build_operator

    operator_dir = args.out / args.op
    try:
        summary, source = load_bundle_inputs(operator_dir)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not summary.get("best"):
        print(f"error: the run in {operator_dir} found no correct kernel; nothing to export", file=sys.stderr)
        return 1
    workload = None
    if summary.get("workload"):
        workload = WorkloadProfile([WorkloadEntry(tuple(entry["shape"]), int(entry["count"])) for entry in summary["workload"]])
    try:
        bundle = build_operator(
            summary["operator"], tuple(summary["shape"]), dtype=summary.get("dtype", "fp32"), workload=workload,
            output_dtype=summary.get("output_dtype"), accumulate_dtype=summary.get("accumulate_dtype"),
        )
    except (KeyError, ValueError) as error:
        print(f"error: cannot rebuild the operator spec from summary.json: {error}", file=sys.stderr)
        return 1
    best = dict(summary["best"])
    best["status"] = TrialStatus(best["status"])
    result = TrialResult(**best)
    record = summary.get("best_candidate") or {}
    candidate = Candidate(
        source=source, origin=record.get("origin", result.origin), params=record.get("params", result.params),
        extra_compile_flags=tuple(record.get("extra_compile_flags", ())), note=record.get("note", ""),
        fast_path_predicate=record.get("fast_path_predicate"),
    )
    backend_name = summary.get("backend", "cpu_c")
    compile_command = None
    launch_abi = summary.get("launch_abi")
    try:
        backend = get_backend(backend_name)
        compile_command = backend.portable_compile_command()
        launch_abi = launch_abi or backend.launch_abi.to_dict()
    except (KeyError, RuntimeError):
        pass
    # A throw-away History only for its parity-test / compile-command helpers: do not touch trials.jsonl.
    history = History.__new__(History)
    history.output_dir = operator_dir
    history.spec = bundle.spec
    history.compile_command = compile_command or []
    baseline = None
    trials_path = operator_dir / "trials.jsonl"
    if trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            try:
                record_dict = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record_dict.get("origin") == "baseline" and record_dict.get("status") == TrialStatus.OK.value:
                record_dict.pop("note", None)
                record_dict["status"] = TrialStatus.OK
                baseline = TrialResult(**record_dict)
                break
    context = BundleContext(backend_name=backend_name, launch_abi=launch_abi or {}, hardware=summary.get("hardware", "unknown"),
                            baseline=baseline, verdict=summary.get("verdict"), peaks=summary.get("peaks"),
                            extra={"fusion_gain": summary["fusion_gain"]} if summary.get("fusion_gain") else {})
    directory = export_bundle(history, candidate, result, context, directory=args.bundle_dir)
    print(f"bundle written to {directory}")
    for name in sorted(path.name for path in directory.iterdir()):
        print(f"  {name}")
    return 0


def _command_list_ops() -> int:
    from ops import OP_REGISTRY

    for name, (builder, default_shape, description) in sorted(OP_REGISTRY.items()):
        bundle = builder(default_shape, dtype="fp32")
        templates = ", ".join(
            f"{template_name}{'*' if template_name == bundle.default_template else ''} ({template.space_size()} configs)"
            for template_name, template in bundle.templates.items()
        )
        flags = []
        if bundle.spec.precision_sensitive:
            flags.append("precision-sensitive")
        if bundle.spec.fused_stages:
            flags.append("fused: " + " -> ".join(bundle.spec.fused_stages))
        print(f"{name:<18} default shape {'x'.join(map(str, default_shape)):<16} {description}" + (f" [{'; '.join(flags)}]" if flags else ""))
        print(f"{'':<18} templates: {templates or 'none'}")
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

    if args.autotune_budget < 0 or args.llm_rounds < 0 or args.repeats < 1 or args.llm_samples < 1 or args.top_k < 1:
        print("budgets must be >= 0; --repeats, --llm-samples and --top-k must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.evolve_fraction <= 1.0:
        print("--evolve-fraction must be within [0, 1]", file=sys.stderr)
        return 2
    if not 0.0 < args.ceiling_fraction <= 1.0:
        print("--ceiling-fraction must be within (0, 1]", file=sys.stderr)
        return 2
    workers = args.workers if args.workers and args.workers > 0 else min(4, os.cpu_count() or 1)

    try:
        workload = WorkloadProfile.load(args.workload) if args.workload else None
        bundle = build_operator(
            args.op, tuple(args.shape) if args.shape else None, dtype=args.dtype, workload=workload,
            output_dtype=args.output_dtype, accumulate_dtype=args.accumulate_dtype,
        )
    except (KeyError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    backend = get_backend(args.backend, threads=args.threads)

    llm = None
    if args.llm_rounds > 0:
        llm_config = LLMConfig.from_env(model=args.model)
        if llm_config is None:
            logger.warning("--llm-rounds given but no API key found; set KOPT_LLM_API_KEY (or OPENAI_API_KEY), KOPT_LLM_BASE_URL, KOPT_LLM_MODEL")
        else:
            llm = LLMGenerator(llm_config, workers=workers)
            logger.info("llm: %s @ %s", llm_config.model, llm_config.base_url)

    config = AgentConfig(
        autotune_budget=args.autotune_budget,
        evolve_fraction=args.evolve_fraction,
        llm_rounds=args.llm_rounds,
        llm_samples=args.llm_samples,
        llm_patience=args.llm_patience,
        workers=workers,
        top_k=args.top_k,
        warm_start=args.warm_start,
        template_name=args.template,
        roofline=not args.no_roofline,
        ceiling_fraction=args.ceiling_fraction,
        stop_at_ceiling=not args.no_stop_at_ceiling,
        allow_reduced_precision=args.allow_reduced_precision,
        fusion_report=not args.no_fusion_report,
        quick_repeats=args.quick_repeats,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
        run_timeout_seconds=args.run_timeout,
        output_dir=args.out,
    )
    try:
        agent = OptimizationAgent(bundle, backend, config, llm=llm)
        history = agent.run()
    except (RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print()
    print(history.leaderboard())
    if agent.verdict is not None:
        print(f"\nroofline verdict: {agent.verdict.describe()}")
    if agent.fusion_gain is not None and agent.fusion_gain.get("fusion_speedup"):
        print(f"fusion gain: {agent.fusion_gain['fusion_speedup']:.2f}x (fused {agent.fusion_gain['fused_best_ms']:.4f} ms vs separate {agent.fusion_gain['separate_total_ms']:.4f} ms)")
    if history.best is not None:
        best_candidate, best_result = history.best
        print(f"\nbest: trial {best_result.trial_id} ({best_candidate.short_label()}, {best_result.numeric_grade}) -> {history.output_dir / 'best.c'}")
        print(f"parity test: {history.output_dir / 'parity_test.py'}")
        if agent.bundle_dir is not None:
            print(f"integration bundle: {agent.bundle_dir} (kernel.c, kernel.h, manifest.json, build.sh, parity_test.py, README.md)")
    print(f"trial log: {history.trials_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list-ops":
        return _command_list_ops()
    if args.command == "show":
        return _command_show(args)
    if args.command == "workload":
        return _command_workload(args)
    if args.command == "export":
        return _command_export(args)
    return _command_run(args)


if __name__ == "__main__":
    sys.exit(main())
