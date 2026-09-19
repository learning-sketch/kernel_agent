"""Export the champion kernel as a self-contained integration bundle.

results/<op>/bundle/
    kernel.c        the winning source, unchanged
    kernel.h        prototype, launch ABI and fast-path flag declaration for the consumer
    manifest.json   machine-readable contract: signature, tensors/dtypes (incl. accumulate),
                    launch ABI, chosen parameters, compile flags, numeric grade, performance
    build.sh        the exact compile command used to validate the kernel
    parity_test.py  independent reference-vs-kernel regression test
    README.md       how to integrate it into an operator library

This is the hand-off point to a downstream operator library: everything it needs to call the
kernel, build it, and re-check it lives in this directory.
"""

from __future__ import annotations

import datetime as _dt
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from kopt_agent.candidate import Candidate
from kopt_agent.evaluator import TrialResult
from kopt_agent.history import History, candidate_record
from kopt_agent.spec import OperatorSpec

BUNDLE_FORMAT = "kopt-bundle/1"
KERNEL_FILE = "kernel.c"
HEADER_FILE = "kernel.h"
MANIFEST_FILE = "manifest.json"
BUILD_FILE = "build.sh"
PARITY_FILE = "parity_test.py"
README_FILE = "README.md"


@dataclass
class BundleContext:
    """Run-level facts that end up in the manifest."""

    backend_name: str
    launch_abi: dict
    hardware: str
    baseline: TrialResult | None = None
    verdict: dict | None = None
    peaks: dict | None = None
    extra: dict = field(default_factory=dict)


def export_bundle(
    history: History,
    candidate: Candidate,
    result: TrialResult,
    context: BundleContext,
    directory: Path | None = None,
) -> Path:
    spec = history.spec
    if spec is None:
        raise ValueError("a spec is required to export a bundle")
    directory = Path(directory) if directory is not None else history.output_dir / "bundle"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / KERNEL_FILE).write_text(candidate.source, encoding="utf-8")
    compile_command = history.parity_compile_command(candidate)
    (directory / HEADER_FILE).write_text(_header(spec, candidate, context), encoding="utf-8")
    (directory / BUILD_FILE).write_text(_build_script(compile_command), encoding="utf-8")
    (directory / BUILD_FILE).chmod(0o755)
    history.write_parity_test(candidate, result, directory / PARITY_FILE, kernel_filename=KERNEL_FILE)
    manifest = build_manifest(spec, candidate, result, context, compile_command)
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / README_FILE).write_text(_readme(spec, candidate, result, context, manifest, compile_command), encoding="utf-8")
    return directory


def build_manifest(spec: OperatorSpec, candidate: Candidate, result: TrialResult, context: BundleContext, compile_command: list[str]) -> dict:
    primary = spec.primary_case()
    tensors = [
        {"name": tensor.name, "role": "input", "dtype": tensor.dtype.name, "c_type": tensor.dtype.c_type, "primary_shape": list(tensor.shape), "const": True}
        for tensor in primary.inputs
    ] + [
        {
            "name": primary.output.name, "role": "output", "dtype": primary.output.dtype.name, "c_type": primary.output.dtype.c_type,
            "primary_shape": list(primary.output.shape), "const": False,
        }
    ]
    baseline = context.baseline
    speedup = (baseline.objective_ms / result.objective_ms) if (baseline and baseline.objective_ms and result.objective_ms) else None
    return {
        "format": BUNDLE_FORMAT,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "operator": {
            "name": spec.name,
            "description": spec.description,
            "symbol": spec.symbol,
            "c_signature": spec.c_signature,
            "scalars": list(spec.scalar_names),
            "primary_shape": list(spec.primary_shape),
            "edge_shapes": [list(shape) for shape in spec.edge_shapes],
            "workload": [{"shape": list(entry.shape), "count": entry.count} for entry in spec.workload.entries] if spec.workload else None,
            "fused_stages": list(spec.fused_stages),
            "precision_sensitive": spec.precision_sensitive,
            "notes": list(spec.notes),
        },
        "precision": {
            "label": spec.precision_label(),
            "input_dtype": spec.dtype.name,
            "output_dtype": spec.out_dtype.name,
            "accumulate_dtype": spec.acc_dtype.name,
            "mixed": spec.mixed_precision,
            "tensor_dtypes": spec.tensor_dtypes(),
        },
        "tensors": tensors,
        "layout": "row-major, contiguous, 64-byte aligned; row stride equals the logical dimension",
        "launch_abi": dict(context.launch_abi),
        "backend": context.backend_name,
        "hardware": context.hardware,
        "candidate": {**candidate_record(candidate), "trial_id": result.trial_id, "source_file": KERNEL_FILE},
        "fast_path": {"predicate": candidate.fast_path_predicate, "flag_symbol": "kopt_fast_path_active" if candidate.fast_path_predicate else None, **result.fast_path},
        "numerics": {
            "grade": result.numeric_grade,
            "scaled_ulp_error": result.scaled_ulp_error,
            "bitwise_match_rate": result.bitwise_match_rate,
            "max_abs_error": result.max_abs_error,
            "atol": spec.policy.atol,
            "rtol": spec.policy.rtol,
            "tight_ulp": spec.policy.tight_ulp,
            "reference": "float64 on the host, rounded to the output dtype",
            "shape_coverage": result.shape_coverage,
        },
        "performance": {
            "timing_source": result.timing_source,
            "primary_latency_ms": result.latency_ms_median,
            "gflops": result.gflops,
            "gbps": result.gbps,
            "objective_ms": result.objective_ms,
            "baseline_objective_ms": baseline.objective_ms if baseline else None,
            "speedup_vs_baseline": speedup,
            "ab_speedup": result.ab_speedup,
            "per_shape": result.per_shape,
            "roofline": result.roofline,
            "verdict": context.verdict,
            "peaks": context.peaks,
            "transfer_ms": result.transfer_ms,
        },
        "build": {
            "compile_command": compile_command,
            "shell_script": BUILD_FILE,
            "extra_compile_flags": list(candidate.extra_compile_flags),
            "source": KERNEL_FILE,
            "header": HEADER_FILE,
        },
        "verification": {"parity_test": PARITY_FILE, "how": f"python {PARITY_FILE}  (or pytest {PARITY_FILE})"},
        "files": [KERNEL_FILE, HEADER_FILE, MANIFEST_FILE, BUILD_FILE, PARITY_FILE, README_FILE],
        **context.extra,
    }


def _header(spec: OperatorSpec, candidate: Candidate, context: BundleContext) -> str:
    guard = f"KOPT_{spec.name.upper()}_KERNEL_H"
    abi = context.launch_abi
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"/* {spec.name}: {spec.description}",
        f" * Precision: {spec.precision_label()}.",
        f" * Launch ABI ({context.backend_name}): pointers are {abi.get('pointer_space', 'host')} pointers; "
        + ("a trailing void* stream argument follows the scalars; " if abi.get("stream_argument") else "no stream argument; ")
        + ("the call is synchronous." if abi.get("synchronous_launch", True) else "the call may return before completion."),
        " * Tensors are row-major, contiguous, 64-byte aligned; row strides equal the logical dimensions.",
        " * The output is fully overwritten (no accumulation into prior contents).",
        " */",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f"{spec.c_signature};",
    ]
    if candidate.fast_path_predicate:
        lines += [
            "",
            f"/* Set to 1 by the kernel when its fast path ran; predicate: {candidate.fast_path_predicate} */",
            "extern int kopt_fast_path_active;",
        ]
    lines += ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */", ""]
    return "\n".join(lines)


def _build_script(compile_command: list[str]) -> str:
    command = " ".join(shlex.quote(token) if token not in ("{source}", "{artifact}") else token for token in compile_command)
    command = command.replace("{source}", '"$HERE/' + KERNEL_FILE + '"').replace("{artifact}", '"${OUT:-$HERE/libkernel.so}"')
    return (
        "#!/usr/bin/env sh\n"
        "# Builds the champion kernel into a shared library (override the output path with OUT=...).\n"
        "set -eu\n"
        'HERE="$(cd "$(dirname "$0")" && pwd)"\n'
        f"{command}\n"
        'echo "built ${OUT:-$HERE/libkernel.so}"\n'
    )


def _readme(spec: OperatorSpec, candidate: Candidate, result: TrialResult, context: BundleContext, manifest: dict, compile_command: list[str]) -> str:
    perf = manifest["performance"]
    speedup = perf["speedup_vs_baseline"]
    per_shape_rows = "\n".join(
        f"| {'x'.join(map(str, entry['shape']))} | {entry['weight']} | {entry['latency_ms']:.4f} | "
        f"{(entry.get('ab_speedup') or float('nan')):.2f}x | {entry.get('roofline', {}).get('bound', '-') if entry.get('roofline') else '-'} |"
        for entry in result.per_shape.values()
    )
    params = ", ".join(f"{key}={value}" for key, value in sorted(candidate.params.items())) or "(hand-written / LLM kernel, no template parameters)"
    fast_path = (
        f"Declared fast path: `{candidate.fast_path_predicate}` over the int scalars; the kernel exports `int kopt_fast_path_active` "
        "and sets it to 1 exactly when the fast path ran. The fallback path is verified on every other shape."
        if candidate.fast_path_predicate
        else "No shape-dependent fast path: one code path handles every shape."
    )
    tensor_lines = "\n".join(f"  - `{t['name']}` ({t['role']}): `{t['c_type']}` = {t['dtype']}" for t in manifest["tensors"])
    speedup_text = f", {speedup:.2f}x vs the naive baseline" if speedup else ""
    ab_text = f", interleaved A/B {result.ab_speedup:.2f}x" if result.ab_speedup else ""
    ulp_text = f"{result.scaled_ulp_error:.2f}" if result.scaled_ulp_error is not None else "n/a"
    match_text = f"{result.bitwise_match_rate * 100:.1f}%" if result.bitwise_match_rate is not None else "n/a"
    error_text = f"{result.max_abs_error:.3g}" if result.max_abs_error is not None else "n/a"
    primary_shape_text = "x".join(map(str, spec.primary_shape))
    sensitive_text = "This operator is precision-sensitive." if spec.precision_sensitive else ""
    scalars_text = ", ".join(spec.scalar_names)
    flags_text = " ".join(candidate.extra_compile_flags) or "(none)"
    command_text = " ".join(compile_command)
    return f"""# {spec.name} champion kernel ({spec.precision_label()})

{spec.description}

## Files

| file | purpose |
| --- | --- |
| `{KERNEL_FILE}` | the kernel source (trial {result.trial_id}, origin `{candidate.origin}`) |
| `{HEADER_FILE}` | prototype + launch ABI for the consumer |
| `{MANIFEST_FILE}` | machine-readable contract (signature, dtypes, ABI, parameters, numerics, performance) |
| `{BUILD_FILE}` | the exact compile command used during validation |
| `{PARITY_FILE}` | independent reference-vs-kernel regression test |

## Contract

```c
{spec.c_signature};
```

- Tensors (row-major, contiguous, 64-byte aligned):
{tensor_lines}
- Int scalars: {scalars_text}.
- Accumulation precision: {spec.acc_dtype.name} (`{spec.acc_dtype.c_type}`).
- Launch ABI ({context.backend_name}): {_abi_sentence(context.launch_abi)}
- The output is fully overwritten; the kernel never reads the output before writing it and never modifies inputs.
- {fast_path}

## Numerics

Grade **{result.numeric_grade}** against a float64 reference rounded to {spec.out_dtype.name}: scaled-ULP error {ulp_text}, bitwise match {match_text}, max |error| {error_text}; acceptance atol={spec.policy.atol}, rtol={spec.policy.rtol}.
{sensitive_text}

## Performance ({perf['timing_source']} timing on: {context.hardware})

Primary shape {primary_shape_text}: {result.latency_ms_median:.4f} ms, {result.gflops:.1f} GFLOP/s, {result.gbps:.1f} GB/s{speedup_text}{ab_text}.

| shape | calls | latency ms | A/B | bound |
| --- | --- | --- | --- | --- |
{per_shape_rows}

Template parameters: {params}.
Extra compile flags: {flags_text}.

## Build

```sh
./{BUILD_FILE}                # -> libkernel.so next to this file
# or, verbatim:
{command_text}
```

## Re-verify after integrating

```sh
python {PARITY_FILE}          # needs numpy, a C compiler and the kernel-opt-agent package (or KOPT_REPO=<checkout>)
```

The parity test recompiles `{KERNEL_FILE}` with the command above and checks benchmark, workload and edge shapes, special values (NaN / Inf / signed zero / denormals), const-input integrity and independence from the output buffer's prior contents.
"""


def _abi_sentence(abi: dict) -> str:
    pointer_space = abi.get("pointer_space", "host")
    parts = [f"the exported symbol is the {abi.get('entry_kind', 'kernel')} and receives {pointer_space} pointers"]
    parts.append("followed by a trailing `void* stream`" if abi.get("stream_argument") else "with no stream argument")
    parts.append("the call is synchronous" if abi.get("synchronous_launch", True) else "the call may return before the kernel completes")
    return "; ".join(parts) + "."


def load_bundle_inputs(operator_dir: Path) -> tuple[dict, str]:
    """Read summary.json + best.c of a finished run (for `kopt export`)."""
    summary_path = operator_dir / "summary.json"
    best_path = operator_dir / "best.c"
    if not summary_path.exists() or not best_path.exists():
        raise FileNotFoundError(f"no finished run in {operator_dir} (need summary.json and best.c)")
    return json.loads(summary_path.read_text(encoding="utf-8")), best_path.read_text(encoding="utf-8")
