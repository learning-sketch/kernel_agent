"""CPU backend: gcc -> shared object -> executed by an isolated runner subprocess.

The runner process is the safety boundary: segfaults, infinite loops and stack overflows in
generated code become a status on the trial instead of killing the agent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from kopt_agent.backends.base import Backend, CompileResult, ProfileReport, RunResult
from kopt_agent.candidate import Candidate
from kopt_agent.dtypes import DType
from kopt_agent.hardware import describe_cpu
from kopt_agent.spec import OperatorSpec, TestCase

DEFAULT_FLAGS = ("-O3", "-march=native", "-fopenmp", "-shared", "-fPIC")
# gcc's vectorizer diagnostics: which loops were vectorized and, more usefully, why others were not.
OPT_REPORT_FLAGS = ("-fopt-info-vec-missed", "-fopt-info-vec-optimized")
OPT_REPORT_PATTERN = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+): (?P<kind>missed|optimized): (?P<text>.+)$", re.MULTILINE)
MAX_OPT_REPORT_LINES = 40
# Libraries go after the sources so --as-needed keeps them (libmvec backs vectorized expf/logf under -ffast-math).
BASE_LINK_LIBS = ("-lm",)
OPTIONAL_LINK_LIBS = ("-lmvec",)


class CpuCBackend(Backend):
    name = "cpu_c"

    def __init__(
        self,
        compiler: str | None = None,
        work_dir: Path | None = None,
        compile_timeout_seconds: float = 60.0,
        threads: int | None = None,
    ) -> None:
        self.compiler = compiler or os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
        if self.compiler is None:
            raise RuntimeError("no C compiler found (set CC or install gcc)")
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="kopt_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.compile_timeout_seconds = compile_timeout_seconds
        self.threads = threads or os.cpu_count() or 1
        self._artifact_cache: dict[str, CompileResult] = {}
        self._cache_lock = threading.Lock()
        self.link_libs = BASE_LINK_LIBS + tuple(lib for lib in OPTIONAL_LINK_LIBS if self._library_links(lib))
        self._dtype_support: dict[str, bool] = {}

    def supports_dtype(self, dtype: DType) -> bool:
        """Probe once whether the compiler accepts the element type with float conversions
        (gcc >= 12 for _Float16, gcc >= 13 for __bf16 on x86)."""
        if dtype.name == "fp32":
            return True
        cached = self._dtype_support.get(dtype.name)
        if cached is not None:
            return cached
        probe_source = self.work_dir / f"probe_{dtype.name}.c"
        probe_source.write_text(
            f"float kopt_probe_load(const {dtype.c_type}* p) {{ return (float)p[0] + 1.0f; }}\n"
            f"void kopt_probe_store({dtype.c_type}* p, float v) {{ p[0] = ({dtype.c_type})v; }}\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [self.compiler, *DEFAULT_FLAGS, str(probe_source), "-o", str(probe_source.with_suffix(".so"))],
                capture_output=True, text=True, timeout=self.compile_timeout_seconds, check=False,
            )
            supported = completed.returncode == 0
        except subprocess.TimeoutExpired:
            supported = False
        self._dtype_support[dtype.name] = supported
        return supported

    def _library_links(self, library_flag: str) -> bool:
        probe_source = self.work_dir / "probe.c"
        probe_source.write_text("int kopt_probe(void) { return 0; }\n", encoding="utf-8")
        probe_artifact = self.work_dir / f"probe{library_flag.replace('-', '_')}.so"
        try:
            completed = subprocess.run(
                [self.compiler, "-shared", "-fPIC", str(probe_source), library_flag, "-o", str(probe_artifact)],
                capture_output=True, text=True, timeout=self.compile_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def portable_compile_command(self) -> list[str]:
        """Command template (with {source} / {artifact} placeholders) for the emitted parity test."""
        return ["gcc", *DEFAULT_FLAGS, "{source}", *self.link_libs, "-o", "{artifact}"]

    def compile(self, candidate: Candidate, spec: OperatorSpec) -> CompileResult:
        with self._cache_lock:
            cached = self._artifact_cache.get(candidate.fingerprint)
        if cached is not None:
            return cached

        source_path = self.work_dir / f"{spec.name}_{candidate.fingerprint}.c"
        artifact_path = source_path.with_suffix(".so")
        source_path.write_text(candidate.source, encoding="utf-8")

        command = [
            self.compiler, *DEFAULT_FLAGS, *OPT_REPORT_FLAGS, *candidate.extra_compile_flags,
            str(source_path), *self.link_libs, "-o", str(artifact_path),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.compile_timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired:
            result = CompileResult(False, None, f"compiler timed out after {self.compile_timeout_seconds:.0f}s", 0.0)
            with self._cache_lock:
                self._artifact_cache[candidate.fingerprint] = result
            return result

        elapsed = time.perf_counter() - started
        raw_log = (completed.stderr or "") + (completed.stdout or "")
        report, log = _split_optimization_report(raw_log, candidate.source)
        if completed.returncode != 0 or not artifact_path.exists():
            result = CompileResult(False, None, log.strip() or f"compiler exited with {completed.returncode}", elapsed)
        else:
            result = CompileResult(True, artifact_path, log.strip(), elapsed, optimization_report=report, profile=_profile_from_report(report))
        with self._cache_lock:
            self._artifact_cache[candidate.fingerprint] = result
        return result

    def run(
        self,
        artifact: Path,
        spec: OperatorSpec,
        case: TestCase,
        inputs: Sequence[np.ndarray],
        warmup: int,
        repeats: int,
        timeout_seconds: float,
        verify: bool = True,
        reference_artifact: Path | None = None,
    ) -> RunResult:
        job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=self.work_dir))
        inputs_path = job_dir / "inputs.npz"
        output_path = job_dir / "output.npy"
        poison_output_path = job_dir / "output_poison.npy"
        np.savez(inputs_path, **{tensor.name: array for tensor, array in zip(case.inputs, inputs)})

        job = {
            "artifact": str(artifact),
            "symbol": spec.symbol,
            "inputs_path": str(inputs_path),
            "input_names": [tensor.name for tensor in case.inputs],
            "output_shape": list(case.output.shape),
            "output_dtype": np.dtype(case.output.dtype.storage).str,
            "scalars": list(case.scalars),
            "warmup": warmup,
            "repeats": repeats,
            "verify": verify,
            "output_path": str(output_path),
            "poison_output_path": str(poison_output_path),
            "ab": {"artifact": str(reference_artifact), "symbol": spec.symbol} if reference_artifact else None,
        }
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(self.threads))
        env.setdefault("OMP_PROC_BIND", "true")

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "kopt_agent.runner"],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(job_dir, ignore_errors=True)
            return RunResult(
                False,
                error=f"kernel did not finish within {timeout_seconds:.0f}s (hang or pathological slowness)",
                error_kind="timeout",
            )

        if completed.returncode != 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            return RunResult(False, error=_describe_crash(completed.returncode, completed.stderr))

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            shutil.rmtree(job_dir, ignore_errors=True)
            return RunResult(False, error=f"runner produced no result: {completed.stderr.strip()[-500:]}")

        if not payload.get("ok"):
            shutil.rmtree(job_dir, ignore_errors=True)
            return RunResult(False, error=payload.get("error", "unknown runner error"), error_kind=payload.get("error_kind", "runtime"))

        output = np.load(output_path) if verify and output_path.exists() else None
        poison_output = np.load(poison_output_path) if poison_output_path.exists() else None
        shutil.rmtree(job_dir, ignore_errors=True)
        return RunResult(
            True,
            output=output,
            timings_ms=payload.get("timings_ms", []),
            host_timings_ms=payload.get("host_timings_ms", []),
            reference_timings_ms=payload.get("reference_timings_ms", []),
            cpu_wall_ratio=payload.get("cpu_wall_ratio"),
            threads_available=payload.get("threads_available"),
            fast_path_active=payload.get("fast_path_active"),
            poison_left_in_output=bool(payload.get("poison_left_in_output", False)),
            prefill_dependent=bool(payload.get("prefill_dependent", False)),
            poison_output=poison_output,
        )

    def hardware_summary(self) -> str:
        return describe_cpu(self.threads)

    def language_guidance(self) -> str:
        return (
            "Target: x86-64 CPU, C11 compiled with gcc "
            + " ".join(DEFAULT_FLAGS[:3])
            + ". You may use OpenMP (#include <omp.h>), <immintrin.h> intrinsics that match the CPU flags "
            "listed in the hardware summary, <math.h>, <string.h>, <stdint.h>, <stdlib.h>. "
            "Do NOT define main(). Do not read files or environment variables. Every input pointer is "
            "row-major, contiguous, 64-byte aligned. Row strides equal the logical dimensions. "
            "16-bit element types are spelled _Float16 (fp16) and __bf16 (bf16); convert to float for arithmetic. "
            "If your kernel has a fast path that only handles some inputs (alignment, multiples of a tile, "
            "size thresholds), you MUST (a) keep a correct fallback for every other input, (b) declare the "
            "activation condition as a comment `// fast_path: <expression over the int scalars>` and (c) "
            "export `int kopt_fast_path_active;` at file scope and set it to 1 when the fast path ran and 0 "
            "otherwise. The harness checks both that the fast path really activates on matching inputs and "
            "that the fallback is correct on the others."
        )


def _split_optimization_report(raw_log: str, source: str) -> tuple[list[str], str]:
    """Separate gcc's vectorizer notes from real diagnostics and attach the offending source line."""
    source_lines = source.splitlines()
    report: list[str] = []
    seen: set[str] = set()
    for match in OPT_REPORT_PATTERN.finditer(raw_log):
        line_number = int(match.group("line"))
        snippet = source_lines[line_number - 1].strip() if 0 < line_number <= len(source_lines) else ""
        entry = f"line {line_number} [{match.group('kind')}] {match.group('text').strip()}" + (f"  // {snippet[:80]}" if snippet else "")
        if entry not in seen:
            seen.add(entry)
            report.append(entry)
    remaining = OPT_REPORT_PATTERN.sub("", raw_log)
    remaining = "\n".join(line for line in remaining.splitlines() if line.strip())
    if len(report) > MAX_OPT_REPORT_LINES:
        report = report[:MAX_OPT_REPORT_LINES] + [f"... {len(report) - MAX_OPT_REPORT_LINES} more vectorizer notes omitted"]
    return report, remaining


VECTOR_WIDTH_PATTERN = re.compile(r"using (\d+) byte vectors")


def _profile_from_report(report: list[str]) -> ProfileReport:
    """Turn gcc's vectorizer notes into the backend-neutral schema."""
    profile = ProfileReport()
    widest = 0
    for entry in report:
        if "[optimized]" in entry:
            match = VECTOR_WIDTH_PATTERN.search(entry)
            if match:
                widest = max(widest, int(match.group(1)) * 8)
            profile.vectorized_loops.append(entry.replace(" [optimized]", ":"))
        elif "[missed]" in entry and "not vectorized" in entry:
            profile.missed_loops.append(entry.replace(" [missed]", ":"))
    profile.vector_width_bits = widest or None
    return profile


def _describe_crash(returncode: int, stderr: str) -> str:
    if returncode < 0:
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = f"signal {-returncode}"
        hint = {
            "SIGSEGV": "out-of-bounds memory access (check tile remainders and index arithmetic)",
            "SIGBUS": "misaligned or invalid memory access",
            "SIGFPE": "integer division by zero",
            "SIGABRT": "runtime abort (assertion / heap corruption)",
            "SIGKILL": "killed, likely out of memory",
        }.get(signal_name, "")
        return f"kernel process crashed with {signal_name}" + (f": {hint}" if hint else "")
    tail = stderr.strip()[-800:]
    return f"runner exited with code {returncode}: {tail}"
