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

from kopt_agent.backends.base import Backend, CompileResult, RunResult
from kopt_agent.candidate import Candidate
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
            result = CompileResult(True, artifact_path, log.strip(), elapsed, optimization_report=report)
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
    ) -> RunResult:
        job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=self.work_dir))
        inputs_path = job_dir / "inputs.npz"
        output_path = job_dir / "output.npy"
        np.savez(inputs_path, **{tensor.name: array for tensor, array in zip(case.inputs, inputs)})

        job = {
            "artifact": str(artifact),
            "symbol": spec.symbol,
            "inputs_path": str(inputs_path),
            "input_names": [tensor.name for tensor in case.inputs],
            "output_shape": list(case.output.shape),
            "output_dtype": np.dtype(case.output.dtype).str,
            "scalars": list(case.scalars),
            "warmup": warmup,
            "repeats": repeats,
            "output_path": str(output_path),
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

        output = np.load(output_path)
        shutil.rmtree(job_dir, ignore_errors=True)
        return RunResult(True, output=output, timings_ms=payload.get("timings_ms", []))

    def hardware_summary(self) -> str:
        return describe_cpu(self.threads)

    def language_guidance(self) -> str:
        return (
            "Target: x86-64 CPU, C11 compiled with gcc "
            + " ".join(DEFAULT_FLAGS[:3])
            + ". You may use OpenMP (#include <omp.h>), <immintrin.h> intrinsics that match the CPU flags "
            "listed in the hardware summary, <math.h>, <string.h>, <stdint.h>, <stdlib.h>. "
            "Do NOT define main(). Do not read files or environment variables. Every input pointer is "
            "row-major, contiguous, 64-byte aligned. Row strides equal the logical dimensions."
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
