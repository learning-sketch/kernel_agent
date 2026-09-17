"""Best-effort description of the target hardware, fed to the LLM and printed in reports."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess

INTERESTING_FLAGS = ("avx512f", "avx512bw", "avx2", "fma", "avx", "sse4_2", "neon", "sve")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _cpu_flags() -> list[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("flags", "Features")):
                    flags = set(line.split(":", 1)[1].split())
                    return [flag for flag in INTERESTING_FLAGS if flag in flags]
    except OSError:
        pass
    return []


def _caches() -> str:
    if shutil.which("lscpu") is None:
        return "unknown"
    try:
        text = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (subprocess.TimeoutExpired, OSError):
        return "unknown"
    entries = re.findall(r"^(L\d[di]? cache):\s+(.+)$", text, flags=re.MULTILINE)
    return ", ".join(f"{name}={value.strip()}" for name, value in entries) or "unknown"


def describe_cpu(threads: int | None = None) -> str:
    thread_count = threads or os.cpu_count() or 1
    flags = _cpu_flags()
    return (
        f"CPU: {_cpu_model()}; usable threads: {thread_count}; "
        f"SIMD: {', '.join(flags) if flags else 'unknown'}; caches: {_caches()}"
    )
