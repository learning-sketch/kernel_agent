from kopt_agent.backends.base import Backend, CompileResult, RunResult
from kopt_agent.backends.cpu_c import CpuCBackend

BACKENDS: dict[str, type[Backend]] = {
    "cpu_c": CpuCBackend,
}


def get_backend(name: str, **kwargs) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend '{name}', available: {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)


__all__ = ["Backend", "CompileResult", "RunResult", "CpuCBackend", "BACKENDS", "get_backend"]
