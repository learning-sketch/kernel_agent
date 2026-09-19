from kopt_agent.backends.base import Backend, CompileResult, LaunchABI, ProfileReport, RunResult
from kopt_agent.backends.cpu_c import CpuCBackend
from kopt_agent.backends.protocol import KernelSession, LaunchJob, ProtocolError, run_protocol

BACKENDS: dict[str, type[Backend]] = {
    "cpu_c": CpuCBackend,
    # Accelerator backends: copy kopt_agent/backends/accelerator_stub.py, fill in the session
    # primitives, and register the class here.
}


def get_backend(name: str, **kwargs) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend '{name}', available: {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)


__all__ = [
    "Backend", "CompileResult", "RunResult", "LaunchABI", "ProfileReport", "KernelSession", "LaunchJob",
    "ProtocolError", "run_protocol", "CpuCBackend", "BACKENDS", "get_backend",
]
