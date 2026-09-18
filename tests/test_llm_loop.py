"""The LLM phase against a fake OpenAI-compatible endpoint: prompts carry the verdict and dead
ends, replies with `// fast_path:` are checked for the activation contract, dead ends are
recorded for the next round (requires gcc)."""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kopt_agent.agent import AgentConfig, OptimizationAgent
from kopt_agent.backends import get_backend
from kopt_agent.generators.llm import LLMConfig, LLMGenerator
from ops import build_operator

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")


class FakeLLM(BaseHTTPRequestHandler):
    prompts: list[str] = []
    replies: list[str] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        FakeLLM.prompts.append(payload["messages"][-1]["content"])
        reply = FakeLLM.replies[min(len(FakeLLM.prompts) - 1, len(FakeLLM.replies) - 1)]
        body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        return


@pytest.fixture
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), FakeLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeLLM.prompts.clear()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_llm_round_trip_with_fast_path_contract_and_dead_ends(fake_server, tmp_path):
    bundle = build_operator("matmul", (32, 32, 16))
    signature = bundle.spec.c_signature
    body = (
        "for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float acc = 0; "
        "for (int k = 0; k < K; k++) acc += A[(size_t)i*K+k] * B[(size_t)k*N+j]; C[(size_t)i*N+j] = acc; }"
    )
    # Round 1: claims a fast path but never sets the flag -> incorrect, becomes a dead end.
    lying = f"```c\n// strategy: aligned fast path\n// fast_path: N % 16 == 0\n#include <stddef.h>\nint kopt_fast_path_active = 0;\n{signature} {{ {body} }}\n```"
    # Round 2: honest fast path.
    honest = (
        f"```c\n// strategy: honest aligned fast path\n// fast_path: N % 16 == 0\n#include <stddef.h>\nint kopt_fast_path_active = 0;\n"
        f"{signature} {{ kopt_fast_path_active = (N % 16 == 0); {body} }}\n```"
    )
    FakeLLM.replies = [lying, honest]

    config = AgentConfig(
        autotune_budget=0, llm_rounds=2, llm_samples=1, llm_patience=5, repeats=2, warmup=1, quick_repeats=1,
        roofline=False, output_dir=tmp_path, stop_at_ceiling=False,
    )
    agent = OptimizationAgent(bundle, get_backend("cpu_c"), config, llm=LLMGenerator(LLMConfig("m", fake_server, "k")))
    history = agent.run()

    statuses = [result.status.value for _, result in history.records]
    assert statuses[0] == "ok"  # baseline
    assert statuses[1] == "incorrect" and "stayed 0" in history.records[1][1].message
    assert statuses[2] == "ok" and history.records[2][1].fast_path["activated_on"]
    assert len(FakeLLM.prompts) == 2
    # The second prompt carries the first round's dead end and the fast-path contract rules.
    assert "Known dead ends" in FakeLLM.prompts[1] and "aligned fast path" in FakeLLM.prompts[1]
    assert "kopt_fast_path_active" in FakeLLM.prompts[1]
    summary = json.loads((tmp_path / "matmul" / "summary.json").read_text())
    assert summary["dead_ends"] and "stayed 0" in summary["dead_ends"][0]
