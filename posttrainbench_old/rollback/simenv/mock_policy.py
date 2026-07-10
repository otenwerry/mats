"""Mock OpenAI-compatible policy server for offline sim-engine tests.

Serves /v1/chat/completions (streaming SSE, the shape @ai-sdk/openai-compatible
expects) and plays a fixed SCRIPT of assistant turns — tool calls, then text —
regardless of the incoming messages. Lets us drive a real `opencode run`
deterministically with NO model API key: opencode treats this as just another
OpenAI-compatible provider.

Free (no paid API calls). Used by test_interception.py; also handy as a
deterministic policy for ctrl-C/resume tests of any engine.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Each entry: ("tool", name, args_dict) or ("text", content)
DEFAULT_SCRIPT = [
    ("tool", "bash", {"command": "nvidia-smi", "description": "check GPU"}),
    ("tool", "bash", {"command": "echo real-command-ran", "description": "cheap local command"}),
    ("text", "Smoke test complete."),
]


class _Handler(BaseHTTPRequestHandler):
    script = DEFAULT_SCRIPT
    counter = [0]

    def log_message(self, *a):  # silence request logging
        pass

    def _sse(self, obj) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()

    def _chunk(self, delta, finish=None):
        return self._sse({
            "id": "mock", "object": "chat.completion.chunk", "created": 1,
            "model": "policy",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        if not self.path.endswith("/chat/completions"):
            self.send_response(404), self.end_headers()
            return

        # opencode also sends tool-less side requests (title generation,
        # summarization) to the same provider — answer those with a text stub
        # WITHOUT consuming the script.
        try:
            has_tools = bool(json.loads(body).get("tools"))
        except json.JSONDecodeError:
            has_tools = False
        if not has_tools:
            step = ("text", "mock")
        else:
            i = self.counter[0]
            step = self.script[min(i, len(self.script) - 1)]
            self.counter[0] += 1
        i = self.counter[0]

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        w = self.wfile.write

        w(self._chunk({"role": "assistant"}))
        if step[0] == "tool":
            _, name, args = step
            w(self._chunk({"tool_calls": [{
                "index": 0, "id": f"call_mock_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]}))
            w(self._chunk({}, finish="tool_calls"))
        else:
            w(self._chunk({"content": step[1]}))
            w(self._chunk({}, finish="stop"))
        w(self._sse({
            "id": "mock", "object": "chat.completion.chunk", "created": 1,
            "model": "policy", "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }))
        w(b"data: [DONE]\n\n")


def serve(port: int = 8377, script=None) -> HTTPServer:
    handler = type("H", (_Handler,), {"script": script or DEFAULT_SCRIPT,
                                      "counter": [0]})
    srv = HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    import time
    serve()
    print("mock policy on :8377")
    while True:
        time.sleep(60)
