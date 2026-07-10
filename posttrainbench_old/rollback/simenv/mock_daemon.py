"""Stub environment-simulator daemon: same HTTP contract as the real one
(exp_simenv_daemon.py) but with canned outputs and a regex router — used by
the offline interception test. Free (no API calls)."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

SIMULATE_RE = re.compile(
    r"nvidia-smi|train|evaluate\.py|torchrun|accelerate|\bsleep\b|timer\.sh", re.I)

CANNED_NVIDIA_SMI = (
    "+-----------------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 (SIMULATED)  |\n"
    "|   0  NVIDIA H100 80GB HBM3   On   | 00000000:01:00.0 Off |   0 MiB / 81559MiB |\n"
    "+-----------------------------------------------------------------------------+\n")


class _Handler(BaseHTTPRequestHandler):
    pending = {}

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/route":
            if SIMULATE_RE.search(req.get("command", "")):
                self.pending[req["callID"]] = req
                self._json({"mode": "simulate", "run_instead": "true"})
            else:
                self._json({"mode": "real"})
        elif self.path == "/simulate":
            req0 = self.pending.pop(req["callID"], {})
            self._json({"output": CANNED_NVIDIA_SMI,
                        "title": req0.get("description", "simulated"),
                        "exit": 0})
        else:
            self.send_response(404), self.end_headers()


def serve(port: int = 8378) -> HTTPServer:
    handler = type("H", (_Handler,), {"pending": {}})
    srv = HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    import time
    serve()
    print("mock daemon on :8378")
    while True:
        time.sleep(60)
