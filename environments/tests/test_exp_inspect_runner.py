"""Free tests for the shared Inspect runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_inspect_runner as runner


class InspectRunnerTests(unittest.TestCase):
    def test_run_eval_returns_complete_saved_logs(self):
        summary_log = SimpleNamespace(samples=[], location="saved.eval")
        saved_log = SimpleNamespace(
            samples=[SimpleNamespace(id="completed-sample")],
            stats=SimpleNamespace(model_usage={}),
        )
        install = SimpleNamespace(to_dict=lambda: {"installed": True})

        with tempfile.TemporaryDirectory() as temporary, patch(
            "inspect_ai.eval_set", return_value=(True, [summary_log])
        ), patch(
            "inspect_ai.log.list_eval_logs", return_value=["saved.eval"]
        ), patch(
            "inspect_ai.log.read_eval_log", return_value=saved_log
        ), patch.object(
            runner, "install_cost_tracking", return_value=install
        ), patch.object(
            runner, "install_inspect_warmup", return_value=True
        ), patch.object(
            runner, "cost_tracking_provenance", return_value={}
        ), patch.object(
            runner,
            "write_cache_report",
            return_value={"verified_models": [], "unverified_models": []},
        ):
            success, logs = runner.run_eval(
                tasks=[object()],
                epochs=1,
                concurrency=1,
                log_dir=temporary,
            )
            manifest = json.loads(
                (Path(temporary) / runner.RUNTIME_MANIFEST_FILENAME).read_text()
            )

        self.assertTrue(success)
        self.assertEqual(logs, [saved_log])
        self.assertEqual(logs[0].samples[0].id, "completed-sample")
        self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
