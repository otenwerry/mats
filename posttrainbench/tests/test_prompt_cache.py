import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


MATS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MATS / "shared"))
sys.path.insert(0, str(MATS / "posttrainbench"))
sys.path.append(str(MATS / "petri" / "lib"))

import prompt_cache
from ask_common import AskUnit
from lib import exp_ask_env


def _usage(*, creation, read, aggregate_read=None, ttl="1h"):
    first = {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
        "cache_creation": {
            "ephemeral_5m_input_tokens": creation if ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": creation if ttl == "1h" else 0,
        },
    }
    return {
        **first,
        "cache_read_input_tokens": read if aggregate_read is None else aggregate_read,
        "iterations": [first],
    }


class PromptCacheAccountingTest(unittest.TestCase):
    def test_verification_uses_first_iteration_not_tool_loop_aggregate(self):
        normalized = prompt_cache.normalize_usage(
            _usage(creation=10_000, read=100, aggregate_read=50_000),
            first_iteration=True,
        )
        self.assertEqual(normalized["cache_read_input_tokens"], 100)
        self.assertEqual(normalized["source"], "first_iteration")

    def test_reuse_check_accepts_large_read_and_rejects_system_only_read(self):
        reference = prompt_cache.normalize_usage(
            _usage(creation=90_000, read=17_000), first_iteration=True)
        hit = prompt_cache.normalize_usage(
            _usage(creation=100, read=106_900), first_iteration=True)
        miss = prompt_cache.normalize_usage(
            _usage(creation=89_000, read=17_000), first_iteration=True)
        self.assertTrue(prompt_cache.assess_reuse(reference, hit)["verified"])
        self.assertFalse(prompt_cache.assess_reuse(reference, miss)["verified"])

    def test_summary_surfaces_missing_usage_and_failed_checks(self):
        records = [
            {"prompt_cache": {
                "mode": "required",
                "usage": prompt_cache.normalize_usage(_usage(creation=10, read=20)),
                "reuse_check": {"verified": True},
            }},
            {"prompt_cache": {
                "mode": "required",
                "usage": prompt_cache.normalize_usage(_usage(creation=30, read=40)),
                "reuse_check": {"verified": False},
            }},
            {"answer": "no cache record"},
        ]
        summary = prompt_cache.summarize(records)
        self.assertEqual(summary["n_records"], 2)
        self.assertEqual(summary["cache_creation_input_tokens"], 40)
        self.assertEqual(summary["cache_read_input_tokens"], 60)
        self.assertEqual(summary["n_reuse_verified"], 1)
        self.assertEqual(summary["n_reuse_failed"], 1)
        self.assertFalse(summary["all_reuse_checks_passed"])


class PTBPrefixCachePolicyTest(unittest.TestCase):
    def test_one_hour_cache_requires_cli_2_1_108(self):
        self.assertFalse(exp_ask_env._cli_supports_generic_1h_cache("2.1.107"))
        self.assertTrue(exp_ask_env._cli_supports_generic_1h_cache("2.1.108"))
        self.assertTrue(exp_ask_env._cli_supports_generic_1h_cache(
            "2.2.0 (Claude Code)"))
        self.assertFalse(exp_ask_env._cli_supports_generic_1h_cache("unknown"))

    def test_required_cache_refuses_opencode_before_launch(self):
        unit = SimpleNamespace(
            label="opencode-run",
            is_baseline=False,
            env={"traj": SimpleNamespace(scaffold="opencode")},
        )
        args = SimpleNamespace(prefix_cache="required", questions="propensity")
        with self.assertRaisesRegex(
            SystemExit, "OpenCode prompt-cache verification is also not implemented"
        ):
            exp_ask_env.prepare([unit], args)

    def _unit(self, root: Path) -> AskUnit:
        workspace = root / exp_ask_env._CACHE_WORKSPACE_NAME
        workspace.mkdir()
        return AskUnit(
            label="test trajectory",
            dirname="test",
            ask_who="test",
            md_who="test",
            md_info="",
            fidelity_base={"flags": []},
            env={
                "prefix_cache_mode": "required",
                "prefix_cache_workspace": workspace,
                "prefix_cache_reference": None,
                "prefix_cache_ask_ordinal": 0,
            },
            out_dir=root,
        )

    def test_stable_workspace_is_snapshotted_and_reset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unit = self._unit(root)
            ask_dir = root / "q__s0"
            cwd, snapshot = exp_ask_env._cache_workspace_for_ask(unit, ask_dir)
            (cwd / "created_by_agent.txt").write_text("preserved")
            exp_ask_env._snapshot_cache_workspace(cwd, snapshot)
            self.assertEqual((ask_dir / "cwd" / "created_by_agent.txt").read_text(),
                             "preserved")
            self.assertTrue(cwd.is_dir())
            self.assertEqual(list(cwd.iterdir()), [])

    def test_cache_tracker_warms_then_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            unit = self._unit(Path(td))
            warm_fidelity = {"flags": []}
            warm, abort = exp_ask_env._record_prompt_cache(
                unit, _usage(creation=90_000, read=17_000), warm_fidelity)
            self.assertIsNone(abort)
            self.assertEqual(warm["phase"], "warmup")
            self.assertIn("prompt_cache_warmup",
                          {x["code"] for x in warm_fidelity["flags"]})

            hit_fidelity = {"flags": []}
            hit, abort = exp_ask_env._record_prompt_cache(
                unit, _usage(creation=100, read=106_900), hit_fidelity)
            self.assertIsNone(abort)
            self.assertTrue(hit["reuse_check"]["verified"])
            self.assertIn("prompt_cache_reuse_verified",
                          {x["code"] for x in hit_fidelity["flags"]})

    def test_cache_tracker_requests_abort_on_expensive_prefix_miss(self):
        with tempfile.TemporaryDirectory() as td:
            unit = self._unit(Path(td))
            exp_ask_env._record_prompt_cache(
                unit, _usage(creation=90_000, read=17_000), {"flags": []})
            fidelity = {"flags": []}
            miss, abort = exp_ask_env._record_prompt_cache(
                unit, _usage(creation=89_000, read=17_000), fidelity)
            self.assertFalse(miss["reuse_check"]["verified"])
            self.assertIsNotNone(abort)
            self.assertIn("prompt_cache_campaign_abort",
                          {x["code"] for x in fidelity["flags"]})

    def test_cache_tracker_refuses_unexpected_five_minute_warmup(self):
        with tempfile.TemporaryDirectory() as td:
            unit = self._unit(Path(td))
            warm, abort = exp_ask_env._record_prompt_cache(
                unit, _usage(creation=90_000, read=17_000, ttl="5m"),
                {"flags": []})
            self.assertEqual(warm["phase"], "verification_failed")
            self.assertIn("one-hour", abort)

    def test_auto_mode_changes_only_propensity(self):
        propensity_args = SimpleNamespace(prefix_cache="auto", questions="propensity")
        em_args = SimpleNamespace(prefix_cache="auto", questions="em")
        self.assertEqual(exp_ask_env._prefix_cache_mode(propensity_args), "required")
        self.assertEqual(exp_ask_env._prefix_cache_mode(em_args), "off")

    def test_two_real_adapter_calls_keep_sessions_fresh_but_cwd_stable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unit = self._unit(root)
            unit.env.update({
                "bundle": object(),
                "orig_version": "2.1.9",
                "config_root": root / "config",
                "cli_cmd": ["claude"],
                "model": "claude-opus-4-6",
                "effort_args": [],
                "env_overrides": {},
            })
            unit.env["config_root"].mkdir()
            usages = [
                _usage(creation=90_000, read=17_000),
                _usage(creation=100, read=106_900),
            ]
            process_cwds = []
            resumed_sessions = []

            def fake_run(cmd, **kwargs):
                process_cwds.append(str(kwargs["cwd"]))
                resumed_sessions.append(cmd[cmd.index("--resume") + 1])
                usage = usages[len(process_cwds) - 1]
                event = {
                    "type": "result",
                    "result": "42",
                    "total_cost_usd": 0.1,
                    "usage": usage,
                    "modelUsage": {"claude-opus-4-6": usage},
                }
                return SimpleNamespace(stdout=json.dumps(event) + "\n",
                                       stderr="", returncode=0)

            emitted_cwds = []

            def fake_emit(_bundle, *, cwd, version):
                emitted_cwds.append(cwd)
                sid = f"fresh-session-{len(emitted_cwds)}"
                return sid, [{"type": "user", "message": {"content": "prefix"}}]

            with (mock.patch.object(exp_ask_env.recon_claude, "emit_session",
                                    side_effect=fake_emit),
                  mock.patch.object(exp_ask_env.recon_claude, "project_dir_name",
                                    return_value="stable-project"),
                  mock.patch.object(exp_ask_env.subprocess, "run",
                                    side_effect=fake_run)):
                first = exp_ask_env._ask_claude(
                    unit, root / "q__s0", {"flags": []}, "question", 60)
                second = exp_ask_env._ask_claude(
                    unit, root / "q__s1", {"flags": []}, "question", 60)

            self.assertEqual(len(set(emitted_cwds)), 1)
            self.assertEqual(len(set(process_cwds)), 1)
            self.assertEqual(emitted_cwds, process_cwds)
            self.assertEqual(resumed_sessions,
                             ["fresh-session-1", "fresh-session-2"])
            self.assertTrue((root / "q__s0" / "cwd").is_dir())
            self.assertTrue((root / "q__s1" / "cwd").is_dir())
            self.assertEqual(first["prompt_cache"]["phase"], "warmup")
            self.assertTrue(second["prompt_cache"]["reuse_check"]["verified"])
            self.assertNotIn("_abort_experiment", second)

    def test_required_cache_aborts_on_nonzero_cli_exit_even_with_cache_usage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unit = self._unit(root)
            unit.env.update({
                "bundle": object(),
                "orig_version": "2.1.9",
                "config_root": root / "config",
                "cli_cmd": ["claude"],
                "model": "claude-opus-4-6",
                "effort_args": [],
                "env_overrides": {},
            })
            unit.env["config_root"].mkdir()
            event = {
                "type": "result", "result": "42", "total_cost_usd": 0.1,
                "usage": _usage(creation=90_000, read=17_000),
                "modelUsage": {},
            }
            failed = SimpleNamespace(stdout=json.dumps(event) + "\n",
                                     stderr="failed", returncode=1)
            with (mock.patch.object(exp_ask_env.recon_claude, "emit_session",
                                    return_value=("fresh-session", [])),
                  mock.patch.object(exp_ask_env.recon_claude, "project_dir_name",
                                    return_value="stable-project"),
                  mock.patch.object(exp_ask_env.subprocess, "run",
                                    return_value=failed)):
                record = exp_ask_env._ask_claude(
                    unit, root / "q__s0", {"flags": []}, "question", 60)
            self.assertIn("_abort_experiment", record)
            self.assertIn("exit 1", record["_abort_experiment"])


if __name__ == "__main__":
    unittest.main()
