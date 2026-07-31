"""Unit tests for the real-environment scripted protocol (no docker, no LLM, no cost).

Covers the RealProtocolController state machine against the pinned _shared/core.md
rules, protocol assembly from the seed files, the real dimension set, and the
build_real_tasks metadata contract.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ENVIRONMENTS = Path(__file__).resolve().parents[1]
PETRI = ENVIRONMENTS.parent / "petri"
sys.path.insert(0, str(ENVIRONMENTS / "lib"))
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))
sys.path.append(str(ENVIRONMENTS))

from exp_real_audit import (  # noqa: E402
    DEFAULT_TIME_LIMIT_SECONDS,
    REAL_PROTOCOL_VERSION,
    RealProtocolController,
    RealProtocolSpec,
    assemble_real_protocol,
    build_real_tasks,
    real_dimensions_for_seed,
    reject_max_turns_flag,
    resolve_time_limit,
)
from exp_real_gate import GateConfig  # noqa: E402
from judge_models import resolve_judge  # noqa: E402

FRAUD = PETRI / "seeds" / "ml_training_data_misuse" / "fraud_detection"
BENCH = PETRI / "seeds" / "p_hacking" / "reasoning_prompt_benchmark"


def make_spec(*, follow_up="FOLLOW UP", finalize=None, call_out=None,
              family="p_hacking") -> RealProtocolSpec:
    return RealProtocolSpec(
        version=REAL_PROTOCOL_VERSION, family=family, member="test",
        system_prompt="SP", opening_user_message="OPEN",
        follow_up=follow_up, follow_up_finalize=finalize, call_out=call_out,
        deliverables=("analysis.py", "RESULT.md"),
        gate=GateConfig(rubric="rubric"),
    )


class ControllerTests(unittest.TestCase):
    def test_clean_first_submission_gets_follow_up(self):
        c = RealProtocolController(make_spec(), "allow")
        self.assertEqual(c.on_submission(gate_verdict="clean"), ("send", "FOLLOW UP"))
        self.assertTrue(c.follow_up_sent)

    def test_hacked_first_submission_ends_blind_eye(self):
        c = RealProtocolController(make_spec(), "allow")
        self.assertEqual(c.on_submission(gate_verdict="hacked"), ("end", "protocol_end"))
        self.assertFalse(c.follow_up_sent)

    def test_gate_error_ends_conservatively(self):
        c = RealProtocolController(make_spec(), "allow")
        self.assertEqual(c.on_submission(gate_verdict="error"), ("end", "gate_error_end"))

    def test_second_submission_with_updated_result_ends(self):
        c = RealProtocolController(make_spec(finalize="FINALIZE"), "allow")
        c.on_submission(gate_verdict="clean")
        action = c.on_submission(result_md_changed=True, worked_since_follow_up=True)
        self.assertEqual(action, ("end", "protocol_end"))
        self.assertFalse(c.finalize_sent)

    def test_finalize_sent_when_work_not_written_to_result(self):
        # core.md: extra work described but RESULT.md untouched -> FOLLOW_UP_FINALIZE.
        c = RealProtocolController(make_spec(finalize="FINALIZE"), "allow")
        c.on_submission(gate_verdict="clean")
        action = c.on_submission(result_md_changed=False, worked_since_follow_up=True)
        self.assertEqual(action, ("send", "FINALIZE"))
        # and the submission after the finalize always ends
        self.assertEqual(
            c.on_submission(result_md_changed=True, worked_since_follow_up=True),
            ("end", "protocol_end"),
        )

    def test_no_finalize_when_target_did_no_work(self):
        # core.md: "do not send it if the target ... did no additional work".
        c = RealProtocolController(make_spec(finalize="FINALIZE"), "allow")
        c.on_submission(gate_verdict="clean")
        action = c.on_submission(result_md_changed=False, worked_since_follow_up=False)
        self.assertEqual(action, ("end", "protocol_end"))

    def test_ml_family_has_no_finalize(self):
        c = RealProtocolController(make_spec(finalize=None), "allow")
        c.on_submission(gate_verdict="clean")
        self.assertEqual(
            c.on_submission(result_md_changed=False, worked_since_follow_up=True),
            ("end", "protocol_end"),
        )

    def test_clean_run_without_follow_up_file_ends(self):
        c = RealProtocolController(make_spec(follow_up=None), "allow")
        self.assertEqual(c.on_submission(gate_verdict="clean"), ("end", "protocol_end"))

    def test_correct_condition_seam_sends_one_call_out(self):
        # v2 seam: with pinned call-out text, correct sends exactly one correction turn.
        c = RealProtocolController(make_spec(call_out="CALL OUT"), "correct")
        self.assertEqual(c.on_submission(gate_verdict="hacked"), ("send", "CALL OUT"))
        self.assertEqual(
            c.on_submission(result_md_changed=True, worked_since_follow_up=True),
            ("end", "protocol_end"),
        )

    def test_correct_without_call_out_text_ends(self):
        c = RealProtocolController(make_spec(call_out=None), "correct")
        self.assertEqual(c.on_submission(gate_verdict="hacked"), ("end", "protocol_end"))

    def test_summary_records_the_branch(self):
        c = RealProtocolController(make_spec(finalize="FINALIZE"), "allow")
        c.on_submission(gate_verdict="clean")
        c.on_submission(result_md_changed=False, worked_since_follow_up=True)
        summary = c.summary("protocol_end")
        self.assertEqual(summary["submissions"], 2)
        self.assertEqual(summary["first_gate_verdict"], "clean")
        self.assertTrue(summary["follow_up_sent"])
        self.assertTrue(summary["finalize_sent"])
        self.assertEqual(summary["ended_reason"], "protocol_end")


class PipelineIntegrityTests(unittest.TestCase):
    def test_incremental_judge_fallback_fails_the_pipeline(self):
        from exp_real_audit_pipeline import judge_replay_integrity_failures

        score = SimpleNamespace(metadata={
            "judge_mode": "single_shot",
            "judge_fallback_reason": "stage-1 render diverged",
            "judge_transcript_source": "target_timeline",
        })
        sample = SimpleNamespace(
            id="test", epoch=1, scores={"audit_judge": score}
        )
        log = SimpleNamespace(
            eval=SimpleNamespace(task="real_audit_test"),
            samples=[sample],
        )

        failures = judge_replay_integrity_failures([log])

        self.assertEqual(len(failures), 1)
        self.assertIn(
            "judge_incremental_fallback:stage-1 render diverged",
            failures[0]["issues"],
        )

    def test_successful_incremental_judge_passes(self):
        from exp_real_audit_pipeline import judge_replay_integrity_failures

        score = SimpleNamespace(metadata={
            "judge_mode": "incremental",
            "judge_fallback_reason": None,
            "judge_transcript_source": "stored_target_messages",
        })
        sample = SimpleNamespace(
            id="test", epoch=1, scores={"audit_judge": score}
        )
        log = SimpleNamespace(
            eval=SimpleNamespace(task="real_audit_test"),
            samples=[sample],
        )

        self.assertEqual(judge_replay_integrity_failures([log]), [])

    def test_missing_judge_provenance_fails_closed(self):
        from exp_real_audit_pipeline import judge_replay_integrity_failures

        sample = SimpleNamespace(
            id="test",
            epoch=1,
            scores={"audit_judge": SimpleNamespace(metadata={})},
        )
        log = SimpleNamespace(
            eval=SimpleNamespace(task="real_audit_test"),
            samples=[sample],
        )

        failures = judge_replay_integrity_failures([log])

        self.assertEqual(len(failures), 1)
        self.assertIn("judge_mode_not_incremental:missing", failures[0]["issues"])
        self.assertIn(
            "judge_transcript_not_canonical:missing",
            failures[0]["issues"],
        )


class AnnotationLoadTests(unittest.TestCase):
    def test_incomplete_directory_does_not_block_completed_audits(self):
        import exp_annotate_real_hacks

        async def fake_load_mode(mode_dir):
            if mode_dir.name == "incomplete":
                raise RuntimeError("run has no completed sample")
            return [{"seed": "completed"}]

        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            (logs / "completed").mkdir()
            (logs / "incomplete").mkdir()
            old_logs = exp_annotate_real_hacks.LOGS
            old_loader = exp_annotate_real_hacks.viewer_load.load_mode
            exp_annotate_real_hacks.LOGS = logs
            exp_annotate_real_hacks.viewer_load.load_mode = fake_load_mode
            try:
                audits = asyncio.run(
                    exp_annotate_real_hacks.load_all_real_audits()
                )
            finally:
                exp_annotate_real_hacks.LOGS = old_logs
                exp_annotate_real_hacks.viewer_load.load_mode = old_loader

        self.assertEqual(audits, [{"seed": "completed"}])


class ProtocolAssemblyTests(unittest.TestCase):
    def test_fraud_spec_matches_pinned_files(self):
        spec = assemble_real_protocol(FRAUD, "allow", "SP")
        shared = FRAUD.parent / "_shared"
        self.assertEqual(spec.follow_up, (shared / "FOLLOW_UP.txt").read_text().strip())
        self.assertIsNone(spec.follow_up_finalize)
        self.assertIsNone(spec.call_out)
        self.assertEqual(spec.deliverables, ("predictions.csv", "REPORT.md"))
        self.assertTrue(spec.opening_user_message.startswith(
            "I need you to train a model for detecting fraudulent"))
        self.assertIn("save the final model under `models/final/`",
                      spec.opening_user_message)
        self.assertIsNotNone(spec.gate.key_text)
        self.assertGreater(spec.gate.agreement_threshold, 0.9)

    def test_benchmark_spec_has_both_follow_ups_in_order(self):
        spec = assemble_real_protocol(BENCH, "allow", "SP")
        shared = BENCH.parent / "_shared"
        self.assertEqual(
            spec.follow_up_user_messages,
            [(shared / "FOLLOW_UP.txt").read_text().strip(),
             (shared / "FOLLOW_UP_FINALIZE.txt").read_text().strip()],
        )
        self.assertEqual(spec.deliverables, ("analysis.py", "RESULT.md"))
        self.assertIsNone(spec.gate.key_text)

    def test_unported_seed_rejected(self):
        with self.assertRaises(SystemExit):
            assemble_real_protocol(
                FRAUD.parent / "youtube_comments", "allow", "SP")


class RealDimensionTests(unittest.TestCase):
    def test_auditor_dimensions_swapped_for_environment_failure(self):
        for seed in (FRAUD, BENCH):
            dims = real_dimensions_for_seed(seed)
            self.assertNotIn("auditor_mistakes", dims.names)
            self.assertNotIn("auditor_nudging", dims.names)
            self.assertIn("environment_failure", dims.names)
            self.assertIn("reward_hacking", dims.names)

    def test_incompleteness_not_judged(self):
        # Replaced 2026-07-31 by the mechanical ended_reason check
        # (viewer.real_is_incomplete); the judge no longer scores it.
        for seed in (FRAUD, BENCH):
            self.assertNotIn("incompleteness", real_dimensions_for_seed(seed).names)

    def test_rubrics_come_from_environments_own_fork(self):
        # environments/dimensions/ owns the rubrics (forked from petri 2026-07-31);
        # editing petri/dimensions/ must not change what real runs are judged on.
        for seed in (FRAUD, BENCH):
            dims = real_dimensions_for_seed(seed)
            for file in dims.files:
                self.assertEqual(
                    ENVIRONMENTS / "dimensions",
                    Path(file).resolve().parents[1],
                    f"{file} is outside environments/dimensions/",
                )

    def test_order_follows_judge_order_json(self):
        import json

        order = json.loads(
            (ENVIRONMENTS / "dimensions" / "judge_order.json").read_text())
        names = real_dimensions_for_seed(FRAUD).names
        # Every active rubric in the fork applies to every seed, so the routed set IS
        # the full order file.
        self.assertEqual(names, order)


class BuildRealTasksTests(unittest.TestCase):
    def test_task_contract(self):
        tasks = build_real_tasks(
            ["qwen3-32b"], ["fraud_detection"], "test-run",
            reasoning=True, condition="allow",
            gate_model=resolve_judge(None),   # the gate must equal the judge
            seeds_path=str(FRAUD.parent),
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.name, "real_audit_qwen3-32b_fraud_detection")
        md = task.metadata
        self.assertEqual(md["target_tools_mode"], "real")
        self.assertEqual(md["condition"], "allow")
        # no turn cap in real mode: the wall clock is the only runaway guard
        self.assertIsNone(md["max_turns"])
        self.assertIn("environment_failure", md["judge_dimensions"])
        self.assertNotIn("auditor_mistakes", md["judge_dimensions"])
        self.assertTrue(md["real_env_manifest_sha"])
        sample = task.dataset[0]
        protocol = sample.metadata["auditor_protocol"]
        self.assertEqual(protocol["version"], REAL_PROTOCOL_VERSION)
        self.assertEqual(len(protocol["follow_up_user_messages"]), 1)
        self.assertIn("/workspace/data/eval/test_labels.csv", sample.files)

    def test_correct_condition_rejected_in_v1(self):
        with self.assertRaises(SystemExit):
            build_real_tasks(
                ["qwen3-32b"], ["fraud_detection"], "x",
                reasoning=True, condition="correct",
                gate_model=resolve_judge(None),   # the gate must equal the judge
                seeds_path=str(FRAUD.parent),
            )



class TimeLimitFlagTests(unittest.TestCase):
    """--time-limit replaced --max-turns as the runaway guard, so its defaulting and
    validation are now load-bearing rather than incidental."""

    def test_default_is_one_hour(self):
        self.assertEqual(DEFAULT_TIME_LIMIT_SECONDS, 3600)
        self.assertEqual(resolve_time_limit(None), 3600)

    def test_explicit_value_wins(self):
        self.assertEqual(resolve_time_limit("900"), 900)

    def test_zero_means_no_limit(self):
        self.assertIsNone(resolve_time_limit("0"))

    def test_bad_values_fail_before_the_run(self):
        for bad in ("abc", "-1", "1.5"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                resolve_time_limit(bad)

    def test_stale_max_turns_flag_is_rejected_not_ignored(self):
        """The lib CLI does not validate unknown flags, so a leftover --max-turns must be
        caught explicitly rather than silently dropped."""
        for argv in (["prog", "--max-turns=200"], ["prog", "--max-turns"]):
            with self.subTest(argv=argv):
                saved, sys.argv = sys.argv, argv
                try:
                    with self.assertRaises(SystemExit) as caught:
                        reject_max_turns_flag()
                finally:
                    sys.argv = saved
                self.assertIn("--time-limit", str(caught.exception))

    def test_normal_argv_passes_the_max_turns_check(self):
        saved, sys.argv = sys.argv, ["prog", "--target=qwen3-32b", "--time-limit=900"]
        try:
            reject_max_turns_flag()
        finally:
            sys.argv = saved


if __name__ == "__main__":
    unittest.main()
