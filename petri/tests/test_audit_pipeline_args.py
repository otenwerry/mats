import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT))

import exp_audit_pipeline


class AuditPipelineArgTests(unittest.TestCase):
    def test_expected_p_hacking_command_parses(self):
        argv = [
            "exp_audit_pipeline.py",
            "--targets=gpt-5.5,kimi-k2.6,glm-5.1,deepseek-v4-pro",
            "--seed-dir=p_hacking",
            "--seeds=all",
            "--epochs=7",
            "--condition=allow",
            "--max-turns=70",
        ]
        with patch.object(sys, "argv", argv):
            config = exp_audit_pipeline._parse_args()

        self.assertEqual(config["max_turns"], 70)
        self.assertEqual(config["epochs"], 7)
        self.assertEqual(
            config["seeds"],
            ["checkout_redesign", "reasoning_prompt_benchmark", "retrieval_practice"],
        )

    def test_pinned_collection_defaults_to_allow(self):
        argv = [
            "exp_audit_pipeline.py",
            "--targets=gpt-5.5",
            "--seed-dir=p_hacking",
            "--seeds=all",
            "--epochs=1",
            "--max-turns=200",
        ]
        with patch.object(sys, "argv", argv):
            config = exp_audit_pipeline._parse_args()

        self.assertEqual(config["condition"], "allow")

    def test_max_turns_is_required(self):
        argv = [
            "exp_audit_pipeline.py",
            "--targets=gpt-5.5",
            "--seed-dir=p_hacking",
            "--seeds=all",
            "--epochs=1",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "--max-turns is required"):
                exp_audit_pipeline._parse_args()

    def test_unknown_value_flag_fails(self):
        with patch.object(sys, "argv", ["exp_audit_pipeline.py", "--turns=70"]):
            with self.assertRaisesRegex(SystemExit, "unknown argument.*--turns=70"):
                exp_audit_pipeline._parse_args()

    def test_known_value_flag_requires_equals_value(self):
        with patch.object(sys, "argv", ["exp_audit_pipeline.py", "--max-turns"]):
            with self.assertRaisesRegex(SystemExit, "--max-turns requires a value"):
                exp_audit_pipeline._parse_args()

    def test_switch_rejects_value(self):
        with patch.object(sys, "argv", ["exp_audit_pipeline.py", "--skip-viewer=yes"]):
            with self.assertRaisesRegex(SystemExit, "--skip-viewer is a switch"):
                exp_audit_pipeline._parse_args()

    def test_integrity_report_catches_missing_judge_and_empty_target_response(self):
        usage = SimpleNamespace(
            input_tokens=5,
            output_tokens=2,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            total_cost=None,
        )
        target_message = SimpleNamespace(content="", tool_calls=[], refusal=None)
        target_event = SimpleNamespace(
            event="model",
            role="target",
            model="provider/target",
            input="request",
            tools=[],
            config={},
            output=SimpleNamespace(
                usage=usage,
                choices=[SimpleNamespace(stop_reason="unknown", message=target_message)],
            ),
            error=None,
        )
        sample = SimpleNamespace(
            id="seed",
            epoch=1,
            scores={},
            role_usage={"target": usage},
            model_usage={"provider/target": usage},
            events=[target_event],
        )
        log = SimpleNamespace(
            eval=SimpleNamespace(
                task="audit_target_seed",
                model_roles={"target": SimpleNamespace(model="provider/target")},
                metadata={"judge_dimensions": ["reward_hacking"]},
            ),
            samples=[sample],
        )

        failures = exp_audit_pipeline.audit_integrity_failures([log])

        self.assertEqual(len(failures), 1)
        self.assertIn("judge_score_missing", failures[0]["issues"])
        self.assertIn(
            "judge_dimensions_missing:reward_hacking",
            failures[0]["issues"],
        )
        self.assertIn(
            "target_provider_empty_response:attempt1",
            failures[0]["issues"],
        )

    def test_visible_target_response_without_usage_is_not_a_provider_failure(self):
        zero_usage = SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            total_cost=None,
        )
        target_event = SimpleNamespace(
            event="model",
            role="target",
            model="provider/target",
            input="request",
            tools=[],
            config={},
            output=SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    stop_reason="stop",
                    message=SimpleNamespace(
                        content="Complete response",
                        tool_calls=[],
                        refusal=None,
                    ),
                )],
            ),
            error=None,
        )
        score = SimpleNamespace(value={"reward_hacking": 1})
        sample = SimpleNamespace(
            id="seed",
            epoch=1,
            scores={"audit_judge": score},
            role_usage={"target": zero_usage},
            model_usage={"provider/target": zero_usage},
            events=[target_event],
        )
        log = SimpleNamespace(
            eval=SimpleNamespace(
                task="audit_target_seed",
                model_roles={"target": SimpleNamespace(model="provider/target")},
                metadata={"judge_dimensions": ["reward_hacking"]},
            ),
            samples=[sample],
        )

        self.assertEqual(exp_audit_pipeline.audit_integrity_failures([log]), [])

    def test_visible_response_repairs_token_only_dead_target_guard(self):
        zero_usage = SimpleNamespace(output_tokens=0)
        target_event = SimpleNamespace(
            event="model",
            role="target",
            model="provider/target",
            input="request",
            tools=[],
            config={},
            output=SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    stop_reason="stop",
                    message=SimpleNamespace(
                        content="Complete response",
                        tool_calls=[],
                        refusal=None,
                    ),
                )],
            ),
            error=None,
        )
        log = SimpleNamespace(
            stats=SimpleNamespace(
                model_usage={"provider/target": zero_usage},
            ),
            eval=SimpleNamespace(
                model_roles={"target": SimpleNamespace(model="provider/target")},
            ),
            samples=[SimpleNamespace(events=[target_event])],
        )

        self.assertEqual(
            exp_audit_pipeline.unrecovered_dead_targets(
                [log],
                ["provider/target"],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
