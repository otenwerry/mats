import sys
import unittest
from unittest.mock import patch

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
        ]
        with patch.object(sys, "argv", argv):
            config = exp_audit_pipeline._parse_args()

        self.assertEqual(config["condition"], "allow")

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


if __name__ == "__main__":
    unittest.main()
