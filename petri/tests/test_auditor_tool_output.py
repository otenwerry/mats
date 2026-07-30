import sys
import unittest
from pathlib import Path

from inspect_ai._util.text import truncate_string_to_bytes


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))

from exp_rh_audit import (  # noqa: E402
    AUDITOR_MAX_TOOL_OUTPUT,
    auditor_generate_config,
)


class AuditorToolOutputTests(unittest.TestCase):
    def test_shared_auditor_config_disables_tool_output_truncation(self):
        config = auditor_generate_config(None)

        self.assertEqual(AUDITOR_MAX_TOOL_OUTPUT, 0)
        self.assertEqual(config.max_tool_output, 0)
        self.assertIsNone(
            truncate_string_to_bytes("x" * 200_000, config.max_tool_output)
        )

    def test_reasoning_effort_does_not_restore_default_tool_cap(self):
        config = auditor_generate_config("high")

        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(config.max_tool_output, 0)


if __name__ == "__main__":
    unittest.main()
