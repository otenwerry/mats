import sys
import unittest
from pathlib import Path


PETRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI / "lib"))

from viewer_load import check_auditor_protocol  # noqa: E402


PROTOCOL = {
    "version": "exact-system-opening-followup-v1",
    "system_prompt": "fixed system",
    "opening_user_message": "open the workspace",
    "follow_up_user_message": "please keep working",
}


def transcript(*messages: tuple[int, str, str]) -> str:
    return "\n\n".join(
        f"[M{turn}] {role.upper()}:\n{content}"
        for turn, role, content in messages
    )


class AuditorProtocolTests(unittest.TestCase):
    def test_non_opted_in_sample_is_not_applicable(self):
        result = check_auditor_protocol("", None)

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["issues"], [])

    def test_exact_opening_without_optional_follow_up_passes(self):
        result = check_auditor_protocol(
            transcript(
                (1, "system", "fixed system"),
                (2, "user", "open the workspace"),
                (3, "assistant", "done"),
            ),
            PROTOCOL,
        )

        self.assertEqual(result["status"], "passed")

    def test_exact_optional_follow_up_passes(self):
        result = check_auditor_protocol(
            transcript(
                (1, "system", "fixed system"),
                (2, "user", "open the workspace"),
                (3, "assistant", "first submission"),
                (4, "user", "please keep working"),
                (5, "assistant", "second submission"),
            ),
            PROTOCOL,
        )

        self.assertEqual(result["status"], "passed")

    def test_wrong_text_reports_exact_turns(self):
        result = check_auditor_protocol(
            transcript(
                (1, "system", "changed system"),
                (2, "user", "changed opening"),
                (3, "assistant", "first submission"),
                (4, "user", "different pressure"),
            ),
            PROTOCOL,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [(issue["turn"], issue["part"]) for issue in result["issues"]],
            [(1, "system"), (2, "opening"), (4, "follow-up")],
        )

    def test_second_follow_up_is_an_extra_user_turn(self):
        result = check_auditor_protocol(
            transcript(
                (1, "system", "fixed system"),
                (2, "user", "open the workspace"),
                (3, "assistant", "first submission"),
                (4, "user", "please keep working"),
                (5, "assistant", "second submission"),
                (6, "user", "please keep working"),
            ),
            PROTOCOL,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [(issue["turn"], issue["part"], issue["problem"])
             for issue in result["issues"]],
            [(6, "user", "extra user message")],
        )

    def test_missing_transcript_is_recorded_as_unavailable_not_deviation(self):
        result = check_auditor_protocol("", PROTOCOL)

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["issues"][0]["problem"], "transcript unavailable")


if __name__ == "__main__":
    unittest.main()
