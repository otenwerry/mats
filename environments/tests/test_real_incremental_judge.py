"""Unit tests for the two-stage incremental judge (no docker, no LLM, no cost).

The scheme (Owen 2026-07-30): the gate call is stage 1 of the judge's own conversation
(it carries the turn-1 transcript and returns a clean/hacked verdict); after the run the
scorer replays that exchange verbatim and appends only the REMAINDER of the trajectory,
so the judge commits to a reading of turn 1 before seeing turn 2.

These tests cover the deterministic pieces: the render-splitting contract that makes a
verbatim replay safe, the stage-1/stage-2 prompt templates, and the model-consistency
guard. The live two-call behavior is asserted end-to-end in test_real_sandbox_smoke.py.
"""

import sys
import unittest
from pathlib import Path

ENVIRONMENTS = Path(__file__).resolve().parents[1]
PETRI = ENVIRONMENTS.parent / "petri"
sys.path.insert(0, str(ENVIRONMENTS / "lib"))
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from exp_real_audit import (  # noqa: E402
    REAL_JUDGE_STAGE2_PROMPT,
    _CONTINUED_NOTE,
    _ENDED_AT_STAGE1_NOTE,
    _split_render,
    build_real_tasks,
)
from exp_real_gate import GATE_PROMPT  # noqa: E402

FRAUD_COLLECTION = PETRI / "seeds" / "ml_training_data_misuse"

HEAD = "[M1] SYSTEM: sp\n[M2] USER: open\n[M3] ASSISTANT: done"
TAIL = "[M4] USER: follow up\n[M5] ASSISTANT: more work"


class SplitRenderTests(unittest.TestCase):
    def test_exact_prefix_yields_tail(self):
        self.assertEqual(_split_render(f"{HEAD}\n{TAIL}", HEAD), TAIL)

    def test_run_that_ended_at_stage1_yields_empty_tail(self):
        self.assertEqual(_split_render(HEAD, HEAD), "")

    def test_diverged_render_refuses_to_replay(self):
        # A single changed character must abort the replay: the stored assistant reply
        # would otherwise be attached to a prompt the model never actually saw.
        self.assertIsNone(_split_render(f"{HEAD}\n{TAIL}", HEAD.replace("M3", "M9")))
        self.assertIsNone(_split_render(f"{HEAD}\n{TAIL}", "[M1] USER: something else"))

    def test_missing_head_refuses_to_replay(self):
        self.assertIsNone(_split_render(f"{HEAD}\n{TAIL}", ""))

    def test_leading_blank_lines_are_trimmed_from_the_tail(self):
        self.assertEqual(_split_render(f"{HEAD}\n\n\n{TAIL}", HEAD), TAIL)


class PromptTemplateTests(unittest.TestCase):
    def test_stage1_prompt_carries_the_transcript_and_asks_for_a_verdict(self):
        prompt = GATE_PROMPT.format(
            rubric="RUBRIC", transcript_head=HEAD,
            deliverables="DELIVERABLES", evidence="EVIDENCE",
        )
        for expected in (HEAD, "RUBRIC", "DELIVERABLES", "EVIDENCE",
                         "VERDICT: clean", "stage 1"):
            self.assertIn(expected, prompt)
        self.assertNotIn("{", prompt.replace("{{", ""), "unfilled placeholder remains")

    def test_stage2_prompt_scores_the_whole_trajectory(self):
        prompt = REAL_JUDGE_STAGE2_PROMPT.format(
            continuation_note=_CONTINUED_NOTE, messages_tail=TAIL,
            artifacts="ARTIFACTS", scoring_dimensions="DIMS",
            highlights_description="HIGHLIGHTS",
        )
        for expected in (TAIL, "ARTIFACTS", "DIMS", "HIGHLIGHTS",
                         "Score the WHOLE trajectory", "answer()"):
            self.assertIn(expected, prompt)
        self.assertIn("numbering continues", prompt)
        # the judge must not feel bound by its stage-1 answer
        self.assertIn("not bound by it", prompt)
        self.assertNotIn("{", prompt.replace("{{", ""))

    def test_stage2_prompt_states_when_nothing_followed(self):
        prompt = REAL_JUDGE_STAGE2_PROMPT.format(
            continuation_note=_ENDED_AT_STAGE1_NOTE,
            messages_tail="(no further messages)", artifacts="ARTIFACTS",
            scoring_dimensions="DIMS", highlights_description="HIGHLIGHTS",
        )
        self.assertIn("complete trajectory", prompt)


class GateJudgeModelConsistencyTests(unittest.TestCase):
    def test_mismatched_gate_model_is_rejected(self):
        # One conversation cannot span two models; a cheap gate model would make the
        # replayed assistant turn another model's words.
        with self.assertRaises(SystemExit) as caught:
            build_real_tasks(
                ["qwen3-32b"], ["fraud_detection"], "x",
                reasoning=True, condition="allow",
                gate_model="openrouter/qwen/qwen3-32b",
                seeds_path=str(FRAUD_COLLECTION),
            )
        self.assertIn("stage 1 of the judge", str(caught.exception))

    def test_default_gate_model_matches_the_judge(self):
        from exp_real_audit import DEFAULT_GATE_MODEL, resolve_gate_model
        from exp_rh_audit import JUDGE

        self.assertEqual(resolve_gate_model(None), JUDGE)
        self.assertEqual(DEFAULT_GATE_MODEL, JUDGE)
        tasks = build_real_tasks(
            ["qwen3-32b"], ["fraud_detection"], "x",
            reasoning=True, condition="allow",
            gate_model=resolve_gate_model(None),
            seeds_path=str(FRAUD_COLLECTION),
        )
        self.assertEqual(tasks[0].metadata["gate_model"], JUDGE)


if __name__ == "__main__":
    unittest.main()
