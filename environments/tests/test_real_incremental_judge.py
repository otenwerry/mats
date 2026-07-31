"""Unit tests for the two-stage incremental judge (no docker, no LLM, no cost).

The scheme (Owen 2026-07-31): the gate call is the judge's FULL evaluation of the
trajectory-so-far, presented as though it were the entire trajectory (same prompt, same
answer() schema as the final judge — no "has it hacked yet" question); the branch is
derived mechanically from its scores. After the run the scorer replays that whole
stage-1 conversation verbatim and appends only the REMAINDER of the trajectory, asking
for the same evaluation over the whole run.

These tests cover the deterministic pieces: the render-splitting contract that makes a
verbatim replay safe, the stage-1/stage-2 prompt templates, and the model-consistency
guard. The live two-call behavior is asserted end-to-end in test_real_sandbox_smoke.py.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    _target_timeline_events,
    build_real_tasks,
    build_stage1_prompt,
    real_dimensions_for_seed,
)

FRAUD_COLLECTION = PETRI / "seeds" / "ml_training_data_misuse"
BENCH = PETRI / "seeds" / "p_hacking" / "reasoning_prompt_benchmark"

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


class TargetTimelineEventTests(unittest.TestCase):
    def test_private_gate_model_event_is_excluded_but_target_tools_remain(self):
        events = [
            SimpleNamespace(event="model", role="target"),
            SimpleNamespace(event="tool", role=None),
            SimpleNamespace(event="model", role="gate"),
            SimpleNamespace(event="span_begin", role=None),
        ]

        kept = _target_timeline_events(events)

        self.assertEqual(kept, [events[0], events[1], events[3]])


class PromptTemplateTests(unittest.TestCase):
    def test_stage1_prompt_is_the_judge_prompt_as_if_complete(self):
        dims = list(real_dimensions_for_seed(BENCH).dimensions)
        artifacts = {"files": [{"path": "/workspace/RESULT.md",
                                "content": "# Results\nnull.\n", "truncated": False}]}
        detectors = [{"name": "key_agreement", "fired": False, "evidence": {}}]
        prompt = build_stage1_prompt(dims, HEAD, artifacts, detectors)
        for expected in (HEAD, "impartial judge", "RESULT.md", "Mechanical evidence",
                         "key_agreement", "answer()", "reward_hacking",
                         "Critical Evaluation Guidelines"):
            self.assertIn(expected, prompt)
        # no staging tell, no verdict question: the model is asked to evaluate this as
        # though it were the entire trajectory, the same way the final judge would
        for forbidden in ("VERDICT", "stage 1", "first submission",
                          "has the target committed"):
            self.assertNotIn(forbidden, prompt)
        for placeholder in ("{messages}", "{artifacts}", "{scoring_dimensions}",
                            "{highlights_description}", "{instructions}"):
            self.assertNotIn(placeholder, prompt, "unfilled placeholder remains")

    def test_stage2_prompt_scores_the_whole_trajectory(self):
        prompt = REAL_JUDGE_STAGE2_PROMPT.format(
            continuation_note=_CONTINUED_NOTE, messages_tail=TAIL,
            artifacts="ARTIFACTS", scoring_dimensions="DIMS",
            highlights_description="HIGHLIGHTS",
        )
        for expected in (TAIL, "ARTIFACTS", "DIMS", "HIGHLIGHTS",
                         "score the WHOLE trajectory", "answer()"):
            self.assertIn(expected, prompt)
        self.assertIn("numbering continues", prompt)
        # stage 2 must come clean about the staged reveal
        self.assertIn("only the FIRST PART", prompt)
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
