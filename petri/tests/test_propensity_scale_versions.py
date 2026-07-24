import argparse
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PETRI = Path(__file__).resolve().parents[1]
SHARED = PETRI.parent / "shared"
sys.path.insert(0, str(PETRI))
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(SHARED))

import propensity  # noqa: E402
import viewer  # noqa: E402
import exp_rh_audit  # noqa: E402
from exp_ask_env import add_cli_args  # noqa: E402


class PropensityScaleVersionTests(unittest.TestCase):
    def test_registry_target_scales_are_uniform_1_to_10(self):
        questions = propensity.load_questions(statuses=None)
        scales = [q["answer_format"] for q in questions if q.get("answer_format")]

        self.assertTrue(scales)
        self.assertTrue(all(fmt["min"] == 1 and fmt["max"] == 10 for fmt in scales))
        self.assertEqual(
            propensity.question_set_metadata()["variant"],
            "target-ratings-1to10-v9",
        )

    def test_legacy_answer_reparses_with_its_saved_1_to_100_scale(self):
        summary = {}
        asks = [{
            "question_id": "agent_metric",
            "question": "Rate this. Answer with only a number from 1 to 100.",
            "answer": "75",
            "closed_form": {"value": 75.0, "parse": "exact"},
        }]

        stats = viewer._reparse_propensity_asks(summary, asks)

        self.assertEqual(asks[0]["closed_form"]["value"], 75.0)
        self.assertEqual(stats["changed"], 0)
        self.assertEqual(summary["n_closed_form_parsed"], 1)

    def test_scale_and_reasoning_conditions_have_distinct_viewer_groups(self):
        old = {
            "summary": {},
            "asks": [{
                "question_id": "agent_metric",
                "question": "Answer with only a number from 1 to 100.",
            }],
        }
        new = {
            "summary": {
                "question_set_metadata": propensity.question_set_metadata(),
                "reasoning": False,
            },
            "asks": [{
                "question_id": "agent_metric",
                "question": "Answer with only a number from 1 to 10.",
                "answer_format": {"type": "scale", "min": 1, "max": 10},
            }],
        }

        old_key, old_label = viewer.propensity_block_version(old)
        new_key, new_label = viewer.propensity_block_version(new)

        self.assertNotEqual(old_key, new_key)
        self.assertIn("1–100", old_label)
        self.assertIn("1–10", new_label)
        self.assertIn("reasoning off", new_label)

    def test_petri_reasoning_flag_is_required(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        self.assertEqual(parser.parse_args(["--reasoning=no"]).reasoning, "no")

    def test_reasoning_off_is_explicit_for_every_provider_route(self):
        sentinel = object()
        with patch.object(exp_rh_audit, "get_model", return_value=sentinel) as get_model:
            model, enabled, _strict = exp_rh_audit.build_target_model(
                "openai/test-model", reasoning_on=False, effort="medium")
            self.assertIs(model, sentinel)
            self.assertIsNone(enabled)
            self.assertEqual(get_model.call_args.kwargs["config"].reasoning_effort, "none")

        with patch.object(exp_rh_audit, "get_model", return_value=sentinel) as get_model:
            model, enabled, _strict = exp_rh_audit.build_target_model(
                "anthropic/test-model", reasoning_on=False, effort="medium")
            self.assertIs(model, sentinel)
            self.assertIsNone(enabled)
            self.assertEqual(get_model.call_args.kwargs["config"].reasoning_effort, "none")

        with patch.object(exp_rh_audit, "get_model", return_value=sentinel) as get_model:
            _model, enabled, strict = exp_rh_audit.build_target_model(
                "openrouter/test/model", reasoning_on=False, effort="medium")
            self.assertFalse(enabled)
            self.assertFalse(strict)
            self.assertFalse(get_model.call_args.kwargs["reasoning_enabled"])


if __name__ == "__main__":
    unittest.main()
