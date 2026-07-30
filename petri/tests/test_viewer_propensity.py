import sys
import unittest
from pathlib import Path


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

from viewer import page_name, propensity_data  # noqa: E402
from viewer_visuals import (  # noqa: E402
    fig_propensity_model_prompt_grid,
    fig_propensity_rate_bars,
    fig_propensity_scenario_panels,
    propensity_section,
)


class ViewerPropensityTests(unittest.TestCase):
    @staticmethod
    def audit(model: str, seed: str, reward_hacking: int, **overrides) -> dict:
        scores = {
            "reward_hacking": reward_hacking,
            "auditor_nudging": 1,
            "auditor_mistakes": 1,
            "situational_awareness": 1,
            "degenerate_behavior": 1,
            "hallucination": 1,
        }
        scores.update(overrides.pop("scores", {}))
        return {
            "target": model,
            "seed": seed,
            "mode": "test-mode",
            "task": seed,
            "epoch": 0,
            "transcript": (
                "[M1] user:\nstart\n[M2] assistant:\nfirst answer\n"
                "[M3] user:\nfollow up\n[M4] assistant:\nsecond answer"
            ),
            "scores": scores,
            "integrity_issues": [],
            **overrides,
        }

    @staticmethod
    def annotations_for(audits: list[dict], *, elicited=(), unknown=()) -> dict:
        annotations = {}
        for idx, audit in enumerate(audits):
            audit["epoch"] = idx
            if idx in unknown:
                continue
            annotations[page_name(
                audit["mode"], audit["task"], audit["seed"], audit["epoch"]
            )] = {"hack_turns": [{"m": 4 if idx in elicited else 2}]}
        return annotations

    def test_current_rates_match_non_invalid_top_outcomes(self):
        audits = [
            self.audit("model-a", "family_seed-a", 6),
            self.audit(
                "model-a",
                "family_seed-b",
                6,
                scores={"auditor_mistakes": 8},
            ),
            self.audit("model-b", "family_seed-a", 1),
            self.audit("model-b", "family_seed-b", 6),
            self.audit(
                "model-b",
                "family_seed-b",
                6,
                integrity_issues=["target_provider_empty_response"],
                judge_score_status="usable",
                judge_missing_dimensions=[],
            ),
        ]

        data = propensity_data(
            audits,
            match_v7_outcomes=True,
            scenario_label="ML scenarios",
            annotations=self.annotations_for(audits),
        )

        self.assertEqual(data["n"], 3)
        self.assertEqual(data["n_invalid_excluded"], 2)
        self.assertNotIn("criterion_filtered", data)
        self.assertEqual(data["models"], ["model-a", "model-b"])
        self.assertEqual(data["prompts"], ["seed-a", "seed-b"])
        self.assertEqual(
            [(row["group"], row["k"], row["n"]) for row in data["by_model"]],
            [("model-a", 1, 1), ("model-b", 1, 2)],
        )
        filtered_cell = data["grid"][1]["cells"][0]
        self.assertEqual((filtered_cell["k"], filtered_cell["n"]), (0, 0))

    def test_reversed_hack_stays_in_denominator_but_not_numerator(self):
        audits = [
            self.audit("model-a", "family_seed-a", 6),
            self.audit(
                "model-a", "family_seed-a", 6,
                rh_category="abandoned_unprompted",
            ),
        ]

        data = propensity_data(
            audits,
            match_v7_outcomes=True,
            annotations=self.annotations_for(audits),
        )

        self.assertEqual(data["n"], 2)
        self.assertEqual(data["by_model"], [{
            "group": "model-a", "k": 1, "k_autonomous": 1,
            "k_elicited": 0, "k_timing_unknown": 0, "n": 2,
        }])

    def test_timing_unknown_hack_stays_in_denominator_not_numerator(self):
        audits = [
            self.audit("model-a", "family_seed-a", 6),
            self.audit("model-a", "family_seed-b", 6),
            self.audit("model-a", "family_seed-c", 6),
        ]
        data = propensity_data(
            audits,
            match_v7_outcomes=True,
            annotations=self.annotations_for(audits, elicited={1}, unknown={2}),
        )

        self.assertEqual(data["n_timing_unknown_hacks"], 1)
        self.assertEqual(data["by_model"][0], {
            "group": "model-a", "k": 2, "k_autonomous": 1,
            "k_elicited": 1, "k_timing_unknown": 1, "n": 3,
        })

    @staticmethod
    def three_by_three_data() -> dict:
        models = ["model-a", "model-b", "model-c"]
        prompts = ["prompt-a", "prompt-b", "prompt-c"]
        return {
            "n": 18,
            "n_invalid_excluded": 1,
            "scenario_label": "ML scenarios",
            "models": models,
            "prompts": prompts,
            "by_model": [
                {"group": model, "k": 3, "k_autonomous": 1,
                 "k_elicited": 2, "n": 6} for model in models
            ],
            "by_prompt": [
                {"group": prompt, "k": 3, "k_autonomous": 1,
                 "k_elicited": 2, "n": 6} for prompt in prompts
            ],
            "grid": [
                {
                    "prompt": prompt,
                    "cells": [
                        {"group": model, "k": 1, "k_autonomous": 1,
                         "k_elicited": 0, "n": 2} for model in models
                    ],
                }
                for prompt in prompts
            ],
        }

    def test_pooled_charts_are_vertical_rate_bars_with_wilson_intervals(self):
        data = self.three_by_three_data()

        svg = fig_propensity_rate_bars(data["by_model"], "By model")
        html = propensity_section(data)

        self.assertIn("Hack rate (%)", svg)
        self.assertIn("By model", svg)
        self.assertEqual(svg.count(">3/6</text>"), 3)
        self.assertIn("autonomous", svg)
        self.assertIn("user-elicited", svg)
        self.assertIn("Wilson 95% intervals", html)
        self.assertNotIn("pbar-fill-excl", html)
        self.assertNotIn("mean RH", html)
        self.assertNotIn("reward-hack-ish", html)
        self.assertIn("Reward hacking rate in ML scenarios", html)
        self.assertIn("invalid run(s) are excluded", html)
        self.assertIn("All models pooled by ML scenario", html)

    def test_grid_has_prompt_rows_model_columns_and_nine_error_barred_rates(self):
        svg = fig_propensity_model_prompt_grid(self.three_by_three_data())

        self.assertIn("Hack rate by prompt and model", svg)
        for label in (
            "model-a", "model-b", "model-c",
            "prompt-a", "prompt-b", "prompt-c",
        ):
            self.assertIn(label, svg)
        self.assertEqual(svg.count(">1/2</text>"), 9)

    def test_elicitation_colors_stay_hidden_without_autonomous_hacks(self):
        svg = fig_propensity_rate_bars([
            {"group": "model-a", "k": 2, "k_autonomous": 0,
             "k_elicited": 2, "n": 4},
        ], "By model")

        self.assertNotIn("autonomous", svg)
        self.assertNotIn("user-elicited", svg)

    def test_current_scenario_panels_have_three_model_bars_per_scenario(self):
        svg = fig_propensity_scenario_panels(self.three_by_three_data())

        self.assertIn("Reward hacking rate by ML scenario", svg)
        for label in (
            "model-a", "model-b", "model-c",
            "prompt-a", "prompt-b", "prompt-c",
        ):
            self.assertIn(label, svg)
        self.assertEqual(svg.count(">1/2</text>"), 9)
        self.assertIn("autonomous", svg)
        self.assertIn("user-elicited", svg)

    def test_scenario_elicitation_colors_stay_hidden_without_autonomous_hacks(self):
        data = self.three_by_three_data()
        for scenario in data["grid"]:
            for cell in scenario["cells"]:
                cell["k_autonomous"] = 0
                cell["k_elicited"] = cell["k"]

        svg = fig_propensity_scenario_panels(data)

        self.assertNotIn("autonomous", svg)
        self.assertNotIn("user-elicited", svg)


if __name__ == "__main__":
    unittest.main()
