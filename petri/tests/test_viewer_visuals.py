import sys
import unittest
from pathlib import Path


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

from viewer import continuation_all_in_cost_data  # noqa: E402
from viewer_visuals import (  # noqa: E402
    CURRENT_VISUAL_TAB_LAYOUT,
    _continuation_analysis_toggle,
    _continuation_panel_parts,
    _cost_section,
    _current_visual_tabs,
    _wilson,
    fig_all_in_cost_by_model,
    fig_continuation_invalids,
    fig_continuation_model_rate,
    fig_continuation_model_scores,
    fig_cost_by_role,
)


class ViewerVisualsTests(unittest.TestCase):
    def test_current_scenario_visual_tabs_come_from_shared_context_layout(self):
        labels = [label for _, label, _ in CURRENT_VISUAL_TAB_LAYOUT["original_audits"]]
        self.assertEqual(labels, ["base rates", "context", "auditor info", "cost"])

        html = _current_visual_tabs("original_audits", {
            "model_outcomes": "<p>rates</p>",
            "context_fullness": "<p>context</p>",
            "user_turns": "<p>auditor</p>",
            "audit_cost": "<p>cost</p>",
        })
        self.assertEqual(html.count('<button class="vsubtab'), 4)

    def test_em_visual_tabs_are_base_rates_and_cost(self):
        labels = [label for _, label, _ in CURRENT_VISUAL_TAB_LAYOUT["EM"]]
        self.assertEqual(labels, ["base rates", "cost"])

    def test_continuation_visuals_open_model_first(self):
        labels = [label for _, label, _ in CURRENT_VISUAL_TAB_LAYOUT["continuations"]]
        self.assertEqual(labels[:2], ["by model", "pooled overview"])

    def test_continuation_analysis_toggle_exposes_both_modes(self):
        filtered = _current_visual_tabs(
            "continuations", {"continuation_models": "<p>filtered</p>"}
        )
        all_runs = _current_visual_tabs(
            "continuations", {"continuation_models": "<p>all</p>"}
        )
        html = _continuation_analysis_toggle(filtered, all_runs)

        self.assertIn("Filtered (current)", html)
        self.assertIn("All trajectories", html)
        self.assertIn('data-contanalysis-panel="filtered"', html)
        self.assertIn('data-contanalysis-panel="all"', html)
        self.assertEqual(html.count('data-vtab="models"'), 2)
        self.assertEqual(html.count('class="vtablayout"'), 2)
        self.assertNotIn('id="vpanel-models"', html)
        self.assertIn("reward_hacking &ge; 5", html)
        self.assertIn("petri-continuation-analysis", html)

    def test_wilson_interval_contains_observed_boundary_rate(self):
        # These sample sizes previously exposed floating-point endpoints just above 0
        # or below 1, which Matplotlib rejected as negative yerr values.
        for k, n in ((0, 11), (6, 6), (21, 21)):
            with self.subTest(k=k, n=n):
                p, lo, hi = _wilson(k, n)
                self.assertLessEqual(lo, p)
                self.assertLessEqual(p, hi)

    @staticmethod
    def cost_data():
        annotation = {
            "total": 4.0,
            "exact": False,
            "n_missing_usage": 0,
        }
        return {
            "by_role": {"auditor": 1.0, "target": 2.0, "judge": 3.0},
            "role_models": {"auditor": [], "target": [], "judge": []},
            "exact": True,
            "annotation": annotation,
            "any_unpriced": False,
            "per_traj": [("Target model", 6.0)],
            "all_in": {
                "total": 10.0,
                "exact": False,
                "by_model": [{
                    "model": "Target model", "n": 1,
                    "auditor": 1.0, "target": 2.0, "judge": 3.0,
                    "annotation": 4.0, "total": 10.0, "mean": 10.0,
                }],
            },
        }

    def test_budget_graph_includes_hack_turn_annotation_bar_and_total(self):
        svg = fig_cost_by_role(self.cost_data())

        self.assertIn("Hack-turn", svg)
        self.assertIn("annotation", svg)
        self.assertIn("Where the budget goes — ~$10.00 total", svg)

    def test_all_in_graph_splits_generation_into_three_roles(self):
        svg = fig_all_in_cost_by_model(self.cost_data()["all_in"])

        for label in ("auditor", "target", "judge", "hack-turn annotation"):
            self.assertIn(label, svg)
        self.assertNotIn("trajectory generation", svg)

    def test_cost_section_has_no_separate_annotation_figure(self):
        html = _cost_section(self.cost_data())

        self.assertNotIn("Hack-turn annotation cost", html)

    def test_continuation_all_in_preserves_each_generation_role(self):
        generation = {
            "by_target": [{
                "model": "Target model", "n": 2,
                "roles": {"auditor": 2.0, "target": 4.0, "judge": 6.0},
                "total": 12.0,
            }],
            "annotation": {
                "total": 3.0, "exact": False, "n_missing_usage": 0,
                "by_target": [{"model": "Target model", "sum": 3.0,
                               "mean": 3.0, "n": 1}],
            },
            "total": 12.0, "n_total": 2, "exact": True,
            "n_missing_generation": 0,
        }

        all_in = continuation_all_in_cost_data(generation, None)
        row = all_in["by_model"][0]

        self.assertEqual(
            {key: row[key] for key in ("auditor", "target", "judge", "annotation")},
            {"auditor": 2.0, "target": 4.0, "judge": 6.0, "annotation": 3.0},
        )
        self.assertEqual(row["total"], 15.0)
        self.assertEqual(row["mean"], 7.5)

    @staticmethod
    def continuation_rows():
        return [{
            "key": "baseline", "label": "No prefix", "is_baseline": True,
            "k": 0, "n": 1, "n_total": 3, "invalid_n": 2,
            "invalid_reasons": {"auditor_nudging": 1, "degenerate_behavior": 2},
            "invalid_breakdown": [
                {"reasons": ["degenerate_behavior"], "n": 1},
                {"reasons": ["auditor_nudging", "degenerate_behavior"], "n": 1},
            ],
            "scores": [1], "fh_rel": [], "frac": [],
        }]

    def test_continuation_invalid_graph_stacks_exact_reason_combinations(self):
        svg = fig_continuation_invalids(self.continuation_rows())

        self.assertIn("Invalid runs excluded from rates", svg)
        self.assertIn("n=2", svg)
        self.assertIn("auditor nudging", svg)
        self.assertIn("degenerate behavior", svg)

    def test_continuation_invalid_graph_is_on_overview_and_model_tabs(self):
        rows = self.continuation_rows()
        parts = _continuation_panel_parts({
            "by_condition": rows,
            "by_model": [{"model": "Test model", "by_condition": rows}],
            "by_seed": [],
            "interesting_categories": [],
        })

        self.assertIn("Invalid runs excluded from rates", parts["continuation_overview"])
        self.assertIn("Invalid runs excluded from rates", parts["continuation_models"])
        self.assertIn('class="modelblock"', parts["continuation_models"])
        self.assertIn('class="figgrid modelpairgrid"', parts["continuation_models"])
        self.assertIn("3 runs &middot; 1 valid &middot; 2 invalid/excluded",
                      parts["continuation_models"])
        self.assertIn("Test model", parts["continuation_models"])

    def test_continuation_model_charts_are_separate_and_each_names_model(self):
        model = {"model": "Test model", "by_condition": self.continuation_rows()}

        score_svg = fig_continuation_model_scores(model)
        rate_svg = fig_continuation_model_rate(model)

        self.assertIn("Test model", score_svg)
        self.assertIn("Test model", rate_svg)
        self.assertIn("reward_hacking by prefix condition", score_svg)
        self.assertIn("Reward-hack rate by prefix condition", rate_svg)


if __name__ == "__main__":
    unittest.main()
