import sys
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

import viewer  # noqa: E402
import viewer_visuals  # noqa: E402
from viewer_load import context_calls_for_role  # noqa: E402


def model_event(role="target", model="model", usage=None, *, request="request",
                error=None, stop_reason=None):
    choices = ([SimpleNamespace(stop_reason=stop_reason)] if stop_reason else [])
    output = SimpleNamespace(usage=usage, choices=choices)
    return SimpleNamespace(
        event="model",
        role=role,
        model=model,
        input=request,
        tools=["bash"],
        config={"temperature": 0},
        output=output,
        error=error,
    )


class ContextUsageTests(unittest.TestCase):
    def test_provider_prompt_parts_are_summed_and_missing_call_is_preserved(self):
        usage = SimpleNamespace(
            input_tokens=7,
            input_tokens_cache_read=101,
            input_tokens_cache_write=13,
        )
        sample = SimpleNamespace(events=[
            model_event(usage=usage),
            model_event(role="auditor", usage=usage),
            model_event(),
        ])

        result = context_calls_for_role(sample, "target", "model")

        self.assertEqual(result["calls"], [121, None])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["missing_calls"], 1)

    def test_identical_successful_retry_is_one_complete_logical_call(self):
        usage = SimpleNamespace(
            input_tokens=7,
            input_tokens_cache_read=101,
            input_tokens_cache_write=13,
        )
        sample = SimpleNamespace(events=[
            model_event(error=ValueError("provider response was not JSON")),
            model_event(usage=usage),
        ])

        result = context_calls_for_role(sample, "target", "model")

        self.assertEqual(result["calls"], [121])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["missing_calls"], 0)
        self.assertEqual(result["recorded_attempts"], 2)
        self.assertEqual(result["logical_calls"], 1)
        self.assertEqual(result["provider_events"], [{
            "kind": "failed_attempt_retried",
            "attempt": 1,
            "succeeded_on_attempt": 2,
            "error": "ValueError",
        }])

    def test_content_filter_stays_partial_and_queryable(self):
        sample = SimpleNamespace(events=[
            model_event(stop_reason="content_filter"),
        ])

        result = context_calls_for_role(sample, "target", "model")

        self.assertEqual(result["calls"], [None])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["provider_events"], [{
            "kind": "content_filter",
            "attempt": 1,
            "stop_reason": "content_filter",
        }])

    def test_content_filter_warning_is_loud_and_specific(self):
        audit = {
            "target_context_usage": {
                "provider_events": [{
                    "kind": "content_filter",
                    "attempt": 10,
                    "stop_reason": "content_filter",
                }],
            },
        }

        html = viewer.content_filter_warning(audit)

        self.assertIn("TARGET CONTENT FILTER TRIGGERED", html)
        self.assertIn("target request 10", html)
        self.assertIn("model provider blocked", html)
        self.assertIn("auditor then continued", html)

    def test_prefix_boundary_is_between_prefix_and_live_calls(self):
        svg = viewer._context_timeline_svg([10_000, 20_000, 30_000], 100_000, 2)

        self.assertIn('class="prefix-boundary"', svg)
        self.assertIn('>prefix ends</text>', svg)
        self.assertIn('call 3: 30,000 tokens (30.00%)', svg)
        # Domain is 0.5..3.5; after two calls the boundary is 2/3 across the plot.
        self.assertIn('x1="554.00"', svg)

    def test_context_fullness_uses_only_complete_exact_target_timelines(self):
        def audit(status, calls, *, role_matching="event_role", dead=False, target="model-a"):
            return {
                "target": target,
                "dead": dead,
                "target_context_usage": {
                    "status": status,
                    "calls": calls,
                    "role_matching": role_matching,
                },
            }

        audits = [
            audit("complete", [10_000, 25_000]),
            audit("partial", [10_000, None]),
            audit("complete", [15_000], role_matching="model_fallback"),
            audit("unavailable", []),
            audit("complete", [12_000], dead=True),
            audit("complete", [5_000], target="unknown-model"),
        ]
        with patch.object(viewer.model_prices, "context_window",
                          side_effect=lambda slug: 100_000 if slug == "model-a" else None):
            data = viewer.context_fullness_data(audits)

        self.assertEqual(data["pcts"], [25.0])
        self.assertEqual(data["n_included"], 1)
        self.assertEqual(data["n_total"], 6)
        self.assertEqual(data["exclusions"], {
            "dead": 1,
            "partial": 1,
            "unavailable": 1,
            "ambiguous_role": 1,
            "unknown_window": 1,
        })

    def test_context_fullness_visuals_have_aggregate_and_per_model_graphs(self):
        data = {
            "n_total": 3,
            "n_included": 3,
            "pcts": [10.0, 20.0, 30.0],
            "by_model": [
                {"model": "model-a", "window": 1_050_000, "pcts": [10.0, 20.0]},
                {"model": "model-b", "window": 262_144, "pcts": [30.0]},
            ],
            "exclusions": {},
        }

        html = viewer_visuals._context_fullness_section(data)

        self.assertEqual(html.count("<figure"), 2)
        self.assertIn("All target models", html)
        self.assertIn("model-a (1.05M)", html)
        self.assertIn("model-b (262K)", html)
        self.assertNotIn("One point per original audit", html)

    def test_trajectory_renderer_never_restores_annotation_summary_prose(self):
        source = inspect.getsource(viewer.write_trajectory_page)

        self.assertNotIn("Hack-turn annotation", source)
        self.assertNotIn("interpretive, spot-check", source)
        self.assertNotIn("tldr", source)

    def test_viewer_discards_historical_annotation_summaries(self):
        raw = {
            "page": {
                "model": "annotator",
                "tldr": "retired trajectory-level prose",
                "hack_turns": [{"m": 3, "title": "marked turn"}],
            }
        }

        loaded = viewer._annotations_for_viewer(raw)

        self.assertNotIn("tldr", loaded["page"])
        self.assertEqual(loaded["page"]["hack_turns"], raw["page"]["hack_turns"])
        self.assertIn("tldr", raw["page"], "the historical source data must not be mutated")

    def test_annotation_jobs_no_longer_request_or_store_summary_prose(self):
        annotator = (PETRI_ROOT / "lib" / "exp_annotate_hacks.py").read_text()
        rollback = (PETRI_ROOT / "lib" / "exp_rollback_judge.py").read_text()

        self.assertNotIn("tldr", annotator)
        self.assertNotIn('["tldr"]', rollback)


if __name__ == "__main__":
    unittest.main()
