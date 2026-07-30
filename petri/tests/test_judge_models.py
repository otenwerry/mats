"""The alignment judge is chosen in ONE place (lib/judge_models.py).

These tests pin the two properties that make varying the judge safe: every judge call
site resolves through the shared helper (so a switch cannot be half-applied), and every
selectable judge is registered for cost + display (so a switch cannot silently break cost
reporting). No LLM calls.
"""

import os
import re
import sys
import unittest
from pathlib import Path

PETRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from judge_models import (  # noqa: E402
    DEFAULT_JUDGE,
    DEFAULT_JUDGE_NAME,
    JUDGE_CHOICES,
    JUDGE_ENV_VAR,
    PREVIOUS_JUDGE_NAME,
    judge_shortname,
    resolve_judge,
)
from model_prices import context_window, price_for  # noqa: E402
from model_routing import route  # noqa: E402


class ResolverTests(unittest.TestCase):
    def test_default_is_luna(self):
        self.assertEqual(DEFAULT_JUDGE_NAME, "gpt-5.6-luna")
        self.assertEqual(DEFAULT_JUDGE, route("openai/gpt-5.6-luna"))
        self.assertEqual(resolve_judge(None), DEFAULT_JUDGE)

    def test_shortname_and_full_slug(self):
        self.assertEqual(resolve_judge("opus-4.8"), route("anthropic/claude-opus-4-8"))
        self.assertEqual(resolve_judge("openrouter/openai/gpt-5.6-sol"),
                         route("openai/gpt-5.6-sol"))

    def test_unknown_name_fails_loudly(self):
        with self.assertRaises(SystemExit):
            resolve_judge("gpt-5.6-lunar")      # a plausible typo
        with self.assertRaises(SystemExit):
            resolve_judge("   ")

    def test_env_override(self):
        previous = os.environ.get(JUDGE_ENV_VAR)
        os.environ[JUDGE_ENV_VAR] = PREVIOUS_JUDGE_NAME
        try:
            self.assertEqual(resolve_judge(), route("anthropic/claude-opus-4-8"))
            # an explicit argument still wins over the environment
            self.assertEqual(resolve_judge("gpt-5.6-luna"), DEFAULT_JUDGE)
        finally:
            if previous is None:
                del os.environ[JUDGE_ENV_VAR]
            else:
                os.environ[JUDGE_ENV_VAR] = previous

    def test_reverse_lookup(self):
        self.assertEqual(judge_shortname(DEFAULT_JUDGE), "gpt-5.6-luna")
        self.assertIsNone(judge_shortname("openrouter/some/unregistered-model"))


class RegistrationTests(unittest.TestCase):
    def test_every_choice_is_priced_and_windowed(self):
        # An unpriced judge reports no cost, which is worse than one fewer option.
        for name, slug in JUDGE_CHOICES.items():
            routed = route(slug)
            self.assertIsNotNone(price_for(routed), f"{name} ({routed}) has no price entry")
            self.assertIsNotNone(context_window(routed), f"{name} has no context window")

    def test_every_choice_has_a_display_name(self):
        import viewer

        for name, slug in JUDGE_CHOICES.items():
            pretty = viewer.pretty_model(route(slug))
            self.assertNotIn("/", pretty, f"{name} falls back to a raw slug: {pretty}")


class SecondaryJudgeTests(unittest.TestCase):
    """The secondary roles (auditor deviation / faithfulness / rollback follow-up /
    mechanism similarity) used to be Anthropic-SDK-only. Since 2026-07-30 they go through
    inspect's model layer (lib/exp_structured_judge.py), so they follow the SAME default
    judge as everything else -- and no judge call site may reach for the Anthropic SDK."""

    JUDGE_FILES = [
        "lib/exp_rollback_judge.py", "lib/exp_resample.py", "lib/exp_continuation.py",
        "tools/exp_deviation_judge.py", "tools/exp_mechanism_similarity.py",
    ]

    def test_no_judge_file_uses_the_anthropic_sdk(self):
        offenders = [f for f in self.JUDGE_FILES
                     if "AsyncAnthropic" in (PETRI / f).read_text()]
        self.assertEqual(offenders, [], "provider-locked judge call site(s)")

    def test_secondary_roles_default_to_the_shared_judge(self):
        import exp_rollback_judge
        import importlib.util

        self.assertEqual(exp_rollback_judge.DEFAULT_MODEL, DEFAULT_JUDGE)
        # the rollback judge keeps its own copy of the pre-agentic annotation schema
        self.assertIn("hack_turns", exp_rollback_judge.HackAnnotation.model_fields)
        self.assertNotIn("{environment_note}", exp_rollback_judge.ANNOTATE_SYSTEM)
        for tool in ("exp_deviation_judge", "exp_mechanism_similarity"):
            spec = importlib.util.spec_from_file_location(
                tool, PETRI / "tools" / f"{tool}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.DEFAULT_MODEL, DEFAULT_JUDGE, tool)

    def test_function_signature_defaults_follow_the_shared_judge(self):
        import inspect as py_inspect

        import exp_continuation
        import exp_resample

        for fn in (exp_resample.run_deviation_for_dir,
                   exp_continuation.run_faithfulness_for_dir):
            default = py_inspect.signature(fn).parameters["model"].default
            self.assertEqual(default, DEFAULT_JUDGE, fn.__name__)

    def test_no_provider_restricted_judge_helper_remains(self):
        import judge_models

        for gone in ("SECONDARY_JUDGE", "resolve_secondary_judge"):
            self.assertFalse(hasattr(judge_models, gone),
                             f"{gone} is dead code now that no judge is provider-locked")


class CallSiteTests(unittest.TestCase):
    """No alignment-judge call site may hard-code a model."""

    # Files that legitimately name the old judge: the price table, display names, the
    # judge-choice list itself, and the judge-comparison tool (which names arms on purpose).
    ALLOWED = {
        "lib/model_prices.py", "viewer.py", "lib/judge_models.py",
        "tools/exp_judge_model_compare.py", "exp_new_judge.py",
        "exp_check_rate_limits.py",          # a rate-limit probe, not a judge
    }
    # exp_rollback_pipeline passes "judge" as a literal CLI default meaning "reuse the
    # run's judge", not a model name.
    SECONDARY = {"exp_rollback_pipeline.py"}

    def test_no_stray_hardcoded_judge_defaults(self):
        pattern = re.compile(r'=\s*"(?:anthropic/)?claude-opus-4-8"')
        offenders = []
        for path in list(PETRI.glob("*.py")) + list(PETRI.glob("lib/*.py")) + list(
                PETRI.glob("tools/*.py")):
            rel = str(path.relative_to(PETRI))
            if rel in self.ALLOWED or rel in self.SECONDARY:
                continue
            for n, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(offenders, [], "hard-coded judge default(s); use resolve_judge()")

    def test_audit_and_rollback_use_the_shared_default(self):
        from exp_rh_audit import JUDGE

        self.assertEqual(JUDGE, DEFAULT_JUDGE)
        import exp_rollback

        self.assertEqual(exp_rollback.JUDGE, DEFAULT_JUDGE)

    def test_build_tasks_stamps_the_resolved_judge(self):
        from exp_rh_audit import build_tasks, global_system_prompt

        sp = global_system_prompt()
        tasks = build_tasks(
            ["qwen3-32b"], ["fraud_detection"], "judge-test",
            reasoning=True, fixed_system_prompt={"fraud_detection": sp},
            seeds_path=str(PETRI / "seeds" / "ml_training_data_misuse"),
            condition="allow", max_turns=200, judge="opus-4.8",
        )
        self.assertEqual(tasks[0].metadata["judge"], route("anthropic/claude-opus-4-8"))
        default_tasks = build_tasks(
            ["qwen3-32b"], ["fraud_detection"], "judge-test",
            reasoning=True, fixed_system_prompt={"fraud_detection": sp},
            seeds_path=str(PETRI / "seeds" / "ml_training_data_misuse"),
            condition="allow", max_turns=200,
        )
        self.assertEqual(default_tasks[0].metadata["judge"], DEFAULT_JUDGE)


if __name__ == "__main__":
    unittest.main()
