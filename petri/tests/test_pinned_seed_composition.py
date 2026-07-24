import sys
import tempfile
import unittest
from pathlib import Path


PETRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from exp_rh_audit import (  # noqa: E402
    assemble_pinned_seed,
    is_pinned_collection,
    pinned_collection_members,
    pinned_conditions,
    resolve_condition,
    resolve_fixed_sp,
)


class PinnedSeedCompositionTests(unittest.TestCase):
    def test_omitted_condition_defaults_to_allow(self):
        collection = PETRI / "seeds" / "p_hacking"
        member = collection / "checkout_redesign"

        self.assertEqual(resolve_condition(None, str(collection)), "allow")
        self.assertEqual(resolve_condition(None, str(member)), "allow")

    def test_active_p_hacking_collection_assembles_all_members(self):
        collection = PETRI / "seeds" / "p_hacking"

        self.assertTrue(is_pinned_collection(collection))
        self.assertEqual(
            pinned_collection_members(collection),
            [
                "checkout_redesign",
                "reasoning_prompt_benchmark",
                "retrieval_practice",
            ],
        )
        system_prompts = resolve_fixed_sp("p_hacking", str(collection))
        self.assertEqual(len(set(system_prompts.values())), 1)

        assembled = {}
        for name in pinned_collection_members(collection):
            member = collection / name
            self.assertEqual(pinned_conditions(member), ["allow", "correct"])
            sample = assemble_pinned_seed(member, "allow")
            self.assertNotIn("{{TASK}}", sample.input)
            self.assertNotIn("{{TOOLING}}", sample.input)
            self.assertNotIn("{{SEED_SPECIFIC_PROMPT}}", sample.input)
            self.assertIn('path="environment/GROUND_TRUTH.md"', sample.input)
            self.assertIn("RESULT.md", sample.input)
            assembled[name] = sample.input

        self.assertIn("mobile", assembled["checkout_redesign"])
        self.assertIn("math questions", assembled["reasoning_prompt_benchmark"])
        self.assertIn(
            "particular kinds of questions",
            assembled["reasoning_prompt_benchmark"],
        )
        self.assertIn("remote participants", assembled["retrieval_practice"])
        self.assertNotEqual(
            assembled["checkout_redesign"], assembled["retrieval_practice"]
        )

    def test_empty_old_local_conditions_do_not_shadow_shared_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            collection = Path(temp) / "collection"
            shared = collection / "_shared"
            member = collection / "member"
            (shared / "conditions").mkdir(parents=True)
            (member / "conditions").mkdir(parents=True)
            (member / "environment").mkdir()
            (shared / "conditions" / "allow.md").write_text("shared condition")
            (shared / "core.md").write_text("{{TASK}}\n{{TOOLING}}")
            (shared / "USER_PROMPT.txt").write_text("start")
            (member / "scenario.md").write_text(
                "[TASK]\nrun the study\n[TOOLING]\nstay consistent"
            )

            self.assertEqual(pinned_conditions(member), ["allow"])

    def test_composed_member_prefixes_the_shared_user_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            collection = Path(temp) / "collection"
            shared = collection / "_shared"
            member = collection / "member"
            (shared / "conditions").mkdir(parents=True)
            (member / "environment").mkdir(parents=True)
            (shared / "conditions" / "allow.md").write_text("shared condition")
            (shared / "core.md").write_text("{{TASK}}\n{{TOOLING}}")
            (shared / "USER_PROMPT.txt").write_text(
                "{{SEED_SPECIFIC_PROMPT}}\n\nshared second part"
            )
            (member / "scenario.md").write_text(
                "[USER_PROMPT]\nseed-specific first part\n\n"
                "[TASK]\nrun the study\n[TOOLING]\nstay consistent"
            )

            sample = assemble_pinned_seed(member, "allow")

            self.assertIn(
                '<pinned_file path="USER_PROMPT.txt">\n'
                "seed-specific first part\n\nshared second part\n"
                "</pinned_file>",
                sample.input,
            )
            self.assertNotIn("{{SEED_SPECIFIC_PROMPT}}", sample.input)


if __name__ == "__main__":
    unittest.main()
