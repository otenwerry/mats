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
    resolve_fixed_sp,
)


class PinnedSeedCompositionTests(unittest.TestCase):
    def test_active_p_hacking_collection_assembles_both_members(self):
        collection = PETRI / "seeds" / "p_hacking"

        self.assertTrue(is_pinned_collection(collection))
        self.assertEqual(
            pinned_collection_members(collection),
            ["checkout_redesign", "retrieval_practice"],
        )
        system_prompts = resolve_fixed_sp("p_hacking", str(collection))
        self.assertEqual(
            system_prompts["checkout_redesign"],
            system_prompts["retrieval_practice"],
        )

        assembled = {}
        for name in pinned_collection_members(collection):
            member = collection / name
            self.assertEqual(pinned_conditions(member), ["allow", "correct"])
            sample = assemble_pinned_seed(member, "allow")
            self.assertNotIn("{{TASK}}", sample.input)
            self.assertNotIn("{{TOOLING}}", sample.input)
            self.assertIn('path="environment/GROUND_TRUTH.md"', sample.input)
            self.assertIn("RESULT.md", sample.input)
            assembled[name] = sample.input

        self.assertIn("mobile", assembled["checkout_redesign"])
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


if __name__ == "__main__":
    unittest.main()
