import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from dimension_routing import (  # noqa: E402
    audit_dimension_names_on_disk,
    audit_dimension_set,
    scoped_dimension_dir,
)


RUBRIC = """---
description: Test dimension.
---
Score this test dimension.
"""


class DimensionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.seeds = root / "seeds"
        self.dimensions = root / "dimensions"
        (self.seeds / "p_hacking" / "checkout_redesign").mkdir(parents=True)
        self.dimensions.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_dim(self, relative: str) -> Path:
        path = self.dimensions / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RUBRIC)
        return path

    def write_order(self, *names: str) -> None:
        (self.dimensions / "judge_order.json").write_text(json.dumps(names))

    def load(self):
        return audit_dimension_set(
            self.seeds / "p_hacking" / "checkout_redesign",
            dimensions_root=self.dimensions,
            seeds_root=self.seeds,
        )

    def test_combines_legacy_global_new_global_and_family_scope(self):
        self.write_dim("legacy.md")
        self.write_dim("global/common.md")
        family = self.write_dim("p_hacking/selective_reporting.md").parent
        self.write_order("selective_reporting", "legacy", "common")

        selected = self.load()

        self.assertEqual(
            selected.names, ["selective_reporting", "legacy", "common"]
        )
        self.assertEqual(selected.scope_dir, family)

    def test_exact_member_scope_wins_over_family_scope(self):
        self.write_dim("global/common.md")
        self.write_dim("p_hacking/family.md")
        exact = self.write_dim(
            "p_hacking/checkout_redesign/member_specific.md"
        ).parent
        self.write_order("member_specific", "family", "common")

        selected = self.load()

        self.assertEqual(selected.names, ["member_specific", "common"])
        self.assertEqual(selected.scope_dir, exact)

    def test_duplicate_name_fails_loudly(self):
        self.write_dim("global/reward_hacking.md")
        self.write_dim("p_hacking/reward_hacking.md")
        self.write_order("reward_hacking")

        with self.assertRaisesRegex(SystemExit, "duplicate judge dimension"):
            self.load()

    def test_missing_scoped_folder_still_uses_global(self):
        common = self.write_dim("global/common.md")
        self.write_order("common")

        selected = self.load()

        self.assertEqual(selected.names, ["common"])
        self.assertEqual(selected.files, (common,))
        self.assertEqual(
            scoped_dimension_dir(
                self.seeds / "p_hacking" / "checkout_redesign",
                dimensions_root=self.dimensions,
                seeds_root=self.seeds,
            ),
            self.dimensions / "p_hacking" / "checkout_redesign",
        )

    def test_empty_tracked_family_folder_is_the_reported_scope(self):
        self.write_dim("global/common.md")
        self.write_order("common")
        family = self.dimensions / "p_hacking"
        family.mkdir()
        (family / ".gitkeep").write_text("")

        selected = self.load()

        self.assertEqual(selected.names, ["common"])
        self.assertEqual(selected.scope_dir, family)

    def test_viewer_union_excludes_special_purpose_folders(self):
        self.write_dim("global/common.md")
        self.write_dim("p_hacking/selective_reporting.md")
        self.write_dim("p_hacking/checkout_redesign/member_specific.md")
        self.write_dim("archived/retired.md")
        self.write_dim("rollback_only/deviation.md")

        self.assertEqual(
            audit_dimension_names_on_disk(dimensions_root=self.dimensions),
            ["common", "member_specific", "selective_reporting"],
        )

    def test_readmes_are_not_treated_as_rubrics(self):
        (self.dimensions / "README.md").write_text("# Dimension notes\n")
        self.write_dim("global/common.md")
        family = self.write_dim("p_hacking/selective_reporting.md").parent
        (family / "README.md").write_text("# P-hacking notes\n")
        self.write_order("selective_reporting", "common")

        selected = self.load()

        self.assertEqual(selected.names, ["selective_reporting", "common"])
        self.assertEqual(
            audit_dimension_names_on_disk(dimensions_root=self.dimensions),
            ["common", "selective_reporting"],
        )

    def test_missing_active_dimension_in_order_fails_loudly(self):
        self.write_dim("global/common.md")
        self.write_dim("p_hacking/selective_reporting.md")
        self.write_order("common")

        with self.assertRaisesRegex(SystemExit, "missing active dimensions"):
            self.load()

    def test_unknown_dimension_in_order_fails_loudly(self):
        self.write_dim("global/common.md")
        self.write_order("common", "typo")

        with self.assertRaisesRegex(SystemExit, "unknown/inactive dimensions"):
            self.load()

    def test_duplicate_dimension_in_order_fails_loudly(self):
        self.write_dim("global/common.md")
        self.write_order("common", "common")

        with self.assertRaisesRegex(SystemExit, "duplicate names"):
            self.load()


if __name__ == "__main__":
    unittest.main()
