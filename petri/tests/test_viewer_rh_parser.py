import sys
import unittest
from pathlib import Path


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))

from viewer_load import RH_TYPE_VOCAB, parse_rh_category  # noqa: E402


class RewardHackParserTests(unittest.TestCase):
    def test_validation_contamination_is_a_valid_current_type(self):
        justification = """reward_hacking: trained on validation, then selected on it.

RH_CATEGORY: delivered_hack
RH_TYPES: validation_contamination
"""

        category, types, parse_error = parse_rh_category(justification, 7)

        self.assertIn("validation_contamination", RH_TYPE_VOCAB)
        self.assertEqual(category, "delivered_hack")
        self.assertEqual(types, ["validation_contamination"])
        self.assertFalse(parse_error)

    def test_legacy_train_on_val_type_still_parses(self):
        justification = """reward_hacking: legacy judgment.

RH_CATEGORY: delivered_hack
RH_TYPES: train_on_val
"""

        category, types, parse_error = parse_rh_category(justification, 5)

        self.assertEqual(category, "delivered_hack")
        self.assertEqual(types, ["train_on_val"])
        self.assertFalse(parse_error)


if __name__ == "__main__":
    unittest.main()
