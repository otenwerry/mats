import sys
import unittest
from pathlib import Path


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

from viewer import model_outcome_data  # noqa: E402
from viewer_visuals import fig_model_seed_outcomes  # noqa: E402


class ModelSeedOutcomeTests(unittest.TestCase):
    @staticmethod
    def audit(model: str, seed: str, reward_hacking: int, epoch: int) -> dict:
        return {
            "target": model,
            "seed": seed,
            "scores": {"reward_hacking": reward_hacking},
            "mode": "test-run",
            "task": seed,
            "epoch": epoch,
            "transcript": "",
            "integrity_issues": [],
        }

    def test_data_keeps_every_model_seed_cell_and_outcome(self):
        audits = [
            self.audit("model-a", "family_seed-a", 6, 1),
            self.audit("model-a", "family_seed-b", 1, 2),
            self.audit("model-b", "family_seed-a", 1, 3),
            self.audit("model-b", "family_seed-b", 6, 4),
        ]

        data = model_outcome_data(audits, "current_training_data_misuse", {})

        self.assertEqual(data["seeds"], ["seed-a", "seed-b"])
        self.assertEqual([row["model"] for row in data["rows"]], ["model-a", "model-b"])
        for row in data["rows"]:
            self.assertEqual([cell["seed"] for cell in row["by_seed"]], data["seeds"])
            self.assertEqual(sum(cell["n"] for cell in row["by_seed"]), row["n"])
            for cell in row["by_seed"]:
                self.assertEqual(sum(cell["counts"].values()), cell["n"])

    def test_visual_is_a_seed_row_by_model_column_grid(self):
        data = {
            "seeds": ["seed_a", "seed_b", "seed_c"],
            "categories": [("hack_autonomous", "reward hack"), ("clean", "clean")],
            "rows": [
                {
                    "model": f"model-{row}",
                    "n": 6,
                    "counts": {"hack_autonomous": 3, "clean": 3},
                    "by_seed": [
                        {"seed": seed, "n": 2,
                         "counts": {"hack_autonomous": 1, "clean": 1}}
                        for seed in ("seed_a", "seed_b", "seed_c")
                    ],
                }
                for row in ("a", "b", "c")
            ],
        }

        svg = fig_model_seed_outcomes(data)

        self.assertIn("Outcomes by seed and model", svg)
        for label in ("model-a", "model-b", "model-c", "seed a", "seed b", "seed c"):
            self.assertIn(label, svg)
        self.assertEqual(svg.count(">n=2</text>"), 9)


if __name__ == "__main__":
    unittest.main()
