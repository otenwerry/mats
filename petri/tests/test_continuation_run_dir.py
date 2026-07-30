import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

from exp_continuation_pipeline import _claim_run_dir  # noqa: E402


class ContinuationRunDirTests(unittest.TestCase):
    def test_same_second_concurrent_runs_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            started = datetime(2026, 7, 29, 0, 54, 51)

            first = _claim_run_dir(root, "continuation", 40, started)
            second = _claim_run_dir(root, "continuation", 40, started)

            self.assertEqual(first.name, "continuation-40x-20260729-005451")
            self.assertEqual(second.name, "continuation-40x-20260729-005451-2")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
