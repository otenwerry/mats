import sys
import unittest
from pathlib import Path


POSTTRAINBENCH = Path(__file__).resolve().parents[1]
MATS = POSTTRAINBENCH.parent
for path in (POSTTRAINBENCH, MATS / "shared", MATS / "petri" / "lib"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib import experiment_readiness
from lib import exp_ask_env


class ExperimentReadinessTest(unittest.TestCase):
    def test_paid_ptb_experiments_are_loudly_blocked(self):
        self.assertFalse(experiment_readiness.PAID_EXPERIMENTS_READY)
        with self.assertRaises(SystemExit) as caught:
            experiment_readiness.require_paid_experiments_ready()
        message = str(caught.exception)
        self.assertIn("PAID EXPERIMENTS ARE NOT READY", message)
        self.assertIn("RUN BLOCKED BEFORE SPEND", message)
        self.assertIn("Do not bypass this", message)

    def test_shared_paid_adapter_checks_block_before_setup(self):
        with self.assertRaises(SystemExit):
            exp_ask_env.prepare([], args=None)


if __name__ == "__main__":
    unittest.main()
