"""95% CI coverage at the Part A validation points.

These runs are the documented validation (R = 10, scaled measurement length).
They take several minutes.
"""

import unittest

from sim.paths import PROJECT_ROOT
from sim.replication import DEFAULT_REPLICATIONS, DEFAULT_SCALE_FACTOR
from sim.validation_study import assert_theory_inside, run_part_a_validation, save_cases


class TestTheoryInside95CI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = run_part_a_validation(
            n_replications=DEFAULT_REPLICATIONS,
            scale_factor=DEFAULT_SCALE_FACTOR,
            echo=True,
        )
        save_cases(cls.cases, PROJECT_ROOT / "output" / "part_a_validation.json")

    def test_all_validation_points(self):
        self.assertEqual(len(self.cases), 4)
        self.assertEqual(self.cases[0]["n_replications"], DEFAULT_REPLICATIONS)
        assert_theory_inside(self.cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
