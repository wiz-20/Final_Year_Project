"""Carried load at the 2% Erlang-B grade of service. The sim class is slow."""

import unittest

from sim.literature_params import BLOCKING_TARGET, SERVER_COUNTS
from sim.mmc_simulation import erlang_b, offered_load_for_blocking
from sim.paths import PROJECT_ROOT
from sim.validation_study import (
    assert_theory_inside,
    carried_metrics,
    run_part_b_carried,
    save_cases,
)


class TestCarriedFormula(unittest.TestCase):
    def test_identity(self):
        got = carried_metrics(offered_a=10.0, blocking=0.02, mu=1.0 / 180.0, c=8)
        self.assertAlmostEqual(got["carried_erlang"], 9.8, places=10)
        self.assertAlmostEqual(got["carried_utilisation"], 9.8 / 8, places=10)
        self.assertAlmostEqual(got["carried_calls_per_hour"], 9.8 / 180.0 * 3600.0, places=8)

    def test_gos_load_is_below_c(self):
        for c in SERVER_COUNTS:
            a = offered_load_for_blocking(c, BLOCKING_TARGET)
            self.assertAlmostEqual(erlang_b(c, a), BLOCKING_TARGET, places=5)
            carried = carried_metrics(a, erlang_b(c, a), 1.0 / 180.0, c)
            self.assertLess(carried["carried_utilisation"], 1.0)


class TestCarriedSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = run_part_b_carried(echo=True)
        save_cases(cls.cases, PROJECT_ROOT / "output" / "part_b_carried.json")

    def test_theory_inside_ci_for_every_site_size(self):
        self.assertEqual(len(self.cases), len(SERVER_COUNTS))
        assert_theory_inside(self.cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
