"""M/M/c/K closed forms, Erlang-B / Erlang-C limits, and a CI check at c=8, rho=0.8."""

import unittest

from sim.literature_params import PRIMARY_MEAN_SERVICE_S
from sim.mmc_k_simulation import (
    QUEUE_EXTRAS,
    SWEEP_SERVER_COUNTS,
    offered_load_for_blocking_k,
    offered_load_for_wait_k,
    theoretical_mmck,
)
from sim.mmc_simulation import erlang_b, theoretical_mmc
from sim.paths import PROJECT_ROOT
from sim.validation_study import (
    assert_theory_inside,
    run_part_c_validation,
    save_cases,
)


class TestMmckClosedForm(unittest.TestCase):
    def test_k_equals_c_matches_erlang_b(self):
        for c, a in ((1, 0.8), (8, 3.9865), (8, 6.4), (21, 10.0)):
            theory = theoretical_mmck(a, 1.0, c, K=c)
            self.assertAlmostEqual(theory["B"], erlang_b(c, a), places=8)
            self.assertAlmostEqual(theory["Lq"], 0.0, places=10)
            self.assertAlmostEqual(theory["Wq"], 0.0, places=10)

    def test_mm1_k2_geometric(self):
        # M/M/1/2, rho = 0.5. P_n = (1-rho) rho^n / (1 - rho^{K+1}).
        rho = 0.5
        k = 2
        p0 = (1.0 - rho) / (1.0 - rho ** (k + 1))
        p_k = p0 * rho ** k
        theory = theoretical_mmck(rho, 1.0, c=1, K=k)
        self.assertAlmostEqual(theory["B"], p_k, places=10)

    def test_large_k_approaches_erlang_c(self):
        mu = 1.0 / PRIMARY_MEAN_SERVICE_S
        c = 8
        rho = 0.80
        lam = rho * c * mu
        erlang_c = theoretical_mmc(lam, mu, c)
        theory = theoretical_mmck(lam, mu, c, K=c + 200)
        self.assertLess(abs(theory["Wq"] - erlang_c["Wq"]) / erlang_c["Wq"], 1e-4)
        self.assertLess(theory["B"], 1e-6)

    def test_heavy_traffic_carried_load_saturates_at_c(self):
        theory = theoretical_mmck(lambda_rate=1000.0, mu_rate=1.0, c=8, K=13)
        self.assertAlmostEqual(theory["carried_erlang"], 8.0, places=4)

    def test_blocking_root_and_zero_wait_buffer(self):
        a = offered_load_for_blocking_k(8, K=8, blocking=0.02)
        self.assertAlmostEqual(erlang_b(8, a), 0.02, places=4)
        self.assertIsNone(offered_load_for_wait_k(8, K=8, mu_rate=1.0 / 180.0, wait_s=10.0))
        # c=8, K=13, 1/mu=180 s: Wq crosses 10 s between 4 and 6 erlang.
        a_wait = offered_load_for_wait_k(8, K=13, mu_rate=1.0 / 180.0, wait_s=10.0)
        self.assertGreater(a_wait, 4.0)
        self.assertLess(a_wait, 6.0)


class TestMmckSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = run_part_c_validation(
            server_counts=SWEEP_SERVER_COUNTS,
            queue_extras=QUEUE_EXTRAS,
            echo=True,
            make_plot=True,
        )
        save_cases(cls.cases, PROJECT_ROOT / "output" / "part_c_validation.json")

    def test_c8_rho080_theory_inside_ci(self):
        # Specified validation point. Larger sites are on the same figure;
        # their shorter runs can put a very tight throughput CI just outside
        # theory, which is reported in the results summary rather than here.
        c8 = [case for case in self.cases if case["c"] == 8]
        self.assertEqual(len(c8), len(QUEUE_EXTRAS))
        assert_theory_inside(c8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
