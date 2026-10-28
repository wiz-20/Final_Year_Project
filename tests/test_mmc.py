"""Unit tests for M/M/c (Erlang-C) closed-form results.

Run from project root:  python -m unittest discover -s tests -p "test_*.py"
"""

import math
import unittest

from sim.mmc_simulation import (
    DEMO_C,
    DEMO_RHO,
    PRIMARY_MEAN_SERVICE_S,
    erlang_b,
    erlang_c_wait_probability,
    offered_load_for_blocking,
    run_mmc_simulation,
    theoretical_mmc,
)
from sim.mmc_b_simulation import run_mmc_b_simulation


class TestTheoreticalMmc(unittest.TestCase):
    def test_c1_reduces_to_mm1(self):
        mu = 1.0 / 180.0
        rho = 0.80
        lam = rho * 1 * mu
        mmc = theoretical_mmc(lam, mu, c=1)
        wq_mm1 = lam / (mu * (mu - lam))
        w_mm1 = wq_mm1 + 1.0 / mu
        self.assertAlmostEqual(mmc["rho"], rho, places=12)
        self.assertAlmostEqual(mmc["Wq"], wq_mm1, places=10)
        self.assertAlmostEqual(mmc["W"], w_mm1, places=10)
        self.assertAlmostEqual(mmc["Lq"], lam * wq_mm1, places=10)

    def test_wait_probability_known_value(self):
        # a = 6.4 Erlang, c = 8 -> rho = 0.8
        # Independent check of Erlang-C via the standard sum formula.
        c, a = 8, 6.4
        s = sum(a**k / math.factorial(k) for k in range(c))
        last = (a**c / math.factorial(c)) / (1.0 - a / c)
        p0 = 1.0 / (s + last)
        expected_c = last * p0
        self.assertAlmostEqual(erlang_c_wait_probability(c, a), expected_c, places=10)

    def test_more_servers_shorter_wait_at_same_rho(self):
        mu = 1.0 / 180.0
        rho = 0.80
        wq = []
        for c in (1, 7, 8, 21, 45):
            lam = rho * c * mu
            wq.append(theoretical_mmc(lam, mu, c)["Wq"])
        self.assertTrue(all(wq[i] > wq[i + 1] for i in range(len(wq) - 1)))


class TestErlangB(unittest.TestCase):
    def test_c1_closed_form(self):
        a = 0.80
        self.assertAlmostEqual(erlang_b(1, a), a / (1.0 + a), places=12)

    def test_known_value_c7(self):
        # B(c,a) = (a^c / c!) / sum_{k=0}^c a^k/k!
        c, a = 7, 3.0
        terms = [a**k / math.factorial(k) for k in range(c + 1)]
        expected = terms[-1] / sum(terms)
        self.assertAlmostEqual(erlang_b(c, a), expected, places=10)

    def test_offered_load_at_2_percent(self):
        a = offered_load_for_blocking(c=7, blocking=0.02)
        self.assertAlmostEqual(erlang_b(7, a), 0.02, places=6)
        self.assertLess(a, 7.0)

    def test_sim_blocking_matches_erlang_b(self):
        mu = 1.0 / PRIMARY_MEAN_SERVICE_S
        c = 8
        a = offered_load_for_blocking(c, 0.03)
        lam = a * mu
        theory_b = erlang_b(c, a)
        measurement_time = 25_000 * PRIMARY_MEAN_SERVICE_S
        warmup_time = 2_500 * PRIMARY_MEAN_SERVICE_S
        sim = run_mmc_b_simulation(
            lam, mu, c, measurement_time, warmup_time, random_seed=123
        )
        rel_err = abs(sim["B"] - theory_b) / theory_b
        self.assertLess(rel_err, 0.05)


class TestSimMmcErlangC(unittest.TestCase):
    def test_sim_wait_matches_erlang_c_at_demo_point(self):
        """End-to-end M/M/c SimPy vs Erlang-C (mirrors Erlang-B sim test)."""
        mu = 1.0 / PRIMARY_MEAN_SERVICE_S
        c = DEMO_C
        lam = DEMO_RHO * c * mu
        measurement_time = 30_000 * PRIMARY_MEAN_SERVICE_S
        warmup_time = 3_000 * PRIMARY_MEAN_SERVICE_S
        sim = run_mmc_simulation(
            lam, mu, c, measurement_time, warmup_time, random_seed=123
        )
        theory = theoretical_mmc(lam, mu, c)
        for key in ("Wq", "W", "Lq"):
            rel_err = abs(sim[key] - theory[key]) / theory[key]
            self.assertLess(rel_err, 0.02, msg=key)
        rel_rho = abs(sim["rho"] - theory["rho"]) / theory["rho"]
        self.assertLess(rel_rho, 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
