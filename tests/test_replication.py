"""Fast checks for the t interval and the 1/(1-rho)^2 run-length scaling."""

import unittest

from sim.replication import (
    apply_arrival_floor,
    confidence_interval,
    length_factor,
    measurement_service_times_after_warmup,
    plan_run,
    student_t_ppf,
    theory_inside_ci,
)


class TestStudentT(unittest.TestCase):
    def test_two_sided_95_df9(self):
        # R = 10 -> df = 9. Published t_0.975,9.
        self.assertAlmostEqual(student_t_ppf(0.975, 9), 2.26215716274, places=6)

    def test_two_sided_95_df1(self):
        self.assertAlmostEqual(student_t_ppf(0.975, 1), 12.7062047364, places=5)

    def test_two_sided_95_df30(self):
        self.assertAlmostEqual(student_t_ppf(0.975, 30), 2.04227245630, places=6)


class TestRunLengthScaling(unittest.TestCase):
    def test_rho_080_is_25_times_base(self):
        # (1-0.8)^2 = 0.04, so 1/(1-rho)^2 = 25.
        self.assertAlmostEqual(length_factor(0.80, scale_factor=1.0), 25.0, places=10)
        self.assertEqual(
            measurement_service_times_after_warmup(1000, 0.80, scale_factor=1.0, c=8),
            25_000,
        )

    def test_rho_090_is_100_times_base(self):
        self.assertAlmostEqual(length_factor(0.90, scale_factor=1.0), 100.0, places=10)
        self.assertEqual(
            measurement_service_times_after_warmup(1000, 0.90, scale_factor=1.0, c=1),
            100_000,
        )

    def test_scale_factor_multiplies(self):
        self.assertAlmostEqual(length_factor(0.80, scale_factor=2.0), 50.0, places=10)

    def test_arrival_floor_for_rare_blocking(self):
        bare = measurement_service_times_after_warmup(1000, 0.0204, c=1)
        floored = apply_arrival_floor(bare, offered_a=0.0204, min_expected_arrivals=8000)
        self.assertGreater(floored, bare)
        self.assertGreaterEqual(floored * 0.0204, 8000)
        plan = plan_run(
            180.0, 0.0204, c=1, offered_a=0.0204, min_expected_arrivals=8000
        )
        self.assertEqual(plan["n_measurement_service_times"], floored)

    def test_large_c_shortens_in_proportion(self):
        n8 = measurement_service_times_after_warmup(1000, 0.80, c=8)
        n16 = measurement_service_times_after_warmup(1000, 0.80, c=16)
        self.assertEqual(n16, n8 // 2)


class TestConfidenceInterval(unittest.TestCase):
    def test_zero_dispersion_contains_the_constant(self):
        interval = confidence_interval([3.0, 3.0, 3.0, 3.0])
        self.assertEqual(interval["mean"], 3.0)
        self.assertEqual(interval["low"], 3.0)
        self.assertEqual(interval["high"], 3.0)
        self.assertTrue(theory_inside_ci(interval, 3.0))
        self.assertFalse(theory_inside_ci(interval, 3.1))

    def test_symmetric_sample_is_centred(self):
        interval = confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(interval["mean"], 3.0, places=10)
        self.assertLess(interval["low"], 3.0)
        self.assertGreater(interval["high"], 3.0)
        self.assertTrue(theory_inside_ci(interval, 3.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
