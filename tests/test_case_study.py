"""Closed-form checks for the published-cell case study. No SimPy."""

import math
import unittest

from sim.capacity_planning import (
    IMPLIED_HOLDING_TIME_S,
    erlang_b_capacity,
    subscribers_at_load,
    years_to_upgrade,
)
from sim.case_study import (
    INCONSISTENT_HALF_RATE,
    analyse_ahmadzai,
    bracketed_by,
    effective_channel_count,
    erlang_b_traffic_at_congestion,
    offered_load_from_carried,
)
from sim.case_study_data import (
    AHMADZAI_GHAG_2018,
    OZOVEHE_ET_AL_2017,
    ahmadzai_observations,
)
from sim.mmc_simulation import erlang_b, offered_load_for_blocking


class TestPublishedInputs(unittest.TestCase):
    def test_table_i_has_fifteen_exact_rows(self):
        observations = ahmadzai_observations()
        self.assertEqual(len(observations), 15)
        self.assertEqual(len(AHMADZAI_GHAG_2018["sectors"]), 5)
        first = observations[0]
        self.assertEqual(first["sector"], "KBLC025_2")
        self.assertEqual(first["row"], 1)
        self.assertEqual(first["net_tch"], 10)
        self.assertEqual(first["carried_erlang"], 7.672)
        self.assertEqual(first["congestion_percent"], 3.697)
        self.assertEqual(first["available_tchs"], 8.825)
        self.assertEqual(first["c_available"], 9)
        self.assertNotIn("pairing", first)
        net = {sector["sector"]: sector["net_tch"] for sector in AHMADZAI_GHAG_2018["sectors"]}
        self.assertEqual(net, {
            "KBLC025_2": 10,
            "PRNP007_1": 24,
            "SMGC004_3": 20,
            "TKRP018_2": 10,
            "KPSP004_1": 25,
        })

    def test_ozovehe_channel_count_is_marked_as_assumption(self):
        self.assertIsNone(OZOVEHE_ET_AL_2017["tch_published"])
        self.assertEqual(OZOVEHE_ET_AL_2017["c_assumption"], 14)
        self.assertIn("MY ASSUMPTION", OZOVEHE_ET_AL_2017["c_assumption_note"])
        self.assertEqual(OZOVEHE_ET_AL_2017["example_traffic_erlang"], 14.0)


class TestConversions(unittest.TestCase):
    def test_offered_load_hand_example(self):
        # 8 erlang carried at 20% blocking is 10 erlang offered.
        self.assertAlmostEqual(offered_load_from_carried(8.0, 0.20), 10.0)

    def test_subscriber_hand_example(self):
        # 10 erlang / 0.025 E per subscriber = 400 subscribers.
        self.assertAlmostEqual(subscribers_at_load(10.0, 0.025), 400.0)

    def test_years_hand_example(self):
        # Load at half of capacity, 10% per year: ln(2) / ln(1.1).
        expected = math.log(2.0) / math.log(1.1)
        self.assertAlmostEqual(years_to_upgrade(10.0, 5.0, 0.10), expected)

    def test_gos_capacity_matches_published_figures(self):
        # Huawei and Parhizgar: 14.04 E at c = 21 and 2.94 E at c = 7, to 0.01 E.
        self.assertAlmostEqual(erlang_b_capacity(21), 14.04, delta=0.01)
        self.assertAlmostEqual(erlang_b_capacity(7), 2.94, delta=0.01)
        self.assertAlmostEqual(offered_load_for_blocking(21, 0.02), 14.04, delta=0.01)
        self.assertAlmostEqual(offered_load_for_blocking(7, 0.02), 2.94, delta=0.01)

    def test_implied_holding_time(self):
        self.assertAlmostEqual(IMPLIED_HOLDING_TIME_S, 0.025 * 3600.0 / 1.64)
        self.assertAlmostEqual(IMPLIED_HOLDING_TIME_S, 54.9, delta=0.05)


class TestAhmadzaiAnalysis(unittest.TestCase):
    def test_three_channel_counts(self):
        row = analyse_ahmadzai()[0]
        self.assertEqual(row["c_fr"], 10)
        self.assertEqual(row["c_hr"], 20)
        self.assertEqual(row["c_av"], 9)
        self.assertAlmostEqual(row["offered_erlang"], 7.672 / (1.0 - 0.03697))

    def test_tkrp_exceeds_net_tch_but_not_rounded_available(self):
        rows = [row for row in analyse_ahmadzai() if row["sector"] == "TKRP018_2"]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertGreater(row["carried_erlang"], row["c_fr"])
            self.assertGreater(row["carried_erlang"], row["c_hr"])
            self.assertLess(row["carried_erlang"], row["c_av"])
            self.assertIn("impossible at full rate", row["flag"])
            self.assertIn("half rate", row["flag"])

    def test_table_iv_matches_authors(self):
        # Erlang B traffic at c = net TCH and the measured congestion.
        rows = analyse_ahmadzai()
        kblc = rows[0]
        tkrp = next(
            row for row in rows if row["sector"] == "TKRP018_2" and row["row"] == 1
        )
        self.assertAlmostEqual(kblc["table_iv_erlang_b_traffic"], 5.79, delta=0.05)
        self.assertAlmostEqual(tkrp["table_iv_erlang_b_traffic"], 9.788, delta=0.05)
        self.assertAlmostEqual(
            erlang_b_traffic_at_congestion(10, 0.03697),
            kblc["table_iv_erlang_b_traffic"],
        )

    def test_effective_channel_count_is_the_smallest(self):
        offered = 7.672 / (1.0 - 0.03697)
        congestion = 0.03697
        c_eff = effective_channel_count(offered, congestion)
        self.assertGreater(c_eff, 1)
        self.assertLessEqual(erlang_b(c_eff, offered), congestion)
        self.assertGreater(erlang_b(c_eff - 1, offered), congestion)

    def test_bracket_uses_full_and_half_rate_only(self):
        self.assertEqual(
            bracketed_by(0.05, 0.12, 0.01),
            "between full rate and half rate",
        )
        self.assertEqual(bracketed_by(0.20, 0.70, 0.41), "half rate")

    def test_tkrp_effective_count_exceeds_half_rate(self):
        rows = [row for row in analyse_ahmadzai() if row["sector"] == "TKRP018_2"]
        for row in rows:
            self.assertGreater(row["c_eff"], 2 * row["net_tch"])
            self.assertEqual(row["configuration_flag"], INCONSISTENT_HALF_RATE)
