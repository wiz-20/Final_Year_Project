"""Published cell measurements used as case-study inputs.

Numbers below are copied from the cited papers. Nothing here is Vodacom data,
and nothing here is a simulated result.
"""

import math

# Ahmadzai & Ghag (2018), "A Study of GSM Voice Traffic Using Measurement
# Performance Technique", International Journal of Science and Research (IJSR)
# 7(5):1847-1849, p.1848 Table I and Table II. Afghanistan GSM network.
# Busy-hour measurements. The paper states that half-rate was disabled.
# Table I tuples are (TCH Traffic Erl, TCH Congestion Rate %, Available TCHs),
# in the order printed. They are not paired from a range.
AHMADZAI_GHAG_2018 = {
    "citation": (
        "Ahmadzai & Ghag (2018), A Study of GSM Voice Traffic Using "
        "Measurement Performance Technique, IJSR 7(5):1847-1849"
    ),
    "table_i": "Table I, p.1848",
    "table_ii": "Table II, p.1848, Net TCHs",
    "network": "Afghanistan GSM",
    "half_rate": "disabled",
    "sectors": (
        {
            "sector": "KBLC025_2",
            "trx": 2,
            "net_tch": 10,
            "table_i": (
                (7.672, 3.697, 8.825),
                (8.483, 2.094, 9.246),
                (8.354, 1.574, 9.186),
            ),
        },
        {
            "sector": "PRNP007_1",
            "trx": 4,
            "net_tch": 24,
            "table_i": (
                (23.586, 20.151, 23.694),
                (23.26, 9.307, 23.444),
                (23.119, 5.558, 23.458),
            ),
        },
        {
            "sector": "SMGC004_3",
            "trx": 3,
            "net_tch": 20,
            "table_i": (
                (19.592, 5.645, 19.762),
                (18.892, 4.253, 19.303),
                (17.433, 3.862, 19.418),
            ),
        },
        {
            "sector": "TKRP018_2",
            "trx": 2,
            "net_tch": 10,
            "table_i": (
                (25.497, 20.48, 25.767),
                (25.765, 17.184, 25.849),
                (25.478, 10.357, 25.676),
            ),
        },
        {
            "sector": "KPSP004_1",
            "trx": 4,
            "net_tch": 25,
            "table_i": (
                (21.52, 1.08, 24.217),
                (22.81, 2.47, 24.207),
                (23.74, 5.24, 24.556),
            ),
        },
    ),
}

# Ozovehe et al. (2017), "Busy hour traffic congestion analysis in mobile
# macrocells", Nigerian Journal of Technology 36(4). 742 cells, 262 BTS,
# November 2012 to September 2014. The example cell is 2 TRX with about
# 14 erlang busy-hour traffic. TCH count is not given.
OZOVEHE_ET_AL_2017 = {
    "citation": (
        "Ozovehe et al. (2017), Busy hour traffic congestion analysis in "
        "mobile macrocells, Nigerian Journal of Technology 36(4)"
    ),
    "n_cells": 742,
    "n_bts": 262,
    "period": "November 2012 - September 2014",
    "example_trx": 2,
    "example_traffic_erlang": 14.0,
    "example_traffic_note": "about 14 E busy-hour traffic",
    "tch_published": None,
    # MY ASSUMPTION. Not in the paper. 2 TRX = 16 timeslots minus signalling.
    "c_assumption": 14,
    "c_assumption_note": (
        "MY ASSUMPTION: c = 14 (2 TRX = 16 timeslots minus signalling). "
        "TCH count is not given by Ozovehe et al."
    ),
}

# Busy-hour traffic per subscriber.
# Parhizgar & Bahmani (2011), Australian Journal of Basic and Applied
# Sciences 5(11):448-468: 25 mE urban.
# Huawei GSM Capacity Planning guide: 25 mE urban, 20 mE suburban, and
# 1.64 busy-hour call attempts per subscriber.
PARHIZGAR_BAHMANI_2011 = (
    "Parhizgar & Bahmani (2011), Australian Journal of Basic and Applied "
    "Sciences 5(11):448-468"
)
HUAWEI_GSM_CAPACITY_PLANNING = "Huawei GSM Capacity Planning guide"

ERLANG_PER_SUBSCRIBER = {
    "urban": 0.025,  # 25 mE; Parhizgar & Bahmani (2011); Huawei guide
    "suburban": 0.020,  # 20 mE; Huawei GSM Capacity Planning guide
}
BUSY_HOUR_CALL_ATTEMPTS_PER_SUBSCRIBER = 1.64  # Huawei GSM Capacity Planning guide


def round_available_tchs(available_tchs):
    """Channel count c = round(Available TCHs) from Table I."""
    return int(math.floor(float(available_tchs) + 0.5))


def ahmadzai_observations():
    """One row per Table I busy hour. Row numbers start at 1 within each sector."""
    rows = []
    for sector in AHMADZAI_GHAG_2018["sectors"]:
        for index, (traffic, congestion_percent, available) in enumerate(sector["table_i"], start=1):
            rows.append({
                "source": AHMADZAI_GHAG_2018["citation"],
                "table": AHMADZAI_GHAG_2018["table_i"],
                "sector": sector["sector"],
                "row": index,
                "trx": sector["trx"],
                "net_tch": sector["net_tch"],
                "carried_erlang": traffic,
                "congestion_percent": congestion_percent,
                "congestion": congestion_percent / 100.0,
                "available_tchs": available,
                "c_available": round_available_tchs(available),
            })
    return rows
