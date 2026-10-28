"""Subscribers at a grade of service, and an illustrative time to upgrade.

Erlang-B capacity is an offered load in erlang. It does not depend on the
holding time. Subscriber counts use the published erlang-per-subscriber
figures. The annual growth rates are not taken from a source.
"""

import math

from sim.case_study_data import (
    AHMADZAI_GHAG_2018,
    BUSY_HOUR_CALL_ATTEMPTS_PER_SUBSCRIBER,
    ERLANG_PER_SUBSCRIBER,
    HUAWEI_GSM_CAPACITY_PLANNING,
    PARHIZGAR_BAHMANI_2011,
)
from sim.literature_params import (
    ACCEPTABLE_WAIT_S,
    BLOCKING_TARGET,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
    SERVER_COUNTS,
)
from sim.mmc_k_simulation import mmck_binding_arrivals
from sim.mmc_simulation import offered_load_for_blocking

# Not sourced. Illustrative only.
ILLUSTRATIVE_GROWTH_PER_YEAR = (0.05, 0.10, 0.20)
LOAD_FRACTIONS_OF_CAPACITY = (0.50, 0.70, 0.90)
QUEUE_EXTRA = 5

# 0.025 E/sub * 3600 s/h / 1.64 attempts. Urban traffic and the Huawei BHCA.
# Displayed as 54.9 s. Sweeps are not rerun at this holding time.
IMPLIED_HOLDING_TIME_S = (
    ERLANG_PER_SUBSCRIBER["urban"]
    * SECONDS_PER_HOUR
    / BUSY_HOUR_CALL_ATTEMPTS_PER_SUBSCRIBER
)


def planning_channel_counts():
    """c in {7, 8, 21, 45} plus the Ahmadzai & Ghag TCH counts."""
    published = {sector["net_tch"] for sector in AHMADZAI_GHAG_2018["sectors"]}
    design = set(SERVER_COUNTS) - {1}
    return tuple(sorted(design | published))


def subscribers_at_load(capacity_erlang, erlang_per_subscriber):
    """Subscribers whose busy-hour traffic sums to capacity_erlang."""
    if erlang_per_subscriber <= 0:
        raise ValueError("erlang_per_subscriber must be positive.")
    return capacity_erlang / erlang_per_subscriber


def years_to_upgrade(capacity, current_load, growth_per_year):
    """years = ln(capacity / current_load) / ln(1 + g).

    growth_per_year is an illustrative assumption, not a sourced forecast.
    A load already at or above capacity has zero years left.
    """
    if capacity <= 0 or current_load <= 0:
        raise ValueError("capacity and current_load must be positive.")
    if growth_per_year <= -1.0 or growth_per_year == 0.0:
        raise ValueError("growth_per_year must be non-zero and greater than -1.")
    if current_load >= capacity:
        return 0.0
    return math.log(capacity / current_load) / math.log(1.0 + growth_per_year)


def erlang_b_capacity(c, blocking=BLOCKING_TARGET):
    """Offered load (erlang) at which Erlang B equals the grade of service."""
    return offered_load_for_blocking(c, blocking)


def arrivals_per_hour(offered_erlang, mean_service_s):
    """Arrivals per hour at a fixed offered load. Changes with the holding time."""
    return offered_erlang * SECONDS_PER_HOUR / mean_service_s


def mmck_binding_erlang(
    c,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    queue_extra=QUEUE_EXTRA,
    wait_s=ACCEPTABLE_WAIT_S,
    blocking=BLOCKING_TARGET,
):
    """Offered erlang at the tighter of B <= blocking and accepted-call Wq <= wait_s.

    Reuses mmck_binding_arrivals. The wait limit is evaluated at mean_service_s,
    so this erlang can change if the holding time changes. The default is the
    study holding time of 180 s.
    """
    mu = 1.0 / mean_service_s
    arrivals_h = mmck_binding_arrivals(c, queue_extra, mu, wait_s, blocking)
    return arrivals_h * mean_service_s / SECONDS_PER_HOUR, arrivals_h


def capacity_rows(channel_counts=None, mean_service_s=PRIMARY_MEAN_SERVICE_S):
    counts = planning_channel_counts() if channel_counts is None else channel_counts
    rows = []
    for c in counts:
        offered = erlang_b_capacity(c)
        binding_erlang, binding_arrivals_h = mmck_binding_erlang(c, mean_service_s)
        rows.append({
            "c": c,
            "erlang_b_capacity": offered,
            "subscribers_urban": subscribers_at_load(offered, ERLANG_PER_SUBSCRIBER["urban"]),
            "subscribers_suburban": subscribers_at_load(
                offered, ERLANG_PER_SUBSCRIBER["suburban"]
            ),
            "mmck_binding_erlang": binding_erlang,
            "mmck_binding_arrivals_h": binding_arrivals_h,
            "mmck_subscribers_urban": subscribers_at_load(
                binding_erlang, ERLANG_PER_SUBSCRIBER["urban"]
            ),
            "mmck_subscribers_suburban": subscribers_at_load(
                binding_erlang, ERLANG_PER_SUBSCRIBER["suburban"]
            ),
            "arrivals_h_180s": arrivals_per_hour(offered, PRIMARY_MEAN_SERVICE_S),
            "arrivals_h_implied": arrivals_per_hour(offered, IMPLIED_HOLDING_TIME_S),
        })
    return rows


def upgrade_rows(
    fractions=LOAD_FRACTIONS_OF_CAPACITY,
    growth_rates=ILLUSTRATIVE_GROWTH_PER_YEAR,
):
    """Years until load reaches capacity. Depends only on the fraction and g."""
    rows = []
    for fraction in fractions:
        for growth in growth_rates:
            # capacity / current = 1 / fraction. Any positive capacity gives the same years.
            rows.append({
                "load_fraction": fraction,
                "growth_per_year": growth,
                "years": years_to_upgrade(1.0, fraction, growth),
            })
    return rows


def planning_markdown(channel_counts=None):
    rows = capacity_rows(channel_counts)
    lines = [
        "Subscribers use 0.025 E (urban) and 0.020 E (suburban) in the busy hour. "
        f"Urban 25 mE: {PARHIZGAR_BAHMANI_2011}; Huawei guide. "
        f"Suburban 20 mE and 1.64 BHCA: {HUAWEI_GSM_CAPACITY_PLANNING}.",
        "",
        "Erlang B capacity is the offered load at 2% blocking. "
        "M/M/c/K capacity is the binding offered load for K−c = 5 that satisfies "
        "both B ≤ 2% and accepted-call Wq ≤ 10 s, evaluated at 1/μ = 180 s.",
        "",
        "| c | Erlang B 2% (E) | Subscribers, 0.025 E | Subscribers, 0.020 E | "
        "M/M/c/K K−c=5 (E) | Subscribers, 0.025 E | Subscribers, 0.020 E |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['c']} | {row['erlang_b_capacity']:.2f} | "
            f"{row['subscribers_urban']:.0f} | {row['subscribers_suburban']:.0f} | "
            f"{row['mmck_binding_erlang']:.2f} | "
            f"{row['mmck_subscribers_urban']:.0f} | "
            f"{row['mmck_subscribers_suburban']:.0f} |"
        )
    lines.extend([
        "",
        f"Implied mean holding time from the urban 0.025 E and 1.64 BHCA: "
        f"0.025 × 3600 / 1.64 = {IMPLIED_HOLDING_TIME_S:.1f} s. "
        "Sweeps were not rerun at this holding time. Erlang B capacity in erlang, "
        "and therefore the subscriber counts, do not change with holding time. "
        "Only arrivals per hour change.",
        "",
        "| c | Arrivals/h at 2% capacity, 1/μ = 180 s | "
        f"Arrivals/h at 2% capacity, 1/μ = {IMPLIED_HOLDING_TIME_S:.1f} s |",
        "|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['c']} | {row['arrivals_h_180s']:.1f} | {row['arrivals_h_implied']:.1f} |"
        )
    lines.extend([
        "",
        "Time to upgrade. Current load is a fraction of capacity, and traffic grows "
        "as (1+g) per year. g is an illustrative assumption, not sourced. "
        "The result depends only on that fraction and on g, so it is the same for every c.",
        "",
        "| Current load | g = 5%/year | g = 10%/year | g = 20%/year |",
        "|---|---:|---:|---:|",
    ])
    upgrades = upgrade_rows()
    for fraction in LOAD_FRACTIONS_OF_CAPACITY:
        cells = [
            f"{item['years']:.2f}"
            for item in upgrades
            if item["load_fraction"] == fraction
        ]
        lines.append(f"| {fraction:.0%} of capacity | " + " | ".join(cells) + " |")
    return "\n".join(lines), rows
