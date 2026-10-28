"""RJIO outgoing to Idea incoming POI, Nandhini and Sivaraj (IJERT 2017).

Busy-hour traffic is carried traffic. Offered load is carried / (1 - B),
as in the GSM case study. c = 31 traffic timeslots per E1; c = 30 * E1 is
a sensitivity column. Erlang B and the M/M/c/c engine are imported, not copied.

The offered load is also estimated from carried traffic alone, by solving
a * (1 - ErlangB(c, a)) = carried at c = 31 * E1. This uses no measured
congestion, so it shows what the model predicts without that input
(column erlangB_pct_strict_31).

The spot-check clock is in mean holding times: discard the first 10, then
measure the next 50. Arrival-count stopping is not used, because c is large.
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from sim.case_study import offered_load_from_carried
from sim.literature_params import PRIMARY_MEAN_SERVICE_S
from sim.mmc_b_simulation import run_mmc_b_simulation
from sim.mmc_simulation import erlang_b, offered_load_for_blocking
from sim.paths import PROJECT_ROOT
from sim.plot_style import (
    LISTED,
    MEASURED,
    MODEL,
    RESULTS_FIGURES,
    THRESHOLD,
    TWO_PANEL,
    apply_style,
    save_figure,
)
from sim.replication import confidence_interval

DATA_PATH = PROJECT_ROOT / "data" / "trunk_case_study.csv"
OUT_PATH = PROJECT_ROOT / "results" / "trunk_case_study.csv"
FIGURE_PATH = RESULTS_FIGURES / "trunk_case_study.png"
FIGURE_PDF = RESULTS_FIGURES / "trunk_case_study.pdf"

TIMESLOTS = (31, 30)
GOS = 0.002
HOLDING_S = PRIMARY_MEAN_SERVICE_S
WARMUP_HOLDINGS = 10
MEASUREMENT_HOLDINGS = 50
SEED_TRUNK = 111_000
REPLICATIONS = 10
REPLICATIONS_SHORT = 5
# One date, all replications. Above this, that date uses the shorter run.
TIME_BUDGET_S = 5 * 60
SPOT_DATES = ("12-Oct-16", "24-Oct-16", "9-Jan-17", "16-Jan-17")

FIELDNAMES = [
    "date",
    "e1",
    "busy_hour_erl",
    "design_erl",
    "congestion_pct",
    "c_31",
    "c_30",
    "carried_per_c_31",
    "carried_per_c_30",
    "offered_a",
    "erlangB_pct_31",
    "erlangB_pct_30",
    "offered_a_strict",
    "erlangB_pct_strict_31",
    "abs_err_pp_strict_31",
    "abs_err_pp_31",
    "abs_err_pp_30",
    "capacity_0p2_31",
    "capacity_0p2_30",
    "design_dev_pct_31",
    "design_dev_pct_30",
    "sim_reps",
    "sim_mean_pct",
    "sim_ci_low_pct",
    "sim_ci_high_pct",
]


def load_observations(path=DATA_PATH):
    rows = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append({
                "date": record["date"],
                "e1": int(record["e1"]),
                "busy_hour_erl": float(record["busy_hour_erl"]),
                "design_erl": float(record["design_erl"]),
                "congestion_pct": float(record["congestion_pct"]),
            })
    return rows


def _channels(e1, timeslots):
    return timeslots * e1


def offered_from_carried_at_c(carried, c, max_iter=100):
    """Offered load a with a * (1 - ErlangB(c, a)) = carried. Uses no measured
    congestion. Returns None when carried >= c (no solution for a loss system).
    """
    if carried >= c:
        return None
    lo = carried
    hi = carried
    while hi * (1.0 - erlang_b(c, hi)) < carried:
        hi *= 1.5
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid * (1.0 - erlang_b(c, mid)) < carried:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyse_row(obs):
    """Closed-form columns for one busy hour. No SimPy."""
    carried = obs["busy_hour_erl"]
    measured = obs["congestion_pct"]
    blocking = measured / 100.0
    offered = offered_load_from_carried(carried, blocking)
    out = dict(obs)
    out["offered_a"] = offered
    for timeslots in TIMESLOTS:
        c = _channels(obs["e1"], timeslots)
        predicted = 100.0 * erlang_b(c, offered)
        capacity = offered_load_for_blocking(c, GOS)
        out[f"c_{timeslots}"] = c
        out[f"carried_per_c_{timeslots}"] = carried / c
        out[f"erlangB_pct_{timeslots}"] = predicted
        out[f"abs_err_pp_{timeslots}"] = abs(predicted - measured)
        out[f"capacity_0p2_{timeslots}"] = capacity
        out[f"design_dev_pct_{timeslots}"] = (capacity - obs["design_erl"]) / obs["design_erl"] * 100.0
    c31 = out["c_31"]
    a_strict = offered_from_carried_at_c(carried, c31)
    if a_strict is None:
        out["offered_a_strict"] = ""
        out["erlangB_pct_strict_31"] = ""
        out["abs_err_pp_strict_31"] = ""
    else:
        strict = 100.0 * erlang_b(c31, a_strict)
        out["offered_a_strict"] = a_strict
        out["erlangB_pct_strict_31"] = strict
        out["abs_err_pp_strict_31"] = abs(strict - measured)
    out["sim_reps"] = ""
    out["sim_mean_pct"] = ""
    out["sim_ci_low_pct"] = ""
    out["sim_ci_high_pct"] = ""
    return out


def _clock_window():
    warmup = WARMUP_HOLDINGS * HOLDING_S
    measurement = warmup + MEASUREMENT_HOLDINGS * HOLDING_S
    return warmup, measurement


def spot_check(offered_a, c, base_seed):
    """R replications on the holding-time clock. Shortens to R = 5 if R = 10
    would run past the time budget. Returns the B interval, R, and the first-run seconds.
    """
    mu = 1.0 / HOLDING_S
    lam = offered_a * mu
    warmup, measurement = _clock_window()

    def once(seed):
        return run_mmc_b_simulation(
            lam, mu, c, measurement, warmup, random_seed=seed,
        )

    started = time.perf_counter()
    first = once(base_seed)
    first_s = time.perf_counter() - started
    n_reps = REPLICATIONS if first_s * REPLICATIONS <= TIME_BUDGET_S else REPLICATIONS_SHORT
    samples = [first["B"]]
    for offset in range(1, n_reps):
        samples.append(once(base_seed + offset)["B"])
    return confidence_interval(samples), n_reps, first_s


def attach_spot_checks(rows):
    """Simulate the four congested dates at c = 31 * E1. Other rows are unchanged."""
    for index, row in enumerate(rows):
        if row["date"] not in SPOT_DATES:
            continue
        print(
            f"Simulating {row['date']}: c = {row['c_31']}, "
            f"a = {row['offered_a']:.2f} erlang ...",
            flush=True,
        )
        interval, n_reps, first_s = spot_check(
            row["offered_a"], row["c_31"], SEED_TRUNK + 100 * index,
        )
        row["sim_reps"] = n_reps
        row["sim_mean_pct"] = 100.0 * interval["mean"]
        row["sim_ci_low_pct"] = 100.0 * interval["low"]
        row["sim_ci_high_pct"] = 100.0 * interval["high"]
        note = ""
        if n_reps != REPLICATIONS:
            note = f" (first run {first_s:.0f} s, so R = {n_reps} to stay near 5 min)"
        print(
            f"  Erlang B {row['erlangB_pct_31']:.2f}%  "
            f"sim {row['sim_mean_pct']:.2f}%  "
            f"95% CI [{row['sim_ci_low_pct']:.2f}, {row['sim_ci_high_pct']:.2f}]  "
            f"R = {n_reps}{note}",
            flush=True,
        )
    return rows


def write_csv(rows, path=OUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDNAMES})


def _percent_or_blank(value):
    if value == "":
        return None
    return float(value)


def _date_label(text):
    """12-Oct-16 -> 12 Oct 16."""
    day, month, year = text.split("-")
    return f"{int(day)} {month} {year}"


def plot_trunk(rows, png_path=FIGURE_PATH, pdf_path=FIGURE_PDF):
    """Carried/c above, measured and predicted blocking below. No title."""
    apply_style()
    dates = [_date_label(row["date"]) for row in rows]
    x = list(range(len(rows)))
    fig, (ax_load, ax_block) = plt.subplots(2, 1, figsize=TWO_PANEL, sharex=True)

    ax_load.plot(
        x,
        [row["carried_per_c_31"] for row in rows],
        color=LISTED,
        marker="o",
        markersize=4.5,
        linewidth=1.0,
        label="Carried / c",
    )
    ax_load.axhline(1.0, color=THRESHOLD, linestyle="--", linewidth=1.0, label="1.0")
    ax_load.set_ylabel("Carried traffic / c")
    ax_load.set_ylim(0.55, 1.08)
    ax_load.legend(loc="lower right", frameon=True)
    ax_load.grid(False)

    measured = [row["congestion_pct"] for row in rows]
    predicted = [row["erlangB_pct_31"] for row in rows]
    ax_block.plot(
        x, predicted,
        linestyle="none",
        marker="o",
        markersize=6.5,
        markerfacecolor="none",
        markeredgecolor=MODEL,
        markeredgewidth=1.2,
        label="Erlang B",
        zorder=3,
    )
    ax_block.plot(
        x, measured,
        linestyle="none",
        marker="o",
        markersize=4.5,
        color=MEASURED,
        label="Measured",
        zorder=4,
    )
    sim_x, sim_y, yerr_lo, yerr_hi = [], [], [], []
    for index, row in enumerate(rows):
        mean = _percent_or_blank(row["sim_mean_pct"])
        low = _percent_or_blank(row["sim_ci_low_pct"])
        high = _percent_or_blank(row["sim_ci_high_pct"])
        if mean is None:
            continue
        sim_x.append(index)
        sim_y.append(mean)
        yerr_lo.append(mean - low)
        yerr_hi.append(high - mean)
    if sim_x:
        ax_block.errorbar(
            sim_x, sim_y,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor=MEASURED,
            elinewidth=0.8,
            capsize=2.5,
            label="Simulation 95% CI",
            zorder=5,
        )
    ax_block.axhline(2.0, color=THRESHOLD, linestyle="--", linewidth=1.0, label="2%")
    ax_block.set_ylabel("Blocking (%)")
    ax_block.set_xlabel("Date")
    ax_block.set_xticks(x)
    ax_block.set_xticklabels(dates, rotation=45, ha="right")
    # Sits above the Nov–Dec hours. Those points are near zero, 24 Oct is
    # to the left, and 9 Jan is to the right.
    ax_block.legend(
        loc="center",
        bbox_to_anchor=(0.42, 0.52),
        frameon=True,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.35,
    )
    fig.tight_layout()
    save_figure(fig, "trunk_case_study")
    plt.close(fig)


def _pct_text(value, width):
    if value == "":
        return f"{'n/a':>{width}}"
    return f"{value:{width}.1f}"


def print_table(rows):
    header = (
        f"{'date':<12}{'c':>7}{'car/c':>7}{'a':>10}{'B pred':>8}{'B strict':>9}{'B meas':>8}"
        f"{'|err|':>7}{'cap 0.2%':>10}{'design':>8}{'dev %':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['date']:<12}{row['c_31']:7d}{row['carried_per_c_31']:7.2f}"
            f"{row['offered_a']:10.1f}{row['erlangB_pct_31']:8.1f}"
            f"{_pct_text(row['erlangB_pct_strict_31'], 9)}{row['congestion_pct']:8.0f}"
            f"{row['abs_err_pp_31']:7.1f}{row['capacity_0p2_31']:10.1f}"
            f"{row['design_erl']:8.0f}{row['design_dev_pct_31']:8.2f}"
        )
    print("B pred: offered load from measured congestion, a = carried / (1 - B_meas).")
    print("B strict: offered load from carried traffic only (no measured congestion).")
    full = [row for row in rows if row["carried_per_c_31"] >= 0.95 and row["congestion_pct"] > 0]
    if full:
        strict_vals = [row["erlangB_pct_strict_31"] for row in full if row["erlangB_pct_strict_31"] != ""]
        if strict_vals:
            print(
                f"Full-link dates (carried/c >= 0.95, n = {len(full)}): "
                f"B strict ranges {min(strict_vals):.2f} to {max(strict_vals):.2f}%, "
                f"measured {min(r['congestion_pct'] for r in full):.0f} to "
                f"{max(r['congestion_pct'] for r in full):.0f}%."
            )
    print()
    print("Sensitivity c = 30 x E1")
    print(f"{'date':<12}{'c':>7}{'car/c':>7}{'B pred':>8}{'cap 0.2%':>10}{'dev %':>8}")
    for row in rows:
        print(
            f"{row['date']:<12}{row['c_30']:7d}{row['carried_per_c_30']:7.2f}"
            f"{row['erlangB_pct_30']:8.1f}{row['capacity_0p2_30']:10.1f}"
            f"{row['design_dev_pct_30']:8.2f}"
        )


def main():
    rows = [analyse_row(obs) for obs in load_observations()]
    print_table(rows)
    rows = attach_spot_checks(rows)
    write_csv(rows)
    plot_trunk(rows)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()