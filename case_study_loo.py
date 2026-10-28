"""
Leave-one-out check on the Ahmadzai & Ghag busy hours.

Calibration. For every busy hour the offered load is estimated as
carried / (1 - B_measured), and a continuous effective channel count c*_eff is
found by interpolating ln B between the two integers that bracket the measured
congestion. Each hour is then predicted from the other two hours of its sector:
c_cal is the rounded mean of their c*_eff values.

Prediction. Two versions are reported for the held-out hour:

  - "meas-a" (as in the thesis draft): the held-out offered load is
    carried / (1 - B_measured). This uses the held-out hour's own measured
    congestion, so the hour is not fully unseen.
  - "strict": the held-out offered load is found from its carried traffic and
    c_cal only, by solving a * (1 - ErlangB(c_cal, a)) = carried. No measured
    congestion from the held-out hour is used. This is the headline result.

Baselines. The listed channel count (c = net TCH) is evaluated in both
versions, and a naive predictor (mean measured congestion of the other two
hours of the sector) shows what can be achieved without any queueing model.

The strict prediction is checked in the existing M/M/c/c engine (R = 10, 95%
Student-t interval). TKRP018_2 is reported on its own and is left out of the
mean absolute error of the four consistent sectors.

Usage (from the project root):
    .\\venv\\Scripts\\python.exe case_study_loo.py
    .\\venv\\Scripts\\python.exe case_study_loo.py --no-sim     (Erlang B only, fast)
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.case_study import offered_load_from_carried, simulate_blocking
from sim.case_study_data import ahmadzai_observations
from sim.mmc_simulation import erlang_b
from sim.plot_style import RESULTS_FIGURES, apply_style, save_figure

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "case_study_loo.csv"
FIGURE_PATH = RESULTS_FIGURES / "case_study_parity.png"
FIGURE_PDF = RESULTS_FIGURES / "case_study_parity.pdf"

EXCLUDED_SECTOR = "TKRP018_2"
# Clear of the case-study seeds (81_000) and the later step-3/5/7/8 blocks.
SEED_LOO = 101_000

COLUMNS = (
    "sector",
    "hour",
    "carried_erl",
    "B_meas",
    "net_tch",
    "offered_a_meas",
    "c_eff_continuous",
    "c_cal",
    "offered_a_strict",
    "erlangB_loo_meas_a",
    "erlangB_loo_strict",
    "sim_mean",
    "sim_ci_low",
    "sim_ci_high",
    "offered_a_strict_listed",
    "erlangB_listed_meas_a",
    "erlangB_listed_strict",
    "naive",
    "abs_err_loo_meas_a_pp",
    "abs_err_loo_strict_pp",
    "abs_err_listed_meas_a_pp",
    "abs_err_listed_strict_pp",
    "abs_err_naive_pp",
    "in_mae",
)

# Error columns summarised at the end: (key, label).
ERROR_KEYS = (
    ("abs_err_loo_strict_pp", "Leave-one-out, strict (headline)"),
    ("abs_err_loo_meas_a_pp", "Leave-one-out, offered load from held-out B"),
    ("abs_err_naive_pp", "Naive: mean of the other two hours"),
    ("abs_err_listed_strict_pp", "Listed TCH, strict"),
    ("abs_err_listed_meas_a_pp", "Listed TCH, offered load from measured B"),
)


# ---------------------------------------------------------------------------
# Queueing helpers
# ---------------------------------------------------------------------------

def continuous_c_eff(offered_a, blocking):
    """Channel count with Erlang B interpolated in log(B) through the measurement.

    c is the smallest integer with Erlang B(c, a) <= B_measured. The continuous
    value lies between c-1 and c.
    """
    c = 1
    while erlang_b(c, offered_a) > blocking:
        c += 1
        if c > 5000:
            raise RuntimeError("No channel count met the measured congestion.")
    if c == 1:
        return 1.0
    b_at_c = erlang_b(c, offered_a)
    b_at_lower = erlang_b(c - 1, offered_a)
    fraction = (math.log(blocking) - math.log(b_at_lower)) / (
        math.log(b_at_c) - math.log(b_at_lower)
    )
    return (c - 1) + fraction


def offered_from_carried_at_c(carried, c, tol=1e-10, max_iter=200):
    """Offered load a with a * (1 - ErlangB(c, a)) = carried.

    Carried traffic of an Erlang loss system rises monotonically with a and
    approaches c, so a solution exists only when carried < c. Returns None
    otherwise (the listed configuration cannot carry the measured traffic).
    Uses no measured congestion.
    """
    if carried >= c:
        return None
    lo = carried
    hi = carried
    while hi * (1.0 - erlang_b(c, hi)) < carried:
        hi *= 2.0
        if hi > 1e7:
            return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid * (1.0 - erlang_b(c, mid)) < carried:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _erlang_b_or_nan(c, offered):
    return float("nan") if offered is None else erlang_b(c, offered)


def _abs_err_pp(predicted, measured):
    if predicted is None or math.isnan(predicted):
        return float("nan")
    return abs(predicted - measured) * 100.0


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------

def leave_one_out_rows(observations):
    """One record per busy hour, with c_cal from the other hours of the sector."""
    grouped = {}
    prepared = []
    for obs in observations:
        offered_meas = offered_load_from_carried(obs["carried_erlang"], obs["congestion"])
        record = {
            "sector": obs["sector"],
            "hour": obs["row"],
            "carried_erl": obs["carried_erlang"],
            "B_meas": obs["congestion"],
            "net_tch": obs["net_tch"],
            "offered_a_meas": offered_meas,
            "c_eff_continuous": continuous_c_eff(offered_meas, obs["congestion"]),
        }
        prepared.append(record)
        grouped.setdefault(obs["sector"], []).append(record)

    for sector, hours in grouped.items():
        for held in hours:
            others = [hour for hour in hours if hour is not held]

            # Calibration uses only the other hours.
            c_cal = int(round(sum(h["c_eff_continuous"] for h in others) / len(others)))
            held["c_cal"] = c_cal

            # Prediction at c_cal.
            held["offered_a_strict"] = offered_from_carried_at_c(held["carried_erl"], c_cal)
            held["erlangB_loo_strict"] = _erlang_b_or_nan(c_cal, held["offered_a_strict"])
            held["erlangB_loo_meas_a"] = erlang_b(c_cal, held["offered_a_meas"])

            # Listed channel count.
            net = held["net_tch"]
            held["offered_a_strict_listed"] = offered_from_carried_at_c(held["carried_erl"], net)
            held["erlangB_listed_strict"] = _erlang_b_or_nan(net, held["offered_a_strict_listed"])
            held["erlangB_listed_meas_a"] = erlang_b(net, held["offered_a_meas"])

            # Naive baseline: no queueing model at all.
            held["naive"] = sum(h["B_meas"] for h in others) / len(others)

            b = held["B_meas"]
            held["abs_err_loo_strict_pp"] = _abs_err_pp(held["erlangB_loo_strict"], b)
            held["abs_err_loo_meas_a_pp"] = _abs_err_pp(held["erlangB_loo_meas_a"], b)
            held["abs_err_listed_strict_pp"] = _abs_err_pp(held["erlangB_listed_strict"], b)
            held["abs_err_listed_meas_a_pp"] = _abs_err_pp(held["erlangB_listed_meas_a"], b)
            held["abs_err_naive_pp"] = _abs_err_pp(held["naive"], b)
            held["in_mae"] = sector != EXCLUDED_SECTOR

            held["sim_mean"] = float("nan")
            held["sim_ci_low"] = float("nan")
            held["sim_ci_high"] = float("nan")
    return prepared


def attach_simulation(rows):
    """Ten replications of the strict prediction at c_cal."""
    for index, row in enumerate(rows):
        if row["offered_a_strict"] is None:
            print(f"Skipping {row['sector']} hour {row['hour']}: carried >= c_cal.")
            continue
        seed = SEED_LOO + 100 * index
        print(
            f"Simulating {row['sector']} hour {row['hour']}  "
            f"c_cal = {row['c_cal']}, a_strict = {row['offered_a_strict']:.3f} ...",
            flush=True,
        )
        interval, _plan = simulate_blocking(row["offered_a_strict"], row["c_cal"], seed)
        row["sim_mean"] = interval["mean"]
        row["sim_ci_low"] = interval["low"]
        row["sim_ci_high"] = interval["high"]
    return rows


def mean_absolute_error(rows, key, included):
    """MAE in pp over rows with in_mae == included. NaN rows are skipped and counted."""
    chosen = [row for row in rows if row["in_mae"] is included]
    values = [row[key] for row in chosen if not math.isnan(row[key])]
    if not values:
        return float("nan"), 0, len(chosen)
    return sum(values) / len(values), len(values), len(chosen)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value, digits):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def csv_row(row):
    out = {}
    for key in COLUMNS:
        value = row[key]
        if key in ("sector", "hour", "c_cal", "net_tch"):
            out[key] = value
        elif key == "in_mae":
            out[key] = "yes" if value else "no"
        elif key.startswith("abs_err"):
            out[key] = _fmt(value, 3)
        elif key.startswith("offered") or key in ("c_eff_continuous", "carried_erl"):
            out[key] = _fmt(value, 4)
        else:
            out[key] = _fmt(value, 6)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(csv_row(row) for row in rows)


def _percent(fraction):
    return 100.0 * fraction


def _finite(rows, key):
    return [row for row in rows if not math.isnan(row[key])]


def plot_parity(rows, simulated):
    """Measured congestion against predicted blocking. No title: the caption is."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    grid = np.logspace(math.log10(0.5), math.log10(100), 300)
    ax.plot(grid, grid, color="0.45", linestyle="--", linewidth=1.0, label="y = x", zorder=1)
    ax.axhline(2.0, color="0.35", linestyle=":", linewidth=0.8, zorder=1)
    ax.axvline(2.0, color="0.35", linestyle=":", linewidth=0.8, label="2%", zorder=1)

    consistent = [row for row in rows if row["in_mae"]]
    excluded = [row for row in rows if not row["in_mae"]]

    listed = _finite(consistent, "erlangB_listed_strict")
    ax.scatter(
        [_percent(row["B_meas"]) for row in listed],
        [_percent(row["erlangB_listed_strict"]) for row in listed],
        s=36,
        facecolors="none",
        edgecolors="#1f4e79",
        linewidths=1.2,
        marker="o",
        label="Listed TCH",
        zorder=3,
    )

    loo = _finite(consistent, "erlangB_loo_strict")
    loo_x = [_percent(row["B_meas"]) for row in loo]
    loo_y = [_percent(row["erlangB_loo_strict"]) for row in loo]
    if simulated:
        # Bars span the simulation 95% interval around the Erlang B marker.
        yerr = [
            [max(0.0, y - _percent(row["sim_ci_low"])) for y, row in zip(loo_y, loo)],
            [max(0.0, _percent(row["sim_ci_high"]) - y) for y, row in zip(loo_y, loo)],
        ]
    else:
        yerr = None
    ax.errorbar(
        loo_x,
        loo_y,
        yerr=yerr,
        fmt="o",
        color="#c45911",
        markersize=5,
        capsize=2,
        elinewidth=0.8,
        label="Leave-one-out (c calibrated on other 2 hours)",
        zorder=4,
    )

    ex_loo = _finite(excluded, "erlangB_loo_strict")
    ax.scatter(
        [_percent(row["B_meas"]) for row in ex_loo],
        [_percent(row["erlangB_loo_strict"]) for row in ex_loo],
        s=42,
        color="black",
        marker="x",
        linewidths=1.2,
        label="TKRP018_2, leave-one-out",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 100)
    ax.set_ylim(0.5, 100)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    labels = ["0.5", "1", "2", "5", "10", "20", "50", "100"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Measured congestion (%)")
    ax.set_ylabel("Predicted blocking (%)")
    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=8.5,
        borderaxespad=0.2,
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=1.15,
        handletextpad=0.35,
    )
    fig.tight_layout()
    save_figure(fig, "case_study_parity")
    plt.close(fig)


def print_table(rows):
    header = (
        f"{'sector':<12}{'hr':>3}{'c_cal':>6}{'c_list':>7}"
        f"{'B meas':>8}{'strict':>8}{'meas-a':>8}{'naive':>8}{'listed':>12}"
        f"{'sim strict [95% CI]':>24}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if math.isnan(row["sim_mean"]):
            sim = "not simulated"
        else:
            sim = (
                f"{_percent(row['sim_mean']):.2f} "
                f"[{_percent(row['sim_ci_low']):.2f}, {_percent(row['sim_ci_high']):.2f}]"
            )
        listed = row["erlangB_listed_strict"]
        listed_txt = "impossible" if math.isnan(listed) else f"{_percent(listed):.2f}"
        print(
            f"{row['sector']:<12}{row['hour']:3d}{row['c_cal']:6d}{row['net_tch']:7d}"
            f"{_percent(row['B_meas']):8.2f}{_percent(row['erlangB_loo_strict']):8.2f}"
            f"{_percent(row['erlangB_loo_meas_a']):8.2f}{_percent(row['naive']):8.2f}"
            f"{listed_txt:>12}{sim:>24}"
        )
    print("All values in %. 'listed' uses the strict offered load at c = net TCH;")
    print("'impossible' means the measured carried traffic exceeds the listed TCH.")


def print_mae(rows):
    print()
    print("Mean absolute error (percentage points)")
    for label, included in (("4 consistent sectors", True), (EXCLUDED_SECTOR, False)):
        print(f"  {label}:")
        for key, name in ERROR_KEYS:
            mae, n_used, n_total = mean_absolute_error(rows, key, included)
            if n_used == 0:
                print(f"    {name:<46} not defined (carried > listed TCH)")
                continue
            note = "" if n_used == n_total else f"  ({n_used} of {n_total} hours)"
            print(f"    {name:<46} {mae:6.2f}{note}")


def check_simulation(rows):
    """Report whether Erlang B lies inside each simulation interval."""
    simulated = [row for row in rows if not math.isnan(row["sim_mean"])]
    if not simulated:
        return
    inside = sum(
        1 for row in simulated
        if row["sim_ci_low"] <= row["erlangB_loo_strict"] <= row["sim_ci_high"]
    )
    print(f"\nErlang B inside the simulation 95% CI at {inside} of {len(simulated)} points.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Skip the M/M/c/c replications (Erlang B results only).",
    )
    args = parser.parse_args()

    rows = leave_one_out_rows(ahmadzai_observations())
    if not args.no_sim:
        rows = attach_simulation(rows)

    write_csv(OUT_PATH, rows)
    plot_parity(rows, simulated=not args.no_sim)
    print()
    print_table(rows)
    print_mae(rows)
    check_simulation(rows)
    print(f"\nSaved {OUT_PATH}")
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    main()