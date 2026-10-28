"""
Step 7: Grow the calibration loads and see when the original channel count overloads.

Loads and channel counts are read from results/step3_calibration.csv. A grown
load is A_g = A * factor. Channels needed for A_g use Erlang B at 2%. Blocking
if the site keeps its original c is Erlang B(c, A_g).

Headroom is the largest offered load the original c can carry at 2% blocking,
divided by A, minus one, in percent.

Spot-checks call run_mmc_b_simulation through step3.simulate_at_load (50,000
measured arrivals, 2,000 warm-up arrivals, 20 replications) at factors 1.10
and 1.50, still on the original c.

Usage (from the project root, after step3_calibrate.py):
    .\\venv\\Scripts\\python.exe step7_growth.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.literature_params import BLOCKING_TARGET
from sim.plot_style import LISTED, MODEL, THRESHOLD, apply_style
from sim.mmc_simulation import erlang_b, offered_load_for_blocking
from sim.replication import theory_inside_ci
from step3_calibrate import SITE_ORDER, channels_for_gos, simulate_at_load

ROOT = Path(__file__).resolve().parent
STEP3_PATH = ROOT / "results" / "step3_calibration.csv"
GROWTH_PATH = ROOT / "results" / "step7_growth.csv"
HEADROOM_PATH = ROOT / "results" / "step7_headroom.csv"

GROWTH_FACTORS = (1.00, 1.10, 1.25, 1.50, 2.00)
SPOT_FACTORS = (1.10, 1.50)
# Clear of step 3 (91_000) and step 5 (93_000).
SEED_STEP7 = 95_000

GROWTH_COLUMNS = (
    "site",
    "load_type",
    "growth_factor",
    "A_erl",
    "A_g",
    "channels_c",
    "channels_needed",
    "blocking_at_c",
    "sim_blocking_mean",
    "sim_ci_low",
    "sim_ci_high",
    "within_ci",
)
HEADROOM_COLUMNS = (
    "site",
    "load_type",
    "A_erl",
    "channels_c",
    "max_load_erl",
    "headroom_pct",
)


def load_step3(path):
    """Calibration rows in site order, busy hour then worst hour."""
    found = {}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            found[(record["site"], record["load_type"])] = {
                "site": record["site"],
                "load_type": record["load_type"],
                "A_erl": float(record["A_erl"]),
                "channels_c": int(record["channels_c"]),
            }
    rows = []
    for site in SITE_ORDER:
        for load_type in ("weekday_busy_hour", "worst_hour"):
            key = (site, load_type)
            if key not in found:
                raise ValueError(f"Step 3 has no row for {site} {load_type}.")
            rows.append(found[key])
    return rows


def growth_row(record, factor, seed):
    """One grown load. Simulate only the spot-check factors."""
    offered = record["A_erl"] * factor
    c = record["channels_c"]
    row = {
        "site": record["site"],
        "load_type": record["load_type"],
        "growth_factor": f"{factor:.2f}",
        "A_erl": f"{record['A_erl']:.5f}",
        "A_g": f"{offered:.5f}",
        "channels_c": c,
        "channels_needed": channels_for_gos(offered),
        "blocking_at_c": f"{erlang_b(c, offered):.6f}",
        "sim_blocking_mean": "",
        "sim_ci_low": "",
        "sim_ci_high": "",
        "within_ci": "",
    }
    if factor not in SPOT_FACTORS:
        return row, seed
    print(
        f"Spot-check {record['site']} {record['load_type']}  "
        f"factor {factor:.2f}, A_g = {offered:.4f} E, c = {c} ...",
        flush=True,
    )
    interval, theory = simulate_at_load(offered, c, seed)
    row["sim_blocking_mean"] = f"{interval['mean']:.6f}"
    row["sim_ci_low"] = f"{interval['low']:.6f}"
    row["sim_ci_high"] = f"{interval['high']:.6f}"
    row["within_ci"] = "yes" if theory_inside_ci(interval, theory) else "no"
    row["abs_diff"] = abs(interval["mean"] - theory)
    return row, seed + 100


def headroom_row(record):
    """Largest load at 2% on the original c, and the spare fraction of A."""
    cap = offered_load_for_blocking(record["channels_c"], BLOCKING_TARGET)
    spare = cap / record["A_erl"] - 1.0
    return {
        "site": record["site"],
        "load_type": record["load_type"],
        "A_erl": f"{record['A_erl']:.5f}",
        "channels_c": record["channels_c"],
        "max_load_erl": f"{cap:.4f}",
        "headroom_pct": f"{100.0 * spare:.4f}",
    }


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_site(site, rows, path):
    """Blocking at the original c against the growth factor, both sizings."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colours = {"weekday_busy_hour": LISTED, "worst_hour": MODEL}
    labels = {
        "weekday_busy_hour": "weekday busy hour",
        "worst_hour": "worst hour",
    }
    for load_type, colour in colours.items():
        series = [row for row in rows if row["load_type"] == load_type]
        c = series[0]["channels_c"]
        ax.plot(
            [float(row["growth_factor"]) for row in series],
            [100.0 * float(row["blocking_at_c"]) for row in series],
            marker="o",
            color=colour,
            linewidth=1.5,
            label=f"{labels[load_type]} (c = {c})",
        )
    ax.axhline(
        100.0 * BLOCKING_TARGET,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="2% grade of service",
    )
    ax.set_xlabel("Growth factor")
    ax.set_ylabel("Erlang B blocking (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_headroom(rows):
    header = f"{'site':<12}{'load_type':<20}{'A':>10}{'c':>4}{'max load':>10}{'headroom':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['site']:<12}{row['load_type']:<20}{float(row['A_erl']):10.3f}"
            f"{row['channels_c']:4d}{float(row['max_load_erl']):10.2f}"
            f"{float(row['headroom_pct']):9.1f}%"
        )


def print_growth(rows):
    header = (
        f"{'site':<12}{'load_type':<20}{'factor':>8}{'A_g':>10}{'c':>4}"
        f"{'c needed':>10}{'B at c':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['site']:<12}{row['load_type']:<20}{float(row['growth_factor']):8.2f}"
            f"{float(row['A_g']):10.3f}{row['channels_c']:4d}"
            f"{row['channels_needed']:10d}{float(row['blocking_at_c']):10.2%}"
        )


def print_spot_checks(rows):
    header = (
        f"{'site':<12}{'load_type':<20}{'factor':>8}{'A_g':>10}{'c':>4}"
        f"{'Erlang B':>12}{'sim mean':>12}{'95% CI':>24}{'|diff|':>10}{'in CI':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["sim_blocking_mean"] == "":
            continue
        ci = f"[{float(row['sim_ci_low']):.4%} , {float(row['sim_ci_high']):.4%}]"
        print(
            f"{row['site']:<12}{row['load_type']:<20}{float(row['growth_factor']):8.2f}"
            f"{float(row['A_g']):10.3f}{row['channels_c']:4d}"
            f"{float(row['blocking_at_c']):12.2%}"
            f"{float(row['sim_blocking_mean']):12.2%}{ci:>24}"
            f"{row['abs_diff']:10.5f}{row['within_ci']:>7}"
        )


def main():
    base = load_step3(STEP3_PATH)
    headroom = [headroom_row(record) for record in base]
    growth = []
    seed = SEED_STEP7
    for record in base:
        for factor in GROWTH_FACTORS:
            row, seed = growth_row(record, factor, seed)
            growth.append(row)

    write_csv(GROWTH_PATH, GROWTH_COLUMNS, growth)
    write_csv(HEADROOM_PATH, HEADROOM_COLUMNS, headroom)
    for site in SITE_ORDER:
        figure = ROOT / "results" / f"step7_{site}.png"
        plot_site(site, [row for row in growth if row["site"] == site], figure)
        print(f"Saved {figure}", flush=True)

    print()
    print_headroom(headroom)
    print()
    print_growth(growth)
    print()
    print_spot_checks(growth)
    print(f"\nSaved {HEADROOM_PATH}")
    print(f"Saved {GROWTH_PATH}")


if __name__ == "__main__":
    main()
