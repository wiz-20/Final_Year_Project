"""
Step 5: Hold the calibration channel counts fixed and score the unseen test hours.

Channel counts are read from results/step3_calibration.csv. Only rows with
split == "test" are used. Empty hours (NaN) are skipped.

For every test hour, Erlang B is evaluated at that hour's voice_erl for both
counts (weekday busy-hour sizing and worst-hour sizing). voice_erl is again
taken as the offered load A.

One spot-check per site and sizing calls the same run_mmc_b_simulation path as
step 3, at the single busiest test hour: warmup_arrivals = 2,000,
max_arrivals = 50,000, and 20 replications.

Usage (from the project root, after step3_calibrate.py):
    .\\venv\\Scripts\\python.exe step5_validate.py
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from sim.literature_params import BLOCKING_TARGET
from sim.plot_style import LISTED, MODEL, THRESHOLD, apply_style
from sim.mmc_simulation import erlang_b
from sim.replication import theory_inside_ci
from step3_calibrate import SITE_ORDER, simulate_at_load

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_clean" / "voice_all_sites.csv"
STEP3_PATH = ROOT / "results" / "step3_calibration.csv"
SUMMARY_PATH = ROOT / "results" / "step5_validation.csv"
HOURLY_PATH = ROOT / "results" / "step5_hourly_blocking.csv"

# Clear of the step-3 seeds (91_000 plus four blocks of 20).
SEED_STEP5 = 93_000
SIZINGS = ("weekday_busy_hour", "worst_hour")

SUMMARY_COLUMNS = (
    "site",
    "sizing",
    "channels_c",
    "n_test_hours",
    "hours_above_2pct",
    "pct_hours_above_2pct",
    "worst_blocking",
    "worst_when",
    "highest_A_erl",
    "highest_when",
    "spot_erlangB",
    "spot_sim_mean",
    "spot_ci_low",
    "spot_ci_high",
    "spot_within_ci",
)

HOURLY_COLUMNS = (
    "datetime",
    "site",
    "voice_erl",
    "blocking_c_busy",
    "blocking_c_worst",
)


def load_channel_counts(path):
    """c from step 3, keyed by (site, load_type)."""
    counts = {}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            counts[(record["site"], record["load_type"])] = int(record["channels_c"])
    for site in SITE_ORDER:
        for sizing in SIZINGS:
            if (site, sizing) not in counts:
                raise ValueError(f"Step 3 has no channel count for {site} {sizing}.")
    return counts


def load_test_rows(path):
    """Test rows with a numeric voice_erl, in file order."""
    grouped = {name: [] for name in SITE_ORDER}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["split"] != "test":
                continue
            if record["voice_erl"].strip() == "":
                continue
            if record["site"] not in grouped:
                raise ValueError(f"Unexpected site {record['site']!r}.")
            record["voice_erl"] = float(record["voice_erl"])
            grouped[record["site"]].append(record)
    return grouped


def hourly_blocking(rows, c_busy, c_worst):
    """Erlang B at each test hour for the two fixed channel counts."""
    scored = []
    for record in rows:
        offered = record["voice_erl"]
        scored.append({
            "datetime": record["datetime"],
            "site": record["site"],
            "voice_erl": offered,
            "blocking_c_busy": erlang_b(c_busy, offered),
            "blocking_c_worst": erlang_b(c_worst, offered),
        })
    return scored


def summarise_sizing(site, sizing, c, scored, blocking_key, base_seed):
    """Hours above 2%, the worst hour, and a simulation at the busiest hour."""
    blocking = [(row[blocking_key], row) for row in scored]
    above = [row for value, row in blocking if value > BLOCKING_TARGET]
    worst_value, worst_row = max(blocking, key=lambda item: item[0])
    busiest = max(scored, key=lambda row: row["voice_erl"])

    print(
        f"Spot-check {site} {sizing}  c = {c}, "
        f"A = {busiest['voice_erl']:.4f} E at {busiest['datetime']} ...",
        flush=True,
    )
    interval, theory = simulate_at_load(busiest["voice_erl"], c, base_seed)
    n_hours = len(scored)
    return {
        "site": site,
        "sizing": sizing,
        "channels_c": c,
        "n_test_hours": n_hours,
        "hours_above_2pct": len(above),
        "pct_hours_above_2pct": f"{100.0 * len(above) / n_hours:.2f}",
        "worst_blocking": f"{worst_value:.6f}",
        "worst_when": worst_row["datetime"],
        "highest_A_erl": f"{busiest['voice_erl']:.5f}",
        "highest_when": busiest["datetime"],
        "spot_erlangB": f"{theory:.6f}",
        "spot_sim_mean": f"{interval['mean']:.6f}",
        "spot_ci_low": f"{interval['low']:.6f}",
        "spot_ci_high": f"{interval['high']:.6f}",
        "spot_within_ci": "yes" if theory_inside_ci(interval, theory) else "no",
        "spot_abs_diff": abs(interval["mean"] - theory),
    }


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_site(site, scored, c_busy, c_worst, path):
    """Test-period Erlang B for both channel counts, with the 2% line."""
    apply_style()
    times = [datetime.fromisoformat(row["datetime"]) for row in scored]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(
        times,
        [100.0 * row["blocking_c_busy"] for row in scored],
        color=LISTED,
        linewidth=1.0,
        label=f"c = {c_busy} (weekday busy hour)",
    )
    ax.plot(
        times,
        [100.0 * row["blocking_c_worst"] for row in scored],
        color=MODEL,
        linewidth=1.0,
        label=f"c = {c_worst} (worst hour)",
    )
    ax.axhline(
        100.0 * BLOCKING_TARGET,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="2% grade of service",
    )
    ax.set_ylabel("Blocking (%)")
    ax.set_xlabel("Test period")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(rows):
    header = (
        f"{'site':<12}{'sizing':<20}{'c':>4}{'hours':>7}{'above 2%':>10}"
        f"{'%':>8}{'worst B':>10}{'when':>22}{'max A':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['site']:<12}{row['sizing']:<20}{row['channels_c']:4d}"
            f"{row['n_test_hours']:7d}{row['hours_above_2pct']:10d}"
            f"{float(row['pct_hours_above_2pct']):8.2f}"
            f"{float(row['worst_blocking']):10.2%}"
            f"{row['worst_when']:>22}{float(row['highest_A_erl']):10.3f}"
        )


def print_spot_checks(rows):
    header = (
        f"{'site':<12}{'sizing':<20}{'when':<22}{'A_erl':>10}{'c':>4}"
        f"{'Erlang B':>12}{'sim mean':>12}{'95% CI':>24}{'|diff|':>10}{'in CI':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{float(row['spot_ci_low']):.4%} , {float(row['spot_ci_high']):.4%}]"
        print(
            f"{row['site']:<12}{row['sizing']:<20}{row['highest_when']:<22}"
            f"{float(row['highest_A_erl']):10.4f}{row['channels_c']:4d}"
            f"{float(row['spot_erlangB']):12.4%}"
            f"{float(row['spot_sim_mean']):12.4%}{ci:>24}"
            f"{row['spot_abs_diff']:10.5f}{row['spot_within_ci']:>7}"
        )


def main():
    counts = load_channel_counts(STEP3_PATH)
    grouped = load_test_rows(DATA_PATH)

    summary = []
    hourly = []
    seed = SEED_STEP5
    for site in SITE_ORDER:
        c_busy = counts[(site, "weekday_busy_hour")]
        c_worst = counts[(site, "worst_hour")]
        scored = hourly_blocking(grouped[site], c_busy, c_worst)
        hourly.extend(scored)
        for sizing, c, key in (
            ("weekday_busy_hour", c_busy, "blocking_c_busy"),
            ("worst_hour", c_worst, "blocking_c_worst"),
        ):
            summary.append(summarise_sizing(site, sizing, c, scored, key, seed))
            seed += 100
        figure = ROOT / "results" / f"step5_{site}.png"
        plot_site(site, scored, c_busy, c_worst, figure)
        print(f"Saved {figure}", flush=True)

    write_csv(HOURLY_PATH, HOURLY_COLUMNS, [
        {
            "datetime": row["datetime"],
            "site": row["site"],
            "voice_erl": f"{row['voice_erl']:.5f}",
            "blocking_c_busy": f"{row['blocking_c_busy']:.6f}",
            "blocking_c_worst": f"{row['blocking_c_worst']:.6f}",
        }
        for row in hourly
    ])
    write_csv(SUMMARY_PATH, SUMMARY_COLUMNS, summary)
    print()
    print_summary(summary)
    print()
    print_spot_checks(summary)
    print(f"\nSaved {SUMMARY_PATH}")
    print(f"Saved {HOURLY_PATH}")


if __name__ == "__main__":
    main()
