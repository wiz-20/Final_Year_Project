"""
Step 3: Size each calibration busy hour and check it in the M/M/c/c simulator.

Only rows with split == "calibration" are used. Empty hours (NaN) are skipped.
voice_erl is the measured carried traffic. This step takes that value as the
offered load A (erlang) and sets lambda = A / h, with h = 180 s, the holding
time already used in the model. Calls per hour are A * 3600 / 180 = 20 * A.

For each site the weekday busy hour is the worked load, and the single worst
calibration hour is the second load. Hoerskool is the worked example; Square
repeats the same steps.

Erlang B and run_mmc_b_simulation are imported. Each replication passes
warmup_arrivals=2,000 and max_arrivals=50,000, so the first 2,000 calls still
take a channel and are left out of the blocking ratio, and the next 50,000
are the measurement. There are 20 replications. A 95% Student-t interval is
expected to miss the true blocking about 1 time in 20.

Usage (from the project root):
    .\\venv\\Scripts\\python.exe step3_calibrate.py
"""

import csv
from pathlib import Path

from sim.literature_params import (
    BLOCKING_TARGET,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
)
from sim.mmc_b_simulation import run_mmc_b_simulation
from sim.mmc_simulation import erlang_b
from sim.replication import run_replications, theory_inside_ci

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_clean" / "voice_all_sites.csv"
OUT_PATH = ROOT / "results" / "step3_calibration.csv"

# Hoerskool first: it is the worked example. Square is the repeat.
SITE_ORDER = ("Hoerskool", "Square")
# Clear of the validation seeds (11_000-61_000) and the case-study block (81_000).
SEED_STEP3 = 91_000
N_WARMUP_ARRIVALS = 2_000
N_MEASURED_ARRIVALS = 50_000
N_REPLICATIONS = 20

COLUMNS = (
    "site",
    "load_type",
    "when",
    "A_erl",
    "lambda_per_hour",
    "channels_c",
    "erlangB_blocking",
    "sim_blocking_mean",
    "sim_ci_low",
    "sim_ci_high",
    "within_ci",
)


def load_calibration(path):
    """Calibration rows with a numeric voice_erl. NaN cells are blank in the CSV."""
    rows = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["split"] != "calibration":
                continue
            if record["voice_erl"].strip() == "":
                continue
            record["voice_erl"] = float(record["voice_erl"])
            record["hour"] = int(record["hour"])
            record["is_weekend"] = record["is_weekend"] == "True"
            rows.append(record)
    return rows


def weekday_busy_hour(rows):
    """Clock hour with the highest mean voice_erl on weekdays. Returns (hour, A)."""
    totals = {}
    counts = {}
    for record in rows:
        if record["is_weekend"]:
            continue
        hour = record["hour"]
        totals[hour] = totals.get(hour, 0.0) + record["voice_erl"]
        counts[hour] = counts.get(hour, 0) + 1
    if not counts:
        raise ValueError("No weekday calibration rows.")
    means = {hour: totals[hour] / counts[hour] for hour in counts}
    hour = max(means, key=means.get)
    return hour, means[hour]


def worst_hour(rows):
    """Single calibration row with the highest voice_erl."""
    return max(rows, key=lambda record: record["voice_erl"])


def channels_for_gos(offered_a, blocking=BLOCKING_TARGET):
    """Smallest integer c with Erlang B(c, offered_a) <= blocking."""
    c = 1
    while erlang_b(c, offered_a) > blocking:
        c += 1
        if c > 5000:
            raise RuntimeError("No channel count met the 2% grade of service.")
    return c


def simulate_at_load(offered_a, c, base_seed):
    """Twenty M/M/c/c replications at lambda = A / h. Returns the blocking interval."""
    holding_s = PRIMARY_MEAN_SERVICE_S
    mu = 1.0 / holding_s
    lam = offered_a / holding_s

    def once(seed):
        result = run_mmc_b_simulation(
            lam,
            mu,
            c,
            0.0,
            0.0,
            random_seed=seed,
            max_arrivals=N_MEASURED_ARRIVALS,
            warmup_arrivals=N_WARMUP_ARRIVALS,
        )
        if result["post_warmup_arrivals"] != N_MEASURED_ARRIVALS:
            raise RuntimeError(
                f"Expected {N_MEASURED_ARRIVALS} measured arrivals, "
                f"got {result['post_warmup_arrivals']}."
            )
        return result

    summary = run_replications(once, N_REPLICATIONS, base_seed, ("B",))
    return summary["B"], erlang_b(c, offered_a)


def calibration_loads(rows):
    """The two loads for one site: weekday busy hour, then the worst hour."""
    hour, busy_a = weekday_busy_hour(rows)
    worst = worst_hour(rows)
    return (
        {
            "load_type": "weekday_busy_hour",
            "when": f"{hour:02d}:00",
            "A_erl": busy_a,
        },
        {
            "load_type": "worst_hour",
            "when": worst["datetime"],
            "A_erl": worst["voice_erl"],
        },
    )


def result_row(site, load, c, interval, theory):
    calls_per_hour = load["A_erl"] * SECONDS_PER_HOUR / PRIMARY_MEAN_SERVICE_S
    inside = theory_inside_ci(interval, theory)
    return {
        "site": site,
        "load_type": load["load_type"],
        "when": load["when"],
        "A_erl": f"{load['A_erl']:.5f}",
        "lambda_per_hour": f"{calls_per_hour:.4f}",
        "channels_c": c,
        "erlangB_blocking": f"{theory:.6f}",
        "sim_blocking_mean": f"{interval['mean']:.6f}",
        "sim_ci_low": f"{interval['low']:.6f}",
        "sim_ci_high": f"{interval['high']:.6f}",
        "within_ci": "yes" if inside else "no",
        "abs_diff": abs(interval["mean"] - theory),
    }


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    header = (
        f"{'site':<12}{'load_type':<20}{'when':<22}{'A_erl':>10}"
        f"{'calls/h':>10}{'c':>4}{'Erlang B':>12}{'sim mean':>12}"
        f"{'95% CI':>24}{'|diff|':>10}{'in CI':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{float(row['sim_ci_low']):.4%} , {float(row['sim_ci_high']):.4%}]"
        print(
            f"{row['site']:<12}{row['load_type']:<20}{row['when']:<22}"
            f"{float(row['A_erl']):10.4f}{float(row['lambda_per_hour']):10.2f}"
            f"{row['channels_c']:4d}{float(row['erlangB_blocking']):12.4%}"
            f"{float(row['sim_blocking_mean']):12.4%}{ci:>24}"
            f"{row['abs_diff']:10.5f}{row['within_ci']:>7}"
        )


def main():
    grouped = {name: [] for name in SITE_ORDER}
    for record in load_calibration(DATA_PATH):
        if record["site"] not in grouped:
            raise ValueError(f"Unexpected site {record['site']!r}.")
        grouped[record["site"]].append(record)

    results = []
    seed = SEED_STEP3
    for site in SITE_ORDER:
        for load in calibration_loads(grouped[site]):
            c = channels_for_gos(load["A_erl"])
            print(
                f"Simulating {site} {load['load_type']}  "
                f"A = {load['A_erl']:.4f} E, c = {c} ...",
                flush=True,
            )
            interval, theory = simulate_at_load(load["A_erl"], c, seed)
            results.append(result_row(site, load, c, interval, theory))
            seed += 100

    write_results(OUT_PATH, results)
    print()
    print_table(results)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
