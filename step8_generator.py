"""
Step 8: A small traffic generator built from the calibration hour-of-week profile.

Calibration rows only (split == "calibration", empty hours skipped) give the
mean voice_erl in each of the 168 weekday-hour slots. A test hour is predicted
by its slot mean. A synthetic hour is the slot mean times a ratio resampled
from that same slot, which is one of the calibration observations for that
weekday and hour. A shared pool of ratios is not used: a quiet hour just
above 1 E can have a ratio of about 5, and applying that to the busy hour
invents a peak the measurements do not contain.

Blocking uses the existing Erlang B function at the step 3 channel counts.
The point is the dimensioning conclusion, not a forecast.

Usage (from the project root, after step 3 and step 5):
    .\\venv\\Scripts\\python.exe step8_generator.py
"""

import csv
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from sim.literature_params import BLOCKING_TARGET
from sim.plot_style import MEASURED, MODEL, THRESHOLD, apply_style
from sim.mmc_simulation import erlang_b
from step3_calibrate import SITE_ORDER

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_clean" / "voice_all_sites.csv"
STEP3_PATH = ROOT / "results" / "step3_calibration.csv"
STEP5_PATH = ROOT / "results" / "step5_validation.csv"
OUT_PATH = ROOT / "results" / "step8_generator.csv"

N_PERIODS = 20
PLOT_HOURS = 168  # one week
MAPE_FLOOR_ERL = 1.0
# Clear of step 3 (91_000), step 5 (93_000) and step 7 (95_000).
SEED_STEP8 = 97_000
SIZINGS = ("weekday_busy_hour", "worst_hour")

COLUMNS = (
    "table",
    "site",
    "sizing",
    "channels_c",
    "mae",
    "mape_pct",
    "mape_n",
    "busy_hour",
    "predicted_erl",
    "actual_erl",
    "mean_hours_above",
    "min_hours_above",
    "max_hours_above",
    "mean_worst_blocking",
    "min_worst_blocking",
    "max_worst_blocking",
    "real_hours_above",
    "real_worst_blocking",
    "cal_hours",
    "cal_hours_above",
    "cal_per_456",
    "cal_worst_blocking",
)
# Calibration counts are also written per 456 hours, the length of the test period.
SCALE_HOURS = 456


def load_voice(path):
    """Rows with a numeric voice_erl, grouped by site. NaN cells are blank."""
    grouped = {name: [] for name in SITE_ORDER}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["voice_erl"].strip() == "":
                continue
            if record["site"] not in grouped:
                raise ValueError(f"Unexpected site {record['site']!r}.")
            record["voice_erl"] = float(record["voice_erl"])
            record["weekday"] = int(record["weekday"])
            record["hour"] = int(record["hour"])
            record["is_weekend"] = record["is_weekend"] == "True"
            grouped[record["site"]].append(record)
    return grouped


def load_channels(path):
    """c and the weekday busy-hour clock time from step 3."""
    channels = {}
    busy_hour = {}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            channels[(record["site"], record["load_type"])] = int(record["channels_c"])
            if record["load_type"] == "weekday_busy_hour":
                busy_hour[record["site"]] = int(record["when"][:2])
    return channels, busy_hour


def load_step5(path):
    """Real test hours above 2% and the worst blocking, from step 5."""
    found = {}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            found[(record["site"], record["sizing"])] = {
                "hours_above": int(record["hours_above_2pct"]),
                "worst": float(record["worst_blocking"]),
            }
    return found


def slot_means(rows):
    """Mean voice_erl for each (weekday, hour). 168 slots when the week is full."""
    totals = {}
    counts = {}
    for record in rows:
        key = (record["weekday"], record["hour"])
        totals[key] = totals.get(key, 0.0) + record["voice_erl"]
        counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in counts}


def slot_observations(rows):
    """Calibration voice_erl values in each weekday-hour. The resample pool."""
    pools = {}
    for record in rows:
        key = (record["weekday"], record["hour"])
        pools.setdefault(key, []).append(record["voice_erl"])
    return pools


def point_check(test_rows, means, busy_clock, calibration_rows):
    """Slot-mean forecast on the test hours, plus the weekday busy-hour mean."""
    abs_err = []
    ape = []
    for record in test_rows:
        predicted = means[(record["weekday"], record["hour"])]
        error = abs(predicted - record["voice_erl"])
        abs_err.append(error)
        if record["voice_erl"] > MAPE_FLOOR_ERL:
            ape.append(error / record["voice_erl"])
    cal_busy = [
        record["voice_erl"]
        for record in calibration_rows
        if (not record["is_weekend"]) and record["hour"] == busy_clock
    ]
    test_busy = [
        record["voice_erl"]
        for record in test_rows
        if (not record["is_weekend"]) and record["hour"] == busy_clock
    ]
    return {
        "mae": sum(abs_err) / len(abs_err),
        "mape_pct": 100.0 * sum(ape) / len(ape),
        "mape_n": len(ape),
        "predicted_erl": sum(cal_busy) / len(cal_busy),
        "actual_erl": sum(test_busy) / len(test_busy),
    }


def synthetic_period(test_rows, means, pools, rng):
    """Slot mean times a ratio observed in that same slot.

    Choosing an observation and dividing by the slot mean is the ratio. Multiplying
    back by the slot mean returns the observation, so the draw is a calibration
    hour from the same weekday and hour.
    """
    loads = []
    for record in test_rows:
        key = (record["weekday"], record["hour"])
        slot = means[key]
        if slot <= 0.0:
            loads.append(0.0)
            continue
        ratio = rng.choice(pools[key]) / slot
        loads.append(slot * ratio)
    return loads


def period_blocking(loads, c):
    """Hours with Erlang B above 2%, and the worst hour's blocking."""
    blocking = [erlang_b(c, load) for load in loads]
    above = sum(value > BLOCKING_TARGET for value in blocking)
    return above, max(blocking)


def calibration_blocking(records, c):
    """Hours above 2% in the calibration period, and that count per 456 hours."""
    loads = [record["voice_erl"] for record in records]
    above, worst = period_blocking(loads, c)
    n_hours = len(loads)
    return {
        "n_hours": n_hours,
        "hours_above": above,
        "per_456": above / n_hours * SCALE_HOURS,
        "worst": worst,
    }


def summarise_periods(samples):
    hours = [item[0] for item in samples]
    worst = [item[1] for item in samples]
    return {
        "mean_hours_above": sum(hours) / len(hours),
        "min_hours_above": min(hours),
        "max_hours_above": max(hours),
        "mean_worst_blocking": sum(worst) / len(worst),
        "min_worst_blocking": min(worst),
        "max_worst_blocking": max(worst),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_site(site, test_rows, synthetic_loads, means, path):
    """First test week, one synthetic week, and the slot-mean profile."""
    apply_style()
    week = test_rows[:PLOT_HOURS]
    times = [datetime.fromisoformat(record["datetime"]) for record in week]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(
        times,
        [record["voice_erl"] for record in week],
        color=MEASURED,
        linewidth=1.0,
        label="Real test week",
    )
    ax.plot(
        times,
        synthetic_loads[:PLOT_HOURS],
        color=MODEL,
        linewidth=1.0,
        label="Synthetic week",
    )
    ax.plot(
        times,
        [means[(record["weekday"], record["hour"])] for record in week],
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label="Slot-mean profile",
    )
    ax.set_ylabel("Voice traffic (erlang)")
    ax.set_xlabel("Test period")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_point_check(rows):
    header = (
        f"{'site':<12}{'MAE (E)':>10}{'MAPE':>8}{'n':>6}"
        f"{'hour':>6}{'predicted':>12}{'actual':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['site']:<12}{float(row['mae']):10.3f}"
            f"{float(row['mape_pct']):7.1f}%{int(row['mape_n']):6d}"
            f"{row['busy_hour']:>6}{float(row['predicted_erl']):12.2f}"
            f"{float(row['actual_erl']):10.2f}"
        )


def print_synthetic(rows):
    header = (
        f"{'site':<12}{'sizing':<20}{'c':>4}"
        f"{'hours above, mean [min, max]':>32}"
        f"{'worst B, mean [min, max]':>28}{'real hours':>12}{'real worst':>12}"
        f"{'cal hours':>16}{'per 456':>10}{'cal worst':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        hours = (
            f"{float(row['mean_hours_above']):5.1f} "
            f"[{int(row['min_hours_above'])}, {int(row['max_hours_above'])}]"
        )
        worst = (
            f"{float(row['mean_worst_blocking']):6.2%} "
            f"[{float(row['min_worst_blocking']):.2%}, {float(row['max_worst_blocking']):.2%}]"
        )
        cal_hours = f"{int(row['cal_hours_above'])} / {int(row['cal_hours'])}"
        print(
            f"{row['site']:<12}{row['sizing']:<20}{row['channels_c']:4d}"
            f"{hours:>32}{worst:>28}"
            f"{int(row['real_hours_above']):12d}"
            f"{float(row['real_worst_blocking']):12.2%}"
            f"{cal_hours:>16}{float(row['cal_per_456']):10.1f}"
            f"{float(row['cal_worst_blocking']):12.2%}"
        )


def main():
    voice = load_voice(DATA_PATH)
    channels, busy_hour = load_channels(STEP3_PATH)
    real = load_step5(STEP5_PATH)
    saved = []

    for index, site in enumerate(SITE_ORDER):
        rows = voice[site]
        calibration = [record for record in rows if record["split"] == "calibration"]
        test = [record for record in rows if record["split"] == "test"]
        means = slot_means(calibration)
        if len(means) != 168:
            raise RuntimeError(f"{site} has {len(means)} slots, expected 168.")
        check = point_check(test, means, busy_hour[site], calibration)
        saved.append({
            "table": "point_check",
            "site": site,
            "sizing": "",
            "channels_c": "",
            "mae": f"{check['mae']:.4f}",
            "mape_pct": f"{check['mape_pct']:.2f}",
            "mape_n": check["mape_n"],
            "busy_hour": f"{busy_hour[site]:02d}:00",
            "predicted_erl": f"{check['predicted_erl']:.4f}",
            "actual_erl": f"{check['actual_erl']:.4f}",
        })

        pools = slot_observations(calibration)
        rng = random.Random(SEED_STEP8 + index)
        periods = [
            synthetic_period(test, means, pools, rng) for _ in range(N_PERIODS)
        ]
        for sizing in SIZINGS:
            c = channels[(site, sizing)]
            summary = summarise_periods([period_blocking(loads, c) for loads in periods])
            observed = real[(site, sizing)]
            calibrated = calibration_blocking(calibration, c)
            saved.append({
                "table": "synthetic",
                "site": site,
                "sizing": sizing,
                "channels_c": c,
                "mean_hours_above": f"{summary['mean_hours_above']:.2f}",
                "min_hours_above": summary["min_hours_above"],
                "max_hours_above": summary["max_hours_above"],
                "mean_worst_blocking": f"{summary['mean_worst_blocking']:.6f}",
                "min_worst_blocking": f"{summary['min_worst_blocking']:.6f}",
                "max_worst_blocking": f"{summary['max_worst_blocking']:.6f}",
                "real_hours_above": observed["hours_above"],
                "real_worst_blocking": f"{observed['worst']:.6f}",
                "cal_hours": calibrated["n_hours"],
                "cal_hours_above": calibrated["hours_above"],
                "cal_per_456": f"{calibrated['per_456']:.2f}",
                "cal_worst_blocking": f"{calibrated['worst']:.6f}",
            })
        figure = ROOT / "results" / f"step8_{site}.png"
        plot_site(site, test, periods[0], means, figure)
        print(f"Saved {figure}", flush=True)

    write_csv(OUT_PATH, saved)
    print()
    print_point_check([row for row in saved if row["table"] == "point_check"])
    print()
    print_synthetic([row for row in saved if row["table"] == "synthetic"])
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
