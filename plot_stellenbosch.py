"""Stellenbosch results figures. Reads the cleaned hours and the step CSVs.

Nothing here resimulates. Erlang B lines use the existing function at the
channel counts and loads already stored in the step 3 and step 7 tables.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Matplotlib's default; it includes the diaeresis used in the site label.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

from sim.mmc_simulation import erlang_b
from sim.plot_style import (
    LISTED,
    MEASURED,
    MODEL,
    THRESHOLD,
    TWO_PANEL,
    apply_style,
    panel_label,
    save_figure,
)
from step3_calibrate import SITE_ORDER
from step8_generator import (
    PLOT_HOURS,
    SEED_STEP8,
    load_voice,
    slot_means,
    slot_observations,
    synthetic_period,
)

ROOT = Path(__file__).resolve().parent
VOICE_PATH = ROOT / "data_clean" / "voice_all_sites.csv"
STEP3_PATH = ROOT / "results" / "step3_calibration.csv"
STEP5_HOURLY = ROOT / "results" / "step5_hourly_blocking.csv"
STEP7_PATH = ROOT / "results" / "step7_growth.csv"

TEST_START = datetime(2026, 9, 20)
TEST_END = datetime(2026, 10, 9)  # exclusive, so 8 Oct 23:00 is inside
HOLIDAY_START = datetime(2026, 9, 24)
HOLIDAY_END = datetime(2026, 10, 6)  # exclusive, so 5 Oct is inside

# Keys stay the data site names. Only the drawn panel text uses the diaeresis.
PANEL = {"Hoerskool": "(a) Hoërskool", "Square": "(b) Square"}


def _parse(text):
    return datetime.fromisoformat(text)


def _load_step3():
    rows = {}
    with STEP3_PATH.open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows[(record["site"], record["load_type"])] = {
                "when": record["when"],
                "A": float(record["A_erl"]),
                "c": int(record["channels_c"]),
            }
    return rows


def _shade_test(ax):
    ax.axvspan(TEST_START, TEST_END, color="0.88", zorder=0)
    # Left of the holiday hatch, still inside the test shade.
    mid = TEST_START + (HOLIDAY_START - TEST_START) / 2
    ax.text(
        mid, 0.98, "test",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9, color="0.3", zorder=5,
    )


def _hatch_holidays(ax, start=HOLIDAY_START, end=HOLIDAY_END, text_y=0.98):
    # Hatch colour is taken from the patch edge. The stroke width is the
    # hatch linewidth, read when the figure is drawn.
    plt.rcParams["hatch.linewidth"] = 0.5
    ax.axvspan(
        start, end,
        facecolor="none",
        edgecolor="#BFBFBF",
        linewidth=0.5,
        hatch="///",
        zorder=1,
    )
    mid = start + (end - start) / 2
    ax.text(
        mid, text_y, "holidays",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9, color="0.25", zorder=5,
    )


def _date_axis(ax, interval=1):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=interval))


def _daily_axis(ax):
    """One tick per day, 20 Sep through 27 Sep, labelled '20 Sep'."""
    ax.set_xlim(datetime(2026, 9, 20), datetime(2026, 9, 27))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def plot_traffic(voice, step3):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=TWO_PANEL, sharex=True)
    for ax, site in zip(axes, SITE_ORDER):
        times = [_parse(record["datetime"]) for record in voice[site]]
        loads = [record["voice_erl"] for record in voice[site]]
        _shade_test(ax)
        _hatch_holidays(ax)
        ax.plot(times, loads, color=MEASURED, linewidth=0.6, zorder=2)
        busy = step3[(site, "weekday_busy_hour")]
        worst = step3[(site, "worst_hour")]
        ax.axhline(
            busy["A"], color=LISTED, linestyle="--", linewidth=1.0, zorder=3,
            label=f"Weekday 16:00 ({busy['A']:.2f} E)",
        )
        ax.plot(
            [_parse(worst["when"])], [worst["A"]],
            marker="o", color=MODEL, linestyle="none", markersize=5, zorder=4,
            label=f"Worst hour ({worst['A']:.2f} E)",
        )
        ax.set_ylabel("Voice traffic (E)")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 1.10)
        panel_label(ax, PANEL[site])
        ax.legend(
            loc="lower left",
            frameon=True,
            borderpad=0.25,
            labelspacing=0.2,
            handlelength=1.4,
            handletextpad=0.35,
        )
        _date_axis(ax, interval=2)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return save_figure(fig, "stb_traffic")


def plot_test_blocking():
    apply_style()
    grouped = {site: [] for site in SITE_ORDER}
    with STEP5_HOURLY.open(newline="") as handle:
        for record in csv.DictReader(handle):
            grouped[record["site"]].append(record)
    step3 = _load_step3()
    fig, axes = plt.subplots(2, 1, figsize=TWO_PANEL, sharex=True)
    for ax, site in zip(axes, SITE_ORDER):
        rows = grouped[site]
        times = [_parse(row["datetime"]) for row in rows]
        c_busy = step3[(site, "weekday_busy_hour")]["c"]
        c_worst = step3[(site, "worst_hour")]["c"]
        _hatch_holidays(ax)
        ax.plot(
            times,
            [100.0 * float(row["blocking_c_busy"]) for row in rows],
            color=LISTED, linewidth=0.8, zorder=2,
            label=f"Busy hour, c = {c_busy}",
        )
        ax.plot(
            times,
            [100.0 * float(row["blocking_c_worst"]) for row in rows],
            color=MODEL, linewidth=0.8, zorder=3,
            label=f"Worst hour, c = {c_worst}",
        )
        ax.axhline(2.0, color=THRESHOLD, linestyle="--", linewidth=1.0, label="2%", zorder=1)
        ax.set_ylabel("Blocking (%)")
        ax.set_ylim(bottom=0)
        panel_label(ax, PANEL[site])
        # Same entries and frame as the growth figure. The box starts slightly
        # further left than that figure's 0.22 anchor so it clears "holidays".
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.20, 1.0),
            borderpad=0.25,
            labelspacing=0.2,
            handlelength=1.4,
            handletextpad=0.35,
        )
        _date_axis(ax)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return save_figure(fig, "stb_test_blocking")


def plot_growth(step3):
    apply_style()
    stored = {site: {"weekday_busy_hour": [], "worst_hour": []} for site in SITE_ORDER}
    with STEP7_PATH.open(newline="") as handle:
        for record in csv.DictReader(handle):
            stored[record["site"]][record["load_type"]].append(record)
    factors = [round(1.0 + i * 0.01, 2) for i in range(101)]
    fig, axes = plt.subplots(2, 1, figsize=TWO_PANEL, sharex=True)
    styles = (
        ("weekday_busy_hour", LISTED, "Busy hour"),
        ("worst_hour", MODEL, "Worst hour"),
    )
    for ax, site in zip(axes, SITE_ORDER):
        labelled_ci = False
        for load_type, colour, name in styles:
            info = step3[(site, load_type)]
            ax.plot(
                factors,
                [100.0 * erlang_b(info["c"], info["A"] * factor) for factor in factors],
                color=colour, linewidth=1.2,
                label=f"{name}, c = {info['c']}",
            )
            spots = [
                row for row in stored[site][load_type]
                if row["sim_blocking_mean"].strip() != ""
            ]
            if not spots:
                continue
            means = [100.0 * float(row["sim_blocking_mean"]) for row in spots]
            lows = [100.0 * float(row["sim_ci_low"]) for row in spots]
            highs = [100.0 * float(row["sim_ci_high"]) for row in spots]
            ax.errorbar(
                [float(row["growth_factor"]) for row in spots],
                means,
                yerr=[
                    [mean - low for mean, low in zip(means, lows)],
                    [high - mean for high, mean in zip(highs, means)],
                ],
                fmt="o",
                color=colour,
                markerfacecolor="white",
                capsize=3,
                linestyle="none",
                label="Simulation 95% CI" if not labelled_ci else None,
            )
            labelled_ci = True
        ax.axhline(2.0, color=THRESHOLD, linestyle="--", linewidth=1.0, label="2%")
        ax.set_ylabel("Blocking (%)")
        ax.set_xlim(1.0, 2.0)
        ax.set_ylim(bottom=0)
        panel_label(ax, PANEL[site])
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.22, 1.0),
            borderpad=0.25,
            labelspacing=0.2,
            handlelength=1.4,
            handletextpad=0.35,
        )
    axes[-1].set_xlabel("Growth factor")
    fig.tight_layout()
    return save_figure(fig, "stb_growth")


def plot_generator(voice):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=TWO_PANEL, sharex=True)
    for index, (ax, site) in enumerate(zip(axes, SITE_ORDER)):
        rows = voice[site]
        calibration = [record for record in rows if record["split"] == "calibration"]
        test = [record for record in rows if record["split"] == "test"]
        means = slot_means(calibration)
        pools = slot_observations(calibration)
        synthetic = synthetic_period(test, means, pools, random.Random(SEED_STEP8 + index))
        week = test[:PLOT_HOURS]
        times = [_parse(record["datetime"]) for record in week]
        week_end = times[-1] + timedelta(hours=1)
        _hatch_holidays(ax, HOLIDAY_START, min(HOLIDAY_END, week_end))
        ax.plot(
            times, [record["voice_erl"] for record in week],
            color=MEASURED, linewidth=0.8, zorder=3, label="Measured",
        )
        ax.plot(
            times, synthetic[:PLOT_HOURS],
            color=MODEL, linewidth=0.8, zorder=2, label="Synthetic week",
        )
        ax.plot(
            times,
            [means[(record["weekday"], record["hour"])] for record in week],
            color=THRESHOLD, linestyle="--", linewidth=1.0, zorder=1,
            label="Hour-of-week mean",
        )
        ax.set_ylabel("Voice traffic (E)")
        ax.set_ylim(bottom=0)
        panel_label(ax, PANEL[site])
        _daily_axis(ax)
    axes[-1].set_xlabel("Date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(fig, "stb_generator")


def main():
    voice = load_voice(VOICE_PATH)
    step3 = _load_step3()
    written = []
    for pair in (
        plot_traffic(voice, step3),
        plot_test_blocking(),
        plot_growth(step3),
        plot_generator(voice),
    ):
        written.extend(pair)
        print(f"Saved {pair[0]}")
    return written


if __name__ == "__main__":
    main()
