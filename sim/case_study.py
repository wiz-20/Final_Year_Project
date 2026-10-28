"""Apply the M/M/c/c model to the Ahmadzai & Ghag Table I busy hours.

Measured traffic is treated as carried traffic. Offered load is
measured / (1 - B_measured). Each row is run at c = net TCH, c = 2 x net TCH,
and c = round(Available TCHs). c_eff is the smallest channel count whose
Erlang B at the offered load is at or below the measured congestion.
Half-rate was disabled on this network, so the doubled channel count is a
counterfactual.
"""

import json

import matplotlib.pyplot as plt

from sim.case_study_data import (
    AHMADZAI_GHAG_2018,
    OZOVEHE_ET_AL_2017,
    ahmadzai_observations,
)
from sim.literature_params import (
    BLOCKING_TARGET,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
)
from sim.mmc_b_simulation import run_mmc_b_simulation
from sim.mmc_simulation import erlang_b, offered_load_for_blocking
from sim.paths import PROJECT_ROOT, figure_path
from sim.plot_style import apply_style
from sim.replication import DEFAULT_REPLICATIONS, plan_run, run_replications
from sim.validation_study import GOS_MIN_EXPECTED_ARRIVALS

# Distinct from the validation seed blocks (11_000 through 75_000).
SEED_CASE = 81_000
CEFF_SEED_OFFSET = 300
LOG_BLOCKING_FLOOR_PERCENT = 0.01
INCONSISTENT_HALF_RATE = (
    "inconsistent with stated configuration (exceeds full half-rate)"
)
FIGURE_SUBTITLE = (
    "Published operator measurements (Afghanistan 2018, Nigeria 2017); "
    "not Vodacom data"
)


def offered_load_from_carried(carried_erlang, blocking):
    """Offered load a = carried / (1 - B). Blocking is a fraction, not a percent."""
    if not 0.0 <= blocking < 1.0:
        raise ValueError("blocking must be in [0, 1).")
    if carried_erlang < 0:
        raise ValueError("carried traffic must be non-negative.")
    return carried_erlang / (1.0 - blocking)


def erlang_b_traffic_at_congestion(c, blocking):
    """Offered load (erlang) at which Erlang B(c, a) equals this congestion.

    Ahmadzai & Ghag Table IV reports this load at c = net TCH and compares
    it with the measured TCH traffic.
    """
    if not 0.0 < blocking < 1.0:
        raise ValueError("blocking must be in (0, 1).")
    hi = float(max(c, 1))
    while erlang_b(c, hi) < blocking:
        hi *= 2.0
        if hi > 1e7:
            raise RuntimeError("Erlang B congestion target was not reached.")
    return offered_load_for_blocking(c, blocking, hi=hi)


def effective_channel_count(offered_a, blocking):
    """Smallest integer c with Erlang B(c, offered_a) <= blocking."""
    if offered_a < 0:
        raise ValueError("offered load must be non-negative.")
    if not 0.0 < blocking < 1.0:
        raise ValueError("blocking must be in (0, 1).")
    c = 1
    while erlang_b(c, offered_a) > blocking:
        c += 1
        if c > 5000:
            raise RuntimeError("No channel count met the measured congestion.")
    return c


def bracketed_by(b_meas, b_fr, b_hr):
    """Where measured congestion sits relative to full-rate and half-rate Erlang B."""
    low = min(b_fr, b_hr)
    high = max(b_fr, b_hr)
    if low <= b_meas <= high:
        return "between full rate and half rate"
    gap_fr = abs(b_fr - b_meas)
    gap_hr = abs(b_hr - b_meas)
    if gap_fr < gap_hr:
        return "full rate"
    if gap_hr < gap_fr:
        return "half rate"
    return "full rate and half rate"


def analyse_observation(obs, blocking_target=BLOCKING_TARGET):
    """Closed-form Erlang-B comparison for one Table I row. No SimPy."""
    carried = obs["carried_erlang"]
    b_meas = obs["congestion"]
    offered = offered_load_from_carried(carried, b_meas)
    channels = {
        "fr": obs["net_tch"],
        "hr": 2 * obs["net_tch"],
        "av": obs["c_available"],
    }
    blocking = {mode: erlang_b(c, offered) for mode, c in channels.items()}
    capacity = {
        mode: offered_load_for_blocking(c, blocking_target)
        for mode, c in channels.items()
    }
    impossible = {mode: carried > c for mode, c in channels.items()}
    table_iv = erlang_b_traffic_at_congestion(obs["net_tch"], b_meas)
    c_eff = effective_channel_count(offered, b_meas)
    return {
        "sector": obs["sector"],
        "row": obs["row"],
        "trx": obs["trx"],
        "net_tch": obs["net_tch"],
        "available_tchs": obs["available_tchs"],
        "c_fr": channels["fr"],
        "c_hr": channels["hr"],
        "c_av": channels["av"],
        "carried_erlang": carried,
        "offered_erlang": offered,
        "B_meas": b_meas,
        "B_erlang_fr": blocking["fr"],
        "B_erlang_hr": blocking["hr"],
        "B_erlang_av": blocking["av"],
        "capacity_fr_erlang": capacity["fr"],
        "capacity_hr_erlang": capacity["hr"],
        "capacity_av_erlang": capacity["av"],
        "utilisation_fr": carried / capacity["fr"],
        "utilisation_hr": carried / capacity["hr"],
        "utilisation_av": carried / capacity["av"],
        "bracketed_by": bracketed_by(b_meas, blocking["fr"], blocking["hr"]),
        "c_eff": c_eff,
        "c_eff_over_net_tch": c_eff / obs["net_tch"],
        "B_erlang_ceff": erlang_b(c_eff, offered),
        "B_erlang_ceff_minus": erlang_b(c_eff - 1, offered) if c_eff > 1 else None,
        "configuration_flag": (
            INCONSISTENT_HALF_RATE if c_eff > 2 * obs["net_tch"] else ""
        ),
        "impossible_at_full_rate": impossible["fr"],
        "impossible_at_half_rate": impossible["hr"],
        "impossible_at_available": impossible["av"],
        "flag": _flag(impossible),
        "table_iv_erlang_b_traffic": table_iv,
        "table_iv_minus_measured": table_iv - carried,
        "source": obs["source"],
    }


def _flag(impossible):
    labels = {
        "fr": "full rate",
        "hr": "half rate",
        "av": "available TCH",
    }
    hit = [labels[mode] for mode in ("fr", "hr", "av") if impossible[mode]]
    if not hit:
        return ""
    if len(hit) == 1:
        return f"impossible at {hit[0]}"
    if len(hit) == 2:
        return f"impossible at {hit[0]} and at {hit[1]}"
    return f"impossible at {hit[0]}, {hit[1]}, and at {hit[2]}"


def analyse_ahmadzai():
    return [analyse_observation(obs) for obs in ahmadzai_observations()]


def _loss_plan(offered_a, c, mean_service_s):
    """Arrival-floor plan for a loss model that may have rho > 1.

    plan_run still supplies the warm-up and the Student-t run. The delay
    factor 1/(1-rho)^2 is started from one mean service time so that it does
    not schedule 10^5 service times when rho is capped at 0.95. The binding
    length is the same 8000-arrival floor used for the 2% GoS loss runs.
    """
    return plan_run(
        mean_service_s,
        offered_a / c,
        c=c,
        base_measurement=1,
        offered_a=offered_a,
        min_expected_arrivals=GOS_MIN_EXPECTED_ARRIVALS,
    )


def simulate_blocking(
    offered_a,
    c,
    base_seed,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    n_replications=DEFAULT_REPLICATIONS,
):
    """R replications of M/M/c/c at this offered load. Returns the B interval and plan."""
    mu = 1.0 / mean_service_s
    lam = offered_a * mu
    plan = _loss_plan(offered_a, c, mean_service_s)

    def once(seed):
        return run_mmc_b_simulation(
            lam, mu, c, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )

    summary = run_replications(once, n_replications, base_seed, ("B",))
    return summary["B"], plan


def apply_simulator(
    rows,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    n_replications=DEFAULT_REPLICATIONS,
    base_seed=SEED_CASE,
):
    """Attach simulated blocking CIs at the four channel counts. Rows are copied."""
    simulated = []
    for index, row in enumerate(rows):
        out = dict(row)
        for mode, c_key, seed_offset in (
            ("fr", "c_fr", 0),
            ("hr", "c_hr", 100),
            ("av", "c_av", 200),
            ("ceff", "c_eff", CEFF_SEED_OFFSET),
        ):
            seed = base_seed + 1_000 * index + seed_offset
            print(
                f"Simulating {row['sector']} row {row['row']} {mode.upper()} "
                f"c = {row[c_key]}, a = {row['offered_erlang']:.3f} ...",
                flush=True,
            )
            interval, plan = simulate_blocking(
                row["offered_erlang"],
                row[c_key],
                seed,
                mean_service_s=mean_service_s,
                n_replications=n_replications,
            )
            out[f"B_sim_{mode}"] = interval["mean"]
            out[f"B_sim_{mode}_low"] = interval["low"]
            out[f"B_sim_{mode}_high"] = interval["high"]
            out[f"sim_{mode}_plan"] = {
                "n_measurement_service_times": plan["n_measurement_service_times"],
                "n_warmup_service_times": plan["n_warmup_service_times"],
                "base_seed": seed,
                "n_replications": n_replications,
                "mean_service_s": mean_service_s,
            }
        simulated.append(out)
    return simulated


def _effective_fields(offered_a, b_meas, net_tch, b_fr, b_hr):
    """Closed-form c_eff fields. Does not touch simulated full/half/available results."""
    c_eff = effective_channel_count(offered_a, b_meas)
    return {
        "bracketed_by": bracketed_by(b_meas, b_fr, b_hr),
        "c_eff": c_eff,
        "c_eff_over_net_tch": c_eff / net_tch,
        "B_erlang_ceff": erlang_b(c_eff, offered_a),
        "B_erlang_ceff_minus": erlang_b(c_eff - 1, offered_a) if c_eff > 1 else None,
        "configuration_flag": INCONSISTENT_HALF_RATE if c_eff > 2 * net_tch else "",
    }


def attach_effective_channels(
    rows,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    n_replications=DEFAULT_REPLICATIONS,
    base_seed=SEED_CASE,
):
    """Add c_eff and its simulation. Existing row fields are kept."""
    attached = []
    for index, row in enumerate(rows):
        out = dict(row)
        out.update(_effective_fields(
            row["offered_erlang"],
            row["B_meas"],
            row["net_tch"],
            row["B_erlang_fr"],
            row["B_erlang_hr"],
        ))
        seed = base_seed + 1_000 * index + CEFF_SEED_OFFSET
        print(
            f"Simulating {row['sector']} row {row['row']} c_eff = {out['c_eff']}, "
            f"a = {row['offered_erlang']:.3f} ...",
            flush=True,
        )
        interval, plan = simulate_blocking(
            row["offered_erlang"],
            out["c_eff"],
            seed,
            mean_service_s=mean_service_s,
            n_replications=n_replications,
        )
        out["B_sim_ceff"] = interval["mean"]
        out["B_sim_ceff_low"] = interval["low"]
        out["B_sim_ceff_high"] = interval["high"]
        out["sim_ceff_plan"] = {
            "n_measurement_service_times": plan["n_measurement_service_times"],
            "n_warmup_service_times": plan["n_warmup_service_times"],
            "base_seed": seed,
            "n_replications": n_replications,
            "mean_service_s": mean_service_s,
        }
        attached.append(out)
    return attached


def plot_case_study_blocking(rows):
    """Measured blocking and four simulated channel counts, on a log axis."""
    apply_style()
    labels = [f"{row['sector']}\n{row['row']}" for row in rows]
    x_values = list(range(len(rows)))
    floor = LOG_BLOCKING_FLOOR_PERCENT

    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    outlier = [i for i, row in enumerate(rows) if row["sector"] == "TKRP018_2"]
    if outlier:
        ax.axvspan(
            outlier[0] - 0.5,
            outlier[-1] + 0.5,
            color="#f6d6c2",
            zorder=0,
            label="TKRP018_2 outlier",
        )
    ax.axhline(
        100.0 * BLOCKING_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label="2% GoS",
    )
    series = (
        ("fr", -0.30, "o", "#1f4e79", r"Full rate, $c$ = net TCH"),
        ("hr", -0.10, "s", "#c0392b", r"Half rate, $c$ = 2$\times$ net TCH"),
        ("av", 0.10, "^", "#117a65", r"$c$ = round(Available TCHs)"),
        ("ceff", 0.30, "P", "#7d3c98", r"$c_{\mathrm{eff}}$"),
    )
    for mode, offset, marker, color, label in series:
        ys, yerr = _log_percent_series(rows, mode, floor)
        ax.errorbar(
            [x + offset for x in x_values],
            ys,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            linestyle="none",
            label=label + " (simulation)",
        )
    ax.plot(
        x_values,
        [max(floor, 100.0 * row["B_meas"]) for row in rows],
        linestyle="none",
        marker="D",
        color="black",
        markersize=5,
        label="Measured",
        zorder=4,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Blocking (%)")
    ax.set_xlabel("Sector and Table I row")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_title(
        "Case-study blocking: measured congestion and M/M/$c$/$c$ predictions\n"
        + FIGURE_SUBTITLE
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = figure_path("case_study_blocking.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def _log_percent_series(rows, mode, floor):
    """Percent scale clipped to `floor` so a log axis can show near-zero blocking."""
    means, low_err, high_err = [], [], []
    for row in rows:
        mean = max(floor, 100.0 * row[f"B_sim_{mode}"])
        low = max(floor, 100.0 * row[f"B_sim_{mode}_low"])
        high = max(mean, 100.0 * row[f"B_sim_{mode}_high"])
        means.append(mean)
        low_err.append(mean - min(low, mean))
        high_err.append(high - mean)
    return means, [low_err, high_err]


def _pct(fraction):
    percent = 100.0 * fraction
    if -0.005 < percent < 0.0:
        percent = 0.0
    return f"{percent:.2f}"


def _ci(row, mode):
    return (
        f"{_pct(row[f'B_sim_{mode}'])} "
        f"[{_pct(row[f'B_sim_{mode}_low'])}, {_pct(row[f'B_sim_{mode}_high'])}]"
    )


def comparison_markdown(rows):
    """Part 2 tables. Blocking columns are percent."""
    lines = [
        "| Sector | Row | Net TCH | Available TCHs | c avail | Measured E | Offered a | B meas % | "
        "B Erlang B FR % | B sim FR % [CI] | B Erlang B HR % | B sim HR % [CI] | "
        "B Erlang B avail % | B sim avail % [CI] | Bracketed by |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sector} | {rownum} | {net} | {avail:.3f} | {cav} | {carried:.3f} | {offered:.2f} | {bmeas} | "
            "{bfr} | {simfr} | {bhr} | {simhr} | {bav} | {simav} | {closer} |".format(
                sector=row["sector"],
                rownum=row["row"],
                net=row["net_tch"],
                avail=row["available_tchs"],
                cav=row["c_av"],
                carried=row["carried_erlang"],
                offered=row["offered_erlang"],
                bmeas=_pct(row["B_meas"]),
                bfr=_pct(row["B_erlang_fr"]),
                simfr=_ci(row, "fr"),
                bhr=_pct(row["B_erlang_hr"]),
                simhr=_ci(row, "hr"),
                bav=_pct(row["B_erlang_av"]),
                simav=_ci(row, "av"),
                closer=row["bracketed_by"],
            )
        )
    lines.extend([
        "",
        "Capacity at 2% GoS is the offered load (erlang) with Erlang B equal to 2%. "
        "Utilisation is measured traffic divided by that capacity.",
        "",
        "| Sector | Row | Flag | c FR | Capacity FR (E) | Measured / capacity FR | "
        "c HR | Capacity HR (E) | Measured / capacity HR | "
        "c avail | Capacity avail (E) | Measured / capacity avail |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['sector']} | {row['row']} | {row['flag'] or '—'} | "
            f"{row['c_fr']} | {row['capacity_fr_erlang']:.2f} | {row['utilisation_fr']:.2f} | "
            f"{row['c_hr']} | {row['capacity_hr_erlang']:.2f} | {row['utilisation_hr']:.2f} | "
            f"{row['c_av']} | {row['capacity_av_erlang']:.2f} | {row['utilisation_av']:.2f} |"
        )
    lines.extend([
        "",
        "### Effective channel count",
        "",
        "c_eff is the smallest integer c for which Erlang B(c, offered a) is at or "
        "below the measured congestion. Offered a is the carried-traffic estimate "
        "already used above. The simulation is M/M/c/c at c_eff.",
        "",
        "| Sector | Row | c_eff | c_eff / net TCH | B at c_eff−1 % | B at c_eff % | "
        "B sim c_eff % [CI] | Configuration |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        below = "n/a" if row["B_erlang_ceff_minus"] is None else _pct(row["B_erlang_ceff_minus"])
        lines.append(
            f"| {row['sector']} | {row['row']} | {row['c_eff']} | "
            f"{row['c_eff_over_net_tch']:.2f} | {below} | {_pct(row['B_erlang_ceff'])} | "
            f"{_ci(row, 'ceff')} | {row['configuration_flag'] or '—'} |"
        )
    lines.extend([
        "",
        "### Table IV check",
        "",
        "Erlang B traffic is the offered load at c = net TCH that produces the "
        "measured congestion. Ahmadzai & Ghag compare that load with measured "
        "TCH traffic. It is not the carried/(1−B) offered load used in the simulation.",
        "",
        "| Sector | Row | Net TCH | Measured congestion % | Measured traffic (E) | "
        "Erlang B traffic (E) | Erlang B − measured (E) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['sector']} | {row['row']} | {row['net_tch']} | {_pct(row['B_meas'])} | "
            f"{row['carried_erlang']:.3f} | {row['table_iv_erlang_b_traffic']:.3f} | "
            f"{row['table_iv_minus_measured']:.3f} |"
        )
    return "\n".join(lines)


def save_case_rows(rows, path=None):
    path = path or (PROJECT_ROOT / "output" / "case_study.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ahmadzai": AHMADZAI_GHAG_2018["citation"],
        "table_i": AHMADZAI_GHAG_2018["table_i"],
        "table_ii": AHMADZAI_GHAG_2018["table_ii"],
        "ozovehe": {
            "citation": OZOVEHE_ET_AL_2017["citation"],
            "example_traffic_erlang": OZOVEHE_ET_AL_2017["example_traffic_erlang"],
            "c_assumption": OZOVEHE_ET_AL_2017["c_assumption"],
            "c_assumption_note": OZOVEHE_ET_AL_2017["c_assumption_note"],
            "omitted_from_blocking_table": (
                "No TCH-congestion percentage is published for the example cell, "
                "so offered load and blocking are not computed."
            ),
        },
        "mean_service_s": PRIMARY_MEAN_SERVICE_S,
        "blocking_unit": "fraction",
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return path


def run_case_study(make_plot=True, resimulate=False):
    """Refresh the figure and summary.

    The saved full-rate, half-rate, and available-TCH simulations are kept.
    Only c_eff is simulated, unless resimulate is true.
    """
    path = PROJECT_ROOT / "output" / "case_study.json"
    if not resimulate and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = attach_effective_channels(saved["rows"])
    else:
        rows = apply_simulator(analyse_ahmadzai())
    save_case_rows(rows)
    figure = plot_case_study_blocking(rows) if make_plot else None
    summary = write_case_study_summary(rows, figure)
    return rows, figure, summary


def write_case_study_summary(rows, figure=None):
    """Write output/case_study_summary.md from computed rows. Does not invent results."""
    from sim.capacity_planning import (
        ILLUSTRATIVE_GROWTH_PER_YEAR,
        IMPLIED_HOLDING_TIME_S,
        planning_markdown,
    )
    from sim.case_study_data import (
        BUSY_HOUR_CALL_ATTEMPTS_PER_SUBSCRIBER,
        ERLANG_PER_SUBSCRIBER,
        HUAWEI_GSM_CAPACITY_PLANNING,
        PARHIZGAR_BAHMANI_2011,
    )

    planning_text, _ = planning_markdown()
    figure_name = "output/figures/case_study_blocking.png"
    parts = [
        "# Case-study summary",
        "",
        "Each row is one busy hour from Ahmadzai & Ghag Table I (p.1848): "
        "TCH traffic, TCH congestion, and Available TCHs. Net TCH is from Table II. "
        "The simulation treats measured traffic as carried traffic, so offered load "
        "a = measured / (1 − B_measured). Three channel counts are run: "
        "c = net TCH (full rate), c = 2× net TCH (half rate), and "
        "c = round(Available TCHs). Half-rate was disabled on this network, "
        "so the half-rate column is a counterfactual.",
        "",
        f"Simulator: M/M/c/c, R = {DEFAULT_REPLICATIONS}, Student-t 95% CI. "
        f"Holding time in the simulator 1/μ = {PRIMARY_MEAN_SERVICE_S:.0f} s. "
        "Post-warm-up length is the existing loss-model floor of at least "
        f"{GOS_MIN_EXPECTED_ARRIVALS} expected arrivals (`plan_run` with "
        "base_measurement = 1, so the delay factor 1/(1−ρ)² does not schedule "
        "hundreds of thousands of service times on cells with ρ > 1). "
        "Seeds start at 81000.",
        "",
        "## Sources",
        "",
        f"- {AHMADZAI_GHAG_2018['citation']}. {AHMADZAI_GHAG_2018['table_i']}; "
        f"{AHMADZAI_GHAG_2018['table_ii']}. {AHMADZAI_GHAG_2018['network']}. "
        "Busy hour. Half-rate disabled.",
        f"- {OZOVEHE_ET_AL_2017['citation']}. "
        f"{OZOVEHE_ET_AL_2017['n_cells']} cells, {OZOVEHE_ET_AL_2017['n_bts']} BTS, "
        f"{OZOVEHE_ET_AL_2017['period']}.",
        f"- {PARHIZGAR_BAHMANI_2011}. Urban 25 mE.",
        f"- {HUAWEI_GSM_CAPACITY_PLANNING}. Urban 25 mE, suburban 20 mE, "
        f"{BUSY_HOUR_CALL_ATTEMPTS_PER_SUBSCRIBER} busy-hour call attempts per subscriber.",
        "",
        "## Assumptions",
        "",
        f"- {OZOVEHE_ET_AL_2017['c_assumption_note']} "
        f"Example traffic: {OZOVEHE_ET_AL_2017['example_traffic_note']}. "
        "No congestion percentage is published for that cell, so it is not in "
        "the blocking table or the figure.",
        "- Measured sector traffic is carried traffic, not offered traffic.",
        "- Growth rates g = "
        + ", ".join(f"{100 * g:.0f}%" for g in ILLUSTRATIVE_GROWTH_PER_YEAR)
        + " per year are an illustrative assumption, not sourced.",
        f"- Implied holding time {IMPLIED_HOLDING_TIME_S:.1f} s uses the urban "
        f"{ERLANG_PER_SUBSCRIBER['urban']} E/subscriber. It is reported only. "
        "No sweep was rerun with it.",
        "",
        "## Blocking comparison",
        "",
        "Bracketed by is \"between full rate and half rate\" when the measured "
        "congestion lies between those two Erlang-B values, and otherwise names "
        "the nearer of the two. c_eff is simulated separately. "
        "The simulation columns are the M/M/c/c estimate and its 95% interval, in percent.",
        "",
        comparison_markdown(rows),
        "",
        "## Subscribers and time to upgrade",
        "",
        planning_text,
        "",
        "## Figures",
        "",
        f"- `{figure_name}`",
        "",
    ]
    path = PROJECT_ROOT / "output" / "case_study_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    run_case_study(make_plot=True)
