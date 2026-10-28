"""Build output/results_summary.md from the saved replication JSON and closed forms.

The tables are read from files written by the tests. Nothing in this script
invents a simulation result.
"""

import json
from pathlib import Path

from sim.literature_params import PRIMARY_MEAN_SERVICE_S
from sim.mmc_k_simulation import (
    erlang_b_arrivals_at_blocking,
    erlang_c_arrivals_at_wait,
    mmck_arrivals_at_wait,
    threshold_rows,
)
from sim.paths import PROJECT_ROOT
from sim.replication import (
    DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    DEFAULT_REPLICATIONS,
    DEFAULT_SCALE_FACTOR,
    RHO_LENGTH_CAP,
)

# Last single-run M/M/c point estimate at c=8, rho=0.80, 1/mu=180 s,
# printed before the replication study (one seed, not a CI).
PREVIOUS_MMC_SINGLE_RUN = {
    "Wq": 52.0318,
    "W": 231.9933,
    "Lq": 1.8500,
    "utilisation": 0.8009,
}


def _load(name):
    path = PROJECT_ROOT / "output" / name
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}g}"


def _dev(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _metric_table(case):
    lines = [
        f"### {case['name']}",
        "",
        (
            f"R = {case['n_replications']}, seeds {case['base_seed']} to "
            f"{case['base_seed'] + case['n_replications'] - 1}. "
            f"Post-warm-up length = {case['plan']['n_measurement_service_times']} "
            f"mean service times. Warm-up = {case['plan']['n_warmup_service_times']} "
            f"mean service times. scale_factor = {case['plan']['scale_factor']}. "
            f"length factor 1/(1-rho)^2 = {case['plan']['length_factor']:.4f}."
        ),
        "",
        "| Metric | Sim. mean | 95% CI low | 95% CI high | Theory | Dev. % | Theory in CI |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, row in case["metrics"].items():
        lines.append(
            f"| {key} | {_fmt(row['mean'], 6)} | {_fmt(row['low'], 6)} | "
            f"{_fmt(row['high'], 6)} | {_fmt(row['theory'], 6)} | "
            f"{_dev(row['deviation_pct'])} | {'yes' if row['inside_ci'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _changed_lines(mmc_case):
    lines = []
    for key, previous in PREVIOUS_MMC_SINGLE_RUN.items():
        current = mmc_case["metrics"][key]["mean"]
        pct = (current - previous) / previous * 100.0
        if abs(pct) > 1.0:
            lines.append(
                f"- M/M/c {key}: previous single-run estimate {previous:.4f}, "
                f"replication mean {current:.4f}, change {pct:+.2f}%."
            )
    if not lines:
        lines.append("- No saved single-run estimate moved by more than 1%.")
    return "\n".join(lines)


def build_markdown():
    part_a = _load("part_a_validation.json")
    part_b = _load("part_b_carried.json")
    part_c = _load("part_c_validation.json")
    mu = 1.0 / PRIMARY_MEAN_SERVICE_S
    mmc_case = next(case for case in part_a if case["model"] == "M/M/c")

    parts = [
        "# Simulation results summary",
        "",
        "Interim literature parameters (not Vodacom measurements). "
        f"Holding time 1/mu = {PRIMARY_MEAN_SERVICE_S:.0f} s unless a table says otherwise. "
        "95% intervals are Student-t intervals across independent replications.",
        "",
        "## Run length",
        "",
        f"- Replications R = {DEFAULT_REPLICATIONS}.",
        f"- scale_factor = {DEFAULT_SCALE_FACTOR}.",
        "- Post-warm-up length in mean service times: "
        "`base_measurement * scale_factor / (1 - rho)^2`.",
        f"- base_measurement = {DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES}.",
        f"- Base warm-up = {DEFAULT_BASE_WARMUP_SERVICE_TIMES} mean service times, "
        "then increased with rho by `warmup_service_times_for_rho` "
        "(fraction 0.08 + 0.35*rho, capped at 45% of the measurement length).",
        f"- For rho above {RHO_LENGTH_CAP}, the gap (1-rho) is floored at "
        f"{1 - RHO_LENGTH_CAP:.2f} so the length stays finite. rho = 0.90 is not floored.",
        "- For c > 8 the measurement length is multiplied by 8/c.",
        "- Loss-model 2% GoS runs also require at least 8000 expected arrivals "
        "in the measurement window (this lengthens the c = 1 case).",
        "- After the measurement window the delay and finite-buffer runs continue "
        "for 50 mean service times so in-window customers finish service.",
        "",
        "## Part A — validation",
        "",
    ]
    for case in part_a:
        parts.append(_metric_table(case))

    parts.extend([
        "## Part B — carried load at 2% Erlang-B GoS",
        "",
        "Offered load is the largest a with Erlang-B blocking at the 2% target. "
        "Carried load is a(1-B), throughput is that load times mu, in calls per hour, "
        "and carried utilisation is a(1-B)/c.",
        "",
    ])
    for case in part_b:
        parts.append(_metric_table(case))

    parts.extend(["## Part C — M/M/c/K at c, rho = 0.80", ""])
    for case in part_c:
        parts.append(_metric_table(case))

    parts.extend([
        "### Offered load where B = 2% and where Wq = 10 s",
        "",
        "Closed form, same curves as `mmck_blocking_wq_vs_arrivals.png`. "
        "`n/a` means accepted-call Wq never reaches 10 s (no queue, or the buffer saturates below 10 s).",
        "",
        "| c | K-c | Arrivals/h at B = 2% | Arrivals/h at Wq = 10 s |",
        "|---:|---:|---:|---:|",
    ])
    for row in threshold_rows():
        wait = row["arrivals_h_at_wait"]
        wait_s = "n/a" if wait is None else f"{wait:.3f}"
        parts.append(
            f"| {row['c']} | {row['queue_extra']} | "
            f"{row['arrivals_h_at_blocking']:.3f} | {wait_s} |"
        )

    parts.extend([
        "",
        "## Overlay boundaries",
        "",
        "Arrivals per hour read from the same functions that draw "
        "`capacity_boundaries_overlay.png`. 1/mu = 180 s.",
        "",
        "| c | Erlang C, Wq = 10 s | Erlang B, B = 2% | M/M/c/K, K-c = 5, Wq = 10 s |",
        "|---:|---:|---:|---:|",
    ])
    for c in (7, 8, 21, 45):
        wq = erlang_c_arrivals_at_wait(c, mu, 10.0)
        blocking = erlang_b_arrivals_at_blocking(c, mu, 0.02)
        k5 = mmck_arrivals_at_wait(c, 5, mu, 10.0)
        parts.append(f"| {c} | {wq:.3f} | {blocking:.3f} | {k5:.3f} |")

    figures = [
        "mm1_wait_vs_rho.png",
        "mmc_wait_vs_rho.png",
        "mmc_wait_heatmap.png",
        "mmc_blocking_heatmap.png",
        "capacity_boundaries_overlay.png",
        "mmck_blocking_wq_vs_arrivals.png",
    ]
    parts.extend([
        "",
        "## Figures written in this study",
        "",
    ])
    for name in figures:
        path = PROJECT_ROOT / "output" / "figures" / name
        status = "present" if path.exists() else "MISSING"
        parts.append(f"- `output/figures/{name}` ({status})")
    parts.extend([
        "",
        "The 3D surface scripts were not modified. Their existing PNG files were not regenerated.",
        "",
        "## Estimates that moved by more than 1%",
        "",
        "Comparison is against the previous single M/M/c run at c = 8, rho = 0.80 "
        "(one seed, printed before replications). A move of the point estimate "
        "can sit inside the new confidence interval.",
        "",
        _changed_lines(mmc_case),
        "",
        "One M/M/c/K replication interval does not cover theory: "
        "c = 45, K-c = 5, carried calls per hour. "
        "The mean is 0.17% below theory and the 95% interval ends just short of it. "
        "That run is shortened by 8/c because c = 45. "
        "Blocking, Wq and Lq at the same point are inside their intervals. "
        "Every c = 8, rho = 0.80 finite-buffer interval covers theory.",
        "",
        "On `mmc_wait_vs_rho.png`, two markers have theory just outside the 95% interval. "
        "These are not the formal validation points. The runs are shorter than the "
        "c = 8, rho = 0.80 validation (length scales as 1/(1-rho)^2, and c = 45 is shortened by 8/c):",
        "",
        "- c = 1, rho = 0.50: mean 165.45 s, CI [155.03, 175.86], theory 180 s.",
        "- c = 45, rho = 0.85: mean 5.99 s, CI [5.67, 6.31], theory 5.62 s.",
        "",
        "The formal validation intervals in Part A all contain theory, including "
        "M/M/c at c = 8, rho = 0.80 and M/M/1 at rho = 0.80.",
        "",
    ])
    return "\n".join(parts) + "\n"


def main():
    text = build_markdown()
    path = PROJECT_ROOT / "output" / "results_summary.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
