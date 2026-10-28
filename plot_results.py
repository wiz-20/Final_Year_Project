"""Redraw the Results-chapter figures. Simulation markers use the existing seeds.

Analytical figures and the Stellenbosch figures do not resimulate.
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np

from case_study_loo import OUT_PATH as LOO_CSV
from case_study_loo import plot_parity
from plot_stellenbosch import main as plot_stellenbosch
from sim.literature_params import (
    ACCEPTABLE_WAIT_S,
    BLOCKING_TARGET,
    MEAN_SERVICE_TIMES_S,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
    SERVER_COUNTS,
)
from sim.mm1_simulation import RANDOM_SEED as MM1_SEED
from sim.mm1_simulation import plot_wait_vs_rho_for_holding_times
from sim.mmc_b_simulation import plot_blocking_heatmap, plot_blocking_surface_3d, plot_blocking_vs_arrivals_family
from sim.mmc_k_simulation import SWEEP_SERVER_COUNTS, plot_capacity_boundaries, plot_mmck_sweep
from sim.mmc_simulation import (
    RANDOM_SEED,
    default_wq_surface_arrivals_h,
    plot_wait_heatmap,
    plot_wait_surface_3d,
    plot_wait_vs_rho_family,
)
from sim.paths import PROJECT_ROOT
from sim.plot_style import RESULTS_FIGURES
from trunk_case_study import OUT_PATH as TRUNK_CSV
from trunk_case_study import plot_trunk

RHO_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
SURFACE_C = range(1, 51)
BLOCKING_ARRIVALS = np.logspace(0, np.log10(2000), 100)


def _note(paths):
    for path in paths:
        print(f"Saved {path}", flush=True)


def analytical():
    arrivals = default_wq_surface_arrivals_h()
    plot_wait_heatmap(PRIMARY_MEAN_SERVICE_S, SURFACE_C, arrivals, ACCEPTABLE_WAIT_S)
    plot_wait_surface_3d(PRIMARY_MEAN_SERVICE_S, SURFACE_C, arrivals, ACCEPTABLE_WAIT_S)
    plot_blocking_heatmap(PRIMARY_MEAN_SERVICE_S, SURFACE_C, BLOCKING_ARRIVALS, BLOCKING_TARGET)
    plot_blocking_surface_3d(PRIMARY_MEAN_SERVICE_S, SURFACE_C, BLOCKING_ARRIVALS, BLOCKING_TARGET)
    plot_capacity_boundaries()


def mmck():
    payload = json.loads((PROJECT_ROOT / "output" / "part_c_validation.json").read_text(encoding="utf-8"))
    markers = []
    for case in payload["cases"]:
        metrics = case["metrics"]
        markers.append({
            "c": case["c"],
            "queue_extra": case["queue_extra"],
            "arrivals_h": case["lambda"] * SECONDS_PER_HOUR,
            "B": metrics["B"]["mean"],
            "B_low": metrics["B"]["low"],
            "B_high": metrics["B"]["high"],
            "Wq": metrics["Wq"]["mean"],
            "Wq_low": metrics["Wq"]["low"],
            "Wq_high": metrics["Wq"]["high"],
        })
    plot_mmck_sweep(sim_markers=markers, server_counts=SWEEP_SERVER_COUNTS)


def parity_and_trunk():
    rows = []
    with LOO_CSV.open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append({
                "in_mae": record["in_mae"] == "yes",
                "B_meas": float(record["B_meas"]),
                "erlangB_full_rate": float(record["erlangB_full_rate"]),
                "erlangB_loo": float(record["erlangB_loo"]),
                "sim_ci_low": float(record["sim_ci_low"]),
                "sim_ci_high": float(record["sim_ci_high"]),
            })
    plot_parity(rows, RESULTS_FIGURES / "case_study_parity.png", RESULTS_FIGURES / "case_study_parity.pdf")

    parsed = []
    float_keys = (
        "carried_per_c_31", "congestion_pct", "erlangB_pct_31",
        "sim_mean_pct", "sim_ci_low_pct", "sim_ci_high_pct",
    )
    with TRUNK_CSV.open(newline="") as handle:
        for record in csv.DictReader(handle):
            row = dict(record)
            for key in float_keys:
                row[key] = float(row[key]) if row[key] != "" else ""
            parsed.append(row)
    plot_trunk(parsed)


def simulated():
    plot_wait_vs_rho_family(
        PRIMARY_MEAN_SERVICE_S,
        SERVER_COUNTS,
        RHO_VALUES,
        ACCEPTABLE_WAIT_S,
        random_seed=RANDOM_SEED,
    )
    plot_blocking_vs_arrivals_family(
        PRIMARY_MEAN_SERVICE_S,
        SERVER_COUNTS,
        RHO_VALUES,
        n_service_times=40_000,
        warmup_service_times=4_000,
        blocking_target=BLOCKING_TARGET,
        random_seed=42,
    )
    plot_wait_vs_rho_for_holding_times(
        list(MEAN_SERVICE_TIMES_S.values()),
        RHO_VALUES,
        ACCEPTABLE_WAIT_S,
        random_seed=MM1_SEED,
    )


def main():
    print("Analytical figures", flush=True)
    analytical()
    print("M/M/c/K from saved replications", flush=True)
    mmck()
    print("Parity and trunk", flush=True)
    parity_and_trunk()
    print("Stellenbosch", flush=True)
    plot_stellenbosch()
    print("Simulation sweeps", flush=True)
    simulated()
    written = sorted(RESULTS_FIGURES.glob("*"))
    print("\nFiles written:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
