"""
M/M/c Queue Simulation (SimPy) with Warm-up Handling and Erlang-C Validation
============================================================================

What this script models:
    A base station as an M/M/c queue (c parallel channels):
        - Poisson arrivals            (rate lambda_rate, per second)
        - Exponential service times   (rate mu_rate = 1 / mean holding time)
        - c servers                   (traffic channels / slots)
        - Infinite buffer             (no calls blocked)
        - Time unit                   seconds

Why this follows the M/M/1 baseline:
    The single-channel ~700 s wait at rho = 0.8, 1/mu = 180 s, is correct for
    c = 1. A real site has several channels; wait then falls. This script
    keeps the same holding time, sets lambda = rho * c * mu, and plots a
    family of Wq(rho) curves for GSM-grounded channel counts:
        c = 1   single-channel baseline
        c = 7   1 TRX traffic channels (8 slots minus 1 signalling)
        c = 8   1 TRX raw TDMA frame (supervisor minimum)
        c = 21  typical 3-TRX cell after BCCH/SDCCH
        c = 45  large 6-TRX macro cell

Notation:
    a   = lambda / mu          offered load (erlang)
    rho = lambda / (c * mu)    mean fraction of busy servers (must be < 1)
"""

import math
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import simpy
from matplotlib.colors import LogNorm

from sim.literature_params import (
    ACCEPTABLE_WAIT_S,
    BLOCKING_TARGET,
    INTERIM_PARAMETER_NOTE,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
    SERVER_COUNTS,
    SERVER_LABELS,
)
from sim.paths import figure_path
from sim.plot_style import SINGLE, THRESHOLD, apply_style, channel_colour, save_figure
from sim.replication import (
    DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    DEFAULT_REPLICATIONS,
    DEFAULT_SCALE_FACTOR,
    log_scale_errorbar,
    plan_run,
    run_replications,
)
from sim.warmup import (
    in_measurement_window,
    scaled_service_time_counts,
    total_run_time,
    warmup_service_times_for_rho,
)


# ==============================================================================
# SECTION: PARAMETERS
# ------------------------------------------------------------------------------
# rho = lambda / (c * mu). For a chosen rho and c, arrivals are
#   lambda = rho * c * mu
# so more servers carry more traffic at the same utilisation.
# ==============================================================================

DEMO_C = 8                       # supervisor's suspected minimum (1 TRX)
DEMO_RHO = 0.80

MU_RATE = 1.0 / PRIMARY_MEAN_SERVICE_S
LAMBDA_RATE = DEMO_RHO * DEMO_C * MU_RATE
assert LAMBDA_RATE < DEMO_C * MU_RATE, "Unstable: need lambda < c * mu."

N_SERVICE_TIMES = 80_000
WARMUP_SERVICE_TIMES = 8_000     # 10% of run discarded before measurement window
DRAIN_SERVICE_TIMES = 50.0       # let in-window customers finish after measurement_time
SIM_TIME = N_SERVICE_TIMES * PRIMARY_MEAN_SERVICE_S
WARMUP_TIME = WARMUP_SERVICE_TIMES * PRIMARY_MEAN_SERVICE_S
RANDOM_SEED = 42


# ==============================================================================
# SECTION: SIMULATION LOGIC
# ==============================================================================

def customer_arrivals(env, server, lambda_rate, mu_rate, records):
    while True:
        yield env.timeout(random.expovariate(lambda_rate))
        env.process(customer(env, server, mu_rate, records))


def customer(env, server, mu_rate, records):
    arrival_time = env.now
    with server.request() as req:
        yield req
        service_start = env.now
        waiting_time = service_start - arrival_time
        service_time = random.expovariate(mu_rate)
        yield env.timeout(service_time)
        service_end = env.now
        time_in_system = service_end - arrival_time
    records.append({
        "arrival_time": arrival_time,
        "service_start": service_start,
        "service_end": service_end,
        "waiting_time": waiting_time,
        "time_in_system": time_in_system,
    })


def run_mmc_simulation(
    lambda_rate,
    mu_rate,
    c,
    measurement_time,
    warmup_time,
    random_seed=None,
    drain_service_times=DRAIN_SERVICE_TIMES,
):
    """One M/M/c run. Returns post-warm-up Wq, W, rho, Lq.

    measurement_time: end of the arrival collection window (seconds).
    warmup_time: arrivals before this time are excluded from delay statistics.
    The SimPy clock runs until measurement_time + drain so customers who arrive
    near the end of the window can complete service (reduces high-rho bias).
    """
    if random_seed is not None:
        random.seed(random_seed)

    mean_service_s = 1.0 / mu_rate
    run_until = total_run_time(measurement_time, mean_service_s, drain_service_times)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=c)
    records = []
    env.process(customer_arrivals(env, server, lambda_rate, mu_rate, records))
    env.run(until=run_until)

    steady = [
        r
        for r in records
        if in_measurement_window(r["arrival_time"], warmup_time, measurement_time)
    ]
    if not steady:
        raise ValueError(
            "No customers in measurement window; increase run length or reduce warm-up."
        )
    waiting_times = [r["waiting_time"] for r in steady]
    system_times = [r["time_in_system"] for r in steady]
    mean_wait = statistics.mean(waiting_times)
    mean_system = statistics.mean(system_times)

    measurement_start, measurement_end = warmup_time, measurement_time
    busy_time = 0.0
    for r in records:
        overlap_start = max(r["service_start"], measurement_start)
        overlap_end = min(r["service_end"], measurement_end)
        if overlap_end > overlap_start:
            busy_time += overlap_end - overlap_start
    window = measurement_end - measurement_start
    utilisation = busy_time / (c * window)

    return {
        "Wq": mean_wait,
        "W": mean_system,
        "rho": utilisation,
        "utilisation": utilisation,
        "B": 0.0,
        "Lq": lambda_rate * mean_wait,
        "num_customers_kept": len(steady),
        "num_customers_total": len(records),
        "measurement_time": measurement_time,
        "run_until": run_until,
    }


# ==============================================================================
# SECTION: ANALYTICAL FORMULAS (Erlang C / M/M/c)
# ==============================================================================

def erlang_c_wait_probability(c, a):
    """P(wait > 0) = C(c, a). Uses an iterative product to avoid huge factorials."""
    if a <= 0:
        return 0.0
    rho = a / c
    if rho >= 1:
        raise ValueError("System unstable: a must be < c (rho < 1).")

    # sum_{k=0}^{c-1} a^k / k!
    term = 1.0
    sum_terms = 1.0
    for k in range(1, c):
        term *= a / k
        sum_terms += term
    # a^c / c!
    term *= a / c
    last = term / (1.0 - rho)
    p0 = 1.0 / (sum_terms + last)
    return last * p0


def theoretical_mmc(lambda_rate, mu_rate, c):
    """Steady-state M/M/c formulae (Kleinrock / standard Erlang-C delay)."""
    a = lambda_rate / mu_rate
    rho = a / c
    wait_prob = erlang_c_wait_probability(c, a)
    wq = wait_prob / (c * mu_rate - lambda_rate)
    w = wq + 1.0 / mu_rate
    lq = lambda_rate * wq
    return {"rho": rho, "a": a, "C": wait_prob, "Wq": wq, "W": w, "Lq": lq}


def erlang_b(c, a):
    """Blocking probability B(c, a) for an M/M/c/c (Erlang-B) loss system."""
    if a <= 0:
        return 0.0
    inv_b = 1.0
    for k in range(1, c + 1):
        inv_b = 1.0 + inv_b * k / a
    return 1.0 / inv_b


def offered_load_for_blocking(c, blocking, lo=1e-9, hi=None, tol=1e-10):
    """Largest offered load a (erlang) with B(c, a) <= blocking. Binary search."""
    if not 0 < blocking < 1:
        raise ValueError("blocking must be in (0, 1).")
    if hi is None:
        hi = float(c)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if erlang_b(c, mid) > blocking:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return lo


# ==============================================================================
# SECTION: VALIDATION OUTPUT
# ==============================================================================

def print_validation_table(sim_results, theory_results, lambda_rate, mu_rate, c, warmup_time):
    metrics = [
        ("Utilisation (rho)", "rho"),
        ("Mean wait in queue Wq (s)", "Wq"),
        ("Mean time in system W (s)", "W"),
        ("Mean queue length (Lq)", "Lq"),
    ]
    mean_service_s = 1.0 / mu_rate
    arrivals_per_hour = lambda_rate * SECONDS_PER_HOUR
    offered_rho = lambda_rate / (c * mu_rate)

    print("=" * 74)
    print("M/M/c VALIDATION  (time unit = seconds)")
    print(f"  Note: {INTERIM_PARAMETER_NOTE}")
    print(f"  c                 = {c} channels")
    print(f"  mean service 1/mu = {mean_service_s:.1f} s  ({mean_service_s / 60:.2f} min)")
    print(f"  mu                = {mu_rate:.6f} /s")
    print(f"  lambda            = {lambda_rate:.6f} /s  ({arrivals_per_hour:.2f} arrivals/hour)")
    print(f"  offered load a    = {lambda_rate / mu_rate:.3f} erlang")
    print(f"  offered rho       = {offered_rho:.3f}  (= lambda / (c mu))")
    dropped = sim_results["num_customers_total"] - sim_results["num_customers_kept"]
    print(
        f"Warm-up: arrivals before {warmup_time:.0f} s excluded; "
        f"measurement window [{warmup_time:.0f}, {sim_results['measurement_time']:.0f}) s; "
        f"SimPy run to {sim_results['run_until']:.0f} s (drain after measurement). "
        f"{dropped} of {sim_results['num_customers_total']} completed customers "
        f"outside the window."
    )
    print("=" * 74)
    print(f"{'Metric':<30}{'Simulated':>13}{'Theoretical':>13}{'% Deviation':>15}")
    print("-" * 74)
    for label, key in metrics:
        sim_val = sim_results[key]
        theory_val = theory_results[key]
        pct_deviation = (sim_val - theory_val) / theory_val * 100
        print(f"{label:<30}{sim_val:>13.4f}{theory_val:>13.4f}{pct_deviation:>14.2f}%")
    print("=" * 74)


def print_family_capacity_table(mean_service_s, server_counts, rho_values, acceptable_wait_s):
    """Theoretical Wq and arrivals/hour at each (c, rho)."""
    mu = 1.0 / mean_service_s
    print()
    print("=" * 86)
    print("FAMILY OF CURVES — theoretical M/M/c  (1/mu = "
          f"{mean_service_s:.0f} s = {mean_service_s / 60:.0f} min)")
    print(f"Acceptable-wait working assumption: Wq <= {acceptable_wait_s:.1f} s")
    print("=" * 86)
    print(
        f"{'c':>6}{'rho':>8}{'a (erlang)':>12}{'arrivals/h':>14}"
        f"{'Wq (s)':>12}{'OK?':>8}"
    )
    print("-" * 86)
    for c in server_counts:
        for rho in rho_values:
            lam = rho * c * mu
            theory = theoretical_mmc(lam, mu, c)
            arrivals_h = lam * SECONDS_PER_HOUR
            ok = "yes" if theory["Wq"] <= acceptable_wait_s else "no"
            print(
                f"{c:6d}{rho:8.2f}{theory['a']:12.2f}{arrivals_h:14.2f}"
                f"{theory['Wq']:12.2f}{ok:>8}"
            )
        print("-" * 86)
    print("=" * 86)


def print_erlang_b_table(mean_service_s, server_counts, blocking):
    """Offered load at a GoS blocking target (no waiting room)."""
    mu = 1.0 / mean_service_s
    print()
    print("=" * 86)
    print(
        f"ERLANG-B CAPACITY  (loss system, blocking target = {100 * blocking:.0f}%, "
        f"1/mu = {mean_service_s:.0f} s)"
    )
    print("No queue: a blocked call is lost. Arrivals/h = a * mu * 3600.")
    print("=" * 86)
    print(
        f"{'c':>6}{'a (erlang)':>14}{'rho=a/c':>12}{'arrivals/h':>14}"
        f"{'B(c,a)':>12}"
    )
    print("-" * 86)
    for c in server_counts:
        if c < 1:
            continue
        a = offered_load_for_blocking(c, blocking)
        arrivals_h = a * mu * SECONDS_PER_HOUR
        print(
            f"{c:6d}{a:14.2f}{a / c:12.3f}{arrivals_h:14.2f}"
            f"{erlang_b(c, a):12.4f}"
        )
    print("=" * 86)


def logspace_rho(rho_min, rho_max, n_points):
    """Log-spaced rho values for dense Erlang-C theory curves (no SimPy)."""
    if rho_min <= 0 or rho_max <= rho_min or n_points < 2:
        raise ValueError("Need 0 < rho_min < rho_max and n_points >= 2.")
    log_min = math.log10(rho_min)
    log_max = math.log10(rho_max)
    step = (log_max - log_min) / (n_points - 1)
    return [10 ** (log_min + i * step) for i in range(n_points)]


def erlang_c_theory_wq_vs_arrivals(c, mu_rate, rho_min, rho_max, n_points):
    """Fine Erlang-C Wq curve vs arrivals/h from theoretical_mmc only."""
    arrivals_h = []
    wq_s = []
    for rho in logspace_rho(rho_min, rho_max, n_points):
        if rho >= 1.0:
            continue
        lam = rho * c * mu_rate
        theory = theoretical_mmc(lam, mu_rate, c)
        arrivals_h.append(lam * SECONDS_PER_HOUR)
        wq_s.append(theory["Wq"])
    return arrivals_h, wq_s


# Representative c values for SimPy markers on Wq vs c family plots.
SIM_C_MARKERS_FOR_FAMILY = [1, 4, 8, 12, 21, 30, 45]

# Fine log-spaced arrivals/h for Wq surface + heatmap (closed-form; cheap to refine).
WQ_SURFACE_ARRIVALS_H_MIN = 1.0
WQ_SURFACE_ARRIVALS_H_MAX = 2000.0
WQ_SURFACE_ARRIVALS_N = 800

WQ_SURFACE_RHO_MIN = 0.01
WQ_SURFACE_RHO_MAX = 0.98
WQ_SURFACE_RHO_N = 400


def default_wq_surface_rho_values(n_points=None):
    """Rho grid for Wq 3D surface; denser as rho -> 1 (log-spaced in 1 - rho)."""
    n = int(n_points if n_points is not None else WQ_SURFACE_RHO_N)
    gap = np.logspace(
        np.log10(1.0 - WQ_SURFACE_RHO_MAX),
        np.log10(1.0 - WQ_SURFACE_RHO_MIN),
        n,
    )
    return 1.0 - gap


def default_wq_surface_arrivals_h(n_points=None):
    """Log-spaced arrivals/hour grid for Wq 3D surface and heatmap."""
    n = int(n_points if n_points is not None else WQ_SURFACE_ARRIVALS_N)
    return np.logspace(
        np.log10(WQ_SURFACE_ARRIVALS_H_MIN),
        np.log10(WQ_SURFACE_ARRIVALS_H_MAX),
        n,
    )


def _build_wq_grid(mean_service_s, c_range, arrivals_h_range):
    """Mesh of theoretical Erlang-C Wq (seconds); NaN where lambda >= c*mu."""
    mu_rate = 1.0 / mean_service_s
    c_vals = np.asarray(list(c_range), dtype=float)
    arrivals_h = np.asarray(arrivals_h_range, dtype=float)
    c_grid, arrivals_grid = np.meshgrid(c_vals, arrivals_h, indexing="ij")
    wq = np.full(c_grid.shape, np.nan, dtype=float)
    lambda_rates = arrivals_h / SECONDS_PER_HOUR
    offered_a = lambda_rates / mu_rate

    for i, c in enumerate(c_vals):
        c_int = int(c)
        stable = offered_a < c - 1e-6
        for j in np.flatnonzero(stable):
            wq[i, j] = theoretical_mmc(lambda_rates[j], mu_rate, c_int)["Wq"]
    return c_vals, arrivals_h, c_grid, arrivals_grid, wq


def _build_wq_grid_rho(mean_service_s, c_range, rho_values):
    """Mesh of Erlang-C Wq over (c, rho); rho < 1 on grid by construction."""
    mu_rate = 1.0 / mean_service_s
    c_vals = np.asarray(list(c_range), dtype=float)
    rho = np.asarray(rho_values, dtype=float)
    c_grid, rho_grid = np.meshgrid(c_vals, rho, indexing="ij")
    wq = np.zeros(c_grid.shape, dtype=float)

    for i, c in enumerate(c_vals):
        c_int = int(c)
        for j, r in enumerate(rho):
            lambda_rate = r * c_int * mu_rate
            wq[i, j] = theoretical_mmc(lambda_rate, mu_rate, c_int)["Wq"]
    return c_vals, rho, c_grid, rho_grid, wq


def _save_wait_surface_views(
    c_grid, log10_arrivals_grid, wq_capped, z_cap, mean_service_s, views
):
    """Write 3D Erlang-C Wq surfaces at multiple camera angles."""
    apply_style()
    z_label = rf"$W_q$ (s), capped at {z_cap:.0f} s near instability"
    wq_masked = np.ma.array(wq_capped, mask=np.isnan(wq_capped))

    for filename, elev, azim in views:
        fig = plt.figure(figsize=SINGLE)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            c_grid,
            log10_arrivals_grid,
            wq_masked,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.12, label=z_label)
        ax.set_xlabel("Number of channels, $c$")
        ax.set_ylabel(r"$\log_{10}$(arrivals/hour)")
        ax.set_zlabel(z_label)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        png, _pdf = save_figure(fig, filename)
        plt.close(fig)
        print(f"Saved {png}")


def _save_wait_surface_views_rho(c_grid, rho_grid, wq_capped, z_cap, mean_service_s, views):
    """Write 3D Erlang-C Wq surfaces over (c, rho)."""
    apply_style()
    z_label = rf"$W_q$ (s), capped at {z_cap:.0f} s near instability"

    for filename, elev, azim in views:
        fig = plt.figure(figsize=SINGLE)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            c_grid,
            rho_grid,
            wq_capped,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=z_label)
        ax.set_xlabel("Number of channels, $c$")
        ax.set_ylabel(r"Utilisation, $\rho = \lambda / (c\mu)$")
        ax.set_zlabel(z_label)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        png, _pdf = save_figure(fig, filename)
        plt.close(fig)
        print(f"Saved {png}")


def plot_wait_surface_3d(
    mean_service_s,
    c_range,
    arrivals_h_range,
    acceptable_wait_s,
    z_cap=None,
):
    """3D surface of capped Erlang-C Wq over (c, arrivals/hour)."""
    if z_cap is None:
        z_cap = 300.0
    _, _, c_grid, arrivals_grid, wq = _build_wq_grid(
        mean_service_s, c_range, arrivals_h_range
    )
    wq_capped = np.where(np.isnan(wq), np.nan, np.minimum(wq, z_cap))
    log10_arrivals = np.log10(np.maximum(arrivals_grid, 1e-9))
    _save_wait_surface_views(
        c_grid,
        log10_arrivals,
        wq_capped,
        z_cap,
        mean_service_s,
        [
            ("mmc_wait_surface_3d.png", 25, -60),
            ("mmc_wait_surface_3d_view2.png", 25, -120),
        ],
    )


def plot_wait_surface_3d_rho(
    mean_service_s,
    c_range,
    rho_values,
    acceptable_wait_s,
    z_cap=None,
):
    """3D surface of capped Erlang-C Wq over (c, rho)."""
    if z_cap is None:
        z_cap = 300.0
    _, _, c_grid, rho_grid, wq = _build_wq_grid_rho(
        mean_service_s, c_range, rho_values
    )
    wq_capped = np.minimum(wq, z_cap)
    _save_wait_surface_views_rho(
        c_grid,
        rho_grid,
        wq_capped,
        z_cap,
        mean_service_s,
        [
            ("mmc_wait_surface_3d_rho.png", 25, -60),
            ("mmc_wait_surface_3d_rho_view2.png", 25, -120),
        ],
    )


def plot_wait_heatmap(mean_service_s, c_range, arrivals_h_range, acceptable_wait_s):
    """2D heatmap of Erlang-C Wq with contour at acceptable wait."""
    c_vals, arrivals_h, _, _, wq = _build_wq_grid(
        mean_service_s, c_range, arrivals_h_range
    )
    wq_plot = wq.T
    positive = wq_plot[np.isfinite(wq_plot) & (wq_plot > 0)]
    vmin = max(float(np.min(positive)), 1e-2) if positive.size else 1e-2
    vmax = float(np.max(positive)) if positive.size else 1.0

    from matplotlib.patches import Patch

    apply_style()
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.set_facecolor("#d9d9d9")
    mesh = ax.pcolormesh(
        c_vals,
        arrivals_h,
        np.ma.masked_invalid(wq_plot),
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(mesh, ax=ax, label=r"Mean waiting time in queue, $W_q$ (s)")
    unstable = np.isnan(wq_plot).astype(float)
    ax.contourf(
        c_vals,
        arrivals_h,
        unstable,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
    )
    contour_data = np.where(np.isfinite(wq_plot), wq_plot, np.nan)
    ax.contour(
        c_vals,
        arrivals_h,
        contour_data,
        levels=[acceptable_wait_s],
        colors="white",
        linewidths=2,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                label=f"$W_q$ = {acceptable_wait_s:.0f} s",
            ),
            Patch(
                facecolor="#d9d9d9",
                edgecolor="0.3",
                hatch="////",
                label=r"unstable ($\lambda \geq c\mu$)",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of channels, $c$")
    ax.set_ylabel("Arrivals per hour, $\\lambda$ (hour$^{-1}$)")
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_wait_heatmap")
    plt.close(fig)
    print(f"Saved {png}")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()


# ==============================================================================
# SECTION: PLOTTING
# ==============================================================================

def plot_wait_vs_rho_family(
    mean_service_s,
    server_counts,
    rho_values,
    acceptable_wait_s,
    random_seed=None,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
):
    """Wq vs rho = lambda/(c mu). Markers are replication means with a 95% CI."""
    apply_style()
    mu_rate = 1.0 / mean_service_s
    markers = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=SINGLE)
    for idx, c in enumerate(server_counts):
        means, lows, highs, theoretical_wq = [], [], [], []
        for i, rho in enumerate(rho_values):
            plan = plan_run(
                mean_service_s, rho, c=c,
                base_measurement=base_measurement,
                scale_factor=scale_factor,
                base_warmup=base_warmup,
            )
            lambda_rate = rho * c * mu_rate
            base_seed = 0 if random_seed is None else random_seed + 1000 * c + 20 * i

            def once(seed, lambda_rate=lambda_rate, plan=plan, c=c):
                return run_mmc_simulation(
                    lambda_rate, mu_rate, c,
                    plan["measurement_time"], plan["warmup_time"],
                    random_seed=seed,
                )

            summary = run_replications(once, n_replications, base_seed, ("Wq",))
            means.append(summary["Wq"]["mean"])
            lows.append(summary["Wq"]["low"])
            highs.append(summary["Wq"]["high"])
            theoretical_wq.append(theoretical_mmc(lambda_rate, mu_rate, c)["Wq"])
            print(
                f"  sweep c={c:2d}  rho={rho:.2f}  "
                f"Wq={summary['Wq']['mean']:.2f} "
                f"CI=[{summary['Wq']['low']:.2f}, {summary['Wq']['high']:.2f}] "
                f"theory={theoretical_wq[-1]:.2f}"
            )

        colour = channel_colour(c)
        label = SERVER_LABELS.get(c, f"$c = {c}$")
        # Below ~10 ms the wait is negligible next to the 10 s target, and the
        # exact Erlang-C value at low rho can be 10^-20 s, which wrecks a log axis.
        theory_y = [y if y >= 1e-2 else float("nan") for y in theoretical_wq]
        ax.plot(rho_values, theory_y, "-", color=colour, label=label)
        log_scale_errorbar(
            ax, rho_values, means, lows, highs,
            fmt=markers[idx], color=colour, markerfacecolor="white",
            capsize=3, linestyle="none",
        )

    ax.axhline(
        acceptable_wait_s,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"{acceptable_wait_s:.0f} s",
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel(r"Utilisation, $\rho = \lambda / (c\mu)$")
    ax.set_ylabel("Mean waiting time in queue, $W_q$ (seconds)")
    ax.legend()
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_wait_vs_rho")
    plt.close(fig)
    print(f"Saved {png}")
    return png


def plot_wait_vs_arrivals_family(
    mean_service_s,
    server_counts,
    rho_values,
    n_service_times,
    warmup_service_times,
    acceptable_wait_s,
    random_seed=None,
    theory_rho_min=0.005,
    theory_rho_max=0.9,
    n_theory_points=300,
):
    """Erlang-C Wq vs arrivals/hour (one curve per c). Log x-axis: c=1 and c=45 differ ~45x in range."""
    apply_style()
    mu_rate = 1.0 / mean_service_s
    markers = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=SINGLE)
    for idx, c in enumerate(server_counts):
        theory_arrivals_h, theory_wq = erlang_c_theory_wq_vs_arrivals(
            c, mu_rate, theory_rho_min, theory_rho_max, n_theory_points
        )

        n_run, warmup_base = scaled_service_time_counts(
            c, n_service_times, warmup_service_times
        )
        measurement_time = n_run * mean_service_s

        sim_arrivals_h = []
        simulated_wq = []

        for i, rho in enumerate(rho_values):
            warmup_run = warmup_service_times_for_rho(warmup_base, n_run, rho)
            warmup_time = warmup_run * mean_service_s
            lambda_rate = rho * c * mu_rate
            arrivals_h = lambda_rate * SECONDS_PER_HOUR
            seed = None if random_seed is None else random_seed + 3000 * c + i
            sim_result = run_mmc_simulation(
                lambda_rate, mu_rate, c, measurement_time, warmup_time, seed
            )
            sim_arrivals_h.append(arrivals_h)
            simulated_wq.append(sim_result["Wq"])

        colour = channel_colour(c)
        label = SERVER_LABELS.get(c, f"$c = {c}$")
        ax.plot(theory_arrivals_h, theory_wq, "-", color=colour, label=label)
        ax.plot(
            sim_arrivals_h,
            simulated_wq,
            markers[idx],
            color=colour,
            markerfacecolor="white",
            label="_nolegend_",
        )

    ax.axhline(
        acceptable_wait_s,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"{acceptable_wait_s:.0f} s",
    )
    ax.set_xlabel("Arrivals per hour, $\\lambda$ (hour$^{-1}$)")
    ax.set_ylabel("Mean waiting time in queue, $W_q$ (seconds)")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_wait_vs_arrivals")
    plt.close(fig)
    print(f"Saved {png}")


def plot_wait_vs_c_family(
    mean_service_s,
    arrival_rates_per_hour,
    c_range,
    n_service_times,
    warmup_service_times,
    acceptable_wait_s,
    random_seed=None,
    sim_c_values=None,
):
    """Erlang-C Wq vs channel count c at fixed arrival rates (theory dense, SimPy sparse)."""
    apply_style()
    mu_rate = 1.0 / mean_service_s
    c_min, c_max = c_range
    c_theory = list(range(int(c_min), int(c_max) + 1))
    if sim_c_values is None:
        sim_c_values = SIM_C_MARKERS_FOR_FAMILY

    markers = ["o", "s", "D", "^", "v"]
    colours = [channel_colour(c) for c in (1, 7, 8, 21, 45)]

    fig, ax = plt.subplots(figsize=SINGLE)
    for idx, arrivals_h in enumerate(arrival_rates_per_hour):
        lambda_rate = arrivals_h / SECONDS_PER_HOUR
        a_fixed = lambda_rate / mu_rate
        c_plot = []
        wq_theory = []
        for c in c_theory:
            # Erlang-C Wq blows up as rho -> 1; require strict stability (a < c).
            if a_fixed >= c - 1e-6:
                continue
            wq_theory.append(theoretical_mmc(lambda_rate, mu_rate, c)["Wq"])
            c_plot.append(c)

        colour = colours[idx % len(colours)]
        label = rf"$\lambda = {arrivals_h:g}$ h$^{{-1}}$"
        ax.plot(c_plot, wq_theory, "-", color=colour, label=label)

        sim_c = [c for c in sim_c_values if a_fixed < c - 1e-6]
        sim_wq = []
        for j, c in enumerate(sim_c):
            n_run, warmup_base = scaled_service_time_counts(
                c, n_service_times, warmup_service_times
            )
            measurement_time = n_run * mean_service_s
            rho_at_c = lambda_rate / (c * mu_rate)
            warmup_run = warmup_service_times_for_rho(warmup_base, n_run, rho_at_c)
            warmup_time = warmup_run * mean_service_s
            seed = None if random_seed is None else random_seed + 4000 * idx + j
            sim_result = run_mmc_simulation(
                lambda_rate, mu_rate, c, measurement_time, warmup_time, seed
            )
            sim_wq.append(sim_result["Wq"])
            print(
                f"  Wq vs c  lambda={arrivals_h:g}/h  c={c:2d}  "
                f"Wq_sim={sim_result['Wq']:.2f} s  "
                f"Wq_th={theoretical_mmc(lambda_rate, mu_rate, c)['Wq']:.2f} s"
            )

        ax.plot(
            sim_c,
            sim_wq,
            markers[idx % len(markers)],
            color=colour,
            markerfacecolor="white",
            label="_nolegend_",
        )

    ax.axhline(
        acceptable_wait_s,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"{acceptable_wait_s:.0f} s",
    )
    ax.set_xlabel("Number of channels, $c$")
    ax.set_ylabel("Mean waiting time in queue, $W_q$ (seconds)")
    ax.legend()
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_wait_vs_c_family")
    plt.close(fig)
    print(f"Saved {png}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    sim_results = run_mmc_simulation(
        LAMBDA_RATE, MU_RATE, DEMO_C, SIM_TIME, WARMUP_TIME, random_seed=RANDOM_SEED
    )
    theory_results = theoretical_mmc(LAMBDA_RATE, MU_RATE, DEMO_C)
    print_validation_table(
        sim_results, theory_results, LAMBDA_RATE, MU_RATE, DEMO_C, WARMUP_TIME
    )

    mm1_at_same_rho = theoretical_mmc(DEMO_RHO * 1 * MU_RATE, MU_RATE, 1)
    print(
        f"\nSame rho = {DEMO_RHO:.2f}, same 1/mu = {PRIMARY_MEAN_SERVICE_S:.0f} s:\n"
        f"  c = 1  -> Wq = {mm1_at_same_rho['Wq']:.1f} s  "
        f"({mm1_at_same_rho['Wq'] / 60:.1f} min), "
        f"{DEMO_RHO * 1 * MU_RATE * SECONDS_PER_HOUR:.1f} arrivals/h\n"
        f"  c = {DEMO_C}  -> Wq = {theory_results['Wq']:.1f} s, "
        f"{LAMBDA_RATE * SECONDS_PER_HOUR:.1f} arrivals/h"
    )

    capacity_rhos = [0.50, 0.70, 0.80, 0.90]
    print_family_capacity_table(
        PRIMARY_MEAN_SERVICE_S, SERVER_COUNTS, capacity_rhos, ACCEPTABLE_WAIT_S
    )
    print_erlang_b_table(PRIMARY_MEAN_SERVICE_S, SERVER_COUNTS, BLOCKING_TARGET)

    SWEEP_N_SERVICE_TIMES = 20_000
    SWEEP_WARMUP_SERVICE_TIMES = 2_000
    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    print("\nRunning family-of-curves sweep (replications vs theory)...")
    plot_wait_vs_rho_family(
        PRIMARY_MEAN_SERVICE_S,
        SERVER_COUNTS,
        rho_values,
        ACCEPTABLE_WAIT_S,
        random_seed=RANDOM_SEED,
    )

    print("\nRunning Erlang-C Wq vs arrivals/hour (log x-axis)...")
    plot_wait_vs_arrivals_family(
        PRIMARY_MEAN_SERVICE_S,
        SERVER_COUNTS,
        rho_values,
        SWEEP_N_SERVICE_TIMES,
        SWEEP_WARMUP_SERVICE_TIMES,
        ACCEPTABLE_WAIT_S,
        random_seed=RANDOM_SEED,
    )

    FAMILY_ARRIVAL_RATES_H = [50, 100, 200, 400, 800]
    print("\nRunning Erlang-C Wq vs c at fixed arrival rates...")
    plot_wait_vs_c_family(
        PRIMARY_MEAN_SERVICE_S,
        FAMILY_ARRIVAL_RATES_H,
        (1, 50),
        SWEEP_N_SERVICE_TIMES,
        SWEEP_WARMUP_SERVICE_TIMES,
        ACCEPTABLE_WAIT_S,
        random_seed=RANDOM_SEED,
    )

    SURFACE_C_RANGE = range(1, 51)
    SURFACE_ARRIVALS_H = default_wq_surface_arrivals_h()
    print(
        f"\nRunning Erlang-C Wq 3D surface and heatmap "
        f"({len(SURFACE_ARRIVALS_H)} log-spaced arrivals/h points)..."
    )
    plot_wait_surface_3d(
        PRIMARY_MEAN_SERVICE_S,
        SURFACE_C_RANGE,
        SURFACE_ARRIVALS_H,
        ACCEPTABLE_WAIT_S,
    )
    SURFACE_RHO = default_wq_surface_rho_values()
    print(
        f"\nRunning Erlang-C Wq 3D surface (rho axis, {len(SURFACE_RHO)} points)..."
    )
    plot_wait_surface_3d_rho(
        PRIMARY_MEAN_SERVICE_S,
        SURFACE_C_RANGE,
        SURFACE_RHO,
        ACCEPTABLE_WAIT_S,
    )
    plot_wait_heatmap(
        PRIMARY_MEAN_SERVICE_S,
        SURFACE_C_RANGE,
        SURFACE_ARRIVALS_H,
        ACCEPTABLE_WAIT_S,
    )
