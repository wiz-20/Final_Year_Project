"""
M/M/c/c Loss System (Erlang B) — SimPy with No Waiting Room
=============================================================

Complements mmc_simulation.py (Erlang C / infinite buffer). Arrivals that find
all c servers busy are blocked (lost), not queued.
"""

import math
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import simpy
from matplotlib.patches import Patch

from sim.paths import figure_path
from sim.plot_style import SINGLE, THRESHOLD, apply_style, channel_colour, save_figure
from sim.replication import DEFAULT_REPLICATIONS, run_replications
from sim.literature_params import (
    BLOCKING_TARGET,
    INTERIM_PARAMETER_NOTE,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
    SERVER_COUNTS,
    SERVER_LABELS,
)
from sim.mmc_simulation import (
    SIM_C_MARKERS_FOR_FAMILY,
    erlang_b,
    offered_load_for_blocking,
)
from sim.warmup import (
    in_measurement_window,
    scaled_service_time_counts_b,
    total_run_time,
    warmup_service_times_for_rho,
)

DRAIN_SERVICE_TIMES = 50.0


def arrivals_per_hour(lambda_rate):
    return lambda_rate * SECONDS_PER_HOUR


def lambda_from_rho_c(rho, c, mu_rate):
    return rho * c * mu_rate


def offered_load_erlangs(lambda_rate, mu_rate):
    return lambda_rate / mu_rate


def logspace_rho(rho_min, rho_max, n_points):
    """Log-spaced rho values for dense Erlang-B theory curves (no SimPy)."""
    if rho_min <= 0 or rho_max <= rho_min or n_points < 2:
        raise ValueError("Need 0 < rho_min < rho_max and n_points >= 2.")
    log_min = math.log10(rho_min)
    log_max = math.log10(rho_max)
    step = (log_max - log_min) / (n_points - 1)
    return [10 ** (log_min + i * step) for i in range(n_points)]


def erlang_b_theory_vs_c(lambda_rate, mu_rate, c_min, c_max):
    """Fine Erlang-B curve vs c at fixed lambda (erlang_b only)."""
    a = offered_load_erlangs(lambda_rate, mu_rate)
    c_list = list(range(int(c_min), int(c_max) + 1))
    blocking_pct = [100.0 * erlang_b(c, a) for c in c_list]
    return c_list, blocking_pct


def erlang_b_theory_vs_arrivals(c, mu_rate, rho_min, rho_max, n_points):
    """Fine Erlang-B curve: arrivals/hour and B(%) from erlang_b only."""
    arrivals_h = []
    blocking_pct = []
    for rho in logspace_rho(rho_min, rho_max, n_points):
        a = rho * c
        lam = rho * c * mu_rate
        arrivals_h.append(arrivals_per_hour(lam))
        blocking_pct.append(100.0 * erlang_b(c, a))
    return arrivals_h, blocking_pct


# ==============================================================================
# SIMULATION (M/M/c/c)
# ==============================================================================

def blocked_customer(
    env, server, mu_rate, stats, warmup_time, measurement_time,
    max_arrivals=None, warmup_arrivals=0,
):
    """Serve if a channel is free at arrival; otherwise count as blocked.

    Blocking is decided at the arrival instant (a free channel exists or not).
    Accepted calls are counted then, not after service ends, so the run can
    stop without dropping in-service calls from the blocking ratio.

    The default window is the clock interval [warmup_time, measurement_time).
    When max_arrivals is set, the window is an arrival count instead: the
    first warmup_arrivals calls still take a channel and are left out of B,
    and the next max_arrivals calls are the measurement.
    """
    arrival_time = env.now
    stats["total_arrivals"] += 1
    if max_arrivals is None:
        in_window = in_measurement_window(arrival_time, warmup_time, measurement_time)
    else:
        index = stats["total_arrivals"]
        in_window = warmup_arrivals < index <= warmup_arrivals + max_arrivals

    if server.count >= server.capacity:
        if in_window:
            stats["blocked"] += 1
        return

    if in_window:
        stats["served"] += 1
    with server.request() as req:
        yield req
        service_start = env.now
        service_time = random.expovariate(mu_rate)
        yield env.timeout(service_time)
        service_end = env.now
    stats["occupancies"].append((service_start, service_end))
    if in_window:
        stats["waits"].append(service_start - arrival_time)
        stats["sojourns"].append(service_end - arrival_time)


def blocked_customer_arrivals(
    env, server, lambda_rate, mu_rate, stats, warmup_time, measurement_time,
    max_arrivals=None, warmup_arrivals=0,
):
    # Arrival mode: SimPy runs the new customer only after this process yields,
    # so the count is one behind at the while-check and one extra call is
    # generated. That call's index is past the measurement window, so it is
    # not included in B. Clock mode is the original infinite loop.
    target = None if max_arrivals is None else warmup_arrivals + max_arrivals
    while target is None or stats["total_arrivals"] < target:
        yield env.timeout(random.expovariate(lambda_rate))
        env.process(
            blocked_customer(
                env, server, mu_rate, stats, warmup_time, measurement_time,
                max_arrivals=max_arrivals, warmup_arrivals=warmup_arrivals,
            )
        )


def run_mmc_b_simulation(
    lambda_rate, mu_rate, c, measurement_time, warmup_time, random_seed=None,
    max_arrivals=None, warmup_arrivals=None,
):
    """One M/M/c/c run. Returns simulated blocking probability B (post warm-up).

    Default: collect arrivals whose clock time is in [warmup_time, measurement_time).
    Optional arrival stop: set max_arrivals to the number of calls included in B
    and warmup_arrivals to the number of leading calls left out of B. The clock
    arguments are then unused for the blocking ratio. Omit both to keep the
    clock window.
    """
    if (max_arrivals is None) != (warmup_arrivals is None):
        raise ValueError("Pass max_arrivals and warmup_arrivals together, or neither.")
    if max_arrivals is not None and (max_arrivals <= 0 or warmup_arrivals < 0):
        raise ValueError("max_arrivals must be positive and warmup_arrivals non-negative.")
    if random_seed is not None:
        random.seed(random_seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=c)
    stats = {
        "total_arrivals": 0,
        "blocked": 0,
        "served": 0,
        "occupancies": [],
        "waits": [],
        "sojourns": [],
    }

    env.process(
        blocked_customer_arrivals(
            env,
            server,
            lambda_rate,
            mu_rate,
            stats,
            warmup_time,
            measurement_time,
            max_arrivals=max_arrivals,
            warmup_arrivals=0 if warmup_arrivals is None else warmup_arrivals,
        )
    )
    mean_service_s = 1.0 / mu_rate
    if max_arrivals is None:
        run_until = total_run_time(measurement_time, mean_service_s, DRAIN_SERVICE_TIMES)
        env.run(until=run_until)
    else:
        env.run()
        run_until = env.now

    post_warmup = stats["blocked"] + stats["served"]
    blocking_sim = stats["blocked"] / post_warmup if post_warmup else 0.0
    a = offered_load_erlangs(lambda_rate, mu_rate)
    blocking_theory = erlang_b(c, a)

    window = measurement_time - warmup_time
    busy_time = 0.0
    for service_start, service_end in stats["occupancies"]:
        overlap_start = max(service_start, warmup_time)
        overlap_end = min(service_end, measurement_time)
        if overlap_end > overlap_start:
            busy_time += overlap_end - overlap_start
    utilisation = busy_time / (c * window) if window > 0 else 0.0
    mean_wait = statistics.mean(stats["waits"]) if stats["waits"] else 0.0
    mean_system = statistics.mean(stats["sojourns"]) if stats["sojourns"] else 0.0
    accepted_rate = (stats["served"] / window) if window > 0 else 0.0

    return {
        "B": blocking_sim,
        "B_theory": blocking_theory,
        "blocked": stats["blocked"],
        "served": stats["served"],
        "post_warmup_arrivals": post_warmup,
        "total_arrivals": stats["total_arrivals"],
        "a": a,
        "rho": a / c,
        "utilisation": utilisation,
        "Wq": mean_wait,
        "W": mean_system,
        "Lq": accepted_rate * mean_wait,
        "measurement_time": measurement_time,
        "run_until": run_until,
    }


# ==============================================================================
# VALIDATION OUTPUT
# ==============================================================================

def print_blocking_validation_table(
    sim_results, lambda_rate, mu_rate, c, warmup_time, blocking_theory
):
    mean_service_s = 1.0 / mu_rate
    arrivals_h = arrivals_per_hour(lambda_rate)
    a = sim_results["a"]
    pct_dev = (sim_results["B"] - blocking_theory) / blocking_theory * 100

    print("=" * 74)
    print("M/M/c/c (ERLANG-B) VALIDATION  (time unit = seconds)")
    print(f"  Note: {INTERIM_PARAMETER_NOTE}")
    print(f"  c                 = {c} channels  (no waiting room)")
    print(f"  mean service 1/mu = {mean_service_s:.1f} s  ({mean_service_s / 60:.2f} min)")
    print(f"  mu                = {mu_rate:.6f} /s")
    print(f"  lambda            = {lambda_rate:.6f} /s  ({arrivals_h:.2f} arrivals/hour)")
    print(f"  offered load a    = {a:.4f} erlang")
    print(f"  rho = a/c         = {a / c:.4f}")
    print(
        f"Warm-up: arrivals before {warmup_time:.0f} s excluded; "
        f"measurement window [{warmup_time:.0f}, {sim_results['measurement_time']:.0f}) s. "
        f"{sim_results['total_arrivals'] - sim_results['post_warmup_arrivals']} "
        f"of {sim_results['total_arrivals']} arrivals outside the window."
    )
    print("=" * 74)
    print(f"{'Metric':<30}{'Simulated':>13}{'Theoretical':>13}{'% Deviation':>15}")
    print("-" * 74)
    print(
        f"{'Blocking probability B':<30}"
        f"{sim_results['B']:>13.6f}{blocking_theory:>13.6f}{pct_dev:>14.2f}%"
    )
    print("=" * 74)
    return pct_dev


def print_family_blocking_sweep_table(mean_service_s, server_counts, rho_values):
    """Erlang-B blocking at each (c, rho) with explicit arrivals/hour."""
    mu = 1.0 / mean_service_s
    print()
    print("=" * 92)
    print(
        "FAMILY OF CURVES — theoretical ERLANG-B (loss, no queue)  "
        f"(1/mu = {mean_service_s:.0f} s)"
    )
    print(f"Working blocking target reference: {100 * BLOCKING_TARGET:.0f}%")
    print("=" * 92)
    print(
        f"{'c':>6}{'rho':>8}{'a (erlang)':>12}{'arrivals/h':>14}"
        f"{'B (%)':>12}{'OK?':>8}"
    )
    print("-" * 92)
    for c in server_counts:
        for rho in rho_values:
            lam = lambda_from_rho_c(rho, c, mu)
            a = lam / mu
            b = erlang_b(c, a)
            arrivals_h = lam * SECONDS_PER_HOUR
            ok = "yes" if b <= BLOCKING_TARGET else "no"
            print(
                f"{c:6d}{rho:8.2f}{a:12.2f}{arrivals_h:14.2f}"
                f"{100 * b:12.3f}{ok:>8}"
            )
        print("-" * 92)
    print("=" * 92)


# ==============================================================================
# PLOTTING
# ==============================================================================

def _build_blocking_grid(mean_service_s, c_range, arrivals_h_range):
    """Mesh of Erlang-B blocking B in percent over (c, arrivals/hour)."""
    mu_rate = 1.0 / mean_service_s
    c_vals = np.asarray(list(c_range), dtype=float)
    arrivals_h = np.asarray(arrivals_h_range, dtype=float)
    c_grid, arrivals_grid = np.meshgrid(c_vals, arrivals_h, indexing="ij")
    blocking_pct = np.zeros(c_grid.shape, dtype=float)

    for i, c in enumerate(c_vals):
        c_int = int(c)
        for j, ah in enumerate(arrivals_h):
            lambda_rate = ah / SECONDS_PER_HOUR
            a = offered_load_erlangs(lambda_rate, mu_rate)
            blocking_pct[i, j] = 100.0 * erlang_b(c_int, a)
    return c_vals, arrivals_h, c_grid, arrivals_grid, blocking_pct


def _save_blocking_surface_views(
    c_grid, log10_arrivals_grid, blocking_pct, mean_service_s, views
):
    apply_style()
    z_label = "Blocking probability"

    for filename, elev, azim in views:
        fig = plt.figure(figsize=SINGLE)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            c_grid,
            log10_arrivals_grid,
            blocking_pct,
            cmap="plasma",
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


def plot_blocking_surface_3d(
    mean_service_s, c_range, arrivals_h_range, blocking_target
):
    """3D surface of Erlang-B blocking over (c, arrivals/hour)."""
    _, _, c_grid, arrivals_grid, blocking_pct = _build_blocking_grid(
        mean_service_s, c_range, arrivals_h_range
    )
    log10_arrivals = np.log10(np.maximum(arrivals_grid, 1e-9))
    _save_blocking_surface_views(
        c_grid,
        log10_arrivals,
        blocking_pct,
        mean_service_s,
        [
            ("mmc_blocking_surface_3d.png", 25, -60),
            ("mmc_blocking_surface_3d_view2.png", 25, -120),
        ],
    )


def plot_blocking_heatmap(
    mean_service_s, c_range, arrivals_h_range, blocking_target
):
    """2D heatmap of Erlang-B blocking with contour at target B."""
    apply_style()
    c_vals, arrivals_h, _, _, blocking_pct = _build_blocking_grid(
        mean_service_s, c_range, arrivals_h_range
    )
    b_plot = blocking_pct.T
    target_pct = 100.0 * blocking_target

    fig, ax = plt.subplots(figsize=SINGLE)
    mesh = ax.pcolormesh(
        c_vals,
        arrivals_h,
        b_plot,
        shading="auto",
        cmap="plasma",
    )
    fig.colorbar(mesh, ax=ax, label="Blocking probability")
    ax.contour(
        c_vals,
        arrivals_h,
        b_plot,
        levels=[target_pct],
        colors="white",
        linewidths=2,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                label=f"$B$ = {target_pct:.0f}%",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of channels, $c$")
    ax.set_ylabel("Arrivals per hour, $\\lambda$ (hour$^{-1}$)")
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_blocking_heatmap")
    plt.close(fig)
    print(f"Saved {png}")


def plot_blocking_vs_arrivals_family(
    mean_service_s,
    server_counts,
    rho_values,
    n_service_times,
    warmup_service_times,
    blocking_target,
    random_seed=None,
    n_replications=DEFAULT_REPLICATIONS,
    theory_rho_min=0.005,
    theory_rho_max=0.9,
    n_theory_points=300,
):
    """Blocking probability vs arrivals/hour (Erlang B), one curve per c.

    Markers are replication means with a 95% CI. The run length is the same
    clock window this sweep has always used.
    """
    apply_style()
    mu_rate = 1.0 / mean_service_s
    markers = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=SINGLE)
    for idx, c in enumerate(server_counts):
        theory_arrivals_h, theory_b_pct = erlang_b_theory_vs_arrivals(
            c, mu_rate, theory_rho_min, theory_rho_max, n_theory_points
        )

        n_run, warmup_base = scaled_service_time_counts_b(
            c, n_service_times, warmup_service_times
        )
        measurement_time = n_run * mean_service_s

        sim_arrivals_h = []
        sim_b = []
        sim_low = []
        sim_high = []

        for i, rho in enumerate(rho_values):
            warmup_run = warmup_service_times_for_rho(warmup_base, n_run, rho)
            warmup_time = warmup_run * mean_service_s
            lambda_rate = lambda_from_rho_c(rho, c, mu_rate)
            a = offered_load_erlangs(lambda_rate, mu_rate)
            arrivals_h = arrivals_per_hour(lambda_rate)
            base_seed = 0 if random_seed is None else random_seed + 2000 * c + 20 * i

            def once(seed, lambda_rate=lambda_rate, c=c, measurement_time=measurement_time, warmup_time=warmup_time):
                return run_mmc_b_simulation(
                    lambda_rate, mu_rate, c, measurement_time, warmup_time, seed
                )

            summary = run_replications(once, n_replications, base_seed, ("B",))
            theory = erlang_b(c, a)
            sim_arrivals_h.append(arrivals_h)
            sim_b.append(100.0 * summary["B"]["mean"])
            sim_low.append(100.0 * summary["B"]["low"])
            sim_high.append(100.0 * summary["B"]["high"])
            print(
                f"  B-sweep c={c:2d}  rho={rho:.2f}  "
                f"arrivals/h={arrivals_h:8.2f}  "
                f"B_sim={sim_b[-1]:.3f}%  "
                f"CI=[{sim_low[-1]:.3f}, {sim_high[-1]:.3f}]  "
                f"B_th={100 * theory:.3f}%",
                flush=True,
            )

        colour = channel_colour(c)
        label = SERVER_LABELS.get(c, f"$c = {c}$")
        ax.plot(theory_arrivals_h, theory_b_pct, "-", color=colour, label=label)
        yerr = [
            [mean - low for mean, low in zip(sim_b, sim_low)],
            [high - mean for mean, high in zip(sim_b, sim_high)],
        ]
        ax.errorbar(
            sim_arrivals_h,
            sim_b,
            yerr=yerr,
            fmt=markers[idx],
            color=colour,
            markerfacecolor="white",
            capsize=3,
            linestyle="none",
        )

    ax.axhline(
        100.0 * blocking_target,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"{100 * blocking_target:.0f}%",
    )
    ax.set_xlabel("Arrivals per hour, $\\lambda$ (hour$^{-1}$)")
    ax.set_ylabel("Blocking probability, $B$ (%)")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmc_blocking_vs_arrivals")
    plt.close(fig)
    print(f"Saved {png}")


def plot_blocking_vs_c_family(
    mean_service_s,
    arrival_rates_per_hour,
    c_range,
    n_service_times,
    warmup_service_times,
    blocking_target,
    random_seed=None,
    sim_c_values=None,
):
    """Erlang-B blocking vs c at fixed arrival rates (theory dense, SimPy sparse)."""
    apply_style()
    mu_rate = 1.0 / mean_service_s
    c_min, c_max = c_range
    if sim_c_values is None:
        sim_c_values = SIM_C_MARKERS_FOR_FAMILY

    markers = ["o", "s", "D", "^", "v"]
    colors = [channel_colour(c) for c in (1, 7, 8, 21, 45)]

    plt.figure(figsize=SINGLE)
    for idx, arrivals_h in enumerate(arrival_rates_per_hour):
        lambda_rate = arrivals_h / SECONDS_PER_HOUR
        c_theory, theory_b_pct = erlang_b_theory_vs_c(
            lambda_rate, mu_rate, c_min, c_max
        )

        label = rf"$\lambda = {arrivals_h:g}$ h$^{{-1}}$"
        plt.plot(
            c_theory, theory_b_pct, "-", color=colors[idx % len(colors)], label=label
        )

        sim_b = []
        for j, c in enumerate(sim_c_values):
            n_run, warmup_base = scaled_service_time_counts_b(
                c, n_service_times, warmup_service_times
            )
            measurement_time = n_run * mean_service_s
            rho_at_c = lambda_rate / (c * mu_rate)
            warmup_run = warmup_service_times_for_rho(warmup_base, n_run, rho_at_c)
            warmup_time = warmup_run * mean_service_s
            seed = None if random_seed is None else random_seed + 5000 * idx + j
            sim_result = run_mmc_b_simulation(
                lambda_rate, mu_rate, c, measurement_time, warmup_time, seed
            )
            a = offered_load_erlangs(lambda_rate, mu_rate)
            theory = erlang_b(c, a)
            sim_b.append(100.0 * sim_result["B"])
            print(
                f"  B vs c  lambda={arrivals_h:g}/h  c={c:2d}  "
                f"B_sim={100 * sim_result['B']:.3f}%  B_th={100 * theory:.3f}%"
            )

        plt.plot(
            sim_c_values,
            sim_b,
            markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            markerfacecolor="white",
            label="_nolegend_",
        )

    plt.axhline(
        100.0 * blocking_target,
        color=THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=f"{100 * blocking_target:.0f}%",
    )
    plt.xlabel("Number of channels, $c$")
    plt.ylabel("Blocking probability, $B$ (%)")
    plt.legend()
    plt.tight_layout()
    png, _pdf = save_figure(plt.gcf(), "mmc_blocking_vs_c_family")
    plt.close()
    print(f"Saved {png}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    mu_rate = 1.0 / PRIMARY_MEAN_SERVICE_S
    validation_c = 8
    validation_blocking_target = 0.03  # ~3% for a non-trivial B in the 2–5% band
    a_val = offered_load_for_blocking(validation_c, validation_blocking_target)
    lambda_val = a_val * mu_rate

    n_service_times = 120_000
    warmup_service_times = 12_000
    measurement_time = n_service_times * PRIMARY_MEAN_SERVICE_S
    warmup_time = warmup_service_times * PRIMARY_MEAN_SERVICE_S

    sim_b = run_mmc_b_simulation(
        lambda_val, mu_rate, validation_c, measurement_time, warmup_time, random_seed=42
    )
    theory_b = erlang_b(validation_c, a_val)
    pct_dev = print_blocking_validation_table(
        sim_b, lambda_val, mu_rate, validation_c, warmup_time, theory_b
    )
    print(f"\nErlang-B validation deviation: {pct_dev:.2f}%")

    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    print_family_blocking_sweep_table(PRIMARY_MEAN_SERVICE_S, SERVER_COUNTS, rho_values)

    print("\nRunning Erlang-B blocking vs arrivals/hour sweep...")
    plot_blocking_vs_arrivals_family(
        PRIMARY_MEAN_SERVICE_S,
        SERVER_COUNTS,
        rho_values,
        n_service_times=40_000,
        warmup_service_times=4_000,
        blocking_target=BLOCKING_TARGET,
        random_seed=42,
    )

    FAMILY_ARRIVAL_RATES_H = [50, 100, 200, 400, 800]
    print("\nRunning Erlang-B blocking vs c at fixed arrival rates...")
    plot_blocking_vs_c_family(
        PRIMARY_MEAN_SERVICE_S,
        FAMILY_ARRIVAL_RATES_H,
        (1, 50),
        n_service_times=40_000,
        warmup_service_times=4_000,
        blocking_target=BLOCKING_TARGET,
        random_seed=42,
    )

    SURFACE_C_RANGE = range(1, 51)
    SURFACE_ARRIVALS_H = np.logspace(0, np.log10(2000), 100)
    print("\nRunning Erlang-B blocking 3D surface and heatmap...")
    plot_blocking_surface_3d(
        PRIMARY_MEAN_SERVICE_S,
        SURFACE_C_RANGE,
        SURFACE_ARRIVALS_H,
        BLOCKING_TARGET,
    )
    plot_blocking_heatmap(
        PRIMARY_MEAN_SERVICE_S,
        SURFACE_C_RANGE,
        SURFACE_ARRIVALS_H,
        BLOCKING_TARGET,
    )
