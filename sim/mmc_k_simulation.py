"""
M/M/c/K with a finite system capacity (K >= c).

K counts customers in service and waiting, so there are K - c queue places.
K = c is the Erlang-B loss model. As K grows with rho < 1 the delay matches
Erlang C. Closed forms use a log-space state distribution so large a and K
do not overflow.
"""

import math
import random
import statistics

import matplotlib.pyplot as plt
import simpy

from sim.literature_params import (
    ACCEPTABLE_WAIT_S,
    BLOCKING_TARGET,
    INTERIM_PARAMETER_NOTE,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
)
from sim.mmc_simulation import (
    erlang_b,
    offered_load_for_blocking,
    theoretical_mmc,
)
from sim.paths import figure_path
from sim.plot_style import (
    LISTED,
    MODEL,
    SINGLE,
    THRESHOLD,
    apply_style,
    channel_colour,
    panel_label,
    save_figure,
)
from sim.warmup import in_measurement_window, total_run_time

DRAIN_SERVICE_TIMES = 50.0
QUEUE_EXTRAS = (0, 2, 5, 10, 20)
SWEEP_SERVER_COUNTS = (8, 21, 45)


def mmck_state_probabilities(c, a, K):
    """Steady-state P_0 ... P_K for M/M/c/K. a = lambda/mu may exceed c."""
    if c < 1 or K < c:
        raise ValueError("Need c >= 1 and K >= c.")
    if a < 0:
        raise ValueError("offered load a must be non-negative.")
    if a == 0:
        probs = [0.0] * (K + 1)
        probs[0] = 1.0
        return probs
    logs = [0.0]
    for n in range(1, K + 1):
        if n <= c:
            logs.append(logs[-1] + math.log(a) - math.log(n))
        else:
            logs.append(logs[-1] + math.log(a) - math.log(c))
    pivot = max(logs)
    weights = [math.exp(value - pivot) for value in logs]
    total = sum(weights)
    return [weight / total for weight in weights]


def theoretical_mmck(lambda_rate, mu_rate, c, K):
    """Blocking P_K, queue length, wait of accepted calls, carried throughput."""
    if mu_rate <= 0 or lambda_rate < 0:
        raise ValueError("Rates must be non-negative and mu > 0.")
    a = lambda_rate / mu_rate
    probs = mmck_state_probabilities(c, a, K)
    blocking = probs[K]
    lq = sum((n - c) * probs[n] for n in range(c, K + 1))
    carried_erlang = a * (1.0 - blocking)
    lambda_eff = lambda_rate * (1.0 - blocking)
    wq = (lq / lambda_eff) if lambda_eff > 0 else 0.0
    return {
        "a": a,
        "rho": a / c,
        "P_K": blocking,
        "B": blocking,
        "Lq": lq,
        "Wq": wq,
        "W": wq + 1.0 / mu_rate,
        "carried_erlang": carried_erlang,
        "carried_calls_per_hour": lambda_eff * SECONDS_PER_HOUR,
        "utilisation": carried_erlang / c,
        "probs": probs,
    }


def offered_load_for_blocking_k(c, K, blocking, lo=1e-9, hi=1.0):
    """Offered load a (erlang) at which P_K = blocking."""
    if not 0 < blocking < 1:
        raise ValueError("blocking must be in (0, 1).")
    while theoretical_mmck(hi, 1.0, c, K)["B"] < blocking:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Blocking target not reached.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if theoretical_mmck(mid, 1.0, c, K)["B"] > blocking:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _wq_at_offered_load(c, K, offered_a, mu_rate):
    """Accepted-call Wq (seconds) at offered load a, for the given service rate."""
    return theoretical_mmck(offered_a * mu_rate, mu_rate, c, K)["Wq"]


def offered_load_for_wait_k(c, K, mu_rate, wait_s, lo=1e-8, hi=1.0):
    """Offered load a (erlang) at which accepted-call Wq = wait_s, or None.

    A pure loss system (K = c) has Wq = 0. A short buffer can saturate below
    the target; that is reported as None rather than a forced root.
    The search is in erlang. Wq scales as 1/mu, so mu_rate is applied inside
    the evaluation rather than by treating a as an arrival rate.
    """
    if K <= c or wait_s <= 0:
        return None

    def wq(a):
        return _wq_at_offered_load(c, K, a, mu_rate)

    if wq(hi) < wait_s:
        hi = max(float(c) * 50.0, hi)
    if wq(hi) < wait_s:
        return None
    while wq(hi) < wait_s:
        hi *= 2.0
        if hi > 1e8:
            return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if wq(mid) < wait_s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mmck_customer(env, server, mu_rate, K, stats, warmup_time, measurement_time):
    """Accept while fewer than K customers are present; otherwise block."""
    arrival_time = env.now
    stats["total_arrivals"] += 1
    in_window = in_measurement_window(arrival_time, warmup_time, measurement_time)
    if server.count + len(server.queue) >= K:
        if in_window:
            stats["blocked"] += 1
        return
    if in_window:
        stats["accepted"] += 1
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


def mmck_arrivals(env, server, lambda_rate, mu_rate, K, stats, warmup_time, measurement_time):
    while True:
        yield env.timeout(random.expovariate(lambda_rate))
        env.process(
            mmck_customer(
                env, server, mu_rate, K, stats, warmup_time, measurement_time
            )
        )


def run_mmck_simulation(
    lambda_rate, mu_rate, c, K, measurement_time, warmup_time, random_seed=None
):
    """One M/M/c/K run. Statistics use arrivals in [warmup_time, measurement_time)."""
    if K < c:
        raise ValueError("K must be at least c.")
    if random_seed is not None:
        random.seed(random_seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=c)
    stats = {
        "total_arrivals": 0,
        "blocked": 0,
        "accepted": 0,
        "occupancies": [],
        "waits": [],
        "sojourns": [],
    }
    env.process(
        mmck_arrivals(
            env, server, lambda_rate, mu_rate, K, stats, warmup_time, measurement_time
        )
    )
    run_until = total_run_time(measurement_time, 1.0 / mu_rate, DRAIN_SERVICE_TIMES)
    env.run(until=run_until)

    window = measurement_time - warmup_time
    attempts = stats["blocked"] + stats["accepted"]
    blocking = stats["blocked"] / attempts if attempts else 0.0
    busy_time = 0.0
    for service_start, service_end in stats["occupancies"]:
        overlap_start = max(service_start, warmup_time)
        overlap_end = min(service_end, measurement_time)
        if overlap_end > overlap_start:
            busy_time += overlap_end - overlap_start
    mean_wait = statistics.mean(stats["waits"]) if stats["waits"] else 0.0
    mean_system = statistics.mean(stats["sojourns"]) if stats["sojourns"] else 0.0
    accepted_rate = stats["accepted"] / window if window > 0 else 0.0
    return {
        "B": blocking,
        "P_K": blocking,
        "Wq": mean_wait,
        "W": mean_system,
        "Lq": accepted_rate * mean_wait,
        "utilisation": busy_time / (c * window) if window > 0 else 0.0,
        "carried_calls_per_hour": accepted_rate * SECONDS_PER_HOUR,
        "carried_erlang": accepted_rate / mu_rate,
        "blocked": stats["blocked"],
        "accepted": stats["accepted"],
        "measurement_time": measurement_time,
        "run_until": run_until,
    }


def threshold_rows(
    server_counts=SWEEP_SERVER_COUNTS,
    queue_extras=QUEUE_EXTRAS,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    blocking=BLOCKING_TARGET,
    wait_s=ACCEPTABLE_WAIT_S,
):
    """Offered load and arrivals/h where B = 2% and where Wq = 10 s."""
    mu = 1.0 / mean_service_s
    rows = []
    for c in server_counts:
        for extra in queue_extras:
            K = c + extra
            a_b = offered_load_for_blocking_k(c, K, blocking)
            a_w = offered_load_for_wait_k(c, K, mu, wait_s)
            rows.append({
                "c": c,
                "K": K,
                "queue_extra": extra,
                "a_at_blocking": a_b,
                "arrivals_h_at_blocking": a_b * mu * SECONDS_PER_HOUR,
                "a_at_wait": a_w,
                "arrivals_h_at_wait": None if a_w is None else a_w * mu * SECONDS_PER_HOUR,
            })
    return rows


def erlang_c_arrivals_at_wait(c, mu_rate, wait_s):
    """Arrivals per hour at which Erlang-C Wq equals wait_s. None if unreachable."""
    # Wq grows without bound as rho -> 1, and is 0 at rho = 0.
    lo, hi = 1e-8, 0.999
    if theoretical_mmc(hi * c * mu_rate, mu_rate, c)["Wq"] < wait_s:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        wq = theoretical_mmc(mid * c * mu_rate, mu_rate, c)["Wq"]
        if wq < wait_s:
            lo = mid
        else:
            hi = mid
    rho = 0.5 * (lo + hi)
    return rho * c * mu_rate * SECONDS_PER_HOUR


def erlang_b_arrivals_at_blocking(c, mu_rate, blocking):
    """Arrivals per hour at the Erlang-B offered load with this blocking probability."""
    a = offered_load_for_blocking(c, blocking)
    return a * mu_rate * SECONDS_PER_HOUR


def mmck_arrivals_at_wait(c, queue_extra, mu_rate, wait_s):
    a = offered_load_for_wait_k(c, c + queue_extra, mu_rate, wait_s)
    if a is None:
        return None
    return a * mu_rate * SECONDS_PER_HOUR


def mmck_arrivals_at_blocking(c, queue_extra, mu_rate, blocking):
    """Arrivals per hour at which M/M/c/K blocking equals the target."""
    a = offered_load_for_blocking_k(c, c + queue_extra, blocking)
    return a * mu_rate * SECONDS_PER_HOUR


def mmck_binding_arrivals(c, queue_extra, mu_rate, wait_s, blocking):
    """Tighter arrival rate of B = blocking and accepted-call Wq = wait_s.

    A missing wait boundary (no queue, or a buffer that saturates below
    wait_s) is treated as infinite and does not bind.
    """
    at_blocking = mmck_arrivals_at_blocking(c, queue_extra, mu_rate, blocking)
    at_wait = mmck_arrivals_at_wait(c, queue_extra, mu_rate, wait_s)
    if at_wait is None:
        return at_blocking
    return min(at_blocking, at_wait)


def plot_capacity_boundaries(
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    c_min=1,
    c_max=50,
    queue_extra=5,
    wait_s=ACCEPTABLE_WAIT_S,
    blocking=BLOCKING_TARGET,
):
    """(c, arrivals/h) plane: Erlang-C wait, Erlang-B loss, and binding M/M/c/K."""
    apply_style()
    mu = 1.0 / mean_service_s
    c_values = list(range(c_min, c_max + 1))
    wq_line = [erlang_c_arrivals_at_wait(c, mu, wait_s) for c in c_values]
    b_line = [erlang_b_arrivals_at_blocking(c, mu, blocking) for c in c_values]
    k_line = [
        mmck_binding_arrivals(c, queue_extra, mu, wait_s, blocking)
        for c in c_values
    ]
    stable = [c * mu * SECONDS_PER_HOUR for c in c_values]
    y_top = max(stable) * 3.0

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.fill_between(
        c_values, stable, y_top,
        color="#d9d9d9", hatch="////", linewidth=0,
        label=r"unstable for Erlang C ($\lambda \geq c\mu$)",
    )
    ax.plot(c_values, wq_line, color=LISTED, linewidth=1.5, label=rf"Erlang C, $W_q$ = {wait_s:.0f} s")
    ax.plot(c_values, b_line, color=MODEL, linewidth=1.5, label=rf"Erlang B, $B$ = {100 * blocking:.0f}%")
    ax.plot(
        c_values, k_line, color=channel_colour(45), linewidth=1.5, linestyle="--",
        label=(
            rf"M/M/$c$/$K$, $K-c$ = {queue_extra}, "
            rf"$B \leq$ {100 * blocking:.0f}% and $W_q \leq$ {wait_s:.0f} s"
        ),
    )
    ax.set_yscale("log")
    ax.set_ylim(min(y for y in wq_line + b_line if y), y_top)
    ax.set_xlabel("Number of channels, $c$")
    ax.set_ylabel("Arrivals per hour")
    ax.legend(loc="lower right")
    fig.tight_layout()
    png, _pdf = save_figure(fig, "capacity_boundaries_overlay")
    plt.close(fig)
    print(f"Saved {png}")
    return {
        "path": png,
        "c": c_values,
        "wq10_arrivals_h": wq_line,
        "blocking_arrivals_h": b_line,
        "mmck_binding_arrivals_h": k_line,
        "stability_arrivals_h": stable,
    }


def plot_mmck_sweep(
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    server_counts=SWEEP_SERVER_COUNTS,
    queue_extras=QUEUE_EXTRAS,
    sim_markers=None,
    n_theory=80,
):
    """Blocking and Wq versus arrivals/h. One column per c. sim_markers optional.

    sim_markers: list of dicts with keys c, queue_extra, arrivals_h, B, B_low,
    B_high, Wq, Wq_low, Wq_high. B is a probability, not a percent.
    """
    apply_style()
    mu = 1.0 / mean_service_s
    colours = [channel_colour(c) for c in (1, 7, 8, 21, 45)]
    # Six panels: wider than the two-panel size so the columns and legend fit.
    fig, axes = plt.subplots(2, len(server_counts), figsize=(9.5, 6.0), sharex="col")
    if len(server_counts) == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, c in enumerate(server_counts):
        arrivals_max = 2.0 * c * mu * SECONDS_PER_HOUR
        arrivals_grid = [
            arrivals_max ** (i / (n_theory - 1)) for i in range(n_theory)
        ]
        # arrivals_max ** 0 = 1, so the grid starts at 1 call/h.
        ax_b = axes[0][col]
        ax_w = axes[1][col]
        for idx, extra in enumerate(queue_extras):
            K = c + extra
            b_pct = []
            wq = []
            for arrivals_h in arrivals_grid:
                theory = theoretical_mmck(arrivals_h / SECONDS_PER_HOUR, mu, c, K)
                b_pct.append(100.0 * theory["B"])
                wq.append(theory["Wq"])
            colour = colours[idx % len(colours)]
            label = f"$K-c = {extra}$"
            ax_b.plot(arrivals_grid, b_pct, color=colour, label=label)
            ax_w.plot(arrivals_grid, wq, color=colour, label=label)
            if sim_markers:
                points = [
                    m for m in sim_markers
                    if m["c"] == c and m["queue_extra"] == extra
                ]
                if points:
                    xs = [m["arrivals_h"] for m in points]
                    ax_b.errorbar(
                        xs,
                        [100.0 * m["B"] for m in points],
                        yerr=[
                            [100.0 * (m["B"] - m["B_low"]) for m in points],
                            [100.0 * (m["B_high"] - m["B"]) for m in points],
                        ],
                        fmt="o",
                        color=colour,
                        markerfacecolor="white",
                        capsize=2,
                        linestyle="none",
                    )
                    ax_w.errorbar(
                        xs,
                        [m["Wq"] for m in points],
                        yerr=[
                            [m["Wq"] - m["Wq_low"] for m in points],
                            [m["Wq_high"] - m["Wq"] for m in points],
                        ],
                        fmt="o",
                        color=colour,
                        markerfacecolor="white",
                        capsize=2,
                        linestyle="none",
                    )
        ax_b.axhline(100.0 * BLOCKING_TARGET, color=THRESHOLD, linestyle="--", linewidth=1)
        ax_w.axhline(ACCEPTABLE_WAIT_S, color=THRESHOLD, linestyle="--", linewidth=1)
        ax_b.set_xscale("log")
        ax_w.set_xscale("log")
        panel_label(ax_b, f"$c = {c}$")
        ax_w.set_xlabel("Arrivals per hour")
    axes[0][0].set_ylabel("Blocking probability, $B$ (%)")
    axes[1][0].set_ylabel("Mean wait of accepted calls, $W_q$ (s)")
    axes[0][0].legend(loc="center left")
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mmck_blocking_wq_vs_arrivals")
    plt.close(fig)
    print(f"Saved {png}")
    return png
