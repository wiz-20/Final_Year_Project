"""
Two explanatory figures, produced with the project's own simulation code.

  sim_trace       One simulated busy hour of an 8-channel cell at rho = 0.80.
                  (a) Loss system: busy channels over time and each blocked call,
                      from sim.mmc_b_simulation.blocked_customer_arrivals.
                  (b) Delay system: busy channels and calls waiting, from
                      sim.mm1_simulation.customer_arrivals on a resource with
                      8 channels (the same customer process as the M/M/1 engine).
  warmup_effect   Welch plot for the M/M/1 queue at rho = 0.80: the waiting time
                  of the i-th call, averaged over many replications that start
                  empty, from sim.mm1_simulation.customer_arrivals.

The engine functions are called unchanged. This script only records the event
times they already produce (occupancy intervals, per-customer records) and, for
the loss system, the time of each blocked call.

Usage (from the project root):
    .\\venv\\Scripts\\python.exe make_extra_figures.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import simpy

from sim.literature_params import PRIMARY_MEAN_SERVICE_S
from sim.mm1_simulation import customer_arrivals, theoretical_mm1
from sim.mmc_b_simulation import blocked_customer_arrivals
from sim.mmc_simulation import erlang_b
from sim.plot_style import THRESHOLD, apply_style, save_figure

ORANGE = "#c45911"
BLUE = "#1f4e79"

C = 8
RHO = 0.80
HOUR_S = 3600.0
DRAIN_S = 50 * PRIMARY_MEAN_SERVICE_S
TRACE_SEED = 7

WELCH_CALLS = 800
WELCH_REPS = 2000
WELCH_WINDOW = 5
WELCH_SEED = 500


class _BlockLog(dict):
    """Stats dict for the loss engine that also notes the time of each blocked call."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.blocked_times = []

    def __setitem__(self, key, value):
        if key == "blocked" and value > self.get("blocked", 0):
            self.blocked_times.append(self.env.now)
        super().__setitem__(key, value)


def _step_from_intervals(intervals, horizon):
    """Number of intervals open at each event time, as a step function on [0, horizon]."""
    events = [(s, +1) for s, e in intervals] + [(e, -1) for s, e in intervals]
    events.sort()
    t, n = [0.0], [0]
    count = 0
    for time, delta in events:
        if time > horizon:
            break
        count += delta
        t.append(time)
        n.append(count)
    t.append(horizon)
    n.append(count)
    return np.array(t), np.array(n)


def loss_trace(lam, mu, seed):
    """One hour of the project's M/M/c/c engine. Returns busy-channel steps and blocked times."""
    random.seed(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=C)
    stats = _BlockLog(env, total_arrivals=0, blocked=0, served=0,
                      occupancies=[], waits=[], sojourns=[])
    env.process(blocked_customer_arrivals(env, server, lam, mu, stats, 0.0, HOUR_S))
    env.run(until=HOUR_S + DRAIN_S)
    t, busy = _step_from_intervals(stats["occupancies"], HOUR_S)
    blocked = np.array([b for b in stats.blocked_times if b < HOUR_S])
    offered = stats["served"] + stats["blocked"]
    return t, busy, blocked, offered


def delay_trace(lam, mu, seed):
    """One hour with the project's customer process on C channels (unlimited queue)."""
    random.seed(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=C)
    records = []
    env.process(customer_arrivals(env, server, lam, mu, records))
    env.run(until=HOUR_S + DRAIN_S)
    busy_iv = [(r["service_start"], r["service_end"]) for r in records]
    wait_iv = [(r["arrival_time"], r["service_start"]) for r in records
               if r["service_start"] > r["arrival_time"]]
    t_b, busy = _step_from_intervals(busy_iv, HOUR_S)
    t_q, queue = _step_from_intervals(wait_iv, HOUR_S)
    return t_b, busy, t_q, queue


def _on_grid(t, y, grid):
    """Value of a post-step function (t, y) at each grid time."""
    idx = np.searchsorted(t, grid, side="right") - 1
    return y[np.clip(idx, 0, len(y) - 1)]


def figure_trace():
    mu = 1.0 / PRIMARY_MEAN_SERVICE_S
    lam = RHO * C * mu
    t_l, b_l, blocked, offered = loss_trace(lam, mu, TRACE_SEED)
    t_b, b_d, t_q, q_d = delay_trace(lam, mu, TRACE_SEED)
    print(f"Loss trace: {len(blocked)} of {offered} calls blocked "
          f"({100 * len(blocked) / offered:.1f}%); Erlang B = {100 * erlang_b(C, lam / mu):.1f}%")

    # Common time grid so the stacked areas line up exactly.
    grid = np.linspace(0.0, HOUR_S, 7201)
    m = grid / 60
    busy_l = _on_grid(t_l, b_l, grid)
    busy_d = _on_grid(t_b, b_d, grid)
    queue_d = _on_grid(t_q, q_d, grid)

    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35]})

    # (a) loss system
    ax1.fill_between(m, 0, busy_l, step="post", color=ORANGE, alpha=0.35, linewidth=0)
    ax1.step(m, busy_l, where="post", color=ORANGE, linewidth=0.9)
    ax1.axhline(C, color="0.3", linestyle="--", linewidth=0.9)
    ax1.text(60.5, C, f"{C} channels", va="center", ha="left", fontsize=8, color="0.3")
    ax1.scatter(blocked / 60, np.full(len(blocked), C + 0.9), marker="v", s=22,
                color="#b00020", zorder=5,
                label=f"blocked call ({len(blocked)} of {offered})")
    ax1.set_ylim(0, C + 2)
    ax1.set_yticks(range(0, C + 1, 2))
    ax1.set_ylabel("Calls in service")
    ax1.legend(loc="lower right", frameon=True, fontsize=8)
    ax1.set_title("(a) Loss system: a call that finds all channels busy is lost",
                  fontsize=9, loc="left")

    # (b) delay system, stacked: in service + waiting = calls in the system
    total = busy_d + queue_d
    ax2.fill_between(m, 0, busy_d, step="post", color=ORANGE, alpha=0.35,
                     linewidth=0, label="in service")
    ax2.fill_between(m, busy_d, total, step="post", color=BLUE, alpha=0.45,
                     linewidth=0, label="waiting")
    ax2.step(m, total, where="post", color=BLUE, linewidth=0.9)
    ax2.axhline(C, color="0.3", linestyle="--", linewidth=0.9)
    ax2.text(60.5, C, f"{C} channels", va="center", ha="left", fontsize=8, color="0.3")
    ax2.set_ylim(0, max(total.max() + 2, C + 2))
    ax2.set_ylabel("Calls in the system")
    ax2.set_xlabel("Time (min)")
    ax2.set_xlim(0, HOUR_S / 60)
    ax2.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)
    ax2.set_title("(b) Delay system: a call that finds all channels busy waits",
                  fontsize=9, loc="left")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    png, _pdf = save_figure(fig, "sim_trace")
    plt.close(fig)
    print(f"Saved {png}")


def mm1_waits_in_order(lam, mu, n_calls, seed):
    """Waiting times of the first n_calls customers of one M/M/1 run started empty."""
    random.seed(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    records = []
    env.process(customer_arrivals(env, server, lam, mu, records))
    horizon = (n_calls + 200) / lam
    while True:
        env.run(until=horizon)
        if len(records) >= n_calls:
            break
        horizon *= 1.5
    records.sort(key=lambda r: r["arrival_time"])
    return np.array([r["waiting_time"] for r in records[:n_calls]])


def figure_warmup():
    mu = 1.0 / PRIMARY_MEAN_SERVICE_S
    lam = RHO * mu
    theory = theoretical_mm1(lam, mu)["Wq"]
    waits = np.zeros((WELCH_REPS, WELCH_CALLS))
    for r in range(WELCH_REPS):
        waits[r] = mm1_waits_in_order(lam, mu, WELCH_CALLS, WELCH_SEED + r)
        if (r + 1) % 250 == 0:
            print(f"  warm-up replications: {r + 1} of {WELCH_REPS}", flush=True)
    kernel = np.ones(WELCH_WINDOW) / WELCH_WINDOW
    ens = waits.mean(axis=0)
    half = 1.96 * waits.std(axis=0, ddof=1) / np.sqrt(WELCH_REPS)
    smooth = np.convolve(ens, kernel, mode="valid")
    lo = np.convolve(ens - half, kernel, mode="valid")
    hi = np.convolve(ens + half, kernel, mode="valid")
    x = np.arange(1, WELCH_CALLS + 1)[WELCH_WINDOW - 1:]
    print(f"Mean waiting time, calls 400 to {WELCH_CALLS}: {ens[399:].mean():.0f} s "
          f"(theory {theory:.0f} s)")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axvspan(0, 150, color="0.92", zorder=0)
    ax.text(75, 120, "start-up\ntransient", ha="center", va="bottom", fontsize=8, color="0.35")
    ax.fill_between(x, lo, hi, color=ORANGE, alpha=0.25, linewidth=0,
                    label="95% CI of the mean")
    ax.plot(x, smooth, color=ORANGE, linewidth=1.6,
            label=f"mean of {WELCH_REPS} replications")
    ax.axhline(theory, color="black", linewidth=1.0, linestyle="--",
               label=f"steady-state theory, {theory:.0f} s")
    ax.set_xlim(0, WELCH_CALLS)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Call number $i$ (calls since the simulation started empty)")
    ax.set_ylabel("Mean waiting time of call $i$ (s)")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png, _pdf = save_figure(fig, "warmup_effect")
    plt.close(fig)
    print(f"Saved {png}")


if __name__ == "__main__":
    figure_trace()
    figure_warmup()