"""
M/M/1 Queue Simulation (SimPy) with Warm-up Handling and Analytical Validation
===============================================================================

What this script models:
    A single base station channel as an M/M/1 queue:
        - Poisson arrivals            (rate lambda_rate, per second)
        - Exponential service times   (rate mu_rate = 1 / mean holding time)
        - 1 server                    (the "1" in M/M/1)
        - Infinite buffer             (no calls are ever blocked/dropped)
        - Time unit                   seconds; 1/mu taken from literature
          (primary baseline 180 s / 3 min)

Why start here:
    This is the baseline before extending to M/M/4 with a finite buffer
    (multiple channels, calls dropped when the buffer is full). M/M/1 has
    exact closed-form formulas, so comparing simulated output against those
    formulas proves the simulation engine itself is correct before it is
    trusted with a more complex model that has no simple closed form.
"""

import random
import statistics

import matplotlib.pyplot as plt
import simpy

from sim.literature_params import (
    DATA_SESSION_CONTRAST_S,
    INTERIM_PARAMETER_NOTE,
    MEAN_SERVICE_TIMES_S,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
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
from sim.warmup import in_measurement_window, total_run_time

DATA_CONTRAST_S = DATA_SESSION_CONTRAST_S

# ==============================================================================
# SECTION: PARAMETERS
# ------------------------------------------------------------------------------
# Time unit is SECONDS. Mean service time 1/mu is the grounded input (from the
# holding-time literature). Arrival rate lambda is then set from a chosen rho
# via lambda = rho * mu, so rho is no longer an arbitrary plug-in number.
# ==============================================================================

DRAIN_SERVICE_TIMES = 50.0
DEMO_RHO = 0.80

MU_RATE = 1.0 / PRIMARY_MEAN_SERVICE_S   # services per second
LAMBDA_RATE = DEMO_RHO * MU_RATE         # arrivals per second
RHO = LAMBDA_RATE / MU_RATE
assert RHO < 1, "Unstable system: lambda_rate must be less than mu_rate."

# Run length is counted in mean service times so the number of observed
# customers stays roughly rho * N regardless of whether 1/mu is 60 s or 900 s.
N_SERVICE_TIMES = 40_000
WARMUP_SERVICE_TIMES = 2_000
SIM_TIME = N_SERVICE_TIMES * PRIMARY_MEAN_SERVICE_S
WARMUP_TIME = WARMUP_SERVICE_TIMES * PRIMARY_MEAN_SERVICE_S
RANDOM_SEED = 42

# Working assumption for "acceptable" access delay (not an operator SLA yet).
ACCEPTABLE_WAIT_S = 10.0


# ==============================================================================
# SECTION: SIMULATION LOGIC
# ------------------------------------------------------------------------------
# SimPy is a discrete-event simulation framework: instead of stepping through
# time in fixed ticks, it jumps from event to event (an arrival, a service
# completion). A "process" here is a Python generator function; every `yield`
# is a point where that process pauses until something happens (e.g. a timer
# expires, or a resource becomes free) and control returns to SimPy's event
# loop to run other processes in the meantime.
# ==============================================================================

def customer_arrivals(env, server, lambda_rate, mu_rate, records):
    """Generates customers as a Poisson arrival process.

    Poisson arrivals mean the time BETWEEN consecutive arrivals is
    exponentially distributed with rate lambda_rate. Drawing that gap and
    yielding a SimPy timeout for it is what actually produces a Poisson
    process over the course of the run.
    """
    while True:
        interarrival_time = random.expovariate(lambda_rate)
        yield env.timeout(interarrival_time)
        env.process(customer(env, server, mu_rate, records))


def customer(env, server, mu_rate, records):
    """Lifecycle of a single customer: arrive, queue if busy, get served, leave.

    `server.request()` asks for the one available "channel". If it's busy,
    the `yield req` line pauses this customer here until SimPy grants the
    request (i.e. until the server frees up) - that pause IS the queueing
    delay, no manual queue bookkeeping required.
    """
    arrival_time = env.now

    with server.request() as req:
        yield req  # blocks here until the server is free -> this is the wait in queue

        service_start = env.now
        waiting_time = service_start - arrival_time

        service_time = random.expovariate(mu_rate)  # exponential service duration
        yield env.timeout(service_time)              # occupy the server for that long

        service_end = env.now
        time_in_system = service_end - arrival_time

    # DATA COLLECTION: one record per customer, stamped with its arrival time
    # so warm-up filtering (below) can later decide whether to keep it.
    records.append({
        "arrival_time": arrival_time,
        "service_start": service_start,
        "service_end": service_end,
        "waiting_time": waiting_time,
        "time_in_system": time_in_system,
    })


def run_mm1_simulation(
    lambda_rate,
    mu_rate,
    measurement_time,
    warmup_time,
    random_seed=None,
    drain_service_times=DRAIN_SERVICE_TIMES,
):
    """Runs one full M/M/1 simulation and returns steady-state statistics.

    Returns a dict with simulated Wq, W, rho and Lq, computed only from the
    post-warm-up portion of the run (see WARM-UP HANDLING below).
    """
    if random_seed is not None:
        random.seed(random_seed)

    mean_service_s = 1.0 / mu_rate
    run_until = total_run_time(measurement_time, mean_service_s, drain_service_times)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)  # capacity=1 -> the "1" in M/M/1
    records = []

    env.process(customer_arrivals(env, server, lambda_rate, mu_rate, records))
    env.run(until=run_until)

    # --------------------------------------------------------------------
    # SECTION: WARM-UP HANDLING
    # ----------------------------------------------------------------------
    # Why warm-up bias matters:
    #   The simulation always starts with an EMPTY, IDLE system. Customers
    #   arriving early on almost never have to wait, because there's no
    #   backlog yet - those early waiting times are systematically too low
    #   compared to steady-state behaviour. If they're included in the
    #   averages, the simulated results will be biased optimistic relative
    #   to the theoretical (steady-state) formulas we're validating against.
    #   The fix: exclude arrivals before warmup_time and before measurement_time;
    #   extend the SimPy run after measurement_time so in-window customers finish.
    # --------------------------------------------------------------------
    steady_state_records = [
        r
        for r in records
        if in_measurement_window(r["arrival_time"], warmup_time, measurement_time)
    ]

    waiting_times = [r["waiting_time"] for r in steady_state_records]
    system_times = [r["time_in_system"] for r in steady_state_records]

    mean_wait_sim = statistics.mean(waiting_times)
    mean_system_time_sim = statistics.mean(system_times)

    # Utilisation: fraction of the post-warm-up measurement window during
    # which the server was actually busy serving someone. Service intervals
    # are taken from ALL records (not just steady_state_records) because a
    # service that started just before warmup_time can still overlap into
    # the measurement window - only the overlapping portion should count.
    measurement_start, measurement_end = warmup_time, measurement_time
    busy_time = 0.0
    for r in records:
        overlap_start = max(r["service_start"], measurement_start)
        overlap_end = min(r["service_end"], measurement_end)
        if overlap_end > overlap_start:
            busy_time += overlap_end - overlap_start
    utilisation_sim = busy_time / (measurement_end - measurement_start)

    # Little's Law (L = lambda * W) converts the measured mean waiting time
    # into a mean queue length without needing to sample queue length directly.
    mean_queue_length_sim = lambda_rate * mean_wait_sim

    return {
        "Wq": mean_wait_sim,
        "W": mean_system_time_sim,
        "rho": utilisation_sim,
        "utilisation": utilisation_sim,
        "B": 0.0,
        "Lq": mean_queue_length_sim,
        "num_customers_kept": len(steady_state_records),
        "num_customers_total": len(records),
        "measurement_time": measurement_time,
        "run_until": run_until,
    }


# ==============================================================================
# SECTION: ANALYTICAL FORMULAS
# ------------------------------------------------------------------------------
# Closed-form M/M/1 results from queueing theory, used as the "ground truth"
# to validate the simulation against. These hold exactly in steady state,
# which is precisely what the warm-up handling above tries to approximate.
# ==============================================================================

def theoretical_mm1(lambda_rate, mu_rate):
    """Standard M/M/1 formulas (see e.g. Kleinrock, Queueing Systems Vol. 1)."""
    rho = lambda_rate / mu_rate                        # server utilisation
    Wq = lambda_rate / (mu_rate * (mu_rate - lambda_rate))  # mean wait in queue
    W = Wq + 1 / mu_rate                                # mean time in system
    Lq = lambda_rate * Wq                               # mean number waiting (Little's Law)
    return {"rho": rho, "Wq": Wq, "W": W, "Lq": Lq}


# ==============================================================================
# SECTION: VALIDATION OUTPUT
# ------------------------------------------------------------------------------
# Prints simulated vs theoretical side by side with % deviation, so it's
# immediately visible whether the simulation is tracking theory closely.
# ==============================================================================

def print_validation_table(sim_results, theory_results, lambda_rate, mu_rate, warmup_time):
    metrics = [
        ("Utilisation (rho)", "rho"),
        ("Mean wait in queue Wq (s)", "Wq"),
        ("Mean time in system W (s)", "W"),
        ("Mean queue length (Lq)", "Lq"),
    ]
    mean_service_s = 1.0 / mu_rate
    arrivals_per_hour = lambda_rate * SECONDS_PER_HOUR

    print("=" * 74)
    print("M/M/1 VALIDATION  (time unit = seconds)")
    print(f"  Note: {INTERIM_PARAMETER_NOTE}")
    print(f"  mean service 1/mu = {mean_service_s:.1f} s  ({mean_service_s / 60:.2f} min)")
    print(f"  mu                = {mu_rate:.6f} /s")
    print(f"  lambda            = {lambda_rate:.6f} /s  ({arrivals_per_hour:.2f} arrivals/hour)")
    print(f"  offered rho       = {lambda_rate / mu_rate:.3f}")
    dropped = sim_results["num_customers_total"] - sim_results["num_customers_kept"]
    print(
        f"Warm-up: arrivals before {warmup_time:.0f} s excluded; "
        f"measurement window [{warmup_time:.0f}, {sim_results['measurement_time']:.0f}) s; "
        f"SimPy run to {sim_results['run_until']:.0f} s (drain). "
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


def print_capacity_table(mean_service_times_s, rho_values, acceptable_wait_s):
    """Shows, for each literature holding time, how many arrivals/hour a
    single channel can take at a given rho, and the resulting theoretical Wq.
    """
    print()
    print("=" * 86)
    print("CAPACITY FROM CHOSEN HOLDING TIMES  (theoretical M/M/1, 1 channel)")
    print(f"Acceptable-wait working assumption: Wq <= {acceptable_wait_s:.1f} s")
    print("=" * 86)
    header = (
        f"{'1/mu (s)':>10}{'1/mu (min)':>12}{'rho':>8}"
        f"{'arrivals/h':>14}{'Wq (s)':>12}{'OK?':>8}"
    )
    print(header)
    print("-" * 86)
    for mean_s in mean_service_times_s:
        mu = 1.0 / mean_s
        for rho in rho_values:
            theory = theoretical_mm1(rho * mu, mu)
            arrivals_h = rho * mu * SECONDS_PER_HOUR
            ok = "yes" if theory["Wq"] <= acceptable_wait_s else "no"
            print(
                f"{mean_s:10.1f}{mean_s / 60:12.2f}{rho:8.2f}"
                f"{arrivals_h:14.2f}{theory['Wq']:12.2f}{ok:>8}"
            )
        print("-" * 86)
    print("=" * 86)


# ==============================================================================
# SECTION: PLOTTING
# ------------------------------------------------------------------------------
# Sweeps traffic intensity rho from low to close to 1 (holding mu_rate fixed
# and increasing lambda_rate) and plots simulated vs theoretical mean waiting
# time. This is the classic M/M/1 result: Wq grows sharply as rho -> 1,
# because the server has almost no spare capacity to absorb random bursts.
# ==============================================================================

def plot_wait_vs_rho_for_holding_times(
    mean_service_times_s,
    rho_values,
    acceptable_wait_s,
    random_seed=None,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
):
    """Wq vs rho for each literature holding time, with a 95% CI on the markers.

    The y-axis is logarithmic. Post-warm-up length follows 1/(1-rho)^2.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=SINGLE)
    markers = ["o", "s", "D", "^"]
    colours = [channel_colour(c) for c in (1, 7, 8, 21)]

    for idx, mean_s in enumerate(mean_service_times_s):
        mu_rate = 1.0 / mean_s
        means, lows, highs, theoretical_wq = [], [], [], []
        for i, rho in enumerate(rho_values):
            plan = plan_run(
                mean_s, rho, c=1,
                base_measurement=base_measurement,
                scale_factor=scale_factor,
                base_warmup=base_warmup,
            )
            lambda_rate = rho * mu_rate
            base_seed = None if random_seed is None else random_seed + 1000 * idx + 10 * i

            def once(seed, lambda_rate=lambda_rate, mu_rate=mu_rate, plan=plan):
                return run_mm1_simulation(
                    lambda_rate, mu_rate,
                    plan["measurement_time"], plan["warmup_time"],
                    random_seed=seed,
                )

            summary = run_replications(once, n_replications, base_seed or 0, ("Wq",))
            means.append(summary["Wq"]["mean"])
            lows.append(summary["Wq"]["low"])
            highs.append(summary["Wq"]["high"])
            theoretical_wq.append(theoretical_mm1(lambda_rate, mu_rate)["Wq"])
            print(
                f"  M/M/1 1/mu={mean_s:.0f}s rho={rho:.2f} "
                f"Wq={summary['Wq']['mean']:.2f} "
                f"CI=[{summary['Wq']['low']:.2f}, {summary['Wq']['high']:.2f}] "
                f"theory={theoretical_wq[-1]:.2f}"
            )

        colour = colours[idx]
        label = f"1/mu = {mean_s:.0f} s ({mean_s / 60:.0f} min)"
        theory_y = [y if y > 0 else float("nan") for y in theoretical_wq]
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
    ax.set_xlabel(r"Traffic intensity, $\rho = \lambda / \mu$")
    ax.set_ylabel("Mean waiting time in queue, $W_q$ (seconds)")
    ax.legend()
    fig.tight_layout()
    png, _pdf = save_figure(fig, "mm1_wait_vs_rho")
    plt.close(fig)
    print(f"Saved {png}")
    return png


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1) Single demo run at the 3-minute baseline, rho = 0.80, vs theory.
    sim_results = run_mm1_simulation(
        LAMBDA_RATE, MU_RATE, SIM_TIME, WARMUP_TIME, random_seed=RANDOM_SEED
    )
    theory_results = theoretical_mm1(LAMBDA_RATE, MU_RATE)
    print_validation_table(
        sim_results, theory_results, LAMBDA_RATE, MU_RATE, WARMUP_TIME
    )

    # 2) Capacity table: lambda (arrivals/hour) and Wq for each holding time.
    capacity_rhos = [0.05, 0.10, 0.20, 0.50, 0.70, 0.80, 0.90]
    holding_times = list(MEAN_SERVICE_TIMES_S.values())
    print_capacity_table(holding_times, capacity_rhos, ACCEPTABLE_WAIT_S)

    data_mu = 1.0 / DATA_CONTRAST_S
    data_theory = theoretical_mm1(DEMO_RHO * data_mu, data_mu)
    print(
        f"Data-session contrast only: 1/mu = {DATA_CONTRAST_S:.1f} s, "
        f"rho = {DEMO_RHO:.2f} -> {DEMO_RHO * data_mu * SECONDS_PER_HOUR:.1f} "
        f"arrivals/hour, Wq = {data_theory['Wq']:.2f} s"
    )

    # 3) Sweep rho at each literature holding time (lambda = rho * mu).
    #    Markers are R replications; length after warm-up scales as 1/(1-rho)^2.
    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    plot_wait_vs_rho_for_holding_times(
        holding_times,
        rho_values,
        ACCEPTABLE_WAIT_S,
        random_seed=RANDOM_SEED,
    )
