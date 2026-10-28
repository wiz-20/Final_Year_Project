"""
Warm-up and run-length helpers for discrete-event queue simulations.

Policy (explainable to examiner):
  1. Run the SimPy model long enough to discard an initial warm-up interval
     (empty-system transient).
  2. Collect statistics only for customers / arrivals whose arrival time falls
     in [warmup_time, measurement_time).
  3. Extend the clock past measurement_time by a drain period so customers who
     arrived during the measurement window can finish service and contribute
     a full waiting-time sample (avoids optimistic bias at high rho).
"""


def time_from_service_times(n_service_times: float, mean_service_s: float) -> float:
    """Convert 'N mean service times' into seconds (run-length scaling)."""
    return n_service_times * mean_service_s


def total_run_time(
    measurement_time: float,
    mean_service_s: float,
    drain_service_times: float = 50.0,
) -> float:
    """SimPy stop time: measurement horizon plus post-window drain."""
    return measurement_time + drain_service_times * mean_service_s


def in_measurement_window(arrival_time: float, warmup_time: float, measurement_time: float) -> bool:
    return warmup_time <= arrival_time < measurement_time


def scaled_service_time_counts(c: int, n_service_times: int, warmup_service_times: int):
    """
    Shorter runs for large c (same rho, higher absolute lambda): keep runtime manageable.
    """
    if c <= 8:
        return n_service_times, warmup_service_times
    n_run = max(6000, int(n_service_times * 8 / c))
    warmup_run = max(600, int(warmup_service_times * 8 / c))
    return n_run, warmup_run


def scaled_service_time_counts_b(c: int, n_service_times: int, warmup_service_times: int):
    """Erlang-B sweeps use slightly longer minimum runs than Erlang-C."""
    if c <= 8:
        return n_service_times, warmup_service_times
    n_run = max(8000, int(n_service_times * 8 / c))
    warmup_run = max(800, int(warmup_service_times * 8 / c))
    return n_run, warmup_run


def warmup_service_times_for_rho(
    base_warmup_service_times: int,
    n_measurement_service_times: int,
    rho: float,
    max_fraction: float = 0.45,
) -> int:
    """
    Warm-up in mean-service-time units, scaled by rho.

    Starting from an empty system, the queue takes longer to reach steady state
    when rho is high; a fixed warm-up fraction otherwise leaves Wq biased low.
    """
    rho = max(0.0, min(float(rho), 0.999))
    fraction = min(max_fraction, 0.08 + 0.35 * rho)
    scaled = int(n_measurement_service_times * fraction)
    return max(base_warmup_service_times, scaled)
