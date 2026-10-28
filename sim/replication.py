"""
Independent replications and Student-t confidence intervals.

Run length after warm-up (mean service times):

    N = base_measurement * scale_factor / (1 - rho)^2

for offered rho < 1. The factor 1/(1-rho)^2 is the usual growth of delay
variance as the queue approaches instability, so rho = 0.85 and 0.90 keep a
comparable number of effective samples. scale_factor is an explicit knob
(default 1). For rho >= RHO_LENGTH_CAP the gap (1-rho) is floored so the
length stays finite; rho = 0.90 is below that cap and is not floored.

Warm-up length still comes from warmup.warmup_service_times_for_rho, applied
to this post-warm-up N. Each replication has its own seed.
"""

import math
import statistics

from sim.warmup import time_from_service_times, warmup_service_times_for_rho

DEFAULT_REPLICATIONS = 10
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES = 1000
DEFAULT_BASE_WARMUP_SERVICE_TIMES = 200
DEFAULT_CONFIDENCE = 0.95
# Floor (1-rho) only above this offered utilisation. 0.90 is unfloored.
RHO_LENGTH_CAP = 0.95
C_LENGTH_REFERENCE = 8


def _betacf(a, b, x, max_iter=200, eps=3e-14):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a, b, x):
    """I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t, df):
    """CDF of Student's t with df degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _regularized_incomplete_beta(0.5 * df, 0.5, x)
    if t >= 0.0:
        return 1.0 - tail
    return tail


def student_t_ppf(p, df, tol=1e-10):
    """Quantile of Student's t. p is the cumulative probability."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1).")
    if df <= 0:
        raise ValueError("df must be positive.")
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, df) > p:
        lo *= 2.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def length_factor(rho, scale_factor=DEFAULT_SCALE_FACTOR, rho_cap=RHO_LENGTH_CAP):
    """Multiplier on the base measurement length: scale_factor / (1-rho)^2."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")
    rho_used = min(float(rho), float(rho_cap))
    gap = max(1.0 - rho_used, 1e-6)
    return scale_factor / (gap * gap)


def measurement_service_times_after_warmup(
    base_measurement,
    rho,
    scale_factor=DEFAULT_SCALE_FACTOR,
    c=1,
    c_reference=C_LENGTH_REFERENCE,
    rho_cap=RHO_LENGTH_CAP,
):
    """Post-warm-up run length in mean service times.

    Large c is shortened by c_reference/c so the arrival count stays in the
    same order as a c_reference run at the same rho (lambda grows with c).
    """
    n = base_measurement * length_factor(rho, scale_factor, rho_cap)
    if c > c_reference:
        n *= c_reference / c
    return max(1, int(round(n)))


def apply_arrival_floor(n_measurement, offered_a, min_expected_arrivals):
    """Raise post-warm-up length so E[arrivals] >= min_expected_arrivals.

    Needed for a rare blocking probability (for example 2% on one channel),
    where 1/(1-rho)^2 does not by itself produce enough losses to estimate B.
    """
    if min_expected_arrivals is None:
        return n_measurement
    if offered_a <= 0:
        raise ValueError("offered_a must be positive when a floor is set.")
    needed = int(math.ceil(min_expected_arrivals / offered_a))
    return max(n_measurement, needed)


def plan_run(
    mean_service_s,
    rho,
    c,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    c_reference=C_LENGTH_REFERENCE,
    rho_cap=RHO_LENGTH_CAP,
    offered_a=None,
    min_expected_arrivals=None,
):
    """Absolute warm-up and measurement-end times (seconds) for one run.

    measurement_time is the end of the arrival window. The post-warm-up
    duration is N mean service times, with N scaled by 1/(1-rho)^2.
    """
    n_meas = measurement_service_times_after_warmup(
        base_measurement, rho, scale_factor, c, c_reference, rho_cap
    )
    if min_expected_arrivals:
        if offered_a is None:
            raise ValueError("offered_a is required when min_expected_arrivals is set.")
        n_meas = apply_arrival_floor(n_meas, offered_a, min_expected_arrivals)
    n_warm = warmup_service_times_for_rho(base_warmup, n_meas, min(float(rho), 0.999))
    warmup_time = time_from_service_times(n_warm, mean_service_s)
    measurement_time = warmup_time + time_from_service_times(n_meas, mean_service_s)
    return {
        "n_measurement_service_times": n_meas,
        "n_warmup_service_times": n_warm,
        "warmup_time": warmup_time,
        "measurement_time": measurement_time,
        "scale_factor": scale_factor,
        "base_measurement": base_measurement,
        "base_warmup": base_warmup,
        "rho": rho,
        "c": c,
        "offered_a": offered_a,
        "min_expected_arrivals": min_expected_arrivals,
        "length_factor": length_factor(rho, scale_factor, rho_cap),
    }


def confidence_interval(samples, confidence=DEFAULT_CONFIDENCE):
    """Mean and two-sided Student-t CI. Zero sample sd gives a degenerate CI."""
    values = [float(x) for x in samples]
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 replications for a t interval.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    df = n - 1
    if sd == 0.0:
        half = 0.0
        tcrit = None
    else:
        tcrit = student_t_ppf(0.5 + 0.5 * confidence, df)
        half = tcrit * sd / math.sqrt(n)
    return {
        "mean": mean,
        "low": mean - half,
        "high": mean + half,
        "half": half,
        "sd": sd,
        "n": n,
        "df": df,
        "tcrit": tcrit,
        "confidence": confidence,
    }


def theory_inside_ci(interval, theory, rtol_edge=1e-9):
    """True when theory lies inside [low, high], with a tiny edge tolerance."""
    eps = rtol_edge * max(1.0, abs(theory), abs(interval["mean"]))
    return (interval["low"] - eps) <= theory <= (interval["high"] + eps)


def deviation_percent(mean, theory):
    """(sim - theory) / theory * 100. None when theory is 0."""
    if theory == 0.0:
        return None if mean != 0.0 else 0.0
    return (mean - theory) / theory * 100.0


def run_replications(run_once, n_replications, base_seed, keys, confidence=DEFAULT_CONFIDENCE):
    """Run run_once(seed) n_replications times. Seeds are base_seed + i.

    run_once must return a dict containing every name in keys.
    """
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2.")
    collected = {key: [] for key in keys}
    for i in range(n_replications):
        result = run_once(base_seed + i)
        for key in keys:
            collected[key].append(result[key])
    return {key: confidence_interval(collected[key], confidence) for key in keys}


def log_scale_errorbar(ax, x_values, means, lows, highs, floor=1e-2, **kwargs):
    """Error bar on a log y-axis.

    Means at or below `floor` are omitted. Bars are clipped to `floor` so a
    confidence interval that crosses zero does not stretch the log axis.
    """
    xs, ys, err_lo, err_hi = [], [], [], []
    for x, mean, low, high in zip(x_values, means, lows, highs):
        if mean <= floor:
            continue
        low_pos = min(max(low, floor), mean)
        high_pos = max(high, mean)
        xs.append(x)
        ys.append(mean)
        err_lo.append(mean - low_pos)
        err_hi.append(high_pos - mean)
    if xs:
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], **kwargs)
    return xs


def metric_row(interval, theory):
    return {
        "mean": interval["mean"],
        "low": interval["low"],
        "high": interval["high"],
        "theory": theory,
        "deviation_pct": deviation_percent(interval["mean"], theory),
        "inside_ci": theory_inside_ci(interval, theory),
    }
