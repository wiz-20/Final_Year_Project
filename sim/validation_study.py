"""
Replication validation points for the literature parameter set.

Seeds and run lengths are fixed so a later results table matches the tests.
Nothing here changes literature_params values.
"""

import json

from sim.literature_params import (
    BLOCKING_TARGET,
    INTERIM_PARAMETER_NOTE,
    PRIMARY_MEAN_SERVICE_S,
    SECONDS_PER_HOUR,
    SERVER_COUNTS,
)
from sim.mm1_simulation import run_mm1_simulation, theoretical_mm1
from sim.mmc_b_simulation import run_mmc_b_simulation
from sim.mmc_k_simulation import (
    QUEUE_EXTRAS,
    SWEEP_SERVER_COUNTS,
    plot_mmck_sweep,
    run_mmck_simulation,
    theoretical_mmck,
    threshold_rows,
)
from sim.mmc_simulation import erlang_b, offered_load_for_blocking, run_mmc_simulation, theoretical_mmc
from sim.replication import (
    DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    DEFAULT_REPLICATIONS,
    DEFAULT_SCALE_FACTOR,
    metric_row,
    plan_run,
    run_replications,
)

DELAY_KEYS = ("Wq", "W", "Lq", "utilisation")
LOSS_KEYS = ("Wq", "W", "Lq", "utilisation", "B")

# Distinct seed blocks so the four cases do not share a stream.
SEED_MM1 = 11_000
SEED_MMC = 21_000
SEED_MMCB_A = 31_000
SEED_MMCB_RHO = 41_000

# Offered load used by the existing ~3% Erlang-B check (not the 2% GoS load).
ERLANG_B_VALIDATION_A = 3.9865

# Part B: enough accepted-or-blocked arrivals to estimate a 2% loss.
GOS_MIN_EXPECTED_ARRIVALS = 8_000
CARRIED_KEYS = ("B", "carried_erlang", "carried_calls_per_hour", "carried_utilisation")
SEED_CARRIED = 51_000
MMCK_KEYS = ("B", "Wq", "Lq", "carried_calls_per_hour")
SEED_MMCK = 61_000


def _case_record(name, model, plan, rows, n_replications, base_seed, extra=None):
    record = {
        "name": name,
        "model": model,
        "note": INTERIM_PARAMETER_NOTE,
        "n_replications": n_replications,
        "base_seed": base_seed,
        "plan": plan,
        "metrics": rows,
    }
    if extra:
        record.update(extra)
    return record


def _print_case(record):
    plan = record["plan"]
    print("=" * 88)
    print(record["name"])
    print(f"  {record['note']}")
    print(
        f"  R = {record['n_replications']}   "
        f"seeds {record['base_seed']} .. {record['base_seed'] + record['n_replications'] - 1}"
    )
    print(
        f"  post-warm-up N = {plan['n_measurement_service_times']} mean service times   "
        f"warm-up N = {plan['n_warmup_service_times']}   "
        f"scale_factor = {plan['scale_factor']}   "
        f"length_factor = {plan['length_factor']:.4f}"
    )
    print(
        f"{'Metric':<16}{'mean':>12}{'95% low':>12}{'95% high':>12}"
        f"{'theory':>12}{'dev %':>10}{'in CI':>8}"
    )
    for key, row in record["metrics"].items():
        dev = row["deviation_pct"]
        dev_s = "n/a" if dev is None else f"{dev:.3f}"
        print(
            f"{key:<16}{row['mean']:12.6g}{row['low']:12.6g}{row['high']:12.6g}"
            f"{row['theory']:12.6g}{dev_s:>10}{'yes' if row['inside_ci'] else 'NO':>8}"
        )
    print("=" * 88)


def validate_mm1(
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    rho=0.80,
    base_seed=SEED_MM1,
):
    """M/M/1 at 1/mu = 180 s, rho = lambda/mu = 0.80."""
    mu = 1.0 / mean_service_s
    lam = rho * mu
    plan = plan_run(
        mean_service_s, rho, c=1,
        base_measurement=base_measurement,
        scale_factor=scale_factor,
        base_warmup=base_warmup,
    )

    def once(seed):
        return run_mm1_simulation(
            lam, mu, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )

    summary = run_replications(once, n_replications, base_seed, DELAY_KEYS)
    theory = theoretical_mm1(lam, mu)
    theory_map = {
        "Wq": theory["Wq"],
        "W": theory["W"],
        "Lq": theory["Lq"],
        "utilisation": theory["rho"],
    }
    rows = {key: metric_row(summary[key], theory_map[key]) for key in DELAY_KEYS}
    return _case_record(
        f"M/M/1  1/mu = {mean_service_s:.0f} s, rho = {rho:.2f}",
        "M/M/1",
        plan,
        rows,
        n_replications,
        base_seed,
        extra={"lambda": lam, "mu": mu, "rho": rho, "c": 1},
    )


def validate_mmc(
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    c=8,
    rho=0.80,
    base_seed=SEED_MMC,
):
    """M/M/c (Erlang C) at c = 8, rho = lambda/(c mu) = 0.80."""
    mu = 1.0 / mean_service_s
    lam = rho * c * mu
    plan = plan_run(
        mean_service_s, rho, c=c,
        base_measurement=base_measurement,
        scale_factor=scale_factor,
        base_warmup=base_warmup,
    )

    def once(seed):
        return run_mmc_simulation(
            lam, mu, c, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )

    summary = run_replications(once, n_replications, base_seed, DELAY_KEYS)
    theory = theoretical_mmc(lam, mu, c)
    theory_map = {
        "Wq": theory["Wq"],
        "W": theory["W"],
        "Lq": theory["Lq"],
        "utilisation": theory["rho"],
    }
    rows = {key: metric_row(summary[key], theory_map[key]) for key in DELAY_KEYS}
    return _case_record(
        f"M/M/c  c = {c}, 1/mu = {mean_service_s:.0f} s, rho = {rho:.2f}",
        "M/M/c",
        plan,
        rows,
        n_replications,
        base_seed,
        extra={"lambda": lam, "mu": mu, "rho": rho, "c": c, "a": lam / mu},
    )


def validate_mmcb(
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    c=8,
    offered_a=None,
    rho=None,
    base_seed=SEED_MMCB_A,
    label=None,
):
    """M/M/c/c. Pass either offered_a (erlang) or offered rho = a/c."""
    if offered_a is None and rho is None:
        raise ValueError("Pass offered_a or rho.")
    mu = 1.0 / mean_service_s
    if offered_a is None:
        offered_a = rho * c
    else:
        rho = offered_a / c
    lam = offered_a * mu
    plan = plan_run(
        mean_service_s, rho, c=c,
        base_measurement=base_measurement,
        scale_factor=scale_factor,
        base_warmup=base_warmup,
    )

    def once(seed):
        return run_mmc_b_simulation(
            lam, mu, c, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )

    summary = run_replications(once, n_replications, base_seed, LOSS_KEYS)
    blocking = erlang_b(c, offered_a)
    theory_map = {
        "Wq": 0.0,
        "W": 1.0 / mu,
        "Lq": 0.0,
        "utilisation": offered_a * (1.0 - blocking) / c,
        "B": blocking,
    }
    rows = {key: metric_row(summary[key], theory_map[key]) for key in LOSS_KEYS}
    name = label or (
        f"M/M/c/c  c = {c}, a = {offered_a:.4f} erlang, "
        f"rho = {rho:.4f}, 1/mu = {mean_service_s:.0f} s"
    )
    return _case_record(
        name,
        "M/M/c/c",
        plan,
        rows,
        n_replications,
        base_seed,
        extra={"lambda": lam, "mu": mu, "rho": rho, "c": c, "a": offered_a},
    )


def carried_metrics(offered_a, blocking, mu, c):
    """a(1-B) in erlang, carried calls per hour, and a(1-B)/c."""
    carried = offered_a * (1.0 - blocking)
    return {
        "carried_erlang": carried,
        "carried_calls_per_hour": carried * mu * SECONDS_PER_HOUR,
        "carried_utilisation": carried / c,
    }


def validate_carried_at_gos(
    c,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    blocking_target=BLOCKING_TARGET,
    min_expected_arrivals=GOS_MIN_EXPECTED_ARRIVALS,
    base_seed=SEED_CARRIED,
):
    """Loss-model carried load at the offered load with Erlang-B = blocking_target."""
    mu = 1.0 / mean_service_s
    offered_a = offered_load_for_blocking(c, blocking_target)
    rho = offered_a / c
    lam = offered_a * mu
    blocking_theory = erlang_b(c, offered_a)
    plan = plan_run(
        mean_service_s,
        rho,
        c=c,
        base_measurement=base_measurement,
        scale_factor=scale_factor,
        base_warmup=base_warmup,
        offered_a=offered_a,
        min_expected_arrivals=min_expected_arrivals,
    )

    def once(seed):
        sim = run_mmc_b_simulation(
            lam, mu, c, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )
        sim.update(carried_metrics(offered_a, sim["B"], mu, c))
        return sim

    summary = run_replications(once, n_replications, base_seed, CARRIED_KEYS)
    theory = carried_metrics(offered_a, blocking_theory, mu, c)
    theory["B"] = blocking_theory
    rows = {key: metric_row(summary[key], theory[key]) for key in CARRIED_KEYS}
    return _case_record(
        (
            f"M/M/c/c carried load  c = {c}, "
            f"GoS target B = {blocking_target:.0%}, a = {offered_a:.4f} erlang"
        ),
        "M/M/c/c",
        plan,
        rows,
        n_replications,
        base_seed,
        extra={
            "lambda": lam,
            "mu": mu,
            "rho": rho,
            "c": c,
            "a": offered_a,
            "blocking_target": blocking_target,
        },
    )


def run_part_b_carried(
    server_counts=SERVER_COUNTS,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    echo=True,
):
    cases = []
    for index, c in enumerate(server_counts):
        if echo:
            print(f"Running carried-load replications for c = {c}...", flush=True)
        case = validate_carried_at_gos(
            c,
            n_replications=n_replications,
            scale_factor=scale_factor,
            base_measurement=base_measurement,
            base_warmup=base_warmup,
            base_seed=SEED_CARRIED + 1_000 * index,
        )
        cases.append(case)
        if echo:
            _print_case(case)
    return cases


def validate_mmck(
    c,
    queue_extra,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    mean_service_s=PRIMARY_MEAN_SERVICE_S,
    rho=0.80,
    base_seed=SEED_MMCK,
):
    """M/M/c/K at a fixed offered rho = lambda/(c mu)."""
    mu = 1.0 / mean_service_s
    K = c + queue_extra
    lam = rho * c * mu
    plan = plan_run(
        mean_service_s, rho, c=c,
        base_measurement=base_measurement,
        scale_factor=scale_factor,
        base_warmup=base_warmup,
    )

    def once(seed):
        return run_mmck_simulation(
            lam, mu, c, K, plan["measurement_time"], plan["warmup_time"], random_seed=seed
        )

    summary = run_replications(once, n_replications, base_seed, MMCK_KEYS)
    theory = theoretical_mmck(lam, mu, c, K)
    theory_map = {key: theory[key] for key in MMCK_KEYS}
    rows = {key: metric_row(summary[key], theory_map[key]) for key in MMCK_KEYS}
    return _case_record(
        (
            f"M/M/c/K  c = {c}, K-c = {queue_extra}, K = {K}, "
            f"rho = {rho:.2f}, 1/mu = {mean_service_s:.0f} s"
        ),
        "M/M/c/K",
        plan,
        rows,
        n_replications,
        base_seed,
        extra={
            "lambda": lam,
            "mu": mu,
            "rho": rho,
            "c": c,
            "K": K,
            "queue_extra": queue_extra,
        },
    )


def run_part_c_validation(
    server_counts=(8,),
    queue_extras=QUEUE_EXTRAS,
    rho=0.80,
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    echo=True,
    make_plot=False,
    plot_server_counts=SWEEP_SERVER_COUNTS,
):
    """CI validation of M/M/c/K. Plot is optional and uses these runs as markers."""
    cases = []
    index = 0
    for c in server_counts:
        for extra in queue_extras:
            if echo:
                print(f"Running M/M/c/K replications c = {c}, K-c = {extra}...", flush=True)
            case = validate_mmck(
                c,
                extra,
                n_replications=n_replications,
                scale_factor=scale_factor,
                base_measurement=base_measurement,
                base_warmup=base_warmup,
                rho=rho,
                base_seed=SEED_MMCK + 1_000 * index,
            )
            cases.append(case)
            index += 1
            if echo:
                _print_case(case)
    if make_plot:
        plot_mmck_sweep(sim_markers=_markers_from_cases(cases), server_counts=plot_server_counts)
    return cases


def _markers_from_cases(cases):
    markers = []
    for case in cases:
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
    return markers


def part_c_thresholds():
    return threshold_rows()


def run_part_a_validation(
    n_replications=DEFAULT_REPLICATIONS,
    scale_factor=DEFAULT_SCALE_FACTOR,
    base_measurement=DEFAULT_BASE_MEASUREMENT_SERVICE_TIMES,
    base_warmup=DEFAULT_BASE_WARMUP_SERVICE_TIMES,
    echo=True,
):
    """The four required validation points. Returns a list of case records."""
    steps = [
        ("M/M/1", lambda: validate_mm1(
            n_replications, scale_factor, base_measurement, base_warmup
        )),
        ("M/M/c", lambda: validate_mmc(
            n_replications, scale_factor, base_measurement, base_warmup
        )),
        ("M/M/c/c a=3.9865", lambda: validate_mmcb(
            n_replications, scale_factor, base_measurement, base_warmup,
            offered_a=ERLANG_B_VALIDATION_A,
            base_seed=SEED_MMCB_A,
            label=(
                f"M/M/c/c  c = 8, a = {ERLANG_B_VALIDATION_A} erlang, "
                f"1/mu = {PRIMARY_MEAN_SERVICE_S:.0f} s"
            ),
        )),
        ("M/M/c/c rho=0.80", lambda: validate_mmcb(
            n_replications, scale_factor, base_measurement, base_warmup,
            rho=0.80,
            base_seed=SEED_MMCB_RHO,
            label=(
                f"M/M/c/c  c = 8, rho = 0.80, "
                f"1/mu = {PRIMARY_MEAN_SERVICE_S:.0f} s"
            ),
        )),
    ]
    cases = []
    for label, runner in steps:
        if echo:
            print(f"Running {label} replications...", flush=True)
        case = runner()
        cases.append(case)
        if echo:
            _print_case(case)
    return cases


def save_cases(cases, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")


def assert_theory_inside(cases):
    """Raise AssertionError listing any metric whose theory is outside the CI."""
    misses = []
    for case in cases:
        for key, row in case["metrics"].items():
            if not row["inside_ci"]:
                misses.append(
                    f"{case['name']} {key}: mean={row['mean']:.6g} "
                    f"CI=[{row['low']:.6g}, {row['high']:.6g}] theory={row['theory']:.6g}"
                )
    if misses:
        raise AssertionError("Theory outside 95% CI:\n" + "\n".join(misses))
