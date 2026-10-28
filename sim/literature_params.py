"""
Interim simulation parameters (literature-sourced, not operator data).

All default holding times, channel counts, and QoS targets in this project use
published teletraffic / cellular sources until Vodacom (Jaco) figures arrive.
When operator data is plugged in, change values here only and re-run the same
simulation scripts — the models themselves stay unchanged.

See docs/progress_report.tex and meeting_brief.tex for the holding-time table.
"""

# Voice holding-time sensitivity sweep (seconds): 1, 3, 10, 15 min baseline grid.
MEAN_SERVICE_TIMES_S = {
    "short_call_1min": 60.0,
    "baseline_3min": 180.0,
    "long_session_10min": 600.0,
    "extreme_15min": 900.0,
}

PRIMARY_MEAN_SERVICE_S = MEAN_SERVICE_TIMES_S["baseline_3min"]
DATA_SESSION_CONTRAST_S = 8.0  # contrast only, not design mu

# GSM-grounded traffic channels (TRX -> TCH counts); interim until site data.
SERVER_COUNTS = (1, 7, 8, 21, 45)
SERVER_LABELS = {
    1: r"$c=1$ (baseline)",
    7: r"$c=7$ (small, 1 TRX TCH)",
    8: r"$c=8$ (1 TRX)",
    21: r"$c=21$ (typical, 3 TRX)",
    45: r"$c=45$ (large, 6 TRX)",
}

# Working QoS assumptions (supervisor / operator may replace).
ACCEPTABLE_WAIT_S = 10.0
BLOCKING_TARGET = 0.02

SECONDS_PER_HOUR = 3600.0

INTERIM_PARAMETER_NOTE = (
    "Interim parameters from literature (holding times, GSM channel counts); "
    "not Vodacom measured traffic."
)
