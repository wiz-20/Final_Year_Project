# Capacity and Throughput Planning for Mobile Base Stations

Discrete-event simulation code for the BEng (Electrical and Electronic Engineering) final-year project
*Capacity and Throughput Planning for Mobile Base Stations: A Queuing Simulation Approach*,
Stellenbosch University, 2026. Supervisor: Prof. R. Wolhuter.

A GSM cell is modelled as an M/M/c/K queue and simulated with [SimPy](https://simpy.readthedocs.io/).
The simulator is verified against the exact queueing results and then used to find the load at
which a cell becomes overloaded, and to test the model against measured traffic.

## Models

| Engine | Queue | System | Reference result |
|---|---|---|---|
| `sim/mm1_simulation.py` | M/M/1 | single channel, unlimited queue | M/M/1 |
| `sim/mmc_simulation.py` | M/M/c | delay system, unlimited queue | Erlang C |
| `sim/mmc_b_simulation.py` | M/M/c/c | loss system, no queue | Erlang B |
| `sim/mmc_k_simulation.py` | M/M/c/K | finite waiting room of K - c places | M/M/c/K stationary distribution |

All engines use the same arrival generator and call process:

1. Calls arrive as a Poisson process with rate λ.
2. A call is blocked if the number in the system has reached K.
3. Otherwise it requests a channel and waits if all c are busy.
4. It holds the channel for an exponential time with mean 1/μ, then releases it.

`K = c` gives the loss system and a very large `K` gives the delay system.

## Method

- **Warm-up removal** (`sim/warmup.py`). Each run starts empty, and observations from the
  start-up transient are discarded.
- **Independent replications** (`sim/replication.py`). Each operating point is run R = 10 times
  with separate fixed seeds and reported as a mean with a 95% Student-t confidence interval.
- **Verification** (`sim/validation_study.py`, `tests/`). A point is accepted when the exact
  value lies inside the confidence interval.
- **Overload analysis.** Capacity is the largest load that meets 2% blocking (loss system) or a
  10 s mean wait (delay system). Literature parameters are in `sim/literature_params.py`.
- **Measured data.** The model is compared with published GSM sector data
  (`sim/case_study.py`, `case_study_loo.py`), an interconnect trunk group
  (`trunk_case_study.py`), and two operational sites (`step*.py`).

## Repository layout

| Path | Contents |
|---|---|
| `sim/` | Simulation engines, replication and confidence intervals, warm-up, parameters, plot style |
| `tests/` | Unit and end-to-end verification tests |
| `case_study_loo.py` | Leave-one-out prediction of the published GSM busy hours |
| `trunk_case_study.py` | Interconnect trunk group case study |
| `step1_clean_data.py` to `step8_generator.py` | Pipeline for the Stellenbosch site data (sizing, held-back test, growth, traffic generator) |
| `plot_results.py`, `plot_stellenbosch.py`, `make_extra_figures.py` | Figures for the report |
| `data/trunk_case_study.csv` | Published trunk group measurements |

## Setup

Requires Python 3.11 or later.

```bash
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root, as modules (`python -m sim....`), not from inside `sim/`.

## Usage

Single models (each prints a comparison with theory and writes figures to `output/figures/`):

```bash
python -m sim.mm1_simulation
python -m sim.mmc_simulation
python -m sim.mmc_b_simulation
python -m sim.mmc_k_simulation
```

The 3D surfaces make `mmc_simulation` and `mmc_b_simulation` slow.

Fast closed-form checks:

```bash
python -m unittest tests.test_replication tests.test_mmc.TestTheoreticalMmc tests.test_mmc_k.TestMmckClosedForm tests.test_carried.TestCarriedFormula -v
```

Full verification study (about 10 minutes, fixed seeds). It writes JSON results to `output/`,
and the second command summarises them in `output/results_summary.md`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m sim.write_results_summary
```

Published-data case studies:

```bash
python case_study_loo.py
python trunk_case_study.py
```

## Data availability

The published GSM and trunk group measurements are taken from the cited papers and are included
(`sim/case_study_data.py`, `data/trunk_case_study.csv`).

The hourly traffic of the two Stellenbosch sites was provided by a South African operator
and is commercially sensitive, so it is **not** included. The `step*.py` scripts and
`plot_stellenbosch.py` are provided for transparency but need that data to run.

## Reproducibility

Every replication uses its own fixed seed, so each result in the report can be regenerated exactly.

## Citation

> W. E. Peru, "Capacity and Throughput Planning for Mobile Base Stations: A Queuing Simulation
> Approach," BEng project report, Department of Electrical and Electronic Engineering,
> Stellenbosch University, 2026.
