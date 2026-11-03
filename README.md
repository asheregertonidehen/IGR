# Innovation-Gated Representations

Code to reproduce the Innovation-Gated Representations experiments in simulation.

## Names in the code

| Paper | Code |
|---|---|
| IGR-sup | `Full` |
| IGR-task | `A3c` |
| correction scale kappa | `corr_scale` |
| ablations | `A1`-`A11`, defined in `models/variants.py` |
| student families | `F1_S1`, `F1_S2` (cell), `F2b` (plain baselines), `F2b_A9` |

## Setup

Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt       # requirements-gpu.txt on a CUDA machine
.venv/bin/python -m pytest
```

## Run

```sh
.venv/bin/python run_all.py --profile smoke     # whole pipeline at tiny sizes on CPU
.venv/bin/python run_all.py --profile full      # the reported experiment
.venv/bin/python analysis/make_numbers.py
.venv/bin/python analysis/make_numbers2.py
```

The full run takes about 5 hours on one NVIDIA L40S. It is resumable and writes to `results/` and `logs/`.

Student settings are in `configs/run.yaml`: 1000 epochs, constant learning rate 1e-4, a 100-step backward window for
the cell and A9, and 500 steps for the plain baselines. Students use `corr_scale` 0.5 and teachers 1.0.

## Layout

| Path | Contents |
|---|---|
| `models/gated_cell.py` | the innovation-gated cell |
| `models/variants.py` | variants, teachers, families |
| `sim/`, `sensing/` | quadrotor simulator, link, IMU and Kalman filter models |
| `train/` | expert, teacher and student training |
| `eval/` | suites, paired counterfactual events, probes, reliance |
| `configs/` | run settings and fault suites |
| `data/` | fixed drone sets |
| `reference/` | reference numbers, figures and selected student checkpoints |

`reference/students/*.pkl` hold the selected students of each family, stacked by model, with model names in the
stored metadata.
