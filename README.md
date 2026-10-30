# Innovation-Gated Representations

Code for a small recurrent quadrotor controller whose memory is updated through an innovation against an internal
expectation of each externally delivered channel. Sensing faults (delay, freezes, reordering, low rate, outages) and
real physical changes (gusts, payloads) are evaluated separately, in simulation.

Memory is supervised on physical targets (state, disturbance, identity, inertial forecast). We compare supervising the
channel expectation as well (**IGR-sup**) with letting the control task shape it (**IGR-task**).

## Main results

Success over 23 held-out fault suites, in percent. Teacher-level 95 % intervals are in `reference/numbers.json`.

| Model | Held-out | Sensing | External-force probe R² |
|---|---:|---:|---:|
| B4: GRU, expert labels | 50.3 | 57.4 | 0.17 |
| B5m: GRU fed by a Kalman filter | 61.7 | 76.3 | 0.28 |
| A7: GRU, teacher labels | 62.8 | 71.5 | 0.29 |
| A9: GRU with physical heads | 61.1 | 69.4 | 0.62 |
| IGR-sup | 58.8 | 69.5 | 0.45 |
| IGR-task | 63.1 | 72.8 | 0.64 |

In these configurations IGR-task matches the strongest GRU in aggregate, adds physical readouts and improves at
5-20 Hz sensing (79.3 % vs. 77.2 %). Its gains are concentrated in sparse sensing and outages rather than uniform. It
combines three changes (likelihood, its distillation term and a stop-gradient) that were not separated. Results rest
on three teacher seeds, with three IGR-task students, and on our own simulated sensing model.

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

The full run took about 5 hours on one NVIDIA L40S. It is resumable and writes to `results/` and `logs/`. Evaluation
uses seed 54321 and selection seed 777.

Student settings are in `configs/run.yaml`: 1000 epochs, constant learning rate 1e-4, a 100-step backward window for
the cell and A9, and 500 steps for the plain baselines. Students use `corr_scale` 0.5 and teachers 1.0. The stability
comparison in `reference/figures/fig_stability.pdf` varies `CORR_SCALE` in `models/variants.py` and `bptt` in
`configs/run.yaml`.

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
| `reference/` | reported numbers, data figures and selected student checkpoints |

`reference/students/*.pkl` hold the selected students of each family, stacked by model, with model names in the
stored metadata. They also contain a few unused parameter groups from training, which the code ignores.
