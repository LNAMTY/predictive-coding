# In-Fluid-PC

Predictive coding trained **without backpropagation**, together with a from-scratch
implementation of *Incompressible-Fluid Networks* (Goertzel, Oct 2025): incompressible
Navier–Stokes transport, Helmholtz–Hodge projection, and Hamilton–Jacobi–Bellman
regularisation, built as components that can be switched on and off one at a time.

Predictive coding stands alone. Everything taken from the paper sits on top of it and can be
removed without touching the PC core:

```
influid_pc/
  pc/          predictive coding.          mandatory. no autograd, no backward chain.
  fluid/       Navier-Stokes transport.    remove: drop --fluid (off by default)
  regularizers/hjb.py                      remove: drop --hjb   (off by default)
  regularizers/transport.py                remove: --transport-alpha 0 --transport-beta 0
  diagnostics/ the measurements that decide whether any of it works
```

## Headline results

| | |
|---|---|
| **PC on full MNIST, zero backprop** | **97.2%** (3-layer, local Hebbian updates only) |
| **PC update vs. true backprop gradient** | cosine **0.99999**, but only under the Fixed Prediction Assumption |
| **Strict PC at 8 hidden layers** | cosine **0.756** and falling, costing 3.5 points of accuracy |
| **Fluid transport invariants** | mass drift `1e-7`, `div u ≈ 1e-6`, CFL pinned to 0.40, exactly zero flux into obstacles |
| **Routing task (paper §8)** | **54%** of the budget delivered, vs **0.05%** for diffusion and **0%** for raw-gradient+projection |
| **Paper's κ=0.3 diffusion warmup** | harmful for classification; collapses a linear probe to chance in one step |
| **Fluid layer on EMNIST** | **loses** to the same network with the layer deleted: fewer params, 21× faster |

Two of these contradict the paper, and one of them is a one-line fix. The full argument, with the
experiment behind every number, is in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

![routing](figures/fig5_routing_task.png)

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# predictive coding alone
.venv/bin/python train.py --dataset mnist --epochs 10 --train-subset 0 \
    --hidden 256 128 --weight-lr 0.1 --track-alignment

# different class counts
.venv/bin/python train.py --dataset mnist --num-classes 3 --epochs 8

# a different dataset with a different class count, plus Navier-Stokes and HJB
.venv/bin/python train.py --dataset emnist_letters --fluid --hjb --epochs 4
```

Tests, experiments and figures:

```bash
.venv/bin/python -m pytest tests/ -q          # 19 tests: invariants, locality, PC vs backprop
PYTHONPATH=. .venv/bin/python scripts/alignment_study.py
PYTHONPATH=. .venv/bin/python scripts/routing_task.py
PYTHONPATH=. .venv/bin/python scripts/run_experiments.py
PYTHONPATH=. .venv/bin/python scripts/make_figures.py
```

## What each piece is

**`pc/`** — Predictive coding. Each layer holds a state, predicts the layer above, and computes a
local error. Inference relaxes the states; learning is Hebbian (`ΔW ∝ error × presynaptic
activity`). The linear connections never call autograd: the update rules are derived by hand, so
the absence of backpropagation is a property of the code rather than a claim about it.
`tests/test_locality.py` patches `torch.autograd` to raise and trains the network anyway.

There are two implementations, and they agree:

* `pc/core.py` — a single self-contained class. Read this one first: the whole algorithm is
  ~150 lines with the mathematics in the docstring, and it depends on nothing else in the repo.
* `pc/network.py` + `pc/connections.py` — the same algorithm with each edge behind a `Connection`
  interface, which is what lets the fluid layer drop in without the PC rules changing. `train.py`
  uses this one.

Two inference modes, and the difference between them matters a great deal:

- `--prediction-mode strict` — top-down predictions are recomputed from the relaxed states each step.
- `--prediction-mode fixed` — predictions are frozen at their feedforward values
  (the *Fixed Prediction Assumption*, Millidge et al. 2020).

**`fluid/`** — The paper's In-Fluid-Net transport layer. Activations become a conserved mass
`ρ ≥ 0, Σρ = 1` on a grid; a learned **stream function** generates a velocity field `u = ∇⊥ψ`
that is divergence-free by construction; the mass is advected by conservative upwind fluxes with
CFL targeting. Mass conservation and zero divergence hold to machine precision rather than
approximately, which `tests/test_fluid_invariants.py` checks.

**`regularizers/hjb.py`** — Stationary Hamilton–Jacobi–Bellman residual `ν∆W − ½‖∇W‖² − V`, with
the running cost `V` taken from the layer's own prediction error, which keeps the regulariser local.

**`diagnostics/`** — `bp_alignment.py` measures the cosine between the PC update and the true
backprop gradient. The fluid layer logs mass drift, `‖div u‖`, the Courant number, and the fraction
of the velocity field that survives projection.

## Removing things

The components are independent, and the defaults are the minimum:

| To drop | Do this |
|---|---|
| Navier–Stokes transport | omit `--fluid` (off by default) |
| HJB regularisation | omit `--hjb` (off by default) |
| Transport regularisers | `--transport-alpha 0 --transport-beta 0` |
| Leray projection | omit `--projection` (off by default; a no-op in stream mode anyway) |
| Everything except PC | `python train.py --dataset mnist` |

Deleting `influid_pc/fluid/` and `influid_pc/regularizers/` leaves a working predictive-coding
implementation, because nothing in `pc/` imports them.

## Reference

Goertzel, B. *Incompressible-Fluid Networks* (v1, October 2025).
Millidge, B., Tschantz, A., Buckley, C. L. *Predictive Coding Approximates Backprop along
Arbitrary Computation Graphs* (2020).
Whittington, J. C. R., Bogacz, R. *An Approximation of the Error Backpropagation Algorithm in a
Predictive Coding Network with Local Hebbian Synaptic Plasticity* (2017).
