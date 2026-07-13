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
```

Every run is one command. `make` on its own lists them.

**The task.** Predictive coding across different class counts, and Navier–Stokes on a dataset that
is neither MNIST nor 10-class:

| command | what it runs |
|---|---|
| `make classes` | PC across 2 / 3 / 5 / 10 classes on MNIST |
| `make fluid` | plus Navier–Stokes transport, on EMNIST-Letters (**26 classes**) |
| `make hjb` | the same, plus Hamilton–Jacobi–Bellman regularisation |

**The finding.**

| command | what it runs |
|---|---|
| `make alignment` | cosine between the PC update and the true backprop gradient, strict vs fixed |
| `make deep` | what that misalignment costs in accuracy, at 6 hidden layers |
| `make routing` | the paper's routing task, with the baselines it lacks |

**Reference points.** `make pc` is plain predictive coding on MNIST with all 10 classes — the
starting point the task moves beyond, and the source of the 97.2%-without-backprop number.
`make bp` is the backprop baseline, given the best of a learning-rate sweep.

Housekeeping: `make test` (19 tests), `make figures`, `make lint`.

Each run prints a header saying exactly what it is training, an aligned per-epoch table, and a
summary. The fluid runs add the transport invariants (mass drift, `‖div u‖`, CFL) as columns, so a
broken flow is visible while it is still training:

```
  predictive coding  ·  fixed predictions (FPA)
  ────────────────────────────────────────────────────────────────────────────
  dataset       emnist_letters  (26 classes, 784-dim)
  network       784 -> [fluid 14x14] -> 128 -> 26   (201,985 params)
  learning      lr 0.1 · nudge 0.2 · 24 inference steps · 4 epochs
  extras        Navier-Stokes transport
  ────────────────────────────────────────────────────────────────────────────

    epoch   test acc  free energy  mass drift     div u    CFL     time
    ─────  ─────────  ───────────  ──────────  ────────  ─────  ───────
        1     52.31%       0.2230     1.7e-07   1.4e-06   0.40    47.2s
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
