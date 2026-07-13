# In-Fluid-PC

Predictive coding trained without backpropagation, plus a from-scratch implementation of
*Incompressible-Fluid Networks* (Goertzel, Oct 2025) — Navier–Stokes transport, Helmholtz–Hodge
projection, and Hamilton–Jacobi–Bellman regularisation.

This is my submission for the training task:

> Train PC with different classification (current MNIST 10 class). + Navier-Stokes (fluid dynamics),
> use a dataset other than MNIST with a different class count.

I built predictive coding first, on its own, and then added each piece of the paper on top of it as
a component I can switch off independently. The point was to be able to defend every line I keep.
Along the way I found two things that contradict the paper, and one of them is a one-line fix.

## How I met the task

| what was asked | what I ran | result |
|---|---|---|
| PC on a **different classification** than MNIST-10 | MNIST at k = 2, 3, 5 classes, and PC on **EMNIST-Letters, 26 classes** | 99.91 / 99.46 / 98.92, and 72.14 on EMNIST-26 |
| Navier–Stokes on **another dataset, different class count** | **EMNIST-Letters, 26 classes** — every fluid run | 68.98 with transport, 64.82 with transport + HJB |

No Navier–Stokes run in this repo uses MNIST, and none uses 10 classes. The 10-class MNIST run is
kept only as the reference point to measure against.

## What I found

| | |
|---|---|
| PC on full MNIST, zero backprop | **97.2%** (3 layers, local Hebbian updates only) |
| PC update vs. the true backprop gradient | cosine **0.99999** — but only under the Fixed Prediction Assumption |
| Strict PC at 8 hidden layers | cosine **0.756** and falling, costing 3.5 points of accuracy |
| Fluid transport invariants | mass drift `1e-7`, `div u ≈ 1e-6`, CFL pinned to 0.40, exactly zero flux into obstacles |
| Routing task (paper §8) | **54%** of the budget delivered, vs **0.05%** for diffusion and **0%** for raw-gradient + projection |
| The paper's κ=0.3 diffusion warmup | harmful for classification: it collapses a linear probe to chance in one step |
| The fluid layer on EMNIST | **loses** to the same network with the layer deleted — fewer params, 21× faster |

The two that contradict the paper:

1. **The envelope theorem only holds if you freeze the top-down predictions.** Implemented strictly,
   as §4.2 specifies, PC converges to a fixed point that is *not* the backprop gradient, and
   shrinking the output nudge does not close the gap. Freezing the predictions fixes it, and that is
   a one-line change.
2. **The fluid layer is dominated on classification.** The mechanism is correct and it wins
   decisively on routing, but a classifier is not a routing problem, and the accuracy it appears to
   add is just the parameters it adds.

The full argument, with the experiment behind every number, is in
**[docs/FINDINGS.md](docs/FINDINGS.md)**. If you want the ideas explained from first principles
instead, read **[docs/UNDERSTANDING.md](docs/UNDERSTANDING.md)**.

![routing](figures/fig5_routing_task.png)

## Running it

Setup:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Every run is a single command. `make` on its own prints this list.

**The task:**

```bash
make classes    # PC across different class counts: 2 / 3 / 5 / 10        (~4 min)
make fluid      # + Navier-Stokes, EMNIST-Letters, 26 classes             (~6 min)
make hjb        # the same, + Hamilton-Jacobi-Bellman regularisation      (~7 min)
```

**The findings:**

```bash
make alignment  # PC vs the true backprop gradient, strict vs fixed       (~3 min)
make deep       # what that misalignment costs in accuracy, at 6 layers   (~5 min)
make routing    # the paper's routing task, with the baselines it lacks   (~1 min)
```

**Reference points:**

```bash
make pc         # plain PC, MNIST 10 classes — the starting point         (~2 min)
make bp         # backprop, given the best of a learning-rate sweep       (~1 min)
```

**Checks:**

```bash
make test       # 19 tests: invariants, locality, PC vs backprop
make figures    # rebuild figures/ from results/
make lint       # ruff
```

Anything not covered by a `make` target can be run through `train.py` directly, which takes the
components as flags (`--fluid`, `--hjb`, `--projection`, `--obstacles`, `--prediction-mode`,
`--num-classes`, and so on — `python train.py --help` lists them all).

Every run prints what it is training, then a table per epoch. The fluid runs carry the transport
invariants as live columns, so if the flow breaks I see it while it is still training rather than
afterwards:

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

## How the code is laid out

```
influid_pc/
  pc/            predictive coding      the mandatory part: no autograd, no backward chain
  fluid/         Navier-Stokes transport
  regularizers/  HJB and transport penalties
  diagnostics/   the measurements that decide whether any of it works
```

**`pc/`** — Each layer holds a state, predicts the layer above, and computes a local error.
Inference relaxes the states; learning is Hebbian (`ΔW ∝ error × presynaptic activity`). The linear
connections never call autograd — I derived the update rules by hand — so "no backpropagation" is a
property of the code and not a claim in a README. `tests/test_locality.py` patches `torch.autograd`
to raise an exception and then trains the network to convergence anyway.

I wrote it twice, deliberately, and the two agree:

* `pc/core.py` — one self-contained class. **Start here.** The whole algorithm is ~150 lines with
  the mathematics in the docstring, and it imports nothing else from this repo.
* `pc/network.py` + `pc/connections.py` — the same algorithm with every edge behind a `Connection`
  interface, which is what lets the fluid layer drop in without the PC rules changing at all.
  `train.py` uses this one.

The inference mode is the switch that carries my main finding:

* `--prediction-mode strict` — predictions recomputed from the relaxed states each step.
* `--prediction-mode fixed` — predictions frozen at their feedforward values (the Fixed Prediction
  Assumption, Millidge et al. 2020). **This is the one that recovers the backprop gradient.**

**`fluid/`** — The paper's transport layer. Activations become a conserved mass (`ρ ≥ 0, Σρ = 1`) on
a grid, a learned stream function generates a velocity field `u = ∇⊥ψ` that is divergence-free *by
construction*, and the mass is advected by conservative upwind fluxes with CFL targeting. Mass
conservation and zero divergence hold to machine precision, not approximately — that is what
`tests/test_fluid_invariants.py` checks.

**`regularizers/hjb.py`** — The stationary HJB residual `ν∆W − ½‖∇W‖² − V`. The paper's toy problem
has a hand-drawn target region to supply the running cost `V`; a classifier has no such thing, so I
take `V` from the layer's own prediction error, which keeps the regulariser local.

**`diagnostics/`** — `bp_alignment.py` measures the cosine between the PC update and the true
backprop gradient. The fluid layer logs mass drift, `‖div u‖`, the Courant number, and the fraction
of the velocity field that survives projection.

## Removing any of it

I built the components so they come out cleanly. The defaults are the minimum:

| to drop | do this |
|---|---|
| Navier–Stokes transport | omit `--fluid` (off by default) |
| HJB regularisation | omit `--hjb` (off by default) |
| transport regularisers | `--transport-alpha 0 --transport-beta 0` |
| Leray projection | omit `--projection` (off by default, and a no-op in stream mode anyway) |
| everything except PC | `python train.py --dataset mnist` |

Deleting `influid_pc/fluid/` and `influid_pc/regularizers/` outright still leaves a working
predictive-coding implementation, because nothing in `pc/` imports them.

## References

Goertzel, B. *Incompressible-Fluid Networks* (v1, October 2025).
Millidge, B., Tschantz, A., Buckley, C. L. *Predictive Coding Approximates Backprop along Arbitrary
Computation Graphs* (2020).
Whittington, J. C. R., Bogacz, R. *An Approximation of the Error Backpropagation Algorithm in a
Predictive Coding Network with Local Hebbian Synaptic Plasticity* (2017).
