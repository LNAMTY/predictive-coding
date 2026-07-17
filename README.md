# In-Fluid-PC

Predictive coding trained without backpropagation, and a measurement of exactly when it does and
does not equal the backprop gradient.

Alongside it, in [`other/`](other/), a from-scratch implementation of *Incompressible-Fluid Networks*
(Goertzel, Oct 2025) — Navier–Stokes transport, Helmholtz–Hodge projection, and Hamilton–Jacobi–Bellman
regularisation.

This is my submission for the training task:

> Train PC with different classification (current MNIST 10 class). + Navier-Stokes (fluid dynamics),
> use a dataset other than MNIST with a different class count.

I built predictive coding first, on its own, and then added each piece of the paper on top of it as
a component I can switch off independently. The point was to be able to defend every line I keep.
Along the way I found two things that contradict the paper, and one of them is a one-line fix.

**The predictive-coding half is the presented result, and it is what the root of this repo is.** The
In-Fluid-Net half is complete and still runs, but it sits under `other/` so that the thing I am
claiming is not mixed up with the thing I am reporting. See [`other/README.md`](other/README.md).

## How I met the task

| what was asked | what I ran | result |
|---|---|---|
| PC on a **different classification** than MNIST-10 | MNIST at k = 2, 3, 5 classes, and PC on **EMNIST-Letters, 26 classes** | 99.91 / 99.46 / 98.92, and 72.14 on EMNIST-26 |
| Navier–Stokes on **another dataset, different class count** | **EMNIST-Letters, 26 classes** — every fluid run (in `other/`) | 68.98 with transport, 64.82 with transport + HJB |

No Navier–Stokes run in this repo uses MNIST, and none uses 10 classes. The 10-class MNIST run is
kept only as the reference point to measure against.

## What I found

| | |
|---|---|
| PC on full MNIST, zero backprop | **97.2%** (3 layers, local Hebbian updates only) |
| PC update vs. the true backprop gradient | cosine **0.99999** — but only under the Fixed Prediction Assumption |
| Strict PC at 8 hidden layers | cosine **0.756** and falling, costing 3.5 points of accuracy |
| The PC–backprop gap vs. class count | widens to −4.36 at 26 classes, and the obvious explanation is *wrong* |

**The finding that contradicts the paper: the envelope theorem only holds if you freeze the top-down
predictions.** Implemented strictly, as §4.2 specifies, PC converges to a fixed point that is *not*
the backprop gradient, and shrinking the output nudge does not close the gap — because at the fixed
point the state displacement and the error signal are both `O(γ)`, so the signal contaminates its own
carrier at the same order. Freezing the predictions severs that feedback, and it is a one-line change.

![alignment](figures/fig1_alignment.png)

The full argument, with the experiment behind every number, is in
**[docs/FINDINGS.md](docs/FINDINGS.md)**. If you want the ideas explained from first principles
instead, read **[docs/UNDERSTANDING.md](docs/UNDERSTANDING.md)**.

The In-Fluid-Net findings — the transport invariants, the κ warm-up, and the routing task where
incompressible transport beats every baseline outright — are in
[`other/docs/FINDINGS.md`](other/docs/FINDINGS.md).

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
make emnist     # PC on EMNIST-Letters, 26 classes                        (~3 min)
```

**The findings:**

```bash
make alignment  # PC vs the true backprop gradient, strict vs fixed       (~3 min)
make deep       # what that misalignment costs in accuracy, at 6 layers   (~5 min)
```

**Reference points:**

```bash
make pc         # plain PC, MNIST 10 classes — the starting point         (~2 min)
make bp         # backprop, given the best of a learning-rate sweep       (~1 min)
```

**Checks:**

```bash
make test       # locality, PC vs backprop, fluid invariants
make figures    # rebuild figures/ from results/
make lint       # ruff
```

**The In-Fluid-Net work** (see [`other/README.md`](other/README.md)):

```bash
make fluid      # + Navier-Stokes, EMNIST-Letters, 26 classes             (~6 min)
make hjb        # the same, + Hamilton-Jacobi-Bellman regularisation      (~7 min)
make routing    # the paper's routing task, with the baselines it lacks   (~1 min)
```

Anything not covered by a `make` target can be run through `train.py` directly, which takes the
components as flags (`--fluid`, `--hjb`, `--projection`, `--obstacles`, `--prediction-mode`,
`--num-classes`, and so on — `python train.py --help` lists them all).

Every run prints what it is training, then a table per epoch:

```
  predictive coding  ·  fixed predictions (FPA)
  ────────────────────────────────────────────────────────────────────────────
  dataset       mnist  (10 classes, 784-dim)
  network       784 -> 256 -> 128 -> 10   (235,146 params)
  learning      lr 0.1 · nudge 0.2 · 24 inference steps · 10 epochs
  extras        none (plain PC)
  ────────────────────────────────────────────────────────────────────────────

    epoch   test acc  free energy   cos(BP)     time
    ─────  ─────────  ───────────  ────────  ───────
        1     94.86%       0.1417    0.9962    13.8s
```

## How the code is laid out

```
influid_pc/
  pc/            predictive coding      the mandatory part: no autograd, no backward chain
  diagnostics/   the measurements that decide whether any of it works

other/           the In-Fluid-Net work — not part of the presented result
  fluid/         Navier-Stokes transport
  regularizers/  HJB and transport penalties
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
  interface, which is what lets an arbitrary module drop in without the PC rules changing at all.
  `train.py` uses this one.

The inference mode is the switch that carries my main finding:

* `--prediction-mode strict` — predictions recomputed from the relaxed states each step.
* `--prediction-mode fixed` — predictions frozen at their feedforward values (the Fixed Prediction
  Assumption, Millidge et al. 2020). **This is the one that recovers the backprop gradient.**

**`diagnostics/`** — `bp_alignment.py` measures the cosine between the PC update and the true
backprop gradient. Its backprop reference is a hand-written backward sweep over the same forward
equations, so the comparison is between two learning rules on one model rather than two models;
`test_backprop_reference_matches_autograd` checks that reference against autograd.

**`other/`** — The paper's transport layer, its regularisers, the routing task, and their tests and
figures. It is imported only when `--fluid` asks for it, so nothing on the PC path depends on it.

## Removing any of it

I built the components so they come out cleanly. The defaults are the minimum:

| to drop | do this |
|---|---|
| Navier–Stokes transport | omit `--fluid` (off by default) |
| HJB regularisation | omit `--hjb` (off by default) |
| transport regularisers | `--transport-alpha 0 --transport-beta 0` |
| Leray projection | omit `--projection` (off by default, and a no-op in stream mode anyway) |
| everything except PC | `python train.py --dataset mnist` |

Deleting `other/` outright still leaves a working predictive-coding implementation, because nothing
in `pc/` imports it.

## References

Goertzel, B. *Incompressible-Fluid Networks* (v1, October 2025).
Millidge, B., Tschantz, A., Buckley, C. L. *Predictive Coding Approximates Backprop along Arbitrary
Computation Graphs* (2020).
Whittington, J. C. R., Bogacz, R. *An Approximation of the Error Backpropagation Algorithm in a
Predictive Coding Network with Local Hebbian Synaptic Plasticity* (2017).
