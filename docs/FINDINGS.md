# Findings

Three claims about predictive coding, each with the experiment that establishes it. One of them
contradicts the paper.

Every number below is reproducible from this repo — the command is given with each result.

The In-Fluid-Net side of the work is not covered here; it lives in `other/docs/FINDINGS.md`.

---

## 1. Predictive coding recovers the backprop gradient — but only under the Fixed Prediction Assumption

The paper's abstract promises "an envelope theorem proving that predictive coding gradients
match the global objective gradient at convergence" (also §4.3: `ΔW ∝ −∂Φ/∂W`, computed from
local quantities only). That is a testable claim, and this section tests it.

For a fixed set of weights and one batch, we compute two things and take the cosine between them:

* the gradient **backprop** would produce, `dL/dW`, by an explicit backward sweep over the same
  architecture (itself verified against autograd — `test_backprop_reference_matches_autograd`);
* the update **predictive coding** produces from purely local signals, after inference converges.

Theory says the PC fixed point satisfies the same error recursion as backprop (`e_l = −δ_l`),
so as the output nudge `γ → 0` the cosine should go to 1.

**It does not — unless you freeze the top-down predictions.**

| output nudge γ | strict PC | fixed-prediction PC |
|---|---|---|
| 1.0   | 0.9722 | 0.9779 |
| 0.1   | 0.9858 | 0.9998 |
| 0.01  | 0.9859 | **1.000000** |
| 0.001 | **0.9859 (plateau)** | **1.000000** |

Inference is genuinely converged in both cases — the fixed-point residual falls to `~1e-6`. Strict
PC simply converges *somewhere else*.

**Why.** At the PC fixed point the hidden-state displacement and the error signal are **both O(γ)**.
So relaxing the states perturbs the top-down prediction at the same order as the signal that
prediction is carrying. Shrinking γ shrinks both together and the ratio never improves — the bias
is structural, not a small-nudge artefact. Freezing the predictions at their feedforward values
(the *Fixed Prediction Assumption*, Millidge et al. 2020) severs exactly that feedback, and the
recursion collapses onto backprop's.

```bash
PYTHONPATH=. .venv/bin/python scripts/alignment_study.py   # -> results/alignment.json
```

---

## 2. The misalignment compounds with depth — and it costs real accuracy

This is the part that matters for the paper's ambition of replacing backprop, because the whole
motivation is *deep* credit assignment.

Cosine against the true gradient, as the network gets deeper:

| hidden layers | strict PC | fixed-prediction PC |
|---|---|---|
| 1 | 0.991 | 1.000 |
| 2 | 0.986 | 1.000 |
| 3 | 0.973 | 1.000 |
| 4 | 0.943 | 1.000 |
| 6 | 0.811 | 1.000 |
| 8 | **0.752** | **1.000** |

And it is not a cosmetic misalignment — **test accuracy tracks it** (MNIST, 20k train, 8 epochs):

| hidden layers | backprop | PC (fixed) | PC (strict) | cos(strict) |
|---|---|---|---|---|
| 1 | 94.34 | 94.18 | 94.86 | 0.991 |
| 2 | 95.60 | 95.26 | 95.32 | 0.986 |
| 3 | 96.40 | 94.68 | 94.70 | 0.973 |
| 4 | 96.28 | 95.22 | 93.42 | 0.943 |
| 6 | 96.62 | **94.28** | **90.78** | **0.811** |

Strict PC loses **3.5 points** to fixed-prediction PC at 6 hidden layers, and the gap is still
widening. Fixed-prediction PC stays within ~2 points of backprop at every depth.

**Takeaway for the research programme.** "Predictive coding ≈ backprop" is not free — it is
purchased with the Fixed Prediction Assumption. If you implement PC literally as the paper's
§4.2 three-phase loop describes (recomputing predictions from relaxed states), the gradient you
get is a systematically biased one, and the bias grows with exactly the depth you were hoping to
scale to. **Anyone building on this paper should freeze predictions during relaxation.** It is a
one-line change and it is the difference between cosine 0.75 and cosine 1.00 at 8 layers.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiments.py exp1
```

---

## 3. What predictive coding alone achieves

The mandatory deliverable, with no fluid and no HJB anywhere in the graph:

| | |
|---|---|
| **MNIST, full 60k, 3 layers, no backprop** | **97.21%** |
| cosine vs. true backprop gradient, during training | 0.996 – 0.999 |
| parameters | 235k |
| wall clock | 140s, CPU only |

Across class counts (the task's "different classification" — MNIST subset to *k* labels):

| classes | backprop | PC (no backprop) |
|---|---|---|
| 2 | 99.91 | **99.91** |
| 3 | 99.30 | **99.46** |
| 5 | 99.10 | 98.92 |
| 10 | 96.28 | 95.36 |

Predictive coding **matches or beats backprop at k = 2 and k = 3** and trails by ~1 point at k = 10,
using only local Hebbian updates — no global backward pass anywhere.

Note the comparison is deliberately **biased against PC**: backprop is given the best of a
learning-rate sweep (`{0.1, 0.05, 0.02, 0.01}`), while PC runs at a single fixed setting. At
depth 1, lr = 0.1 diverges for backprop, and reporting that as "PC beats backprop 94.9 to 9.8"
would have been a cheat.

```bash
.venv/bin/python train.py --dataset mnist --epochs 10 --train-subset 0 \
    --hidden 256 128 --weight-lr 0.1 --track-alignment
```

### 3b. The PC–backprop gap widens with the number of classes

Extending to a different dataset with a different class count — **EMNIST-Letters, 26 classes**
(20k train, 1×128 hidden, 8 epochs):

| classes | dataset | backprop | PC | gap |
|---|---|---|---|---|
| 2 | MNIST | 99.91 | 99.91 | **0.00** |
| 3 | MNIST | 99.30 | 99.46 | **+0.16** |
| 5 | MNIST | 99.10 | 98.92 | −0.18 |
| 10 | MNIST | 96.28 | 95.36 | −0.92 |
| 26 | EMNIST-Letters | 76.50 | 72.14 | **−4.36** |

This is not undertuning. We swept PC's inference steps (24, 48) and learning rate (0.05, 0.1):
the number moves by less than half a point.

**We had a hypothesis, and it was wrong.** The natural explanation is that PC drives learning
through a nudge `γ` at the output layer, and with `k` classes the one-hot target concentrates that
nudge on 1 unit out of `k` while the other `k−1` receive only weak suppression — so the effective
drive should dilute as `k` grows, and raising `γ` with `k` should compensate. It does not:

| nudge γ | lr 0.02 | lr 0.05 |
|---|---|---|
| 0.2 | 66.9 | 71.0 |
| 0.4 | 69.5 | **71.2** |
| 0.6 | 70.7 | 69.6 |
| 1.0 | 55.0 | 4.4 (diverged) |

Nothing beats the 72.1 baseline, and large `γ` destabilises inference outright. So the dilution
story is not the mechanism, and the cause of the widening gap remains **open**. Recording this
matters: it eliminates the most obvious explanation and tells the next person not to spend a day
on it. The remaining suspects are the MSE-on-one-hot energy (which couples all `k` output units
through the same quadratic) and the fact that PC's output layer is relaxed rather than clamped at
test time.

---

## Summary for the research programme

| the paper says | verdict | we find |
|---|---|---|
| PC gradients match the global gradient at convergence | **partly** | Only under the Fixed Prediction Assumption. Strict PC plateaus at cosine 0.986 and **decays to 0.75 by 8 layers**, costing 3.5 points of accuracy. |
| Local learning eliminates the backprop chain | **confirmed** | Verified in code: `torch.autograd` is patched to raise and the network trains anyway. |

**The thing worth acting on: freeze the predictions during relaxation.** It is a one-line change,
and it is the difference between a credit-assignment rule that holds at 8 layers (cosine 1.000) and
one that quietly degrades (0.752). Anyone building on §4.2 of the paper as written is training on a
biased gradient without knowing it.
