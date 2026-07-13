# Findings

Four claims, each with the experiment that establishes it. Two of them contradict the paper.

Every number below is reproducible from this repo — the command is given with each result.

---

## 1. Predictive coding recovers the backprop gradient — but only under the Fixed Prediction Assumption

The paper's abstract promises "an envelope theorem proving that predictive coding gradients
match the global objective gradient at convergence" (also §4.3: `ΔW ∝ −∂Φ/∂W`, computed from
local quantities only). This is a testable claim, so we tested it instead of believing it.

For a fixed set of weights and one batch, we compute two things and take the cosine between them:

* the gradient **backprop** would produce, `dL/dW`, by an explicit backward sweep over the same
  architecture (itself verified against autograd — `test_backprop_reference_matches_autograd`);
* the update **predictive coding** produces from purely local signals, after inference converges.

Theory says the PC fixed point satisfies the same error recursion as backprop (`e_l = −δ_l`),
so as the output nudge `γ → 0` the cosine should go to 1.

**It does not — unless you freeze the top-down predictions.**

| output nudge γ | strict PC | fixed-prediction PC |
|---|---|---|
| 1.0   | 0.970 | 0.999 |
| 0.1   | 0.985 | 0.99975 |
| 0.01  | 0.985 | 0.999996 |
| 0.001 | **0.985 (plateau)** | **1.000000** |

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
| 2 | 0.985 | 1.000 |
| 3 | 0.975 | 1.000 |
| 4 | 0.943 | 1.000 |
| 6 | 0.815 | 1.000 |
| 8 | **0.756** | **1.000** |

And it is not a cosmetic misalignment — **test accuracy tracks it** (MNIST, 20k train, 8 epochs):

| hidden layers | backprop | PC (fixed) | PC (strict) | cos(strict) |
|---|---|---|---|---|
| 1 | 96.0 | 94.2 | 94.9 | 0.998 |
| 2 | 95.5 | 95.3 | 95.3 | 0.979 |
| 3 | 96.4 | 94.7 | 94.7 | 0.928 |
| 4 | 96.3 | 95.2 | 93.4 | 0.857 |
| 6 | 96.6 | **94.3** | **90.8** | **0.676** |

Strict PC loses **3.5 points** to fixed-prediction PC at 6 hidden layers, and the gap is still
widening. Fixed-prediction PC stays within ~2 points of backprop at every depth.

**Takeaway for the research programme.** "Predictive coding ≈ backprop" is not free — it is
purchased with the Fixed Prediction Assumption. If you implement PC literally as the paper's
§4.2 three-phase loop describes (recomputing predictions from relaxed states), the gradient you
get is a systematically biased one, and the bias grows with exactly the depth you were hoping to
scale to. **Anyone building on this paper should freeze predictions during relaxation.** It is a
one-line change and it is the difference between cosine 0.76 and cosine 1.00 at 8 layers.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiments.py exp1
```

---

## 3. The fluid machinery works exactly as specified — and the paper's own warning is the load-bearing one

The paper repeatedly insists (§1.3, §2.1.3, §3.1.3, §8.7) that you must **parameterise inside the
divergence-free subspace** rather than build a raw gradient field and project it, because the Leray
projector is precisely the operator that annihilates gradient fields. We implemented both routes
to measure the difference rather than take it on faith.

Building `u = ∇⊥ψ` from a learned stream function on a staggered grid gives, at machine precision:

| invariant | measured |
|---|---|
| mass drift over 200 advection steps | `< 1e-12` (float64) / `1.5e-7` (float32 training) |
| `‖div u‖` | `~1e-6` **with the projector switched off entirely** |
| energy retained through projection | **1.00** |
| Courant number | pinned to 0.40 by construction |
| flux into obstacle cells | `0` exactly |

Zero divergence here is an *algebraic identity* of the discrete curl on a staggered grid (the four
`ψ` terms in the divergence stencil cancel), not a numerical approximation. Obstacles are handled by
pinning `ψ` to a constant on the nodes they touch, which makes every obstacle face have equal `ψ` at
both endpoints — a no-through wall that costs nothing in incompressibility.

And the warning is real. Feeding a **pure gradient field** through the projector:

> `retained_energy < 0.05` — over 95% of the field is destroyed.
> (`test_projection_annihilates_a_pure_gradient_field`)

So the projector is a *safety net*, and in stream-function mode it is a **provable no-op** — which
is also a 3.3× speedup, since we can stop paying for a Poisson solve on every step and simply audit
the divergence instead.

![fluid](../figures/fig3_fluid_transport.png)

---

## 4. The paper's diffusion warm-up (κ: 0.3 → 0) is actively harmful for classification

§8.1 and §3.1.4 recommend annealing diffusion `κ₀ = 0.3 → 0` over the first 30–50% of steps —
"explore first (let mass lift off and feel the winds), sharpen later."

**In a classifier this destroys the representation.** Probing how much class information survives
the transport layer (linear probe on the layer's output):

| transport steps | κ = 0.0 | κ = 0.3 |
|---|---|---|
| 0 | 92.2% | 93.0% |
| 1 | 93.8% | **13.3%** |
| 4 | 88.7% | **12.1%** |
| 8 | 87.9% | **22.3%** |

Chance is 10%. Diffusion drives the probe to chance after a *single* step.

**Why the paper is not wrong, and why it still doesn't transfer.** In the paper's toy task, `ρ` is a
*routing budget* seeded as a single node — nearly a delta — and the target is a hand-drawn band.
Diffusion there is genuinely useful: it lets mass lift off the seed and feel alternative corridors.
In a classifier, `ρ` **is the representation**. Every bit of class information lives in the
deviations of `ρ` from uniform, and diffusion is a low-pass filter on exactly those deviations. The
`κ` warm-up is a good idea whose precondition — "the density is a budget, not a signal" — silently
fails on the transfer to classification.

This is the single most important thing to know before putting In-Fluid-Net on a real dataset, and
it is not something the toy experiment could have revealed.

**Two consequences for the design:**

1. `κ` defaults to **0** here, not 0.3.
2. The transport layer is **residual**: it emits `x + gain · Δlog ρ`, with `gain` initialised to 0,
   so switching the fluid on starts as the exact identity and the network opens the valve only
   insofar as routing earns its keep. Without this, inserting the transport layer drops MNIST to
   **~9%** (chance) regardless of κ, because a softmax-normalised density simply is not a
   drop-in replacement for a hidden activation vector.

---

## 5. What predictive coding alone achieves

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

### 5b. The PC–backprop gap widens with the number of classes

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

| the paper says | we find |
|---|---|
| PC gradients match the global gradient at convergence | Only under the Fixed Prediction Assumption. Strict PC plateaus at cosine 0.985 and **decays to 0.76 by 8 layers**. |
| Parameterise inside the divergence-free subspace; projection is a safety net | **Confirmed, emphatically.** Projection kills >95% of a raw gradient field; a stream function retains 100% and makes the projector a no-op. |
| Anneal diffusion κ: 0.3 → 0 to "explore then sharpen" | **Does not transfer.** In a classifier the density *is* the signal, and κ=0.3 drives a linear probe to chance in one step. Use κ=0. |
| Target CFL ∈ [0.3, 0.45] | **Confirmed** — and cheap to enforce exactly, since a per-sample rescale cannot introduce divergence. |
| Local learning eliminates the backprop chain | **True, and verified in code** — the linear connections never call autograd. |

The most actionable of these is the first: it is a one-line change to the inference loop, and it is
the difference between a credit-assignment rule that scales with depth and one that quietly does not.
