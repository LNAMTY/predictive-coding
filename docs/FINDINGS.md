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

## 6. Does the fluid layer actually help? Not at classification — but decisively at routing

This is the question the paper leaves open ("the research hasn't been implemented fully to see
the full effect"), so it deserves a straight answer in both directions.

### At classification, incompressible transport does not earn its keep

EMNIST-Letters, 26 classes, 20k train, **matched at 4 epochs** so the comparison is fair:

| model | params | test acc | s/epoch |
|---|---|---|---|
| PC, 1×128 hidden | 103,834 | 63.54 | 4.5 |
| **PC + fluid transport** | 192,128 | **68.98** | 180.4 |
| PC + fluid + HJB | 192,129 | 64.82 | 217.0 |
| **same net, fluid layer deleted** (784→196→128→26) | **182,430** | **70.14** | **8.5** |

Adding the transport layer looks like a **+5.4-point win** over the small baseline — until you ask
what the extra parameters alone would have bought. The last row is the exact same architecture with
the fluid layer *removed and nothing put in its place*. It has **fewer parameters** (182k vs 192k),
runs **21× faster**, and scores **1.2 points higher**.

The same control on **Fashion-MNIST** (10 classes), to rule out an EMNIST artefact:

| model | params | test acc | s/epoch |
|---|---|---|---|
| PC + fluid transport | 190,064 | 84.38 | 164.6 |
| same net, fluid layer deleted | 180,366 | 84.26 | 8.1 |

A **+0.12** difference — noise — for **20× the compute**.

**The incompressible transport layer is strictly dominated on classification.** Its apparent gain
is entirely explained by the width it adds, and the HJB regulariser costs a further 4.2 points on
top. We went looking for this to come out the other way, on two datasets; it does not.

### At routing, it wins outright — and nothing else even functions

But classification is not the task In-Fluid-Net was designed for. The paper's own toy problem is
*routing a conserved budget around obstacles into a target region*, which is where directed,
mass-conserving transport should matter. We reproduced it (24×24 grid, seed at top-left, target
band bottom-right, a pillar and a bar in the way) and — unlike the paper — ran baselines:

| mechanism | band mass ↑ | E[distance] ↓ | ‖div u‖ | CFL |
|---|---|---|---|---|
| isotropic diffusion | 0.0005 | 12.67 | 0 | 0 |
| raw gradient + Leray projection | **0.0000** | 22.00 | ~0 | 0 |
| **stream function + greedy controller** | **0.5400** | **2.64** | 9e-07 | 0.40 |

**54% of the entire budget is delivered into the band.** Diffusion delivers 0.05% — it spreads
into the nearest corner and never arrives, exactly the failure the paper's introduction describes.

![routing](../figures/fig5_routing_task.png)

And the raw-gradient route delivers **nothing at all** — because the projector destroys **100%** of
the field. Note this was run with the *ideal* value function (the true graph distance to the band),
not a learned one. So this is not a training failure; it is structural. A Leray projector is the
operator that annihilates gradient fields, and `−∇W` is a gradient field. **This single number is
the strongest confirmation in the repo of the paper's most important practical warning**, and it is
why the stream-function parameterisation is not a stylistic preference but the thing that makes the
method work at all.

### Where that leaves In-Fluid-Net

The mechanism does what it claims — conserves mass exactly, routes around hard obstacles, stays
stable — and it beats the alternatives *on transport problems*. What we could not find is evidence
that a classifier is a transport problem. Bolting the layer onto MNIST/EMNIST buys nothing a plain
layer of equal size does not buy more cheaply.

**The productive next step is therefore not "scale In-Fluid-Net on ImageNet". It is to find a task
whose structure actually is routing under a conserved budget** — sparse-reward RL credit
assignment, attention-budget allocation over a graph, or the ECAN-style economics the paper's
introduction cites — and test it there. On that class of problem the routing figure above suggests
it will win, and win big.

---

## Summary for the research programme

| the paper says | we find |
|---|---|
| PC gradients match the global gradient at convergence | ⚠️ Only under the Fixed Prediction Assumption. Strict PC plateaus at cosine 0.985 and **decays to 0.76 by 8 layers**, costing 3.5 points of accuracy. |
| Parameterise inside the divergence-free subspace; projection is a safety net | ✅ **Confirmed, emphatically.** The projector destroys **100%** of an ideal raw-gradient drift. A stream function retains 100% and makes the projector a no-op (and 3.3× faster). |
| Anneal diffusion κ: 0.3 → 0 to "explore then sharpen" | ❌ **Does not transfer to classification.** There the density *is* the signal, and κ=0.3 drives a linear probe to chance in one step. Use κ=0. |
| Target CFL ∈ [0.3, 0.45] | ✅ **Confirmed** — and free to enforce exactly, since a per-sample rescale cannot introduce divergence. |
| Local learning eliminates the backprop chain | ✅ **True, and verified in code** — we sabotage `torch.autograd` and train anyway. |
| Incompressible transport beats undirected diffusion for routing | ✅ **Confirmed and quantified**: 54% of the budget delivered vs 0.05% for diffusion. |
| (implied) this should make a better network | ❌ **Not at classification.** At matched capacity a plain MLP beats the fluid layer at 1/20th the cost. HJB regularisation makes it worse still. |

**The two things worth acting on:**

1. **Freeze the predictions during relaxation.** It is a one-line change, and it is the difference
   between a credit-assignment rule that holds at 8 layers (cosine 1.000) and one that quietly
   degrades (0.756). Anyone building on §4.2 of the paper as written is training on a biased
   gradient without knowing it.

2. **Stop evaluating In-Fluid-Net on classification.** The transport machinery is correct and it
   wins decisively on routing, but a classifier is not a routing problem, and the accuracy gains it
   appears to give are just the parameters it adds. Test it where the budget metaphor is literally
   true — sparse-reward credit assignment, attention allocation over a graph — and the routing
   result suggests it will be worth the compute.
