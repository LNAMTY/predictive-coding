# Understanding the pieces

Written to be read top to bottom. Says what the idea *is*, why it is shaped that way, and which file
it lives in.

This covers predictive coding, which is the presented result. The In-Fluid-Net transport layer and
the HJB regulariser are explained in `other/docs/UNDERSTANDING.md`.

---

## Predictive coding

### The problem it solves

Backpropagation computes `dL/dW` for a weight buried deep in a network by applying the chain rule
backwards through every layer above it. That means the update for layer 3 cannot be computed until
the error has been passed down from layer 10. It is a *global* algorithm: one long dependency chain,
and every link needs to know the derivative of the link above it.

Predictive coding asks: **what if every layer only ever talked to its immediate neighbours?**

### The setup

Give every layer a **state** `a_l` — a vector of numbers that the layer is currently "committed to".
This is the key departure. In a normal network, layer 3's activation is *forced* to be whatever
`W₂·f(a₂)` says it is. In predictive coding, `a₃` is a **free variable** that can disagree with
what layer 2 predicts.

Each connection makes a **prediction** of the layer above it:

```
â_{l+1} = W_l · f(a_l) + b_l
```

and the disagreement between prediction and actual state is the **prediction error**:

```
e_{l+1} = a_{l+1} − â_{l+1}
```

The whole network's objective is just: **be less surprised**. Minimise the total squared error,
called the free energy:

```
F = Σ_l ½‖e_l‖²
```

### Two timescales

The trick is that `F` is minimised over *two* different sets of variables, on two different
timescales.

**Fast — inference.** Freeze the weights; let the states move. Ask: how should `a_l` change to
reduce `F`? Layer `l` appears in `F` in exactly two places — as the thing being predicted from
below (`e_l`) and as the thing predicting above (`e_{l+1}`). Differentiate:

```
∂F/∂a_l  =  e_l  −  f'(a_l) ⊙ (W_lᵀ · e_{l+1})
             ↑            ↑
   "I disagree      "the layer above me
    with what        disagrees, and I am
    my input          partly to blame"
    predicted"
```

so we relax the state with `a_l ← a_l − η · ∂F/∂a_l`, repeatedly, until the two pressures balance.

Notice what that expression contains: `e_l` (this layer's error), `e_{l+1}` (the layer above's
error), and `W_l` (the weights between them). **Nothing else.** No term from layer 10 appears in
layer 3's update. This is the whole point.

**Slow — learning.** Now freeze the states and let the weights move:

```
∂F/∂W_l = −e_{l+1} · f(a_l)ᵀ        →        ΔW_l ∝ e_{l+1} · f(a_l)ᵀ
```

Read that in words: **change the weight in proportion to (the error above) × (the activity below).**
That is a Hebbian rule — a synapse changes based only on the two neurons it connects. A biological
synapse could actually do this. It cannot do backpropagation.

**Where it lives:** `influid_pc/pc/core.py` (read this one first — it is the whole algorithm in one
class, and the maths above is the docstring). `influid_pc/pc/connections.py` is the same thing
refactored so an arbitrary module can be swapped in as a connection.

### Where the labels come in

During training we **clamp** the top layer to the target: `a_L = one_hot(y)`. Now the top layer has
a prediction error — the difference between what the network guessed and the truth — and that error
propagates *downwards* through the relaxation, layer by layer, as the states settle.

That settling **is** the credit assignment. Backprop does it in one exact backward sweep;
predictive coding does it by letting a dynamical system come to equilibrium.

`--output-nudge γ` controls how hard we clamp: `a_L ← a_L + γ·(target − a_L)`. Small `γ` means a
gentle nudge.

### The thing everybody gets wrong (and the repo's main finding)

Here is the subtlety. During relaxation, the states `a_l` move. So when you recompute the prediction
`â_{l+1} = W_l·f(a_l)`, you are computing it from a state that has **already been perturbed by the
error you are trying to propagate**. The signal contaminates its own carrier.

Two ways to handle it:

* **strict** — recompute predictions from the current, relaxed states, every step. This is
  predictive coding as the paper (and most descriptions) literally specify it.
* **fixed** — compute the predictions *once*, from the initial feedforward pass, and hold them
  frozen for the whole relaxation. (The "Fixed Prediction Assumption".)

They sound almost identical. They are not. **Only `fixed` recovers the backprop gradient.** `strict`
converges to a systematically different answer, and the error grows with depth — at 8 hidden layers
its update has cosine **0.752** with the true gradient, and it loses real accuracy for it. See
[FINDINGS.md](FINDINGS.md) §1–2. This is the single most useful thing in this repo.

---

## What this repo keeps, and why

**`pc/` and the alignment diagnostic are the core.** Together they are a complete, verified,
backprop-free learner, and they carry the one finding that changes what someone building on the
paper should do: `strict` versus `fixed` prediction mode.

Everything else — the In-Fluid-Net transport layer, the HJB regulariser, the routing task — is
optional, defaults to off, and lives under `other/`. Predictive coding is the *learning rule*; the
fluid layer is a *module* that learns under that rule. The two are orthogonal, which is why either
can be kept without the other, and why deleting `other/` breaks nothing here.
