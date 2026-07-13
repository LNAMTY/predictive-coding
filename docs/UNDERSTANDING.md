# Understanding the pieces

Written to be read top to bottom. Each section says what the idea *is*, why it is shaped that way,
and which file it lives in — so you can decide what you actually want to keep.

---

## 1. Predictive coding

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
refactored so a fluid layer can be swapped in.

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
its update has cosine **0.756** with the true gradient, and it loses real accuracy for it. See
[FINDINGS.md](FINDINGS.md) §1–2. This is the single most useful thing in this repo.

---

## 2. Navier–Stokes / In-Fluid-Net

### The idea in one paragraph

Treat the activations in a layer not as "numbers" but as a **fluid** — a quantity of "stuff"
(`ρ`, a density) sitting on a 2-D grid. The total amount of stuff is fixed at 1 (a *conserved
budget*). Then, instead of transforming activations with a matrix multiply, you **move the stuff
around** by blowing a wind over the grid. The network learns the wind.

Why bother? Because a matrix multiply can create and destroy activation freely, whereas moving a
conserved budget forces the network to make *choices* — to spend attention here rather than there.
That is the paper's "budget faithfulness".

### The three ingredients

**1. Conservation (the continuity equation).**

```
∂ρ/∂t + ∇·(ρu) = κ∇²ρ
```

The first two terms say: the density at a point changes only because stuff *flowed in or out*. It
cannot appear from nowhere. Discretely, we compute the flux across each cell face and subtract what
leaves from what arrives. Sum over all cells and the interior fluxes cancel in pairs — so total mass
is conserved **exactly**, not approximately. (`fluid/advection.py`.)

**2. Incompressibility (`∇·u = 0`).**

If the wind had *sources* — points where it blows outward in all directions — stuff would pile up
and the "conserved budget" idea would break down locally. Requiring `∇·u = 0` means the wind
neither creates nor destroys; it only rearranges.

**How you get it matters enormously.** Two routes:

* *Project*: build any wind field `g` you like, then subtract off its divergent part by solving a
  Poisson equation (Helmholtz–Hodge / Leray projection). `fluid/projection.py`.
* *Construct*: build the wind as the **curl of a stream function**, `u = ∇⊥ψ = (∂ψ/∂y, −∂ψ/∂x)`.
  The divergence of a curl is *identically zero* — it is an algebraic identity, so it holds to
  machine precision for free. `fluid/operators.py: curl_from_stream`.

The paper insists on the second, and it is right, and the reason is worth internalising: **a Leray
projector is precisely the operator that annihilates gradient fields.** So if you build your wind as
`−∇W` (the natural thing to do — it is the gradient of a value function!) and then project it, you
get **nothing back**. We measured it: **100% of the field is destroyed** (FINDINGS §6). The naive
route does not merely work badly, it works *not at all*.

**3. Stability (the CFL condition).**

If the wind blows stuff more than one grid cell per timestep, the discrete update overshoots and the
simulation explodes (or produces negative mass, which is meaningless). The Courant number
`CFL = Δt · max|u|` must stay below 1. The paper targets `0.3–0.45`: too small and nothing moves,
too large and it breaks. We simply rescale `u` every step to hit `0.4` exactly. A per-sample rescale
is a scalar multiply, so it cannot introduce divergence — the incompressibility survives it.

### Obstacles

To carve a wall into the grid, pin `ψ` to a constant on every node the wall touches. Then every face
of the wall has *equal `ψ` at both endpoints*, so the velocity across it is exactly zero — a
no-through boundary. And because the field is still a curl, it is still divergence-free. You get
walls for free, with no special-casing. This is elegant and it is the kind of thing that makes the
staggered-grid formulation worth the bookkeeping.

**Where it lives:** `influid_pc/fluid/`. Remove the directory and nothing else breaks.

---

## 3. Hamilton–Jacobi–Bellman regularisation

### Where it comes from

This is the "why" behind the wind. The paper's chain of reasoning:

1. Pick a **value function** `W(x)` — how *desirable* is it for activation to be at location `x`?
2. The optimal thing to do is move **downhill on value**: `u = −∇W`. (This is just gradient
   descent, but on a landscape of "goodness" rather than a loss.)
3. If you set this up as an optimal-control problem on the space of *volume-preserving*
   rearrangements, the value function must satisfy a **Hamilton–Jacobi–Bellman** equation, and —
   this is the paper's cute result — differentiating it gives you back the **Navier–Stokes
   equations**, with pressure appearing as the Lagrange multiplier that enforces incompressibility.

So "Navier–Stokes" is not an analogy here. It is what optimal control *becomes* when you constrain
it to conserve volume.

### What we actually regularise

The stationary HJB equation is:

```
ν·∇²W  −  ½‖∇W‖²  =  V
```

where `V` is the *running cost* — how expensive it is to be at each location. We penalise the
squared residual of that equation, which pushes the learned value field to be a *consistent* value
field rather than an arbitrary scalar map.

The honest difficulty: **what is `V` in a classifier?** The paper's toy problem has a hand-drawn
target region, so `V` is obvious. A classifier has no such thing. Our choice: `V` = the layer's own
squared prediction error. Cells whose mass incurs prediction error are expensive. This keeps the
regulariser **local** (it reads only this layer's error and this layer's value net), which is what
lets it coexist with predictive coding instead of sneaking a global gradient back in.

**Does it help?** No. It costs 4.2 points of accuracy on EMNIST (FINDINGS §6). It is off by default.
I would keep the code (it is the paper's central theoretical claim, and it is only ~60 lines) and
leave the flag off.

**Where it lives:** `influid_pc/regularizers/hjb.py`.

---

## 4. How they fit together

The paper's system is a **reaction–diffusion–advection** loop:

| phase | what it does | who does it here |
|---|---|---|
| **reaction** | fast local error correction | predictive coding's inference relaxation |
| **advection** | long-range, goal-directed routing | the fluid transport layer |
| **diffusion** | gentle smoothing for stability | the `κ` term (which we find you should set to 0) |

Predictive coding is the *learning rule*; the fluid layer is a *module* that learns under that rule.
They are orthogonal, which is why you can keep one and throw away the other.

---

## 5. What to keep

If you want the defensible minimum, keep **`pc/`** and the alignment diagnostic. That is a complete,
verified, backprop-free learner with a real finding attached (`strict` vs `fixed`), and you can
explain every line of it.

If you want the full story, keep the fluid layer too — but present it honestly: the mechanism is
*correct and beautiful* (exact conservation, exact incompressibility, obstacle routing) and it
**wins decisively on the routing task it was designed for**, while **losing to a plain layer on
classification**. That contrast is a more interesting thing to present than a fake win would be.
