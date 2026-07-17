# Understanding the pieces — In-Fluid-Net

The transport layer and the HJB regulariser. Predictive coding, which is the learning rule all of
this runs under, is explained in `docs/UNDERSTANDING.md`.

---

## 1. Navier–Stokes / In-Fluid-Net

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
is conserved **exactly**, not approximately. The diffusion term on the right is masked to the open
faces for the same reason, so that conservation survives on a grid with obstacles in it.
(`fluid/advection.py`.)

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
get **nothing back**. We measured it: **100% of the field is destroyed** ([FINDINGS.md](FINDINGS.md)
§3). The naive route does not merely work badly, it works *not at all*.

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

The one place this does *not* come for free is diffusion, which is not built from `ψ` at all: it
needs its flux masked to the open faces explicitly, or it leaks mass into the walls.
(`fluid/advection.py: diffusion`.)

**Where it lives:** `other/fluid/`. Remove the directory and nothing on the PC path breaks.

---

## 2. Hamilton–Jacobi–Bellman regularisation

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

**Does it help?** No. It costs 4.2 points of accuracy on EMNIST ([FINDINGS.md](FINDINGS.md) §3), so
it is off by default. The code stays in the repo because it is the paper's central theoretical claim
and only takes ~60 lines, but nothing here relies on it.

**Where it lives:** `other/regularizers/hjb.py`.

---

## 3. How they fit together

The paper's system is a **reaction–diffusion–advection** loop:

| phase | what it does | who does it here |
|---|---|---|
| **reaction** | fast local error correction | predictive coding's inference relaxation |
| **advection** | long-range, goal-directed routing | the fluid transport layer |
| **diffusion** | gentle smoothing for stability | the `κ` term, which is set to 0 by default |

Predictive coding is the *learning rule*; the fluid layer is a *module* that learns under that rule.
The two are orthogonal, which is why either can be kept without the other.

The transport layer plugs in as a `Connection` (`fluid/connection.py`), so the inference and
learning rules never change to accommodate it. It learns from the prediction error measured at its
own output and nothing else — it never sees the task loss, the label, or any other layer's error.

---

## 4. Why this is in `other/`

The mechanism is correct — exact mass conservation, exact incompressibility, exact no-through
obstacles — and it wins decisively on the routing task it was designed for. On classification it
loses to a plain layer of the same size. Both halves of that are in [FINDINGS.md](FINDINGS.md),
because the contrast is the more useful result: it says the machinery works and is being evaluated
on the wrong kind of problem.

It sits under `other/` because it was not part of the presented result, not because it is unfinished.
It still runs, and it is still tested: `make fluid`, `make routing`, and the invariant suite in
`other/tests/`.
