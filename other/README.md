# other/ — the In-Fluid-Net work

A from-scratch implementation of *Incompressible-Fluid Networks* (Goertzel, Oct 2025): Navier–Stokes
transport as a network layer, Helmholtz–Hodge / Leray projection, Hamilton–Jacobi–Bellman
regularisation, and the paper's §8 routing task with the baselines it does not report.

**Why it is here and not in the main tree.** It was not part of what I presented, which was
predictive coding and the strict-vs-fixed comparison against backprop. Keeping it separate means the
result I am claiming is not tangled with the result I am only reporting. It is not unfinished and it
is not broken: it still runs, and it is still tested.

## What is in it

```
fluid/         the transport layer: staggered-grid operators, upwind advection,
               Leray projection, the stream/value velocity nets, and the
               PC Connection that wraps it all
regularizers/  the HJB residual and the transport (control-cost) penalty
scripts/       routing_task.py, make_fluid_figures.py, run_fluid_experiments.py
tests/         the invariant suite: incompressibility, mass conservation,
               positivity, no-through obstacles, CFL, the projector's backward
docs/          FINDINGS.md and UNDERSTANDING.md for this half
figures/       fig3 (transport), fig4 (ablations), fig5 (routing)
results/       the fluid experiment grid
```

## Running it

From the repo root:

```bash
make fluid              # PC + Navier-Stokes, EMNIST-Letters, 26 classes      (~6 min)
make hjb                # the same, + HJB regularisation                      (~7 min)
make routing            # the routing task, with baselines                    (~1 min)
make other-figures      # rebuild other/figures/ from other/results/
make other-experiments  # the full fluid grid (slow)
```

The invariant tests run as part of `make test` at the root.

## What it depends on

`fluid/connection.py` imports `ModuleConnection` from `influid_pc.pc.connections` — the transport
layer plugs in as a predictive-coding connection, so it learns under the same local rule as
everything else, from the prediction error at its own output and nothing more.

The dependency runs one way only. `influid_pc/build.py` imports from here lazily, inside the
`--fluid` branch, so the predictive-coding path never touches this directory. Delete `other/` and
the root repo still works.

## The findings

In [`docs/FINDINGS.md`](docs/FINDINGS.md). The short version:

* The transport machinery is **exact**, not approximate: mass drift `<1e-12` in float64, `div u`
  identically zero by algebraic identity, zero flux into obstacles.
* The paper's warning about projecting a raw gradient field is **the load-bearing one**: the
  projector destroys **100%** of an ideal `−∇W` drift. Parameterise inside the solenoidal subspace.
* The paper's κ = 0.3 diffusion warm-up **does not transfer** to classification, and the residual
  wrapper — not the κ value — is what makes the layer safe to insert at all.
* On **routing** the layer wins outright: 54% of the budget delivered, vs 0.35% for diffusion and 0%
  for raw-gradient + projection.
* On **classification** it is strictly dominated: at matched capacity a plain layer beats it, 21×
  faster.

## A correction worth reading

The routing table originally reported **0.0005** band mass for the diffusion baseline. That was
wrong, and it was wrong in the direction that flattered my own conclusion.

`laplacian_cell` enforces zero flux at the domain border but knows nothing about obstacles, so the
diffusion term pushed mass across obstacle walls into blocked cells, which the advection step then
deleted — destroying ~30% of the budget over a rollout, in a task whose whole premise is a *conserved*
budget. The stream-function and raw-gradient routes were unaffected (they run at κ = 0), so the one
mechanism silently losing mass was the baseline I was arguing against.

Fixed in `fluid/advection.py::diffusion`, which masks the diffusive flux to the open faces using
`face_masks_from_cells` — a helper that already existed in `operators.py` and had never been wired
up. Diffusion now conserves mass exactly and delivers **0.0035**, seven times more than reported.
The conclusion is unchanged: the stream function still wins by ~150×.

The tests missed it because they covered κ > 0 *without* obstacles
(`test_advection_conserves_mass_exactly`) and obstacles *without* κ
(`test_obstacles_are_no_through_and_still_divergence_free`), and never both.
`test_diffusion_conserves_mass_with_obstacles` now covers that cell.
