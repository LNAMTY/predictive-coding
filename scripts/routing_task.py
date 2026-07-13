"""The paper's toy task (section 8), reproduced and given the baselines it lacks.

Setup as specified: a 24x24 grid, a single seed node near the top-left, a target band
in the bottom-right, and two obstacles (a vertical pillar and a horizontal bar). A
conserved activation budget must be routed around the obstacles and into the band.

The paper reports its four diagnostic scalars for its own method only. Without a
baseline, a rise in band mass is not evidence that incompressible transport achieved
anything a simpler mechanism could not, so three mechanisms are run here on identical
geometry and identical budgets:

  diffusion    isotropic spreading, the baseline the paper's introduction argues against
  raw-gradient a learned potential, projected  (u = P[-grad W])
  stream       a learned stream function       (u = curl(psi)), the paper's recommendation

reporting band mass, expected distance, divergence, and CFL for each.

Routing, rather than classification, is the regime In-Fluid-Net was designed for.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from influid_pc.fluid.advection import advect, renormalise, total_mass
from influid_pc.fluid.operators import (
    cfl_number,
    curl_from_stream,
    divergence,
    gradient,
    node_mask_from_cells,
    rescale_to_cfl,
)
from influid_pc.fluid.projection import leray_project

ROOT = Path(__file__).resolve().parent.parent
G, T, DT = 24, 240, 0.5


def geometry():
    free = torch.ones(G, G, dtype=torch.bool)
    free[6:18, 11:13] = False          # vertical pillar
    free[15:17, 4:20] = False          # horizontal bar

    band = torch.zeros(G, G, dtype=torch.bool)
    band[13:22, 13:22] = True
    band &= free

    seed = torch.zeros(G, G)
    seed[2, 2] = 1.0
    return free, band, seed


def band_distance(free, band):
    """BFS graph distance to the band, respecting obstacles."""
    d = torch.full((G, G), float("inf"))
    q = deque()
    for i in range(G):
        for j in range(G):
            if band[i, j]:
                d[i, j] = 0.0
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < G and 0 <= b < G and free[a, b] and d[a, b] == float("inf"):
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    d[~free] = 0.0
    finite = d[torch.isfinite(d)]
    d[torch.isinf(d)] = float(finite.max())
    return d


def rollout(ux, uy, seed, free, kappa=0.0):
    rho = renormalise((seed * free).unsqueeze(0))
    frames = [rho[0].clone()]
    for _ in range(T):
        rho = advect(rho, ux, uy, DT, kappa=kappa, free_cell=free)
        frames.append(rho[0].clone())
    return rho, frames


def metrics(rho, ux, uy, band, dist):
    return {
        "band_mass": float(rho[0][band].sum()),
        "expected_distance": float((rho[0] * dist).sum()),
        "div_norm": float(divergence(ux, uy).norm()),
        "cfl": float(cfl_number(ux, uy, DT).mean()),
        "mass": float(total_mass(rho)),
    }


def stream_basis(free, sigma=2.0, stride=2):
    """A basis of localised stream-function bumps. Each mode is a curl, hence
    divergence-free by construction, as is any linear combination of them."""
    node_mask = node_mask_from_cells(free).float()
    ii, jj = torch.meshgrid(torch.arange(G + 1.0), torch.arange(G + 1.0), indexing="ij")

    modes = []
    for ci in range(0, G + 1, stride):
        for cj in range(0, G + 1, stride):
            bump = torch.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * sigma**2))
            psi = (bump * node_mask).unsqueeze(0)
            ux, uy = curl_from_stream(psi)
            if float(ux.abs().max() + uy.abs().max()) > 1e-8:
                modes.append((ux, uy))
    return modes


def greedy_drift(modes, rho, dist, lam=1e-3):
    """The paper's greedy, distance-shaped controller (section 8, 'Controller').

        alpha_k  ∝  sum_{e=(i->j)} U_{k,e} * a_i * (d(i) - d(j))  /  (||U_k||^2 + lambda)

    Modes that carry mass down the graph distance to the band receive positive weight.
    This is a one-step surrogate for a receding-horizon controller, and it shapes the
    wind before any mass has arrived, which is where backpropagating through the rollout
    has no signal to work with.
    """
    a = rho[0]
    # oriented edge drops: x-faces are (left -> right), y-faces are (up -> down)
    dx_drop = torch.zeros(G, G + 1)
    dy_drop = torch.zeros(G + 1, G)
    dx_drop[:, 1:-1] = dist[:, :-1] - dist[:, 1:]
    dy_drop[1:-1, :] = dist[:-1, :] - dist[1:, :]

    ax = torch.zeros(G, G + 1)
    ay = torch.zeros(G + 1, G)
    ax[:, 1:-1] = a[:, :-1]          # source cell of a rightward flux
    ay[1:-1, :] = a[:-1, :]          # source cell of a downward flux

    ux = torch.zeros(1, G, G + 1)
    uy = torch.zeros(1, G + 1, G)
    for mx, my in modes:
        num = float((mx[0] * ax * dx_drop).sum() + (my[0] * ay * dy_drop).sum())
        den = float(mx.pow(2).sum() + my.pow(2).sum()) + lam
        alpha = num / den
        ux = ux + alpha * mx
        uy = uy + alpha * my
    return ux, uy


def gradient_drift(dist, free):
    """A raw potential drift, then projection.

    The potential used is the ideal one, the true distance-to-band field, which is the
    best value function any learner could hope to find. This is therefore a generous
    version of "learn W, take -grad W, project": if it still collapses, the collapse is
    structural rather than a training failure.
    """
    w = (-dist).unsqueeze(0)
    gx, gy = gradient(w)
    ux, uy = -gx, -gy
    ux, uy = ux.clone(), uy.clone()
    ux[:, :, 0] = ux[:, :, -1] = 0
    uy[:, 0, :] = uy[:, -1, :] = 0

    raw = float((ux.pow(2).sum() + uy.pow(2).sum()).sqrt())
    ux, uy, st = leray_project(ux, uy, iters=800)
    st["retained_energy"] = float((ux.pow(2).sum() + uy.pow(2).sum()).sqrt()) / (raw + 1e-12)
    return ux, uy, st


def main() -> None:
    torch.manual_seed(0)
    free, band, seed = geometry()
    dist = band_distance(free, band)
    modes = stream_basis(free)
    print(f"solenoidal basis: {len(modes)} modes\n")

    results = {}
    fields = {}

    # 1. isotropic diffusion: no drift at all, just spreading.
    zx = torch.zeros(1, G, G + 1)
    zy = torch.zeros(1, G + 1, G)
    rho, frames = rollout(zx, zy, seed, free, kappa=0.30)
    results["diffusion"] = metrics(rho, zx, zy, band, dist)
    fields["diffusion"] = (zx, zy, frames)

    # 2. raw gradient drift + projection, using the ideal value function as a static field.
    gx, gy, gstat = gradient_drift(dist, free)
    retained = gstat["retained_energy"]
    print(f"raw gradient field: {retained:.4f} of its energy survives projection")
    if retained > 1e-3:
        gx, gy, _ = rescale_to_cfl(gx, gy, DT, 0.4)
    else:
        # The projector annihilated the field. Rescaling a numerically-zero vector to a
        # target Courant number would amplify rounding noise into a fake wind, so the
        # field is left as it is.
        print("  -> field collapsed; no rescale (rescaling zero would amplify noise)")
    rho, frames = rollout(gx, gy, seed, free, kappa=0.0)
    results["raw-gradient + projection"] = metrics(rho, gx, gy, band, dist)
    fields["raw-gradient + projection"] = (gx, gy, frames)
    results["_raw_gradient_energy_retained_by_projection"] = retained

    # 3. stream function + the paper's greedy distance-shaped controller, recomputed
    #    each step so the wind adapts as the budget moves.
    rho = renormalise((seed * free).unsqueeze(0))
    frames = [rho[0].clone()]
    last = (torch.zeros(1, G, G + 1), torch.zeros(1, G + 1, G))
    for _ in range(T):
        ux, uy = greedy_drift(modes, rho, dist)
        if float(ux.abs().max() + uy.abs().max()) > 1e-9:
            ux, uy, _ = rescale_to_cfl(ux, uy, DT, 0.4)
            last = (ux, uy)
        rho = advect(rho, ux, uy, DT, kappa=0.0, free_cell=free)
        frames.append(rho[0].clone())
    results["stream function (greedy)"] = metrics(rho, *last, band, dist)
    fields["stream function (greedy)"] = (*last, frames)

    print(f"{'mechanism':28s} {'band mass':>10s} {'E[dist]':>9s} {'||div u||':>10s} {'CFL':>6s}")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"{k:28s} {v['band_mass']:10.4f} {v['expected_distance']:9.2f} "
              f"{v['div_norm']:10.2e} {v['cfl']:6.2f}")
    print(f"\nraw gradient field energy surviving projection: "
          f"{results['_raw_gradient_energy_retained_by_projection']:.3f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "routing_task.json").write_text(json.dumps(results, indent=2))

    # figure
    names = [k for k in results if not k.startswith("_")]
    fig, axes = plt.subplots(len(names), 4, figsize=(11, 2.7 * len(names)))
    blocked = (~free).numpy()
    for r, name in enumerate(names):
        ux, uy, frames = fields[name]
        for c, t in enumerate([0, T // 3, 2 * T // 3, T]):
            ax = axes[r][c]
            m = frames[t].numpy()
            ax.imshow(np.ma.masked_where(blocked, m), cmap="magma",
                      interpolation="nearest", vmin=0, vmax=max(m.max() * 0.6, 1e-6))
            ax.imshow(np.ma.masked_where(~blocked, blocked.astype(float)),
                      cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(band.numpy().astype(float), levels=[0.5], colors="#1baf7a", linewidths=1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t}", fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
        axes[r][3].set_title(
            f"band mass {results[name]['band_mass']:.3f}", fontsize=10,
            color="#2a78d6", fontweight="bold",
        )
    fig.suptitle(
        "Routing a conserved budget around obstacles into the target band (green)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig5_routing_task.png", dpi=170, bbox_inches="tight")
    print("wrote figures/fig5_routing_task.png")


if __name__ == "__main__":
    main()
