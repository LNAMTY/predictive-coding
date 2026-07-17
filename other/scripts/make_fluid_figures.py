"""Figures for the transport layer. Reads other/results/*.json, writes other/figures/*.png.

Split out of scripts/make_figures.py, which now covers only the predictive-coding
figures. Same house style, so the two sets still sit together in a report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

BLUE, RED, VIOLET, AQUA = "#2a78d6", "#e34948", "#4a3aa7", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "grid.color": "#e8e7e4",
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def _load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _finish(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


# -- figure 3: transport invariants of the fluid layer ------------------------

def fig_fluid() -> None:
    import torch.nn.functional as F

    from influid_pc import data
    from influid_pc.build import _obstacle_mask
    from other.fluid.advection import advect, renormalise
    from other.fluid.layer import IncompressibleRouting

    torch.manual_seed(3)
    bundle = data.load("mnist", train_subset=64, batch_size=8)
    x, y = next(iter(bundle.train))

    G = 14
    proj = torch.nn.Linear(784, G * G)
    h = proj(x)

    def rollout(layer, h):
        b = h.shape[0]
        rho = F.softmax(layer.softmax_beta * h, dim=1).reshape(b, G, G)
        rho = renormalise(rho * layer.free_cell)
        ux, uy, stats = layer.velocity(rho)
        frames = [rho.clone()]
        for _ in range(layer.steps):
            rho = advect(rho, ux, uy, layer.dt, kappa=0.0, free_cell=layer.free_cell)
            frames.append(rho.clone())
        return frames, ux, uy, stats

    free_layer = IncompressibleRouting(grid=G, steps=12, kappa0=0.0)
    obs_mask = _obstacle_mask(G)
    obs_layer = IncompressibleRouting(grid=G, steps=12, kappa0=0.0, obstacles=obs_mask)
    obs_layer.stream.load_state_dict(free_layer.stream.state_dict())

    fig, axes = plt.subplots(2, 5, figsize=(14, 5.8))
    i = 0

    for row, (layer, title) in enumerate(
        [(free_layer, "open grid"), (obs_layer, "with obstacles")]
    ):
        with torch.no_grad():
            frames, ux, uy, stats = rollout(layer, h)

        # cell-centred velocity for streamplot
        uxc = 0.5 * (ux[i, :, 1:] + ux[i, :, :-1])
        uyc = 0.5 * (uy[i, 1:, :] + uy[i, :-1, :])
        speed = (uxc**2 + uyc**2).sqrt()
        gy, gx = np.mgrid[0:G, 0:G]

        blocked = (~layer.free_cell.bool()).numpy()

        for k, t in enumerate([0, 4, 8, 12]):
            ax = axes[row][k]
            m = frames[t][i].numpy().copy()
            m_show = np.ma.masked_where(blocked, m)
            ax.imshow(m_show, cmap="magma", interpolation="nearest")
            if blocked.any():
                ax.imshow(np.ma.masked_where(~blocked, blocked.astype(float)),
                          cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"t={t}" + ("  (input)" if t == 0 else ""), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(title, fontsize=10, fontweight="bold")

        ax = axes[row][4]
        ax.streamplot(gx, gy, uxc.numpy(), uyc.numpy(), color=speed.numpy(),
                      cmap="viridis", density=1.1, linewidth=1.0, arrowsize=0.7)
        if blocked.any():
            ax.imshow(np.ma.masked_where(~blocked, blocked.astype(float)),
                      cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlim(0, G - 1)
        ax.set_ylim(G - 1, 0)
        ax.set_title(
            f"velocity  |  div={stats['div_final']:.1e}\n"
            f"CFL={stats['cfl']:.2f}  kept={stats['retained_energy']:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Incompressible transport of a conserved activation budget\n"
        "(mass drift < 1e-6, div u ≈ 1e-6)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_fluid_transport.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig3_fluid_transport.png")


# -- figure 4: In-Fluid-Net ablations -----------------------------------------

def fig_ablation() -> None:
    # Colour follows the result, not the expectation: MUTED where the knob turns out to
    # make no difference, BLUE/RED only for the one contrast that actually decides the
    # layer's fate.
    rows = [
        ("kappa = 0 (default)", "exp4-kappa0.0", MUTED),
        ("kappa = 0.1", "exp4-kappa0.1", MUTED),
        ("kappa = 0.3 (paper)", "exp4-kappa0.3", MUTED),
        ("CFL 0.05", "exp4-cfl0.05", MUTED),
        ("CFL 0.4 (default)", "exp4-base", MUTED),
        ("CFL 0.9", "exp4-cfl0.9", MUTED),
        ("HJB regulariser", "exp4-hjb", MUTED),
        ("obstacles", "exp4-obstacles", MUTED),
        ("residual readout", "exp4-base", BLUE),
        ("replace readout", "exp4-no-residual", RED),
    ]
    have = [(n, _load(t), c) for n, t, c in rows]
    have = [(n, r["final_acc"], c) for n, r, c in have if r]
    if not have:
        print("  skip ablation (no results)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(have) + 1.6))
    names = [n for n, _, _ in have]
    vals = [v for _, v, _ in have]
    cols = [c for _, _, c in have]
    ypos = np.arange(len(have))[::-1]

    ax.barh(ypos, vals, color=cols, height=0.62, zorder=3)
    for yp, v in zip(ypos, vals):
        ax.text(v + 0.7, yp, f"{v:.1f}%", va="center", fontsize=9,
                color=INK2, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("test accuracy (%)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title(
        "Inside the transport layer, only the residual wrapper decides anything\n"
        "(MNIST, 3 epochs; diffusion, CFL and obstacles are within noise, HJB costs ~1 point)",
        loc="left", fontsize=11,
    )
    _finish(ax, grid_axis="x")

    fig.tight_layout()
    fig.savefig(FIGS / "fig4_fluid_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig4_fluid_ablation.png")


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig_fluid()
    fig_ablation()


if __name__ == "__main__":
    main()
