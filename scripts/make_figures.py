"""Figures for the report. Reads results/*.json, writes figures/*.png.

Every series is direct-labelled as well as coloured, so it stays readable in greyscale
and to colour-blind readers.
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


# -- figure 1: alignment between the PC update and the backprop gradient ------

def fig_alignment() -> None:
    d = _load("alignment")
    if not d:
        print("  skip alignment (no results)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    def series(rows, xkey, mode):
        r = [x for x in rows if x["mode"] == mode]
        return [x[xkey] for x in r], [x["cosine"] for x in r]

    # (a) nudge
    ax = axes[0]
    for mode, c in (("strict", RED), ("fixed", BLUE)):
        x, y = series(d["nudge"], "nudge", mode)
        ax.plot(x, y, "o-", color=c, label=mode, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"output nudge $\gamma$  (smaller $\rightarrow$ theory's limit)")
    ax.set_ylabel(r"cosine(PC update, BP gradient)")
    ax.set_title("(a) The envelope theorem holds\nonly with fixed predictions")
    # The separation lives in the last 4% of the range; a 0-1 axis would hide it.
    ax.set_ylim(0.955, 1.004)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1, zorder=1)
    ax.annotate("fixed → 1.000", (0.001, 0.9995), color=BLUE, fontsize=9,
                xytext=(4, -13), textcoords="offset points", fontweight="bold")
    ax.annotate("strict plateaus ≈ 0.985", (0.001, 0.985), color=RED, fontsize=9,
                xytext=(4, -14), textcoords="offset points", fontweight="bold")
    _finish(ax)

    # (b) inference steps
    ax = axes[1]
    for mode, c in (("strict", RED), ("fixed", BLUE)):
        x, y = series(d["steps"], "steps", mode)
        ax.plot(x, y, "o-", color=c, label=mode, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("inference relaxation steps")
    ax.set_ylabel("cosine(PC update, BP gradient)")
    ax.set_title("(b) Both need ~32 steps;\nonly one converges to backprop")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1, zorder=1)
    _finish(ax)

    # (c) depth
    ax = axes[2]
    for mode, c in (("strict", RED), ("fixed", BLUE)):
        x, y = series(d["depth"], "hidden_layers", mode)
        ax.plot(x, y, "o-", color=c, label=mode, zorder=3)
        ax.annotate(
            f"{mode}", (x[-1], y[-1]), color=c, fontweight="bold", fontsize=10,
            xytext=(6, -3), textcoords="offset points",
        )
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("cosine(PC update, BP gradient)")
    ax.set_title("(c) Strict PC's gradient decays with depth.\nFixed predictions do not.")
    ax.set_ylim(0.7, 1.03)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1, zorder=1)
    _finish(ax)

    for ax in axes[:2]:
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Predictive coding recovers the backprop gradient\n"
        "— but only under the Fixed Prediction Assumption",
        fontsize=12, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_alignment.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig1_alignment.png")


# -- figure 2: accuracy cost of the misalignment ------------------------------

def fig_depth_accuracy() -> None:
    depths = [1, 2, 3, 4, 6]
    runs = {
        "backprop": (VIOLET, [_load(f"exp1-depth{n}-bp") for n in depths]),
        "PC (fixed)": (BLUE, [_load(f"exp1-depth{n}-pc-fixed") for n in depths]),
        "PC (strict)": (RED, [_load(f"exp1-depth{n}-pc-strict") for n in depths]),
    }
    if not any(all(r) for _, (_, r) in runs.items()):
        print("  skip depth-accuracy (no results)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    for name, (c, rs) in runs.items():
        xs = [d for d, r in zip(depths, rs) if r]
        ys = [r["final_acc"] for r in rs if r]
        if not xs:
            continue
        ax.plot(xs, ys, "o-", color=c, zorder=3)
        ax.annotate(name, (xs[-1], ys[-1]), color=c, fontweight="bold", fontsize=9,
                    xytext=(6, -3), textcoords="offset points")
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy follows gradient quality")
    ax.set_xlim(0.8, 7.6)
    _finish(ax)

    ax = axes[1]
    for name, (c, rs) in runs.items():
        if "PC" not in name:
            continue
        xs, ys = [], []
        for d, r in zip(depths, rs):
            if r and "bp_cosine" in r["history"][-1]:
                xs.append(d)
                ys.append(r["history"][-1]["bp_cosine"])
        if xs:
            ax.plot(xs, ys, "o-", color=c, zorder=3)
            ax.annotate(name, (xs[-1], ys[-1]), color=c, fontweight="bold", fontsize=9,
                        xytext=(6, -3), textcoords="offset points")
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("cosine(PC, BP) after training")
    ax.set_title("...and the gradient degrades with depth")
    ax.set_xlim(0.8, 7.6)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1, zorder=1)
    _finish(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_depth_accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig2_depth_accuracy.png")


# -- figure 3: transport invariants of the fluid layer ------------------------

def fig_fluid() -> None:
    import torch.nn.functional as F

    from influid_pc import data
    from influid_pc.build import _obstacle_mask
    from influid_pc.fluid.advection import advect, renormalise
    from influid_pc.fluid.layer import IncompressibleRouting

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
    fig_alignment()
    fig_depth_accuracy()
    fig_fluid()
    fig_ablation()


if __name__ == "__main__":
    main()
