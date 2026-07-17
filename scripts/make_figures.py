"""Predictive-coding figures. Reads results/*.json, writes figures/*.png.

Every series is direct-labelled as well as coloured, so it stays readable in greyscale
and to colour-blind readers.

The transport-layer figures live in other/scripts/make_fluid_figures.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig_alignment()
    fig_depth_accuracy()


if __name__ == "__main__":
    main()
