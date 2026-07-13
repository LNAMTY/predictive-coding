"""Measures cosine(PC update, BP gradient). Writes results/alignment.json.

Three sweeps:

  nudge   as the output nudge gamma -> 0, for both prediction modes
  steps   as inference relaxation runs longer
  depth   how the above degrades as the network gets deeper

The paper asserts an envelope theorem: predictive-coding gradients match the global
objective gradient at convergence. These sweeps establish when that holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from influid_pc import data
from influid_pc.diagnostics.bp_alignment import _cosine, backprop_gradients_net
from influid_pc.pc.connections import LinearConnection
from influid_pc.pc.network import PCNetwork, PCTrainConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def make_net(dims, mode, nudge, steps, seed=0):
    torch.manual_seed(seed)
    conns = [
        LinearConnection(dims[i], dims[i + 1], activation="tanh")
        for i in range(len(dims) - 1)
    ]
    cfg = PCTrainConfig(
        inference_steps=steps, inference_lr=0.1, output_nudge=nudge, prediction_mode=mode
    )
    return PCNetwork(conns, cfg)


def cosine_for(net, x, t):
    bp = backprop_gradients_net(net, x, t)
    out = net.infer(x, target=t)
    states, errors = out["states"], out["errors"]
    pc = [c.weight_gradient(states[l], errors[l + 1]) for l, c in enumerate(net.conns)]
    per_layer = [_cosine(pc[l], bp[l]) for l in range(net.n_layers)]
    glob = _cosine(
        torch.cat([g.flatten() for g in pc]), torch.cat([g.flatten() for g in bp])
    )
    return glob, per_layer


def main() -> None:
    bundle = data.load("mnist", train_subset=512, batch_size=256)
    x, y = next(iter(bundle.train))
    t = data.one_hot(y, 10)

    dims = [784, 128, 128, 10]
    out: dict = {"dims": dims, "nudge": [], "steps": [], "depth": []}

    print("== nudge sweep (inference_steps=256)")
    for mode in ("strict", "fixed"):
        for gamma in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001):
            net = make_net(dims, mode, gamma, 256)
            g, per = cosine_for(net, x, t)
            out["nudge"].append(
                {"mode": mode, "nudge": gamma, "cosine": g, "layer_cosine": per}
            )
            print(f"  {mode:6s} gamma={gamma:<6} cos={g:.6f}")

    print("== inference-step sweep (gamma=0.02)")
    for mode in ("strict", "fixed"):
        for steps in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            net = make_net(dims, mode, 0.02, steps)
            g, per = cosine_for(net, x, t)
            out["steps"].append(
                {"mode": mode, "steps": steps, "cosine": g, "layer_cosine": per}
            )
            print(f"  {mode:6s} steps={steps:<5} cos={g:.6f}")

    print("== depth sweep (gamma=0.02, steps=256)")
    for n_hidden in (1, 2, 3, 4, 6, 8):
        d = [784, *([128] * n_hidden), 10]
        for mode in ("strict", "fixed"):
            net = make_net(d, mode, 0.02, 256)
            g, per = cosine_for(net, x, t)
            out["depth"].append(
                {"mode": mode, "hidden_layers": n_hidden, "cosine": g, "layer_cosine": per}
            )
            per_str = [f"{c:.3f}" for c in per]
            print(f"  {mode:6s} hidden={n_hidden} cos={g:.6f}  per-layer={per_str}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "alignment.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'alignment.json'}")


if __name__ == "__main__":
    main()
