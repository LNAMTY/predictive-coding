"""Measures how closely a predictive-coding update tracks the backprop gradient.

The paper claims an envelope theorem: predictive-coding gradients match the global
objective gradient at convergence. This module tests that claim. Holding one set of
weights, on a single batch, it computes:

  * the gradient backprop would produce, dL/dW_l, by an explicit hand-written
    backward sweep over the same architecture;
  * the update predictive coding produces from purely local quantities, after n
    steps of inference relaxation;

and reports the cosine similarity and relative magnitude per layer, as a function of
inference steps and of the output nudge gamma.

Theory (Millidge et al. 2020) holds that the PC fixed point satisfies the same error
recursion as backprop, e_l = -delta_l, so the cosine should approach 1 as gamma -> 0
and steps -> inf. How that degrades with depth and with a finite nudge is what these
functions measure.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor

from ..pc.core import PredictiveCodingNet


def backprop_gradients(net: PredictiveCodingNet, x: Tensor, target: Tensor) -> List[Tensor]:
    """Explicit backward sweep for L = 0.5 * ||a_L - target||^2, mean over batch.

    Hand-written rather than autograd so that it uses exactly the same forward
    equations as `PredictiveCodingNet`, making the comparison one of two learning
    rules on one model rather than of two different models.
    """
    act = net.act
    batch = x.shape[0]

    states = net.forward(x)
    delta = states[-1] - target

    grads: List[Tensor] = [torch.zeros(0)] * net.n_layers
    for l in reversed(range(net.n_layers)):
        grads[l] = delta.T @ act.f(states[l]) / batch
        if l > 0:
            delta = (delta @ net.W[l]) * act.df(states[l])
    return grads


def backprop_gradients_net(net, x: Tensor, target: Tensor) -> List[Tensor]:
    """Same backward sweep, for a `PCNetwork` built from LinearConnections."""
    batch = x.shape[0]

    states = net.forward(x)
    delta = states[-1] - target

    grads: List[Tensor] = [torch.zeros(0)] * net.n_layers
    for l in reversed(range(net.n_layers)):
        c = net.conns[l]
        grads[l] = delta.T @ c.act.f(states[l]) / batch
        if l > 0:
            delta = (delta @ c.W) * c.act.df(states[l])
    return grads


def _cosine(a: Tensor, b: Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def alignment_report(
    net: PredictiveCodingNet,
    x: Tensor,
    target: Tensor,
    step_grid: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> Dict[str, object]:
    """Cosine(PC update, BP gradient) per layer, swept over inference steps."""
    bp = backprop_gradients(net, x, target)

    rows = []
    for steps in step_grid:
        out = net.infer(x, target=target, steps=steps)
        pc = net.pc_weight_gradients(out["states"], out["errors"])  # type: ignore[arg-type]

        per_layer_cos = [_cosine(pc[l], bp[l]) for l in range(net.n_layers)]
        per_layer_ratio = [
            float(pc[l].norm() / bp[l].norm()) if bp[l].norm() > 0 else float("nan")
            for l in range(net.n_layers)
        ]
        flat_pc = torch.cat([g.flatten() for g in pc])
        flat_bp = torch.cat([g.flatten() for g in bp])

        rows.append(
            {
                "steps": steps,
                "global_cosine": _cosine(flat_pc, flat_bp),
                "layer_cosine": per_layer_cos,
                "layer_ratio": per_layer_ratio,
                "free_energy": out["energy"],
            }
        )
    return {"nudge": net.cfg.output_nudge, "n_layers": net.n_layers, "rows": rows}


def nudge_sweep(
    net: PredictiveCodingNet,
    x: Tensor,
    target: Tensor,
    nudges: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02),
    steps: int = 64,
) -> List[Dict[str, object]]:
    """Alignment as the output nudge gamma shrinks toward the theoretical limit."""
    bp = backprop_gradients(net, x, target)
    flat_bp = torch.cat([g.flatten() for g in bp])

    original = net.cfg.output_nudge
    results: List[Dict[str, object]] = []
    try:
        for gamma in nudges:
            net.cfg.output_nudge = gamma
            out = net.infer(x, target=target, steps=steps)
            pc = net.pc_weight_gradients(out["states"], out["errors"])  # type: ignore[arg-type]
            flat_pc = torch.cat([g.flatten() for g in pc])
            results.append(
                {
                    "nudge": gamma,
                    "global_cosine": _cosine(flat_pc, flat_bp),
                    # PC's update scales with gamma; normalise it out to compare shape.
                    "scale_ratio": float(flat_pc.norm() / (gamma * flat_bp.norm() + 1e-12)),
                    "layer_cosine": [_cosine(pc[l], bp[l]) for l in range(net.n_layers)],
                }
            )
    finally:
        net.cfg.output_nudge = original
    return results
