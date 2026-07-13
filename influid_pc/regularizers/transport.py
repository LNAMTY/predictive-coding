"""Transport regularisation from eq. (3.3) / (6.4):

    L_trans = alpha * E[ 0.5 ||u||^2 ]  +  beta * E[ ||div u||^2 ]

The alpha term is the control cost, penalising large velocities. The beta term softly
reinforces incompressibility.

When the velocity comes from a stream function, div u is zero by construction, so the
beta term is identically zero and contributes no gradient. This is expected, and it is
the practical argument for parameterising inside the solenoidal subspace rather than
penalising a field toward it. The term is still computed and logged so that its being
inert is visible.
"""

from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

from ..fluid.operators import divergence


def transport_loss(
    ux: Tensor,
    uy: Tensor,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    kinetic = 0.5 * (ux.pow(2).mean() + uy.pow(2).mean())
    div = divergence(ux, uy, dx, dy)
    div_sq = div.pow(2).mean()

    loss = alpha * kinetic + beta * div_sq
    return loss, {
        "kinetic": float(kinetic),
        "div_norm": float(div.norm()),
        "div_max": float(div.abs().max()),
    }
