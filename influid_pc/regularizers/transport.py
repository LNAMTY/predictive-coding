"""Transport regularisation from eq. (3.3) / (6.4):

    L_trans = alpha * E[ 0.5 ||u||^2 ]  +  beta * E[ ||div u||^2 ]

The alpha term is the control cost: it penalises the network for spending large
velocities to achieve its routing. The beta term softly reinforces incompressibility.

Note that when the velocity comes from a stream function, div u is zero by
construction, so the beta term is *identically zero and contributes no gradient*.
That is not a bug -- it is the cleanest possible statement of why parameterising
inside the solenoidal subspace is better than penalising your way toward it. We
still compute and log it, because when beta stops being able to do anything, that
is worth being able to see.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
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
