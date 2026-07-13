"""Hamilton-Jacobi-Bellman residual regularisation.

The paper's chain is: a value function W solves an HJB equation on the group of
volume-preserving diffeomorphisms; its negative gradient is the optimal control;
projecting that control to physical space yields incompressible Navier-Stokes
(eqs 2.1-2.5). Section 2.3 says the practical version of this is "learn a
value-like function W_theta encoding task objectives".

We take the stationary form of (2.1) -- drop the d/dt term, since our routing
layer runs to a fixed horizon rather than tracking a time-varying value:

    nu * lap(W) - 0.5 * ||grad W||^2 = V

and penalise the squared residual. V is the running cost. In a classification
network there is no hand-drawn "target band" to fly toward, so we define the cost
from the only task signal that is legitimately available to this layer: its own
local prediction error. Cells whose mass incurs large prediction error are
expensive; the value field is pushed to route mass away from them.

This keeps the regulariser strictly local -- it reads the layer's error and its
own value net, nothing else -- which is what lets it coexist with predictive
coding instead of smuggling a global gradient back in.
"""

from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor

from ..fluid.operators import gradient, laplacian_cell


def hjb_residual(
    value: Tensor,
    cost: Tensor,
    nu: float = 0.01,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Stationary HJB residual  nu*lap(W) - 0.5*||grad W||^2 - V,  as a scalar loss.

    value: [B, H, W]   the learned value field
    cost:  [B, H, W]   the running cost V
    """
    lap = laplacian_cell(value, dx, dy)

    gx, gy = gradient(value, dx, dy)
    # Face gradients -> cell-centred squared magnitude (average the two faces
    # bounding each cell, which is the standard collocated reconstruction).
    gx_c = 0.5 * (gx[:, :, 1:] + gx[:, :, :-1])
    gy_c = 0.5 * (gy[:, 1:, :] + gy[:, :-1, :])
    grad_sq = gx_c.pow(2) + gy_c.pow(2)

    residual = nu * lap - 0.5 * grad_sq - cost
    loss = residual.pow(2).mean()

    return loss, {
        "hjb_residual": float(residual.abs().mean()),
        "hjb_lap": float(lap.abs().mean()),
        "hjb_gradsq": float(grad_sq.mean()),
    }


def error_cost(error_grid: Tensor, normalise: bool = True) -> Tensor:
    """Running cost from the layer's own prediction error: high error = high cost."""
    cost = error_grid.pow(2)
    if normalise:
        b = cost.shape[0]
        scale = cost.reshape(b, -1).amax(dim=1).clamp_min(1e-8).view(-1, 1, 1)
        cost = cost / scale
    return cost
