"""Staggered (MAC) grid operators.

Cell-centred scalars carry mass / pressure:      a, p    [B, H, W]
Face-normal velocities live on cell faces:       ux      [B, H, W+1]   (vertical faces)
                                                 uy      [B, H+1, W]   (horizontal faces)
Node-centred stream function lives on corners:   psi     [B, H+1, W+1]

The staggering is not decoration. It is what makes the discrete identity
div(curl(psi)) == 0 hold to machine precision rather than to truncation error,
which in turn is what lets us claim exact incompressibility instead of
approximate incompressibility.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def divergence(ux: Tensor, uy: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """Cell-centred divergence: [B, H, W]."""
    dux = (ux[:, :, 1:] - ux[:, :, :-1]) / dx
    duy = (uy[:, 1:, :] - uy[:, :-1, :]) / dy
    return dux + duy


def gradient(p: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Face-centred gradient of a cell scalar, with zero-flux (Neumann) borders.

    Adjoint of `divergence` up to sign, which is exactly what the Poisson solve
    needs for the projection to be an orthogonal projection.
    """
    b, h, w = p.shape
    gx = torch.zeros(b, h, w + 1, device=p.device, dtype=p.dtype)
    gy = torch.zeros(b, h + 1, w, device=p.device, dtype=p.dtype)
    gx[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / dx
    gy[:, 1:-1, :] = (p[:, 1:, :] - p[:, :-1, :]) / dy
    return gx, gy


def laplacian_cell(p: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """div(grad(p)) with Neumann borders. Used for the pressure Poisson equation
    and for the diffusion term; both need the same zero-flux boundary."""
    gx, gy = gradient(p, dx, dy)
    return divergence(gx, gy, dx, dy)


def curl_from_stream(psi: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tuple[Tensor, Tensor]:
    """u = nabla^perp psi = (d psi/dy, -d psi/dx), evaluated on faces.

    psi: [B, H+1, W+1] on nodes  ->  ux: [B, H, W+1], uy: [B, H+1, W]

    Divergence of the result is identically zero: the four psi terms in the
    divergence stencil cancel algebraically, not numerically.
    """
    ux = (psi[:, 1:, :] - psi[:, :-1, :]) / dy
    uy = -(psi[:, :, 1:] - psi[:, :, :-1]) / dx
    return ux, uy


def cfl_number(ux: Tensor, uy: Tensor, dt: float, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """Courant number, per batch element."""
    b = ux.shape[0]
    mx = ux.reshape(b, -1).abs().max(dim=1).values / dx
    my = uy.reshape(b, -1).abs().max(dim=1).values / dy
    return dt * torch.maximum(mx, my)


def rescale_to_cfl(
    ux: Tensor,
    uy: Tensor,
    dt: float,
    target: float = 0.4,
    dx: float = 1.0,
    dy: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rescale the velocity so every sample hits a target Courant number.

    Section 8 of the paper reports this as one of four decisive ingredients:
    CFL << 0.05 means nothing moves, CFL near 1 means the integrator breaks.
    Rescaling is a per-sample scalar, so it cannot introduce divergence.
    """
    cfl = cfl_number(ux, uy, dt, dx, dy)
    scale = target / (cfl + eps)
    s = scale.view(-1, 1, 1)
    return ux * s, uy * s, cfl


def face_masks_from_cells(free_cell: Tensor) -> Tuple[Tensor, Tensor]:
    """Faces are open only when both adjacent cells are open. free_cell: [H, W] bool."""
    h, w = free_cell.shape
    fx = torch.zeros(h, w + 1, dtype=torch.bool, device=free_cell.device)
    fy = torch.zeros(h + 1, w, dtype=torch.bool, device=free_cell.device)
    fx[:, 1:-1] = free_cell[:, 1:] & free_cell[:, :-1]
    fy[1:-1, :] = free_cell[1:, :] & free_cell[:-1, :]
    return fx, fy


def node_mask_from_cells(free_cell: Tensor) -> Tensor:
    """Nodes touching any blocked cell (or the domain border) are pinned.

    Pinning psi to a constant on these nodes forces every face on an obstacle
    boundary to have equal psi at both endpoints, hence zero normal velocity --
    a no-through wall that costs us nothing in incompressibility, because the
    field is still an exact curl.
    """
    h, w = free_cell.shape
    blocked = ~free_cell
    node_blocked = torch.zeros(h + 1, w + 1, dtype=torch.bool, device=free_cell.device)
    node_blocked[:-1, :-1] |= blocked
    node_blocked[:-1, 1:] |= blocked
    node_blocked[1:, :-1] |= blocked
    node_blocked[1:, 1:] |= blocked

    border = torch.zeros_like(node_blocked)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    return ~(node_blocked | border)
