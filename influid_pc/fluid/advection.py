"""Conservative transport of an activation "mass" field.

    d(rho)/dt + div(rho * u) = kappa * lap(rho),     div(u) = 0

Discretised with donor-cell (first-order upwind) fluxes. Two properties matter
and both are structural rather than approximate:

  * Mass conservation. The update is a difference of face fluxes, so summing over
    cells telescopes and only boundary fluxes survive. Those are zero (the stream
    function is pinned on the border), so total mass is conserved to machine
    precision. This is Lemma 7.1.

  * Positivity. Upwind flux takes mass from the donor cell only, so under
    CFL <= 1 no cell can be driven negative. "Mass" stays interpretable as mass.

Diffusion uses the same Neumann Laplacian, which is also exactly conservative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .operators import laplacian_cell


def upwind_fluxes(rho: Tensor, ux: Tensor, uy: Tensor) -> Tuple[Tensor, Tensor]:
    """Donor-cell fluxes on faces. Flux is carried by the upstream cell's density."""
    b, h, w = rho.shape

    # Vertical faces: face j separates cell j-1 (left) and cell j (right).
    left = torch.zeros(b, h, w + 1, device=rho.device, dtype=rho.dtype)
    right = torch.zeros(b, h, w + 1, device=rho.device, dtype=rho.dtype)
    left[:, :, 1:] = rho
    right[:, :, :-1] = rho
    flux_x = torch.where(ux > 0, ux * left, ux * right)

    # Horizontal faces: face i separates cell i-1 (up) and cell i (down).
    up = torch.zeros(b, h + 1, w, device=rho.device, dtype=rho.dtype)
    down = torch.zeros(b, h + 1, w, device=rho.device, dtype=rho.dtype)
    up[:, 1:, :] = rho
    down[:, :-1, :] = rho
    flux_y = torch.where(uy > 0, uy * up, uy * down)

    return flux_x, flux_y


def advect(
    rho: Tensor,
    ux: Tensor,
    uy: Tensor,
    dt: float,
    kappa: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
    free_cell: Optional[Tensor] = None,
) -> Tensor:
    """One conservative transport step. Returns the new density."""
    flux_x, flux_y = upwind_fluxes(rho, ux, uy)

    div_flux = (flux_x[:, :, 1:] - flux_x[:, :, :-1]) / dx + (
        flux_y[:, 1:, :] - flux_y[:, :-1, :]
    ) / dy

    rho_new = rho - dt * div_flux

    if kappa > 0:
        rho_new = rho_new + dt * kappa * laplacian_cell(rho, dx, dy)

    if free_cell is not None:
        rho_new = rho_new * free_cell

    # Upwind + CFL should already guarantee this; clamp defends against a caller
    # that ignored the CFL bound rather than silently producing negative mass.
    return rho_new.clamp_min(0.0)


def total_mass(rho: Tensor) -> Tensor:
    return rho.reshape(rho.shape[0], -1).sum(dim=1)


def renormalise(rho: Tensor, eps: float = 1e-12) -> Tensor:
    """Restore a unit budget after a non-conservative step (e.g. a PC reaction).

    Lemma 7.3: the PC reaction phase is locally non-conservative, so we renormalise
    before the next transport step, which then conserves exactly.
    """
    m = total_mass(rho).view(-1, 1, 1)
    return rho / (m + eps)


def kappa_schedule(step: int, total_steps: int, kappa0: float, warm_frac: float = 0.5) -> float:
    """Anneal diffusion kappa0 -> 0 over the first `warm_frac` of the rollout.

    Section 8.1's "explore, then sharpen": early diffusion lets mass lift off the
    seed and feel alternative routes; late diffusion would only blur the answer.
    """
    if kappa0 <= 0:
        return 0.0
    cutoff = max(1, int(warm_frac * total_steps))
    if step >= cutoff:
        return 0.0
    return kappa0 * (1.0 - step / cutoff)
