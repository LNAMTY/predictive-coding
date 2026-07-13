"""Helmholtz-Hodge / Leray projection onto the divergence-free subspace.

Given a raw field g, solve the pressure Poisson equation

    lap(p) = div(g),      u = g - grad(p),      so that div(u) = 0.

The paper is emphatic (sections 1.3, 2.1.3, 3.1.3, 8.7) that this should be a
*safety net*, not the primary construction: if you build g as a raw gradient
field and then project, the projector can annihilate almost all of it, because a
gradient field is exactly the thing a Leray projector kills.

`retained_energy` below is the number that makes that concrete: it is the
fraction of the field that survives projection. We report it so the failure mode
is visible rather than mysterious.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .operators import divergence, gradient, laplacian_cell


def cg_poisson(
    rhs: Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    iters: int = 200,
    tol: float = 1e-8,
) -> Tuple[Tensor, int, float]:
    """Conjugate gradient for lap(p) = rhs with Neumann boundaries.

    With pure Neumann conditions the Laplacian is singular (constants are in the
    null space), so we project the mean out of both the right-hand side and the
    iterate. That is Lemma 7.2 in the paper: pressure is unique only up to an
    additive constant, and the resulting velocity does not care.
    """
    b = rhs.shape[0]

    def demean(v: Tensor) -> Tensor:
        return v - v.reshape(b, -1).mean(dim=1).view(-1, 1, 1)

    r = demean(rhs.clone())
    p = torch.zeros_like(rhs)
    d = r.clone()
    rs = (r.reshape(b, -1) * r.reshape(b, -1)).sum(dim=1)

    it = 0
    for it in range(1, iters + 1):
        ad = demean(laplacian_cell(d, dx, dy))
        dad = (d.reshape(b, -1) * ad.reshape(b, -1)).sum(dim=1)
        alpha = rs / (dad + 1e-30)
        p = p + alpha.view(-1, 1, 1) * d
        r = r - alpha.view(-1, 1, 1) * ad
        rs_new = (r.reshape(b, -1) * r.reshape(b, -1)).sum(dim=1)
        # rs is a squared norm; the tolerance is on the residual norm.
        if float(rs_new.max().detach().sqrt()) < tol:
            rs = rs_new
            break
        d = r + (rs_new / (rs + 1e-30)).view(-1, 1, 1) * d
        rs = rs_new

    return demean(p), it, float(rs.max().sqrt())


def leray_project(
    ux: Tensor,
    uy: Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    iters: int = 200,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Project (ux, uy) onto {div u = 0}. Returns the projected field and diagnostics."""
    raw_norm = _norm(ux, uy)
    div_before = divergence(ux, uy, dx, dy)

    p, n_iter, residual = cg_poisson(div_before, dx, dy, iters=iters)
    gx, gy = gradient(p, dx, dy)
    ux_p, uy_p = ux - gx, uy - gy

    div_after = divergence(ux_p, uy_p, dx, dy)
    proj_norm = _norm(ux_p, uy_p)

    stats = {
        "div_before": float(div_before.norm()),
        "div_after": float(div_after.norm()),
        "retained_energy": float(proj_norm / (raw_norm + 1e-12)),
        "cg_iters": float(n_iter),
        "cg_residual": residual,
    }
    return ux_p, uy_p, stats


def _norm(ux: Tensor, uy: Tensor) -> Tensor:
    return (ux.pow(2).sum() + uy.pow(2).sum()).sqrt()
