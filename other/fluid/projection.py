"""Helmholtz-Hodge / Leray projection onto the divergence-free subspace.

Given a raw field g, solve the pressure Poisson equation

    lap(p) = div(g),      u = g - grad(p),      so that div(u) = 0.

The paper is emphatic (sections 1.3, 2.1.3, 3.1.3, 8.7) that this should be a safety
net rather than the primary construction. Building g as a raw gradient field and then
projecting can annihilate almost all of it, since a gradient field is precisely what a
Leray projector kills.

`retained_energy` reports the fraction of the field that survives projection, which
makes that failure mode visible.
"""

from __future__ import annotations

from typing import Dict, Tuple

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

    Under pure Neumann conditions the Laplacian is singular (constants lie in its null
    space), so the mean is projected out of both the right-hand side and the iterate.
    Paper, Lemma 7.2: the pressure is unique only up to an additive constant, and the
    resulting velocity is unaffected by it.
    """
    b = rhs.shape[0]
    in_dtype = rhs.dtype

    # In float32 conjugate gradients loses orthogonality on this singular, poorly
    # conditioned operator and the residual starts growing after ~200 iterations
    # (measured: 0.06 -> 1.6e3 -> NaN). The solve therefore runs in float64 regardless
    # of the caller's dtype; on a grid this size that costs nothing.
    work = rhs.to(torch.float64)

    def demean(v: Tensor) -> Tensor:
        return v - v.reshape(b, -1).mean(dim=1).view(-1, 1, 1)

    r = demean(work)
    p = torch.zeros_like(work)
    d = r.clone()
    rs = (r.reshape(b, -1) * r.reshape(b, -1)).sum(dim=1)

    best_p, best_res = p, float(rs.max().detach().sqrt())

    it = 0
    for it in range(1, iters + 1):
        ad = demean(laplacian_cell(d, dx, dy))
        dad = (d.reshape(b, -1) * ad.reshape(b, -1)).sum(dim=1)
        alpha = rs / (dad + 1e-30)
        p = p + alpha.view(-1, 1, 1) * d
        r = r - alpha.view(-1, 1, 1) * ad
        rs_new = (r.reshape(b, -1) * r.reshape(b, -1)).sum(dim=1)

        # rs is a squared norm; the tolerance is on the residual norm.
        res = float(rs_new.max().detach().sqrt())
        if res < best_res:
            best_res, best_p = res, p
        if res < tol:
            break
        # Never return a worse iterate than the best one seen, in case of breakdown.
        if res > 10.0 * best_res:
            break

        d = r + (rs_new / (rs + 1e-30)).view(-1, 1, 1) * d
        rs = rs_new

    return demean(best_p).to(in_dtype), it, best_res


def _zero_boundary_faces(ux: Tensor, uy: Tensor) -> Tuple[Tensor, Tensor]:
    """No flux through the domain wall. Also the condition under which the projector
    is orthogonal rather than oblique; see `_Leray`."""
    ux, uy = ux.clone(), uy.clone()
    ux[:, :, 0] = 0
    ux[:, :, -1] = 0
    uy[:, 0, :] = 0
    uy[:, -1, :] = 0
    return ux, uy


def _project_raw(ux: Tensor, uy: Tensor, dx: float, dy: float, iters: int):
    ux, uy = _zero_boundary_faces(ux, uy)
    p, n_iter, residual = cg_poisson(divergence(ux, uy, dx, dy), dx, dy, iters=iters)
    gx, gy = gradient(p, dx, dy)
    return ux - gx, uy - gy, n_iter, residual


class _Leray(torch.autograd.Function):
    """Projection as a linear operator with a hand-written backward.

    On the space of face fields with zero boundary faces (the physically meaningful
    space, since no flux may cross the domain wall) `gradient` is the negative adjoint
    of `divergence`. There P = I - grad L^-1 div is an orthogonal projection: linear,
    idempotent, self-adjoint. Its Jacobian is P itself, so its vector-Jacobian product
    is P applied to the incoming gradient.

    Off that subspace this fails: `gradient` pins the boundary faces to zero while
    `divergence` reads them, the adjoint identity breaks, and P becomes oblique. The
    operator therefore projects onto the subspace on the way in rather than assume the
    caller has done so. Verified against finite differences in
    `test_leray_custom_backward_matches_autograd_through_the_solver`.

    The alternative, letting autograd unroll the conjugate-gradient loop, builds a graph
    with one node per CG iteration and backpropagates through the whole solver. This
    costs one extra solve in the backward pass and is exact.
    """

    @staticmethod
    def forward(ctx, ux, uy, dx, dy, iters):
        ctx.cfg = (dx, dy, iters)
        ux_p, uy_p, _, _ = _project_raw(ux, uy, dx, dy, iters)
        return ux_p, uy_p

    @staticmethod
    def backward(ctx, gux, guy):
        dx, dy, iters = ctx.cfg
        gx, gy, _, _ = _project_raw(gux.contiguous(), guy.contiguous(), dx, dy, iters)
        return gx, gy, None, None, None


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

    ux_p, uy_p = _Leray.apply(ux, uy, dx, dy, iters)

    with torch.no_grad():
        div_after = divergence(ux_p, uy_p, dx, dy)
        proj_norm = _norm(ux_p, uy_p)

    stats = {
        "div_before": float(div_before.norm()),
        "div_after": float(div_after.norm()),
        "retained_energy": float(proj_norm / (raw_norm + 1e-12)),
        "cg_iters": float(iters),
        "cg_residual": 0.0,
    }
    return ux_p, uy_p, stats


def _norm(ux: Tensor, uy: Tensor) -> Tensor:
    return (ux.pow(2).sum() + uy.pow(2).sum()).sqrt()
