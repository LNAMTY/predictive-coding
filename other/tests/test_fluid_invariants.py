"""Structural invariants of the transport layer: incompressibility, mass conservation,
positivity, no-through obstacles, CFL targeting, and the projector's backward pass."""

import torch

from other.fluid.advection import advect, renormalise, total_mass
from other.fluid.operators import (
    cfl_number,
    curl_from_stream,
    divergence,
    node_mask_from_cells,
    rescale_to_cfl,
)
from other.fluid.projection import leray_project

torch.manual_seed(0)
B, H, W = 4, 24, 24


def test_stream_function_is_exactly_divergence_free():
    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64)
    ux, uy = curl_from_stream(psi)
    div = divergence(ux, uy)
    assert div.abs().max() < 1e-12, div.abs().max()


def test_pinned_border_means_no_flux_leaves_the_domain():
    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64)
    free = torch.ones(H, W, dtype=torch.bool)
    psi = psi * node_mask_from_cells(free)
    ux, uy = curl_from_stream(psi)
    assert ux[:, :, 0].abs().max() == 0
    assert ux[:, :, -1].abs().max() == 0
    assert uy[:, 0, :].abs().max() == 0
    assert uy[:, -1, :].abs().max() == 0


def test_obstacles_are_no_through_and_still_divergence_free():
    free = torch.ones(H, W, dtype=torch.bool)
    free[6:18, 10:13] = False          # a pillar
    free[15:17, 4:20] = False          # a bar

    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64) * node_mask_from_cells(free)
    ux, uy = curl_from_stream(psi)

    assert divergence(ux, uy).abs().max() < 1e-12

    # No mass may cross into a blocked cell.
    rho = torch.rand(B, H, W, dtype=torch.float64) * free
    rho = renormalise(rho)
    ux, uy, _ = rescale_to_cfl(ux, uy, dt=0.5, target=0.4)
    for _ in range(50):
        rho = advect(rho, ux, uy, dt=0.5, kappa=0.0, free_cell=free)
    assert rho[:, ~free].abs().max() < 1e-14


def test_advection_conserves_mass_exactly():
    free = torch.ones(H, W, dtype=torch.bool)
    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64) * node_mask_from_cells(free)
    ux, uy = curl_from_stream(psi)
    ux, uy, _ = rescale_to_cfl(ux, uy, dt=0.5, target=0.4)

    rho = torch.rand(B, H, W, dtype=torch.float64)
    rho = renormalise(rho)
    m0 = total_mass(rho)

    for _ in range(200):
        rho = advect(rho, ux, uy, dt=0.5, kappa=0.05)
        assert (rho >= 0).all()

    drift = (total_mass(rho) - m0).abs().max()
    assert drift < 1e-12, drift


def test_diffusion_conserves_mass_with_obstacles():
    """Diffusion and obstacles together: the case each of the tests above misses.

    `test_advection_conserves_mass_exactly` runs kappa > 0 on an open grid, and
    `test_obstacles_are_no_through` runs obstacles at kappa = 0. Between them they let a
    leak through: an unmasked Laplacian diffuses mass into blocked cells, which `advect`
    then zeroes, destroying ~30% of the budget over a rollout.
    """
    free = torch.ones(H, W, dtype=torch.bool)
    free[6:18, 10:13] = False
    free[15:17, 4:20] = False

    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64) * node_mask_from_cells(free)
    ux, uy = curl_from_stream(psi)
    ux, uy, _ = rescale_to_cfl(ux, uy, dt=0.5, target=0.4)

    rho = renormalise(torch.rand(B, H, W, dtype=torch.float64) * free)
    m0 = total_mass(rho)
    for _ in range(50):
        rho = advect(rho, ux, uy, dt=0.5, kappa=0.3, free_cell=free)

    assert (total_mass(rho) - m0).abs().max() < 1e-12, (total_mass(rho) - m0).abs().max()
    assert rho[:, ~free].abs().max() == 0


def test_masked_diffusion_reduces_to_the_plain_laplacian_on_an_open_grid():
    from other.fluid.advection import diffusion
    from other.fluid.operators import laplacian_cell

    rho = torch.rand(B, H, W, dtype=torch.float64)
    free = torch.ones(H, W, dtype=torch.bool)
    assert torch.allclose(diffusion(rho, free_cell=None), laplacian_cell(rho))
    assert torch.allclose(diffusion(rho, free_cell=free), laplacian_cell(rho))


def test_cfl_targeting_hits_the_target():
    psi = torch.randn(B, H + 1, W + 1, dtype=torch.float64)
    ux, uy = curl_from_stream(psi)
    ux, uy, _ = rescale_to_cfl(ux, uy, dt=0.5, target=0.4)
    assert torch.allclose(cfl_number(ux, uy, dt=0.5), torch.full((B,), 0.4, dtype=torch.float64))


def test_leray_projection_removes_divergence():
    ux = torch.randn(B, H, W + 1, dtype=torch.float64)
    uy = torch.randn(B, H + 1, W, dtype=torch.float64)
    ux[:, :, 0] = ux[:, :, -1] = 0
    uy[:, 0, :] = uy[:, -1, :] = 0

    ux_p, uy_p, stats = leray_project(ux, uy, iters=2000)
    assert stats["div_after"] < 1e-8 * max(stats["div_before"], 1.0)


def test_projection_annihilates_a_pure_gradient_field():
    """The paper's warning: project a raw gradient field and almost nothing survives."""
    from other.fluid.operators import gradient

    phi = torch.randn(B, H, W, dtype=torch.float64)
    gx, gy = gradient(phi)
    _, _, stats = leray_project(gx, gy, iters=1000)
    assert stats["retained_energy"] < 0.05, stats["retained_energy"]


def test_leray_custom_backward_matches_autograd_through_the_solver():
    """The projector has a hand-written backward, checked here against finite differences.

    Not against autograd-through-the-solver: an early-terminating CG loop does not
    necessarily differentiate to the right thing, so it is a reference rather than the
    ground truth. Finite differences is the ground truth.
    """
    from other.fluid.projection import _project_raw, _zero_boundary_faces, leray_project

    torch.manual_seed(0)

    def rand_field():
        return _zero_boundary_faces(
            torch.randn(1, 8, 9, dtype=torch.float64),
            torch.randn(1, 9, 8, dtype=torch.float64),
        )

    ux0, uy0 = rand_field()
    s0, s1 = rand_field()
    d0, d1 = rand_field()

    def loss(ux, uy):
        px, py, _, _ = _project_raw(ux, uy, 1.0, 1.0, 600)
        return (px * s0).sum() + (py * s1).sum()

    eps = 1e-6
    fd = (loss(ux0 + eps * d0, uy0 + eps * d1) - loss(ux0 - eps * d0, uy0 - eps * d1)) / (2 * eps)

    a, b = ux0.clone().requires_grad_(True), uy0.clone().requires_grad_(True)
    px, py, _ = leray_project(a, b, iters=600)
    ((px * s0).sum() + (py * s1).sum()).backward()
    ours = (a.grad * d0).sum() + (b.grad * d1).sum()

    assert torch.allclose(ours, fd, rtol=1e-5), (float(ours), float(fd))
