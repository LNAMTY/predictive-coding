"""The invariants the whole fluid story rests on. If these fail, nothing above them means anything."""

import torch

from influid_pc.fluid.advection import advect, renormalise, total_mass
from influid_pc.fluid.operators import (
    cfl_number,
    curl_from_stream,
    divergence,
    node_mask_from_cells,
    rescale_to_cfl,
)
from influid_pc.fluid.projection import leray_project

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
    """The paper's warning, made concrete: project a raw gradient and almost nothing survives."""
    from influid_pc.fluid.operators import gradient

    phi = torch.randn(B, H, W, dtype=torch.float64)
    gx, gy = gradient(phi)
    _, _, stats = leray_project(gx, gy, iters=1000)
    assert stats["retained_energy"] < 0.05, stats["retained_energy"]
