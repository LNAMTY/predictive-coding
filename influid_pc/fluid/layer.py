"""In-Fluid-Net's transport layer, as a drop-in predictive-coding connection.

The layer treats its input as a conserved activation budget on a GxG grid, builds
an incompressible velocity field, and advects the budget for T steps:

    rho_0   = softmax(beta * x)          non-negative, sums to 1  ("budget faithful")
    u       = velocity(rho_0)            divergence-free
    rho_t+1 = advect(rho_t, u, dt, kappa(t))
    out     = readout(rho_T)

There are two ways to build u, and the difference between them is the paper's central
practical lesson (secs 1.3, 2.1.3, 8.7):

  velocity_mode="stream"  u = curl(psi_theta), divergence-free by construction.
  velocity_mode="value"   u = Leray_project(-grad W_theta): build a raw gradient field,
                          then project the divergence out.

The paper predicts the second collapses, a Leray projector being the operator that
annihilates gradient fields. Both are implemented here so that the collapse can be
measured via `retained_energy`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..regularizers.hjb import error_cost, hjb_residual
from ..regularizers.transport import transport_loss
from .advection import advect, kappa_schedule, renormalise, total_mass
from .nets import StreamNet, ValueNet
from .operators import (
    cfl_number,
    curl_from_stream,
    divergence,
    gradient,
    node_mask_from_cells,
    rescale_to_cfl,
)
from .projection import leray_project


class IncompressibleRouting(nn.Module):
    def __init__(
        self,
        grid: int,
        steps: int = 8,
        dt: float = 0.5,
        target_cfl: float = 0.4,
        kappa0: float = 0.3,
        kappa_warm_frac: float = 0.5,
        velocity_mode: str = "stream",       # "stream" | "value"
        use_projection: bool = True,          # Leray safety net (and its diagnostic)
        use_hjb: bool = False,
        hjb_weight: float = 0.01,
        hjb_nu: float = 0.01,
        transport_alpha: float = 1e-3,
        transport_beta: float = 1e-3,
        softmax_beta: float = 1.0,
        readout: str = "scaled",              # "scaled" | "log"
        residual: bool = True,
        width: int = 32,
        obstacles: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.G = grid
        self.steps = steps
        self.dt = dt
        self.target_cfl = target_cfl
        self.kappa0 = kappa0
        self.kappa_warm_frac = kappa_warm_frac
        self.velocity_mode = velocity_mode
        # A raw gradient drift has divergence by definition, so "value" mode is only
        # meaningful with the projector attached. In "stream" mode the field is already
        # solenoidal and the projector is an expensive no-op, so it stays opt-in and
        # the divergence is audited instead.
        self.use_projection = use_projection or velocity_mode == "value"
        self.use_hjb = use_hjb
        self.hjb_weight = hjb_weight
        self.hjb_nu = hjb_nu
        self.transport_alpha = transport_alpha
        self.transport_beta = transport_beta
        self.softmax_beta = softmax_beta
        self.readout = readout
        self.residual = residual

        # Residual mode: the layer emits x + gain * (transport-induced change in
        # log-density). At gain=0 this is exactly the identity, so enabling the fluid
        # cannot destroy the representation before the layer has learned to route.
        self.gain = nn.Parameter(torch.zeros(1))

        if velocity_mode == "stream":
            self.stream = StreamNet(1, width)
        elif velocity_mode == "value":
            self.value = ValueNet(1, width)
        else:
            raise ValueError(f"unknown velocity_mode {velocity_mode!r}")

        # An HJB value field can accompany either drift: in "value" mode it is the
        # drift, in "stream" mode it is an auxiliary critic shaping the flow.
        if use_hjb and velocity_mode != "value":
            self.value = ValueNet(1, width)

        free = torch.ones(grid, grid, dtype=torch.bool) if obstacles is None else ~obstacles
        self.register_buffer("free_cell", free)
        self.register_buffer("node_mask", node_mask_from_cells(free).to(torch.float32))

        self.last_stats: Dict[str, float] = {}
        self.aux_loss: Optional[Tensor] = None

    # -- velocity -----------------------------------------------------------

    def velocity(self, rho: Tensor) -> tuple[Tensor, Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}
        x = rho.unsqueeze(1)

        if self.velocity_mode == "stream":
            psi = self.stream(x) * self.node_mask
            ux, uy = curl_from_stream(psi)
        else:
            w = self.value(x)
            gx, gy = gradient(w, 1.0, 1.0)
            ux, uy = -gx, -gy

        raw_energy = (ux.pow(2).sum() + uy.pow(2).sum()).sqrt()

        if self.use_projection:
            ux, uy, pstats = leray_project(ux, uy, iters=100)
            stats.update({f"proj/{k}": v for k, v in pstats.items()})
        else:
            stats["proj/div_before"] = float(divergence(ux, uy).norm())

        # Obstacles / borders: no flux through a wall.
        ux, uy = self._apply_walls(ux, uy)

        proj_energy = (ux.pow(2).sum() + uy.pow(2).sum()).sqrt()
        retained = float(proj_energy / (raw_energy + 1e-12))
        stats["retained_energy"] = retained

        # If the projector annihilated the drift, as it does to a raw gradient field,
        # all that remains is floating-point noise. Rescaling that to hit a target
        # Courant number amplifies it by ~1e9 and returns a field that is not even
        # divergence-free (measured: div 0.96, against 6e-07 for a stream function).
        # The CFL step is only safe on a field with real energy in it.
        collapsed = retained < 1e-6
        stats["collapsed"] = float(collapsed)
        if collapsed:
            stats["cfl_pre_rescale"] = 0.0
            stats["cfl"] = 0.0
            stats["div_final"] = float(divergence(ux, uy).norm())
            return ux, uy, stats

        ux, uy, cfl_pre = rescale_to_cfl(ux, uy, self.dt, self.target_cfl)
        stats["cfl_pre_rescale"] = float(cfl_pre.mean())
        stats["cfl"] = float(cfl_number(ux, uy, self.dt).mean())
        stats["div_final"] = float(divergence(ux, uy).norm())

        return ux, uy, stats

    def _apply_walls(self, ux: Tensor, uy: Tensor) -> tuple[Tensor, Tensor]:
        ux = ux.clone()
        uy = uy.clone()
        ux[:, :, 0] = 0
        ux[:, :, -1] = 0
        uy[:, 0, :] = 0
        uy[:, -1, :] = 0
        return ux, uy

    # -- forward ------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        rho0 = F.softmax(self.softmax_beta * x, dim=1).reshape(b, self.G, self.G)
        rho0 = renormalise(rho0 * self.free_cell)
        rho = rho0
        m0 = total_mass(rho)

        ux, uy, stats = self.velocity(rho)

        aux, aux_stats = self._aux_losses(rho, ux, uy)
        stats.update(aux_stats)
        self.aux_loss = aux

        for t in range(self.steps):
            kappa = kappa_schedule(t, self.steps, self.kappa0, self.kappa_warm_frac)
            rho = advect(rho, ux, uy, self.dt, kappa=kappa, free_cell=self.free_cell)

        drift = (total_mass(rho) - m0).abs().max()
        stats["mass_drift"] = float(drift)
        stats["mass"] = float(total_mass(rho).mean())
        stats["mass_entropy"] = float(
            -(rho.clamp_min(1e-12) * rho.clamp_min(1e-12).log()).sum(dim=(1, 2)).mean()
        )
        stats["gain"] = float(self.gain)
        self.last_stats = stats

        if self.residual:
            # Transport expressed as a change in log-density, added back to the input.
            # Since log softmax(beta*x) = beta*x - logsumexp, at gain=0 and zero velocity
            # this reduces to the identity map on x.
            delta = rho.clamp_min(1e-12).log() - rho0.clamp_min(1e-12).log()
            return x + self.gain * delta.reshape(b, -1)

        out = rho.reshape(b, -1)
        if self.readout == "log":
            return out.clamp_min(1e-12).log()
        return out * (self.G * self.G)

    # -- regularisers -------------------------------------------------------

    def _aux_losses(self, rho: Tensor, ux: Tensor, uy: Tensor):
        stats: Dict[str, float] = {}
        loss = torch.zeros((), device=rho.device)

        t_loss, t_stats = transport_loss(
            ux, uy, alpha=self.transport_alpha, beta=self.transport_beta
        )
        loss = loss + t_loss
        stats.update(t_stats)

        if self.use_hjb:
            w = self.value(rho.unsqueeze(1))
            cost = error_cost(self._pending_error_grid(rho))
            h_loss, h_stats = hjb_residual(w, cost, nu=self.hjb_nu)
            loss = loss + self.hjb_weight * h_loss
            stats.update(h_stats)

        return loss, stats

    def set_error_grid(self, e: Optional[Tensor]) -> None:
        """The PC error at this layer's output, reshaped to the grid: the HJB cost."""
        self._error_grid = e

    def _pending_error_grid(self, rho: Tensor) -> Tensor:
        """The HJB cost is only defined during a local update, when an error exists.

        `forward` also runs during plain prediction, and on the last ragged batch, so a
        stale or wrongly shaped error falls back to zero cost rather than reshaping
        another batch's error into this one.
        """
        e = getattr(self, "_error_grid", None)
        if e is None or e.shape[0] != rho.shape[0] or e.numel() != rho.numel():
            return torch.zeros_like(rho)
        return e.detach().reshape(rho.shape[0], self.G, self.G)
