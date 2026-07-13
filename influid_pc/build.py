"""Assemble a network from flags. Every component is independently removable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .fluid.connection import FluidConnection
from .fluid.layer import IncompressibleRouting
from .pc.connections import Connection, LinearConnection
from .pc.network import PCNetwork, PCTrainConfig


@dataclass
class ModelSpec:
    input_dim: int
    num_classes: int
    hidden: List[int]
    activation: str = "tanh"
    weight_lr: float = 1e-3
    momentum: float = 0.9

    # fluid
    fluid: bool = False
    fluid_grid: int = 14
    fluid_steps: int = 8
    fluid_dt: float = 0.5
    fluid_cfl: float = 0.4
    fluid_kappa: float = 0.0
    velocity_mode: str = "stream"
    projection: bool = False
    fluid_lr: float = 1e-3
    readout: str = "scaled"
    residual: bool = True

    # hjb
    hjb: bool = False
    hjb_weight: float = 0.01
    hjb_nu: float = 0.01

    # transport regularisers
    transport_alpha: float = 1e-3
    transport_beta: float = 1e-3

    obstacles: bool = False


def _obstacle_mask(grid: int) -> torch.Tensor:
    """The paper's toy geometry: a pillar and a bar that the flow must route around."""
    m = torch.zeros(grid, grid, dtype=torch.bool)
    c = grid // 2
    m[grid // 4 : 3 * grid // 4, c : c + max(1, grid // 12)] = True
    m[c + grid // 8 : c + grid // 8 + max(1, grid // 14), grid // 6 : 5 * grid // 6] = True
    return m


def build(spec: ModelSpec, cfg: PCTrainConfig, device: str = "cpu") -> PCNetwork:
    conns: List[Connection] = []

    if not spec.fluid:
        dims = [spec.input_dim, *spec.hidden, spec.num_classes]
        for i in range(len(dims) - 1):
            conns.append(
                LinearConnection(
                    dims[i],
                    dims[i + 1],
                    activation=spec.activation,
                    lr=spec.weight_lr,
                    momentum=spec.momentum,
                    device=device,
                )
            )
        return PCNetwork(conns, cfg)

    # With the fluid layer, one hidden layer is the GxG transport grid.
    g = spec.fluid_grid
    grid_dim = g * g

    conns.append(
        LinearConnection(
            spec.input_dim,
            grid_dim,
            activation=spec.activation,
            lr=spec.weight_lr,
            momentum=spec.momentum,
            device=device,
        )
    )

    layer = IncompressibleRouting(
        grid=g,
        steps=spec.fluid_steps,
        dt=spec.fluid_dt,
        target_cfl=spec.fluid_cfl,
        kappa0=spec.fluid_kappa,
        velocity_mode=spec.velocity_mode,
        use_projection=spec.projection,
        use_hjb=spec.hjb,
        hjb_weight=spec.hjb_weight,
        hjb_nu=spec.hjb_nu,
        transport_alpha=spec.transport_alpha,
        transport_beta=spec.transport_beta,
        readout=spec.readout,
        residual=spec.residual,
        obstacles=_obstacle_mask(g) if spec.obstacles else None,
    )
    conns.append(FluidConnection(layer, lr=spec.fluid_lr, device=device))

    tail = [*spec.hidden, spec.num_classes]
    dims = [grid_dim, *tail]
    for i in range(len(dims) - 1):
        conns.append(
            LinearConnection(
                dims[i],
                dims[i + 1],
                activation=spec.activation,
                lr=spec.weight_lr,
                momentum=spec.momentum,
                device=device,
            )
        )

    return PCNetwork(conns, cfg)
