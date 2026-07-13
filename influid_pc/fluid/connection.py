"""The fluid transport layer, exposed as a predictive-coding connection.

The layer learns from `e`, the prediction error measured at its own output, plus its own
transport and HJB regularisers. It never sees the task loss, the label, or any other
layer's error. Removing this connection from the stack leaves the rest of the network
unchanged.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from ..pc.connections import ModuleConnection
from .layer import IncompressibleRouting


class FluidConnection(ModuleConnection):
    def __init__(
        self,
        layer: IncompressibleRouting,
        activation: str = "identity",
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        dim = layer.G * layer.G
        super().__init__(layer, dim, dim, activation=activation, lr=lr, device=device)
        self.layer = layer

    def local_update(self, a: Tensor, e: Tensor) -> Dict[str, float]:
        self.layer.set_error_grid(e)

        self.opt.zero_grad(set_to_none=True)
        out = self.layer(self.act.f(a.detach()))

        # Descending 0.5||e||^2 in the layer's parameters means ascending <e, out>.
        surrogate = -(e.detach() * out).sum() / a.shape[0]
        aux = self.layer.aux_loss
        total = surrogate + aux if aux is not None else surrogate
        total.backward()
        self.opt.step()

        self._capture_diagnostics()
        stats = dict(self.layer.last_stats)
        stats["aux_loss"] = float(aux) if aux is not None else 0.0
        return stats

    def diagnostics(self) -> Dict[str, float]:
        return dict(self.layer.last_stats)
