"""Small conv nets that produce the two fields the transport layer needs.

StreamNet emits psi on grid *nodes* ((G+1)x(G+1)), because u = curl(psi) then lands
exactly on the faces where the advection scheme wants it. The final conv uses
kernel 2 with padding 1, which takes GxG to (G+1)x(G+1) with no interpolation.

ValueNet emits W on grid *cells* (GxG): the value/desirability landscape whose
negative gradient is the raw drift, and on which the HJB residual is evaluated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StreamNet(nn.Module):
    """rho (and optional extra channels) -> stream function psi on nodes."""

    def __init__(self, in_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(1)


class ValueNet(nn.Module):
    """rho -> scalar value field W on cells."""

    def __init__(self, in_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(1)
