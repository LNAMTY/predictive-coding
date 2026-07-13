"""Activations with hand-written derivatives, since PC never calls autograd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import Tensor


@dataclass(frozen=True)
class Activation:
    name: str
    f: Callable[[Tensor], Tensor]
    df: Callable[[Tensor], Tensor]


def _tanh_df(x: Tensor) -> Tensor:
    return 1.0 - torch.tanh(x).pow(2)


def _relu_df(x: Tensor) -> Tensor:
    return (x > 0).to(x.dtype)


def _leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    return torch.where(x > 0, x, slope * x)


def _leaky_relu_df(x: Tensor, slope: float = 0.01) -> Tensor:
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, slope))


def _sigmoid_df(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


ACTIVATIONS: Dict[str, Activation] = {
    "tanh": Activation("tanh", torch.tanh, _tanh_df),
    "relu": Activation("relu", torch.relu, _relu_df),
    "leaky_relu": Activation("leaky_relu", _leaky_relu, _leaky_relu_df),
    "sigmoid": Activation("sigmoid", torch.sigmoid, _sigmoid_df),
    "identity": Activation("identity", lambda x: x, torch.ones_like),
}
