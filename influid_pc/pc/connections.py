"""Connections: the interface between one predictive-coding layer and the next.

A connection between layer l and l+1 supplies three quantities:

    predict(a_l)          the expected value of a_{l+1}
    vjp(a_l, e_{l+1})     how a_l must move to reduce that error   (d ahat/d a_l)^T e
    local_update(...)     how the parameters must move to reduce it (d ahat/d theta)^T e

All three are local to the connection: none can see the loss, the output layer, or
any other connection. An exotic connection (the fluid transport layer, say) can
therefore be swapped in without reintroducing backpropagation, since only the local
map changes and the credit assignment stays predictive coding.

`LinearConnection` computes all three in closed form and never touches autograd.
`ModuleConnection` wraps an arbitrary nn.Module and obtains them from autograd, but
invokes it inside a single connection on detached inputs, so the computation graph
never spans two layers. `tests/test_locality.py` checks this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .activations import ACTIVATIONS, Activation


class Connection(ABC):
    """One directed edge in the predictive-coding hierarchy."""

    in_dim: int
    out_dim: int

    @abstractmethod
    def predict(self, a: Tensor) -> Tensor: ...

    @abstractmethod
    def vjp(self, a: Tensor, e: Tensor) -> Tensor:
        """(d predict / d a)^T @ e: the top-down signal that moves the state below."""

    @abstractmethod
    def local_update(self, a: Tensor, e: Tensor) -> Dict[str, float]:
        """Move parameters to reduce 0.5*||e||^2, using only `a` and `e`."""

    def prepare_vjp(self, a: Tensor) -> None:
        """Hint that many `vjp` calls are coming at this same `a` (fixed-prediction mode)."""

    def release_vjp(self) -> None:
        """Drop anything `prepare_vjp` cached."""

    def diagnostics(self) -> Dict[str, float]:
        return {}

    def parameters(self) -> Iterable[Tensor]:
        return []


class LinearConnection(Connection):
    """ahat = W f(a) + b. Hand-derived; the update is Hebbian (error x presynaptic)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "tanh",
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.in_dim, self.out_dim = in_dim, out_dim
        self.act: Activation = ACTIVATIONS[activation]
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay

        bound = (2.0 / in_dim) ** 0.5
        self.W = torch.randn(out_dim, in_dim, device=device) * bound
        self.b = torch.zeros(out_dim, device=device)
        self._vW = torch.zeros_like(self.W)
        self._vb = torch.zeros_like(self.b)

    def predict(self, a: Tensor) -> Tensor:
        return self.act.f(a) @ self.W.T + self.b

    def vjp(self, a: Tensor, e: Tensor) -> Tensor:
        return (e @ self.W) * self.act.df(a)

    def local_update(self, a: Tensor, e: Tensor) -> Dict[str, float]:
        batch = a.shape[0]
        dW = e.T @ self.act.f(a) / batch
        db = e.mean(dim=0)
        if self.weight_decay:
            dW = dW - self.weight_decay * self.W

        self._vW.mul_(self.momentum).add_(dW)
        self._vb.mul_(self.momentum).add_(db)
        self.W += self.lr * self._vW
        self.b += self.lr * self._vb
        return {"dW": float(dW.norm())}

    def weight_gradient(self, a: Tensor, e: Tensor) -> Tensor:
        """The implied loss gradient (i.e. -dW), for comparison against backprop."""
        return -(e.T @ self.act.f(a)) / a.shape[0]

    def parameters(self) -> Iterable[Tensor]:
        return [self.W, self.b]


class ModuleConnection(Connection):
    """Wraps an nn.Module as a PC connection. ahat = module(f(a)).

    Autograd differentiates this one module on detached inputs. The graph is created
    and destroyed inside `vjp` / `local_update`, so it cannot reach any other layer:
    autograd is a convenient local differentiator here, not a backward chain.
    """

    def __init__(
        self,
        module: nn.Module,
        in_dim: int,
        out_dim: int,
        activation: str = "identity",
        lr: float = 1e-3,
        optimiser: str = "adam",
        device: torch.device | str = "cpu",
    ) -> None:
        self.module = module.to(device)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.act: Activation = ACTIVATIONS[activation]
        opt = torch.optim.Adam if optimiser == "adam" else torch.optim.SGD
        self.opt = opt(self.module.parameters(), lr=lr)
        self._diag: Dict[str, float] = {}
        self._cache: Optional[tuple[Tensor, Tensor]] = None

    def predict(self, a: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.module(self.act.f(a))
        self._capture_diagnostics()
        return out

    def prepare_vjp(self, a: Tensor) -> None:
        """Build the graph once and reuse it for every inference step.

        Under the Fixed Prediction Assumption the linearisation point does not move
        during relaxation, so every inference step asks for a vector-Jacobian product
        at the same `a` with a different `e`. Rebuilding the graph each time would
        re-run the whole transport rollout in the fluid layer's case.
        """
        a_leaf = a.detach().requires_grad_(True)
        out = self.module(self.act.f(a_leaf))
        self._cache = (a_leaf, out)

    def release_vjp(self) -> None:
        self._cache = None

    def vjp(self, a: Tensor, e: Tensor) -> Tensor:
        if self._cache is not None:
            a_leaf, out = self._cache
            (g,) = torch.autograd.grad(out, a_leaf, grad_outputs=e, retain_graph=True)
            return g

        a_leaf = a.detach().requires_grad_(True)
        out = self.module(self.act.f(a_leaf))
        (g,) = torch.autograd.grad(out, a_leaf, grad_outputs=e, retain_graph=False)
        return g

    def local_update(self, a: Tensor, e: Tensor) -> Dict[str, float]:
        # dF/dtheta = -(d ahat/d theta)^T e, so descending F means ascending <e, ahat>.
        self.opt.zero_grad(set_to_none=True)
        out = self.module(self.act.f(a.detach()))
        surrogate = -(e.detach() * out).sum() / a.shape[0]
        surrogate.backward()
        self.opt.step()
        self._capture_diagnostics()
        gnorm = sum(
            float(p.grad.norm()) for p in self.module.parameters() if p.grad is not None
        )
        return {"dW": gnorm}

    def _capture_diagnostics(self) -> None:
        stats = getattr(self.module, "last_stats", None)
        if isinstance(stats, dict):
            self._diag = dict(stats)

    def diagnostics(self) -> Dict[str, float]:
        return self._diag

    def parameters(self) -> Iterable[Tensor]:
        return list(self.module.parameters())
