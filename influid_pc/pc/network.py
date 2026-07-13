"""Predictive coding over a stack of arbitrary connections.

Same algorithm as `core.py`, but each edge is a `Connection`, so a fluid transport
layer can be inserted without the inference or learning rules changing at all.
That is the design goal: `--fluid off` and `--fluid on` run the *same* PC code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .connections import Connection, LinearConnection


@dataclass
class PCTrainConfig:
    inference_steps: int = 32
    inference_lr: float = 0.1
    output_nudge: float = 1.0
    prediction_mode: str = "strict"   # "strict" | "fixed"
    grad_clip: float = 10.0
    ff_init: bool = True


class PCNetwork:
    def __init__(self, connections: Sequence[Connection], cfg: PCTrainConfig) -> None:
        self.conns: List[Connection] = list(connections)
        self.cfg = cfg
        self.n_layers = len(self.conns)

    # -- forward ------------------------------------------------------------

    def forward(self, x: Tensor) -> List[Tensor]:
        states = [x]
        for c in self.conns:
            states.append(c.predict(states[-1]))
        return states

    def logits(self, x: Tensor) -> Tensor:
        return self.forward(x)[-1]

    def errors(self, states: List[Tensor], preds: Optional[List[Tensor]] = None) -> List[Tensor]:
        errs: List[Tensor] = [torch.zeros_like(states[0])]
        for l, c in enumerate(self.conns):
            pred = preds[l] if preds is not None else c.predict(states[l])
            errs.append(states[l + 1] - pred)
        return errs

    @staticmethod
    def free_energy(errors: Sequence[Tensor]) -> float:
        return float(sum(0.5 * e.pow(2).sum(dim=1).mean() for e in errors[1:]))

    # -- inference ----------------------------------------------------------

    def infer(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        steps: Optional[int] = None,
        record: bool = False,
    ) -> Dict[str, object]:
        cfg = self.cfg
        steps = cfg.inference_steps if steps is None else steps
        fixed = cfg.prediction_mode == "fixed"

        ff = self.forward(x)
        states = [s.clone() for s in ff]
        if target is not None:
            states[-1] = states[-1] + cfg.output_nudge * (target - states[-1])

        preds = [self.conns[l].predict(ff[l]) for l in range(self.n_layers)] if fixed else None
        # Under FPA the top-down signal is also evaluated at the feedforward state.
        vjp_at = ff if fixed else states

        if fixed:
            for l in range(1, self.n_layers):
                self.conns[l].prepare_vjp(ff[l])

        trace: List[float] = []
        try:
            for _ in range(steps):
                errs = self.errors(states, preds)
                for l in range(1, self.n_layers):
                    lateral = self.conns[l].vjp(vjp_at[l], errs[l + 1])
                    grad = errs[l] - lateral
                    if cfg.grad_clip > 0:
                        grad = grad.clamp(-cfg.grad_clip, cfg.grad_clip)
                    states[l] = states[l] - cfg.inference_lr * grad
                if target is None and not fixed:
                    states[-1] = self.conns[-1].predict(states[-2])
                if record:
                    trace.append(self.free_energy(self.errors(states, preds)))
        finally:
            for c in self.conns:
                c.release_vjp()

        errs = self.errors(states, preds)
        return {
            "states": states,
            "errors": errs,
            "ff_states": ff,
            "energy": self.free_energy(errs),
            "energy_trace": trace,
        }

    # -- learning -----------------------------------------------------------

    def local_update(self, states: List[Tensor], errors: List[Tensor]) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for l, c in enumerate(self.conns):
            out = c.local_update(states[l], errors[l + 1])
            for k, v in out.items():
                stats[f"L{l}/{k}"] = v
            for k, v in c.diagnostics().items():
                stats[f"L{l}/{k}"] = v
        return stats

    @property
    def num_parameters(self) -> int:
        return sum(int(p.numel()) for c in self.conns for p in c.parameters())
