"""Deep predictive coding: inference relaxation + strictly local weight updates.

Discriminative ("forward-prediction") formulation, following Whittington & Bogacz
(2017) and Millidge et al. (2020):

    layer states   a_0 ... a_L        (a_0 clamped to input, a_L clamped to target)
    prediction     ahat_{l+1} = W_l @ f(a_l) + b_l
    error          e_{l+1}    = a_{l+1} - ahat_{l+1}
    free energy    F          = sum_l 0.5 * ||e_l||^2

Inference (fast, states move, weights frozen):

    dF/da_l = e_l - f'(a_l) * (W_l^T @ e_{l+1})
    a_l <- a_l - eta * dF/da_l

Learning (slow, weights move, states frozen):

    dF/dW_l = -e_{l+1} @ f(a_l)^T          <- purely local: pre-synaptic activity
    W_l <- W_l + lr * e_{l+1} @ f(a_l)^T      times post-synaptic error

No autograd, no backward chain: every update above touches only quantities that
live on the two layers a synapse connects. That is the whole point, and it is
what `diagnostics/bp_alignment.py` independently verifies against true backprop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .activations import ACTIVATIONS, Activation


@dataclass
class PCConfig:
    layer_sizes: Sequence[int]
    activation: str = "tanh"
    inference_steps: int = 32
    inference_lr: float = 0.1
    weight_lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    # Initialise hidden states with a feedforward sweep instead of zeros.
    # This is what makes PC converge in few steps.
    ff_init: bool = True
    # Scales the output-layer error during training. gamma -> 0 recovers the
    # "infinitesimal nudge" regime where PC provably equals backprop.
    output_nudge: float = 1.0
    grad_clip: float = 10.0

    # "strict": top-down predictions are recomputed from the relaxed states at
    #   every inference step. This is predictive coding as literally specified.
    # "fixed": predictions (and their local derivatives) are frozen at their
    #   feedforward values for the whole relaxation -- the Fixed Prediction
    #   Assumption of Millidge et al. (2020).
    #
    # The two agree to first order but converge to *different* fixed points, and
    # only "fixed" reproduces the backprop gradient. See docs/FINDINGS.md.
    prediction_mode: str = "strict"


class PredictiveCodingNet:
    """Fully-connected predictive coding network trained without backpropagation."""

    def __init__(self, config: PCConfig, device: torch.device | str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        self.act: Activation = ACTIVATIONS[config.activation]

        sizes = list(config.layer_sizes)
        self.n_layers = len(sizes) - 1

        self.W: List[Tensor] = []
        self.b: List[Tensor] = []
        for l in range(self.n_layers):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            bound = (2.0 / fan_in) ** 0.5
            self.W.append(torch.randn(fan_out, fan_in, device=self.device) * bound)
            self.b.append(torch.zeros(fan_out, device=self.device))

        self._vW = [torch.zeros_like(w) for w in self.W]
        self._vb = [torch.zeros_like(b) for b in self.b]

    # -- prediction ---------------------------------------------------------

    def predict(self, a_l: Tensor, l: int) -> Tensor:
        return self.act.f(a_l) @ self.W[l].T + self.b[l]

    def forward(self, x: Tensor) -> List[Tensor]:
        """Feedforward sweep. Also the test-time inference path (no relaxation)."""
        states = [x]
        for l in range(self.n_layers):
            states.append(self.predict(states[l], l))
        return states

    def logits(self, x: Tensor) -> Tensor:
        return self.forward(x)[-1]

    # -- inference ----------------------------------------------------------

    def infer(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        steps: Optional[int] = None,
        record: bool = False,
        hook: Optional[Callable[[int, List[Tensor], List[Tensor]], None]] = None,
    ) -> Dict[str, object]:
        """Relax hidden states to (approximately) minimise free energy.

        `x` is clamped at layer 0. If `target` is given it is clamped at layer L
        (supervised / "nudged" phase). Hidden layers 1..L-1 are free.
        """
        steps = self.cfg.inference_steps if steps is None else steps
        fixed = self.cfg.prediction_mode == "fixed"

        ff = self.forward(x)
        states = [s.clone() for s in ff] if self.cfg.ff_init else self._zero_init(x)
        if target is not None:
            states[-1] = states[-1] + self.cfg.output_nudge * (target - states[-1])

        # Under the Fixed Prediction Assumption the predictions -- and the local
        # derivatives f'(a_l) that gate the top-down error -- are pinned to the
        # feedforward pass and never see the relaxed states.
        preds = [self.predict(ff[l], l) for l in range(self.n_layers)] if fixed else None
        dfs = [self.act.df(ff[l]) for l in range(self.n_layers)] if fixed else None

        energies: List[float] = []

        for step in range(steps):
            errors = self.errors(states, preds)

            # Hidden states only: layer 0 is the clamped input, layer L is either
            # clamped to the target or is itself just the top prediction.
            for l in range(1, self.n_layers):
                # dF/da_l = e_l - f'(a_l) * (W_l^T e_{l+1})
                df_l = dfs[l] if fixed else self.act.df(states[l])
                lateral = (errors[l + 1] @ self.W[l]) * df_l
                grad = errors[l] - lateral
                if self.cfg.grad_clip > 0:
                    grad = grad.clamp(-self.cfg.grad_clip, self.cfg.grad_clip)
                states[l] = states[l] - self.cfg.inference_lr * grad

            if target is None and not fixed:
                # Unclamped output relaxes toward its own prediction.
                states[-1] = self.predict(states[-2], self.n_layers - 1)

            if record or hook is not None:
                errs = self.errors(states, preds)
                if record:
                    energies.append(self.free_energy(errs))
                if hook is not None:
                    hook(step, states, errs)

        errors = self.errors(states, preds)
        return {
            "states": states,
            "errors": errors,
            "ff_states": ff,
            "energy": self.free_energy(errors),
            "energy_trace": energies,
        }

    def _zero_init(self, x: Tensor) -> List[Tensor]:
        states = [x]
        for l in range(self.n_layers):
            states.append(torch.zeros(x.shape[0], self.W[l].shape[0], device=x.device))
        return states

    def errors(
        self, states: List[Tensor], preds: Optional[List[Tensor]] = None
    ) -> List[Tensor]:
        """errors[l] is the prediction error at layer l; errors[0] is unused (zeros).

        If `preds` is given, those frozen predictions are used instead of ones
        recomputed from the current states (Fixed Prediction Assumption).
        """
        errors: List[Tensor] = [torch.zeros_like(states[0])]
        for l in range(self.n_layers):
            pred = preds[l] if preds is not None else self.predict(states[l], l)
            errors.append(states[l + 1] - pred)
        return errors

    @staticmethod
    def free_energy(errors: Sequence[Tensor]) -> float:
        return float(sum(0.5 * e.pow(2).sum(dim=1).mean() for e in errors[1:]))

    # -- learning -----------------------------------------------------------

    def local_update(self, states: List[Tensor], errors: List[Tensor]) -> Dict[str, float]:
        """Hebbian update. dW_l = e_{l+1} outer f(a_l), averaged over the batch."""
        cfg = self.cfg
        stats: Dict[str, float] = {}
        batch = states[0].shape[0]

        for l in range(self.n_layers):
            pre = self.act.f(states[l])          # presynaptic activity
            post = errors[l + 1]                 # postsynaptic error
            dW = post.T @ pre / batch
            db = post.mean(dim=0)

            if cfg.weight_decay:
                dW = dW - cfg.weight_decay * self.W[l]

            self._vW[l].mul_(cfg.momentum).add_(dW)
            self._vb[l].mul_(cfg.momentum).add_(db)

            self.W[l] += cfg.weight_lr * self._vW[l]
            self.b[l] += cfg.weight_lr * self._vb[l]

            stats[f"dW{l}"] = float(dW.norm())
        return stats

    # -- pc gradient (for comparison against backprop) ----------------------

    def pc_weight_gradients(self, states: List[Tensor], errors: List[Tensor]) -> List[Tensor]:
        """The gradient PC *implies*, sign-matched to a descent direction on the loss.

        Returned as -dW so it is directly comparable to a backprop gradient.
        """
        batch = states[0].shape[0]
        return [-(errors[l + 1].T @ self.act.f(states[l])) / batch for l in range(self.n_layers)]

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {"W": [w.cpu() for w in self.W], "b": [b.cpu() for b in self.b], "cfg": self.cfg}

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        self.W = [w.to(self.device) for w in sd["W"]]  # type: ignore[index]
        self.b = [b.to(self.device) for b in sd["b"]]  # type: ignore[index]
        self._vW = [torch.zeros_like(w) for w in self.W]
        self._vb = [torch.zeros_like(b) for b in self.b]

    @property
    def num_parameters(self) -> int:
        return sum(w.numel() for w in self.W) + sum(b.numel() for b in self.b)
