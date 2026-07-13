"""Predictive coding must recover the backprop gradient from purely local signals.

This is the load-bearing claim of the whole "training without backpropagation"
story, so it gets a test, not a paragraph.
"""

import pytest
import torch

from influid_pc.diagnostics.bp_alignment import alignment_report, backprop_gradients, nudge_sweep
from influid_pc.pc.core import PCConfig, PredictiveCodingNet

torch.manual_seed(0)


def _net(nudge=1.0, steps=64, depth=(30, 40, 40, 10), mode="fixed"):
    cfg = PCConfig(
        layer_sizes=depth,
        activation="tanh",
        inference_steps=steps,
        inference_lr=0.1,
        output_nudge=nudge,
        prediction_mode=mode,
    )
    return PredictiveCodingNet(cfg)


def _batch(net, b=16):
    x = torch.randn(b, net.cfg.layer_sizes[0], dtype=torch.float32)
    t = torch.randn(b, net.cfg.layer_sizes[-1], dtype=torch.float32)
    return x, t


def test_backprop_reference_matches_autograd():
    """Our hand-written backward sweep must agree with autograd, or it proves nothing."""
    net = _net()
    x, t = _batch(net)

    ours = backprop_gradients(net, x, t)

    Ws = [w.clone().requires_grad_(True) for w in net.W]
    bs = [b.clone().requires_grad_(True) for b in net.b]
    a = x
    for l in range(net.n_layers):
        a = net.act.f(a) @ Ws[l].T + bs[l]
    loss = 0.5 * (a - t).pow(2).sum(dim=1).mean()
    loss.backward()

    for l in range(net.n_layers):
        assert torch.allclose(ours[l], Ws[l].grad, atol=1e-6), l


def test_fixed_prediction_pc_recovers_backprop_exactly():
    """Under the Fixed Prediction Assumption, gamma -> 0 drives cosine(PC, BP) -> 1.

    This is the paper's envelope theorem, and it does hold -- in this regime.
    """
    net = _net(mode="fixed")
    x, t = _batch(net, b=64)

    rows = nudge_sweep(net, x, t, nudges=(1.0, 0.1, 0.01, 0.001), steps=512)
    cosines = [r["global_cosine"] for r in rows]

    assert cosines[-1] > 0.9999, cosines
    assert cosines == sorted(cosines), f"alignment should improve monotonically: {cosines}"


def test_strict_pc_does_not_recover_backprop_even_as_nudge_shrinks():
    """Strict PC converges, but to a fixed point that is NOT the backprop gradient.

    The residual misalignment is structural: at the fixed point the hidden-state
    displacement and the error signal are both O(gamma), so relaxing the states
    perturbs the top-down prediction at the same order as the signal it carries.
    Shrinking gamma therefore does not close the gap -- it plateaus.
    """
    net = _net(mode="strict")
    x, t = _batch(net, b=64)

    rows = nudge_sweep(net, x, t, nudges=(0.1, 0.01, 0.001), steps=512)
    cosines = [r["global_cosine"] for r in rows]

    assert all(c < 0.97 for c in cosines), cosines
    # It plateaus rather than converging to 1: the last two are within a hair.
    assert abs(cosines[-1] - cosines[-2]) < 0.01, cosines


def test_more_inference_steps_improve_alignment():
    net = _net(nudge=0.01, mode="fixed")
    x, t = _batch(net, b=64)

    rep = alignment_report(net, x, t, step_grid=(1, 4, 32, 512))
    cos = [r["global_cosine"] for r in rep["rows"]]
    assert cos[-1] > cos[0], cos
    assert cos[-1] > 0.99, cos


def test_free_energy_decreases_during_inference():
    """Only strict PC is gradient descent on the free energy.

    Under the Fixed Prediction Assumption the predictions do not depend on the
    states, so the relaxation is not descending any energy -- it is a bespoke
    error-propagation rule that happens to land on the backprop gradient.
    """
    torch.manual_seed(1)
    net = _net(nudge=1.0, steps=100, mode="strict")
    x, t = _batch(net, b=64)
    trace = net.infer(x, target=t, record=True)["energy_trace"]
    assert trace[-1] < trace[0], (trace[0], trace[-1])
    assert all(b <= a + 1e-6 for a, b in zip(trace, trace[1:])), "energy should be monotone"


@pytest.mark.parametrize("mode", ["strict", "fixed"])
def test_local_update_reduces_loss(mode):
    torch.manual_seed(1)
    net = _net(nudge=0.5, steps=32, mode=mode)
    net.cfg.weight_lr = 0.05
    x, t = _batch(net, b=64)

    def loss():
        return float(0.5 * (net.logits(x) - t).pow(2).sum(dim=1).mean())

    before = loss()
    for _ in range(50):
        out = net.infer(x, target=t)
        net.local_update(out["states"], out["errors"])
    assert loss() < before * 0.5, (before, loss())
