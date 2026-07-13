"""Prove there is no backpropagation, rather than asserting it in a comment.

"Trains without backprop" is the central claim, and it is the kind of claim that is
easy to make and easy to quietly violate (one stray `.backward()` inside a helper and
the whole thing is a lie). So we check it three ways:

  1. autograd is *physically unavailable* -- we sabotage torch.autograd and
     torch.Tensor.backward, then train anyway;
  2. no tensor in the PC path carries a computation graph;
  3. a layer's update is invariant to non-adjacent layers' errors -- credit is
     genuinely local, not a backward chain in disguise.
"""

import pytest
import torch

from influid_pc.data import one_hot
from influid_pc.pc.connections import LinearConnection
from influid_pc.pc.network import PCNetwork, PCTrainConfig


def _net(dims=(64, 32, 32, 5), mode="fixed"):
    torch.manual_seed(0)
    conns = [
        LinearConnection(dims[i], dims[i + 1], activation="tanh", lr=0.05)
        for i in range(len(dims) - 1)
    ]
    return PCNetwork(conns, PCTrainConfig(inference_steps=16, output_nudge=0.3,
                                          prediction_mode=mode))


def _batch(n=16, d=64, k=5):
    return torch.randn(n, d), one_hot(torch.randint(0, k, (n,)), k)


@pytest.mark.parametrize("mode", ["strict", "fixed"])
def test_training_works_with_autograd_sabotaged(monkeypatch, mode):
    """The strongest form of the claim: break backprop, then train to convergence anyway."""

    def explode(*a, **k):
        raise AssertionError("backpropagation was used")

    monkeypatch.setattr(torch.autograd, "grad", explode)
    monkeypatch.setattr(torch.autograd, "backward", explode)
    monkeypatch.setattr(torch.Tensor, "backward", explode)

    net = _net(mode=mode)
    x, t = _batch()

    def loss():
        return float(0.5 * (net.logits(x) - t).pow(2).sum(dim=1).mean())

    before = loss()
    for _ in range(60):
        out = net.infer(x, target=t)
        net.local_update(out["states"], out["errors"])

    assert loss() < before * 0.6, (before, loss())


def test_no_tensor_in_the_pc_path_carries_a_graph():
    net = _net()
    x, t = _batch()
    out = net.infer(x, target=t)

    for name in ("states", "errors"):
        for i, tensor in enumerate(out[name]):
            assert not tensor.requires_grad, f"{name}[{i}] requires grad"
            assert tensor.grad_fn is None, f"{name}[{i}] carries a graph"

    for c in net.conns:
        for p in c.parameters():
            assert not p.requires_grad
            assert p.grad is None


def test_update_is_invariant_to_non_adjacent_errors():
    """Layer l's update may see states[l] and errors[l+1]. Nothing else may reach it."""
    net = _net()
    x, t = _batch()
    out = net.infer(x, target=t)
    states, errors = out["states"], out["errors"]

    l = 0
    ref = net.conns[l].weight_gradient(states[l], errors[l + 1])

    # Corrupt every error the connection is not allowed to see.
    tampered = [e.clone() for e in errors]
    for j in range(len(tampered)):
        if j != l + 1:
            tampered[j] = torch.randn_like(tampered[j]) * 10

    after = net.conns[l].weight_gradient(states[l], tampered[l + 1])
    assert torch.equal(ref, after), "layer 0's update changed when a distant error changed"
