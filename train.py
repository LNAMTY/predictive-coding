"""Train a predictive-coding network, with every extra component independently removable.

Baselines and ablations
-----------------------
  --learner bp                      backprop reference (same architecture)
  --learner pc                      predictive coding, local updates only        [default]
  --prediction-mode strict|fixed    strict PC, or the Fixed Prediction Assumption
  --fluid                           insert the incompressible transport layer
  --velocity-mode stream|value      solenoidal by construction, or raw-gradient+projection
  --projection                      run the Leray safety net (forced on in value mode)
  --hjb                             add the Hamilton-Jacobi-Bellman residual regulariser
  --obstacles                       carve the paper's pillar+bar into the transport grid

Examples
--------
  python train.py --dataset mnist --num-classes 10 --epochs 5
  python train.py --dataset mnist --num-classes 3  --epochs 5
  python train.py --dataset emnist_letters --fluid --hjb --epochs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from influid_pc import data, report
from influid_pc.build import ModelSpec, build
from influid_pc.diagnostics.bp_alignment import _cosine
from influid_pc.pc.connections import LinearConnection
from influid_pc.pc.network import PCNetwork, PCTrainConfig

RESULTS = Path(__file__).parent / "results"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=sorted(data.DATASETS))
    p.add_argument("--num-classes", type=int, default=None, help="subset to k classes")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--learner", default="pc", choices=["pc", "bp"])
    p.add_argument("--hidden", type=int, nargs="*", default=[128])
    p.add_argument("--activation", default="tanh")
    p.add_argument("--weight-lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--inference-steps", type=int, default=24)
    p.add_argument("--inference-lr", type=float, default=0.1)
    p.add_argument("--output-nudge", type=float, default=0.2)
    p.add_argument("--prediction-mode", default="fixed", choices=["strict", "fixed"])

    p.add_argument("--fluid", action="store_true")
    p.add_argument("--fluid-grid", type=int, default=14)
    p.add_argument("--fluid-steps", type=int, default=8)
    p.add_argument("--fluid-dt", type=float, default=0.5)
    p.add_argument("--fluid-cfl", type=float, default=0.4)
    p.add_argument("--fluid-kappa", type=float, default=0.0,
                   help="diffusion warmup; the paper suggests 0.3, but 0 is better here")
    p.add_argument("--fluid-lr", type=float, default=1e-3)
    p.add_argument("--velocity-mode", default="stream", choices=["stream", "value"])
    p.add_argument("--projection", action="store_true",
                   help="run the Leray safety net every step; a no-op in stream mode")
    p.add_argument("--readout", default="scaled", choices=["scaled", "log"])
    p.add_argument("--obstacles", action="store_true")
    p.add_argument("--no-residual", action="store_true",
                   help="fluid layer replaces its input instead of perturbing it")

    p.add_argument("--hjb", action="store_true")
    p.add_argument("--hjb-weight", type=float, default=0.01)
    p.add_argument("--hjb-nu", type=float, default=0.01)
    p.add_argument("--transport-alpha", type=float, default=1e-3)
    p.add_argument("--transport-beta", type=float, default=1e-3)

    p.add_argument("--track-alignment", action="store_true",
                   help="log cosine(PC update, BP gradient) each epoch")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def accuracy(net: PCNetwork, loader) -> float:
    correct = total = 0
    for x, y in loader:
        pred = net.logits(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def measure_alignment(net: PCNetwork, x, t) -> float:
    """Only meaningful for an all-linear stack, where a backprop reference exists."""
    if not all(isinstance(c, LinearConnection) for c in net.conns):
        return float("nan")

    from influid_pc.diagnostics.bp_alignment import backprop_gradients_net

    bp = backprop_gradients_net(net, x, t)
    out = net.infer(x, target=t)
    states, errors = out["states"], out["errors"]
    pc = [
        c.weight_gradient(states[l], errors[l + 1])
        for l, c in enumerate(net.conns)
    ]
    return _cosine(
        torch.cat([g.flatten() for g in pc]), torch.cat([g.flatten() for g in bp])
    )


def _train_bp_once(args, bundle, lr: float) -> Dict[str, object]:
    import torch.nn as nn

    torch.manual_seed(args.seed)
    dims = [bundle.input_dim, *args.hidden, bundle.num_classes]
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(acts[args.activation]())
        layers.append(nn.Linear(dims[i], dims[i + 1]))
    model = nn.Sequential(*layers)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in bundle.train:
            t = data.one_hot(y, bundle.num_classes)
            opt.zero_grad(set_to_none=True)
            loss = 0.5 * (model(x) - t).pow(2).sum(dim=1).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            correct = sum(int((model(x).argmax(1) == y).sum()) for x, y in bundle.test)
            total = sum(int(y.numel()) for _, y in bundle.test)
        acc = 100.0 * correct / total
        history.append({"epoch": epoch, "test_acc": acc, "elapsed": time.time() - t0})

    return {"history": history, "final_acc": history[-1]["test_acc"], "lr": lr}


def train_bp(args, bundle) -> Dict[str, object]:
    """Backprop reference: same architecture, same MSE-on-one-hot loss.

    Backprop gets the best of a learning-rate sweep while PC runs at a single fixed
    setting. Without this the comparison can be won by handing the baseline a learning
    rate that diverges, which is what lr=0.1 does at depth 1.
    """
    report.header(
        "backprop baseline  ·  best of a learning-rate sweep",
        [
            ("dataset", f"{bundle.name}  ({bundle.num_classes} classes, {bundle.input_dim}-dim)"),
            ("network", report.arch(bundle.input_dim, args.hidden, bundle.num_classes, None)),
            ("training", f"SGD · momentum {args.momentum} · {_plural(args.epochs, 'epoch')}"),
        ],
    )

    cols: List[report.Column] = [
        ("learning rate", 15, "g", ""), ("test acc", 11, ".2f", "%"), ("time", 9, ".1f", "s"),
    ]
    report.table_header(cols)

    candidates = sorted({args.weight_lr, 0.1, 0.05, 0.02, 0.01}, reverse=True)
    runs = []
    for lr in candidates:
        r = _train_bp_once(args, bundle, lr)
        runs.append(r)
        report.table_row(cols, {
            "learning rate": lr,
            "test acc": r["final_acc"],
            "time": r["history"][-1]["elapsed"],
        })

    best = max(runs, key=lambda r: r["final_acc"])
    best["lr_sweep"] = {str(r["lr"]): r["final_acc"] for r in runs}
    return best


def train_pc(args, bundle) -> Dict[str, object]:
    cfg = PCTrainConfig(
        inference_steps=args.inference_steps,
        inference_lr=args.inference_lr,
        output_nudge=args.output_nudge,
        prediction_mode=args.prediction_mode,
    )
    spec = ModelSpec(
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        hidden=list(args.hidden),
        activation=args.activation,
        weight_lr=args.weight_lr,
        momentum=args.momentum,
        fluid=args.fluid,
        fluid_grid=args.fluid_grid,
        fluid_steps=args.fluid_steps,
        fluid_dt=args.fluid_dt,
        fluid_cfl=args.fluid_cfl,
        fluid_kappa=args.fluid_kappa,
        fluid_lr=args.fluid_lr,
        velocity_mode=args.velocity_mode,
        projection=args.projection,
        readout=args.readout,
        residual=not args.no_residual,
        obstacles=args.obstacles,
        hjb=args.hjb,
        hjb_weight=args.hjb_weight,
        hjb_nu=args.hjb_nu,
        transport_alpha=args.transport_alpha,
        transport_beta=args.transport_beta,
    )
    net = build(spec, cfg)

    mode = "fixed predictions (FPA)" if args.prediction_mode == "fixed" else "strict predictions"
    components = [c for c, on in (
        ("Navier-Stokes transport", args.fluid),
        ("HJB regulariser", args.hjb),
        ("Leray projection", args.projection),
        ("obstacles", args.obstacles),
    ) if on]
    grid = args.fluid_grid if args.fluid else None
    shape = report.arch(bundle.input_dim, args.hidden, bundle.num_classes, grid)
    report.header(
        f"predictive coding  ·  {mode}",
        [
            ("dataset", f"{bundle.name}  ({bundle.num_classes} classes, {bundle.input_dim}-dim)"),
            ("network", f"{shape}   ({net.num_parameters:,} params)"),
            ("learning", f"lr {args.weight_lr} · nudge {args.output_nudge} · "
                         f"{args.inference_steps} inference steps · "
                         f"{_plural(args.epochs, 'epoch')}"),
            ("extras", ", ".join(components) if components else "none (plain PC)"),
        ],
    )

    cols: List[report.Column] = [
        ("epoch", 7, "d", ""), ("test acc", 11, ".2f", "%"), ("free energy", 13, ".4f", ""),
    ]
    if args.track_alignment:
        cols.append(("cos(BP)", 10, ".4f", ""))
    if args.fluid:
        cols += [("mass drift", 12, ".1e", ""), ("div u", 10, ".1e", ""), ("CFL", 7, ".2f", "")]
    cols.append(("time", 9, ".1f", "s"))
    report.table_header(cols)

    probe_x, probe_y = next(iter(bundle.train))
    probe_t = data.one_hot(probe_y, bundle.num_classes)

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_stats: Dict[str, float] = {}
        n_batches = 0

        for x, y in bundle.train:
            t = data.one_hot(y, bundle.num_classes)
            out = net.infer(x, target=t)
            stats = net.local_update(out["states"], out["errors"])  # type: ignore[arg-type]
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v
            epoch_stats["free_energy"] = epoch_stats.get("free_energy", 0.0) + out["energy"]  # type: ignore[operator]
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_stats.items()}
        acc = accuracy(net, bundle.test)
        row = {"epoch": epoch, "test_acc": acc, "elapsed": time.time() - t0, **avg}

        if args.track_alignment:
            row["bp_cosine"] = measure_alignment(net, probe_x, probe_t)

        history.append(row)

        cells: Dict[str, object] = {
            "epoch": epoch,
            "test acc": acc,
            "free energy": avg.get("free_energy", 0.0),
            "time": time.time() - t0,
        }
        if args.track_alignment:
            cells["cos(BP)"] = row["bp_cosine"]
        if args.fluid:
            cells["mass drift"] = avg.get("L1/mass_drift")
            cells["div u"] = avg.get("L1/div_final")
            cells["CFL"] = avg.get("L1/cfl")
        report.table_row(cols, cells)

    return {"history": history, "final_acc": history[-1]["test_acc"], "params": net.num_parameters}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    bundle = data.load(
        args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        seed=args.seed,
    )

    result = train_bp(args, bundle) if args.learner == "bp" else train_pc(args, bundle)
    result["config"] = vars(args)
    result["dataset"] = {"name": bundle.name, "num_classes": bundle.num_classes}

    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or (
        f"{args.dataset}-k{bundle.num_classes}-{args.learner}"
        + (f"-{args.prediction_mode}" if args.learner == "pc" else "")
        + ("-fluid" if args.fluid else "")
        + ("-hjb" if args.hjb else "")
    )
    path = Path(args.out) if args.out else RESULTS / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2))

    rows = [("final", f"{result['final_acc']:.2f}%  test accuracy")]
    if args.learner == "bp":
        rows.append(("best lr", f"{result['lr']}  (of {len(result['lr_sweep'])} swept)"))
    if args.learner == "pc" and not args.fluid:
        rows.append(("credit", "local Hebbian updates only, no backpropagation"))
    shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    rows.append(("saved", str(shown)))
    report.summary(rows)


if __name__ == "__main__":
    main()
