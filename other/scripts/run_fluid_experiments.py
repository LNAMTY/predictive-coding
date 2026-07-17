"""The transport-layer experiment grid. Writes one JSON per run into other/results/.

  exp3f  transfer: EMNIST-Letters / Fashion-MNIST with the fluid layer inserted, against
                   the matched no-fluid controls. The plain PC and backprop runs these
                   are compared with live in scripts/run_experiments.py (exp3).
  exp4   fluid:    ablations of the In-Fluid-Net machinery itself.

Run:  PYTHONPATH=. .venv/bin/python other/scripts/run_fluid_experiments.py [exp3f exp4]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # other/
REPO = HERE.parent                                  # repo root
PY = str(REPO / ".venv" / "bin" / "python")
RESULTS = HERE / "results"


def run(tag: str, **flags) -> None:
    out = RESULTS / f"{tag}.json"
    if out.exists():
        print(f"  skip {tag} (exists)")
        return

    cmd = [PY, str(REPO / "train.py"), "--tag", tag, "--out", str(out)]
    for k, v in flags.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd += [flag, *[str(i) for i in v]]
        else:
            cmd += [flag, str(v)]

    t0 = time.time()
    print(f"  run  {tag} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        print(f"  FAIL {tag}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        return
    acc = json.loads(out.read_text())["final_acc"] if out.exists() else float("nan")
    print(f"  done {tag}: {acc:.2f}%  ({time.time()-t0:.0f}s)")


# -- exp3f: the fluid layer on a non-MNIST dataset ------------------------------

def exp3f_transfer() -> None:
    print("[exp3f] EMNIST-Letters / Fashion-MNIST, with transport")
    common = dict(
        dataset="emnist_letters", train_subset=20000, test_subset=5000,
        weight_lr=0.1, output_nudge=0.2, hidden=[128],
    )
    run("exp3-emnist-pc-fluid", learner="pc", prediction_mode="fixed", epochs=4,
        fluid=True, fluid_lr=0.01, **common)
    run("exp3-emnist-pc-fluid-hjb", learner="pc", prediction_mode="fixed", epochs=4,
        fluid=True, fluid_lr=0.01, hjb=True, **common)

    # Same comparison on Fashion-MNIST, to check it is not an EMNIST artefact.
    fcommon = dict(common, dataset="fashion")
    run("exp3-fashion-pc-fluid", learner="pc", prediction_mode="fixed", epochs=4,
        fluid=True, fluid_lr=0.01, **fcommon)


# -- exp4: In-Fluid-Net ablations ----------------------------------------------

def exp4_fluid() -> None:
    print("[exp4] fluid ablations")
    base = dict(
        dataset="mnist", train_subset=8000, test_subset=2000, epochs=3,
        learner="pc", prediction_mode="fixed", hidden=[128],
        weight_lr=0.1, output_nudge=0.2, fluid=True, fluid_lr=0.01,
    )
    run("exp4-base", **base)

    # The paper's recommended diffusion warmup.
    for kappa in (0.0, 0.1, 0.3):
        run(f"exp4-kappa{kappa}", **dict(base, fluid_kappa=kappa))

    # The residual wrapper, which is what makes the layer safe to insert at all.
    run("exp4-no-residual", **dict(base, no_residual=True, readout="log"))

    # CFL band: too small and nothing moves, too large and the integrator degrades.
    for cfl in (0.05, 0.9):
        run(f"exp4-cfl{cfl}", **dict(base, fluid_cfl=cfl))

    # HJB regulariser and obstacles.
    run("exp4-hjb", **dict(base, hjb=True))
    run("exp4-obstacles", **dict(base, obstacles=True))

    # Not run end-to-end: velocity_mode="value" and projection=True.
    #
    # Both force a Poisson solve, and under the Fixed Prediction Assumption each of the
    # 24 inference steps asks for a VJP, costing another solve in the projector's
    # backward. That is ~20x slower than the rest of this grid for no new information:
    # the projector annihilates the raw-gradient drift (retained_energy = 0.0000,
    # measured in the layer), the transport term is identically zero, and the residual
    # layer degenerates to the identity. other/scripts/routing_task.py shows the collapse
    # more sharply, where the same drift, built from the ideal value function, delivers
    # 0.0000 band mass against the stream function's 0.5400.


EXPERIMENTS = {
    "exp3f": exp3f_transfer,
    "exp4": exp4_fluid,
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    which = sys.argv[1:] or list(EXPERIMENTS)
    for name in which:
        EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
