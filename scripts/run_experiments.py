"""The full experiment grid. Writes one JSON per run into results/.

  exp1  depth:      whether the strict-PC gradient decay (see alignment_study) costs
                    accuracy as the network deepens. PC-strict vs PC-fixed vs BP.
  exp2  classes:    predictive coding across class counts on MNIST (k = 2, 3, 5, 10).
  exp3  transfer:   a non-MNIST dataset with a different class count (EMNIST-Letters,
                    26 classes).

Run:  PYTHONPATH=. .venv/bin/python scripts/run_experiments.py [exp1 exp2 ...]

The transport-layer grid lives in other/scripts/run_fluid_experiments.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
RESULTS = ROOT / "results"


def run(tag: str, **flags) -> None:
    out = RESULTS / f"{tag}.json"
    if out.exists():
        print(f"  skip {tag} (exists)")
        return

    cmd = [PY, str(ROOT / "train.py"), "--tag", tag]
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
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(f"  FAIL {tag}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        return
    acc = json.loads(out.read_text())["final_acc"] if out.exists() else float("nan")
    print(f"  done {tag}: {acc:.2f}%  ({time.time()-t0:.0f}s)")


# -- exp1: accuracy cost of gradient misalignment as depth grows ---------------

def exp1_depth() -> None:
    print("[exp1] depth: strict vs fixed vs backprop")
    common = dict(
        dataset="mnist", train_subset=20000, test_subset=5000, epochs=8,
        weight_lr=0.1, output_nudge=0.2, inference_steps=32, track_alignment=True,
    )
    depths = {1: [128], 2: [128, 128], 3: [128, 128, 128],
              4: [128, 128, 128, 128], 6: [128] * 6}
    for n, hidden in depths.items():
        run(f"exp1-depth{n}-bp", learner="bp", hidden=hidden, **common)
        for mode in ("strict", "fixed"):
            run(f"exp1-depth{n}-pc-{mode}", learner="pc", prediction_mode=mode,
                hidden=hidden, **common)


# -- exp2: predictive coding across different class counts ---------------------

def exp2_classes() -> None:
    print("[exp2] class counts on MNIST")
    common = dict(
        dataset="mnist", train_subset=20000, test_subset=5000, epochs=8,
        hidden=[256, 128], weight_lr=0.1, output_nudge=0.2, track_alignment=True,
    )
    for k in (2, 3, 5, 10):
        run(f"exp2-mnist-k{k}-bp", learner="bp", num_classes=k, **common)
        run(f"exp2-mnist-k{k}-pc", learner="pc", prediction_mode="fixed",
            num_classes=k, **common)


# -- exp3: non-MNIST dataset, different class count -----------------------------

def exp3_transfer() -> None:
    print("[exp3] EMNIST-Letters (26 classes)")
    common = dict(
        dataset="emnist_letters", train_subset=20000, test_subset=5000,
        weight_lr=0.1, output_nudge=0.2, hidden=[128],
    )
    run("exp3-emnist-bp", learner="bp", epochs=8, **common)
    run("exp3-emnist-pc", learner="pc", prediction_mode="fixed", epochs=8, **common)

    # Same dataset swap on Fashion-MNIST, to check it is not an EMNIST artefact.
    fcommon = dict(common, dataset="fashion")
    run("exp3-fashion-pc", learner="pc", prediction_mode="fixed", epochs=8, **fcommon)


EXPERIMENTS = {
    "exp1": exp1_depth,
    "exp2": exp2_classes,
    "exp3": exp3_transfer,
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    which = sys.argv[1:] or list(EXPERIMENTS)
    for name in which:
        EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
