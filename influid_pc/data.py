"""Datasets, with the class count as a first-class knob.

The assignment asks for two separable things:

  * predictive coding across *different numbers of classes* (MNIST is 10);
  * the Navier-Stokes variant on a dataset that is *not* MNIST and has a
    different class count.

So every dataset here exposes `num_classes`, and `--num-classes k` will subset any
of them down to k labels. EMNIST-Letters (26) is the default non-MNIST target: it
is a different dataset *and* a different class count, while staying 28x28
greyscale so the fluid grid machinery transfers without reshaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

DATASETS = {
    "mnist": (datasets.MNIST, 10, (1, 28, 28)),
    "fashion": (datasets.FashionMNIST, 10, (1, 28, 28)),
    "kmnist": (datasets.KMNIST, 10, (1, 28, 28)),
    "emnist_letters": (datasets.EMNIST, 26, (1, 28, 28)),
    "emnist_balanced": (datasets.EMNIST, 47, (1, 28, 28)),
    "cifar10": (datasets.CIFAR10, 10, (3, 32, 32)),
}


@dataclass
class DataBundle:
    name: str
    num_classes: int
    input_dim: int
    shape: Tuple[int, int, int]
    train: DataLoader
    test: DataLoader


def _load_raw(name: str, root: str, train: bool):
    cls, _, _ = DATASETS[name]
    tf = transforms.ToTensor()
    if name == "emnist_letters":
        ds = cls(root=root, split="letters", train=train, download=True, transform=tf)
    elif name == "emnist_balanced":
        ds = cls(root=root, split="balanced", train=train, download=True, transform=tf)
    else:
        ds = cls(root=root, train=train, download=True, transform=tf)
    return ds


def _to_tensors(ds, shape) -> Tuple[Tensor, Tensor]:
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs).reshape(-1, int(torch.tensor(shape).prod()))
    y = torch.cat(ys)
    return x, y


def load(
    name: str,
    root: str = "./data",
    batch_size: int = 64,
    num_classes: Optional[int] = None,
    train_subset: int = 0,
    test_subset: int = 0,
    seed: int = 0,
) -> DataBundle:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    _, full_classes, shape = DATASETS[name]

    xtr, ytr = _to_tensors(_load_raw(name, root, True), shape)
    xte, yte = _to_tensors(_load_raw(name, root, False), shape)

    # EMNIST-Letters labels are 1..26; everything else is 0-indexed.
    if name == "emnist_letters":
        ytr, yte = ytr - 1, yte - 1

    k = full_classes if num_classes is None else min(num_classes, full_classes)
    if k < full_classes:
        keep_tr, keep_te = ytr < k, yte < k
        xtr, ytr = xtr[keep_tr], ytr[keep_tr]
        xte, yte = xte[keep_te], yte[keep_te]

    g = torch.Generator().manual_seed(seed)
    if train_subset > 0 and train_subset < len(xtr):
        idx = torch.randperm(len(xtr), generator=g)[:train_subset]
        xtr, ytr = xtr[idx], ytr[idx]
    if test_subset > 0 and test_subset < len(xte):
        idx = torch.randperm(len(xte), generator=g)[:test_subset]
        xte, yte = xte[idx], yte[idx]

    # Standardise: PC relaxation is sensitive to input scale, and an unnormalised
    # input silently changes the effective inference learning rate.
    mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
    xtr, xte = (xtr - mean) / std, (xte - mean) / std

    return DataBundle(
        name=name,
        num_classes=k,
        input_dim=xtr.shape[1],
        shape=shape,
        train=DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True),
        test=DataLoader(TensorDataset(xte, yte), batch_size=512, shuffle=False),
    )


def one_hot(y: Tensor, k: int) -> Tensor:
    return torch.nn.functional.one_hot(y, k).to(torch.float32)
