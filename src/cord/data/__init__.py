"""Dataset registry. Add a dataset by writing a loader returning :class:`ImageDataset` and
registering it here (GTSRB, Tiny-ImageNet and the imbalance tier plug in the same way)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cord.data.base import ImageDataset
from cord.data.cifar10 import load_cifar10

Loader = Callable[[Path], ImageDataset]

_REGISTRY: dict[str, Loader] = {"cifar10": load_cifar10}


def register_dataset(name: str) -> Callable[[Loader], Loader]:
    def deco(fn: Loader) -> Loader:
        if name in _REGISTRY:
            raise KeyError(f"dataset {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def load_dataset(name: str, root: Path) -> ImageDataset:
    try:
        loader = _REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(_REGISTRY)}") from err
    return loader(Path(root))


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["ImageDataset", "available_datasets", "load_dataset", "register_dataset"]
