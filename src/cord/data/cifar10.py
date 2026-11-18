"""CIFAR-10 from the official "python version" archive, verified by MD5 and loaded without
executing pickle code."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cord.data.base import ImageDataset
from cord.data.safe_pickle import safe_load_pickle

_DIRNAME = "cifar-10-batches-py"
_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _resolve_root(root: Path) -> Path:
    root = Path(root).expanduser()
    for candidate in (root, root / _DIRNAME):
        if (candidate / "data_batch_1").is_file():
            return candidate
    raise FileNotFoundError(
        f"CIFAR-10 batches not found under {root}. Download "
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract it there."
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    entry = safe_load_pickle(path)
    data = np.asarray(entry[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(entry[b"labels"], dtype=np.int64)
    return np.ascontiguousarray(data), labels


def load_cifar10(root: Path, verify: bool = True) -> ImageDataset:
    base = _resolve_root(root)
    if verify:
        for name, expected in _MD5.items():
            got = _md5(base / name)
            if got != expected:
                raise ValueError(f"checksum mismatch for {name}: {got} != {expected}")
    parts = [_read_batch(base / f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = _read_batch(base / "test_batch")
    return ImageDataset("cifar10", x_train, y_train, x_test, y_test, CLASSES, MEAN, STD)
