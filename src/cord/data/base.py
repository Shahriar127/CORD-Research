"""Array-based dataset container shared by every loader.

Images are kept as ``uint8`` arrays in NHWC layout. Attacks, splits and the GPU-resident trainer
all operate on these arrays, so a new dataset only needs a loader that returns this container.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageDataset:
    name: str
    x_train: np.ndarray  # uint8 (N, H, W, C) -- the untrusted pool before poisoning
    y_train: np.ndarray  # int64 (N,)
    x_test: np.ndarray  # uint8 (M, H, W, C) -- used only for clean accuracy and ASR
    y_test: np.ndarray  # int64 (M,)
    class_names: tuple[str, ...]
    mean: tuple[float, ...]  # per-channel normalisation statistics in [0, 1] units
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("x_train", "x_test"):
            arr = getattr(self, name)
            if arr.dtype != np.uint8 or arr.ndim != 4:
                raise ValueError(f"{name} must be uint8 NHWC, got {arr.dtype} {arr.shape}")
        if len(self.x_train) != len(self.y_train) or len(self.x_test) != len(self.y_test):
            raise ValueError("images and labels have different lengths")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        h, w, c = self.x_train.shape[1:]
        return int(h), int(w), int(c)

    def subset(self, n_train: int | None, n_test: int | None, rng: np.random.Generator) -> ImageDataset:
        """Uniform random sub-sample (used for smoke tests and CPU development)."""

        def pick(n: int | None, total: int) -> np.ndarray:
            if n is None or n >= total:
                return np.arange(total)
            return np.sort(rng.choice(total, size=n, replace=False))

        tr = pick(n_train, len(self.y_train))
        te = pick(n_test, len(self.y_test))
        return ImageDataset(
            self.name,
            self.x_train[tr],
            self.y_train[tr],
            self.x_test[te],
            self.y_test[te],
            self.class_names,
            self.mean,
            self.std,
        )
