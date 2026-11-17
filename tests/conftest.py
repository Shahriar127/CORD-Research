from __future__ import annotations

import numpy as np
import pytest

from cord.data import ImageDataset


@pytest.fixture
def tiny_dataset() -> ImageDataset:
    """A synthetic 10-class, 16x16 RGB dataset (no download needed)."""
    rng = np.random.default_rng(0)
    n, m = 600, 200
    return ImageDataset(
        "tiny",
        rng.integers(0, 256, (n, 16, 16, 3), dtype=np.uint8),
        rng.integers(0, 10, n).astype(np.int64),
        rng.integers(0, 256, (m, 16, 16, 3), dtype=np.uint8),
        rng.integers(0, 10, m).astype(np.int64),
        tuple(str(i) for i in range(10)),
        (0.5, 0.5, 0.5),
        (0.25, 0.25, 0.25),
    )
