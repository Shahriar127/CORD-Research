"""By-products: poisoning-rate estimate (Storey, 2002) and per-class breakdown of selections."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def storey_null_proportion(pvalues: ArrayLike, lam: float = 0.5) -> float:
    """``pi0 = #{p > lambda} / (n (1 - lambda))``, clipped to [0, 1]."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 1.0
    return float(np.clip((p > lam).sum() / (p.size * (1.0 - lam)), 0.0, 1.0))


def estimate_poison_rate(pvalues: ArrayLike, lam: float = 0.5) -> float:
    return 1.0 - storey_null_proportion(pvalues, lam)


def per_class_counts(selected: ArrayLike, labels: ArrayLike, num_classes: int) -> np.ndarray:
    """Count of selected samples per given label; the arg-max points to the target class."""
    sel = np.asarray(selected, dtype=bool)
    lab = np.asarray(labels, dtype=np.int64)
    return np.bincount(lab[sel], minlength=num_classes)
