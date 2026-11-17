"""Multiple testing procedures: Benjamini–Hochberg, Benjamini–Yekutieli and e-BH."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]


def _step_up(sorted_ok: NDArray[np.bool_]) -> int:
    """Largest 1-based index k with ``sorted_ok[k-1]`` true, or 0."""
    hits = np.flatnonzero(sorted_ok)
    return int(hits[-1] + 1) if hits.size else 0


def benjamini_hochberg(pvalues: ArrayLike, alpha: float) -> BoolArray:
    """BH step-up procedure. Returns the rejection mask.

    Rejects every hypothesis with ``p_i <= alpha * k / n`` where
    ``k = max{k : p_(k) <= alpha k / n}``; this handles ties exactly.
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    p_sorted = np.sort(p)
    k = _step_up(p_sorted <= alpha * np.arange(1, n + 1) / n)
    if k == 0:
        return np.zeros(n, dtype=bool)
    return p <= alpha * k / n


def benjamini_yekutieli(pvalues: ArrayLike, alpha: float) -> BoolArray:
    """BY procedure: BH at ``alpha / H_n``; valid under arbitrary dependence."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    harmonic = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    return benjamini_hochberg(p, alpha / harmonic)


def e_benjamini_hochberg(evalues: ArrayLike, alpha: float) -> BoolArray:
    """e-BH (Wang and Ramdas, 2022). Controls FDR at ``alpha`` under arbitrary dependence.

    With ``e_(1) >= ... >= e_(n)``, selects the hypotheses with ``e_i >= n / (alpha k)`` where
    ``k = max{k : e_(k) >= n / (alpha k)}``.
    """
    e = np.asarray(evalues, dtype=np.float64).reshape(-1)
    n = e.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if np.any(e < 0):
        raise ValueError("e-values must be non-negative")
    e_sorted = -np.sort(-e)
    k = _step_up(e_sorted >= n / (alpha * np.arange(1, n + 1)))
    if k == 0:
        return np.zeros(n, dtype=bool)
    return e >= n / (alpha * k)


def false_discovery_proportion(selected: ArrayLike, is_null: ArrayLike) -> float:
    """``|F ∩ H0| / max(|F|, 1)`` (eq. 1)."""
    sel = np.asarray(selected, dtype=bool)
    null = np.asarray(is_null, dtype=bool)
    return float((sel & null).sum() / max(int(sel.sum()), 1))


def true_positive_proportion(selected: ArrayLike, is_null: ArrayLike) -> float:
    """Share of non-null hypotheses selected (power); 0 when there are no non-nulls."""
    sel = np.asarray(selected, dtype=bool)
    alt = ~np.asarray(is_null, dtype=bool)
    return float((sel & alt).sum() / max(int(alt.sum()), 1))
