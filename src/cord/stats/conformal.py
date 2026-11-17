"""Conformal p-values and conformal e-values.

Convention: larger scores are more suspicious. For a test score ``s`` and calibration scores
``c_1..c_m`` the conformal p-value is (eq. 2 of the proposal)

    p(s) = (1 + #{k : c_k >= s}) / (1 + m).

Ties are counted against the test point, which keeps the p-value super-uniform for clean
points under exchangeability (Bates et al., 2023) without randomisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cord.stats.multiple_testing import benjamini_hochberg

FloatArray = NDArray[np.float64]


def _as_1d(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def count_at_least(test_scores: ArrayLike, cal_scores: ArrayLike) -> NDArray[np.int64]:
    """Number of calibration scores ``>=`` each test score (vectorised, O((n+m) log m))."""
    test = _as_1d(test_scores, "test_scores")
    cal = np.sort(_as_1d(cal_scores, "cal_scores"))
    return (cal.size - np.searchsorted(cal, test, side="left")).astype(np.int64)


def conformal_pvalues(test_scores: ArrayLike, cal_scores: ArrayLike) -> FloatArray:
    """Marginal conformal p-values of ``test_scores`` against ``cal_scores``."""
    cal = _as_1d(cal_scores, "cal_scores")
    if cal.size == 0:
        raise ValueError("calibration set is empty")
    return np.asarray((1.0 + count_at_least(test_scores, cal)) / (1.0 + cal.size), dtype=np.float64)


@dataclass(frozen=True)
class ConformalEValues:
    """Conformal e-values of one fold, built from a BH run at level ``alpha_bh`` (Bashari et al., 2023).

    Attributes:
        evalues: e-value of every test point; within a fold all nonzero e-values are equal.
        threshold: score threshold ``T`` (smallest score among BH rejections, ``inf`` if none).
        n_bh_rejections: number of BH rejections that defined ``T``.
    """

    evalues: FloatArray
    threshold: float
    n_bh_rejections: int


def conformal_evalues(test_scores: ArrayLike, cal_scores: ArrayLike, alpha_bh: float) -> ConformalEValues:
    """Conformal e-values (eq. 3): ``e_i = (1+m) 1{s_i >= T} / (1 + #{c >= T})``.

    ``T`` is the smallest test score among the BH rejections at level ``alpha_bh``. With no
    rejection every e-value is zero. The e-values satisfy ``sum_{i null} E[e_i] <= n`` under
    exchangeability, which is the condition e-BH needs (Wang and Ramdas, 2022).
    """
    if not 0.0 < alpha_bh < 1.0:
        raise ValueError("alpha_bh must lie in (0, 1)")
    test = _as_1d(test_scores, "test_scores")
    cal = _as_1d(cal_scores, "cal_scores")
    pvals = conformal_pvalues(test, cal)
    rejected = benjamini_hochberg(pvals, alpha_bh)
    n_rej = int(rejected.sum())
    if n_rej == 0:
        return ConformalEValues(np.zeros_like(test), float("inf"), 0)
    threshold = float(test[rejected].min())
    n_cal_above = int((cal >= threshold).sum())
    e_value = (1.0 + cal.size) / (1.0 + n_cal_above)
    evalues = np.where(test >= threshold, e_value, 0.0)
    return ConformalEValues(evalues, threshold, n_rej)
