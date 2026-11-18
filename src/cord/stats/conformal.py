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
        threshold: stopping threshold ``T`` (first pooled score, scanning upwards, at which the BH
            condition holds; ``inf`` if BH rejects nothing). It may be a calibration score.
        n_bh_rejections: number of BH rejections; the nonzero e-values are exactly these points.
    """

    evalues: FloatArray
    threshold: float
    n_bh_rejections: int


def conformal_evalues(test_scores: ArrayLike, cal_scores: ArrayLike, alpha_bh: float) -> ConformalEValues:
    """Conformal e-values: ``e_i = (1+m) 1{s_i >= T} / (1 + #{c >= T})``.

    ``T`` is the first value of the *pooled* scores (test and calibration), scanning upwards, at
    which the BH condition ``(1 + #{c >= t}) / (1 + m) <= alpha_bh * #{test >= t} / n`` holds.
    ``{i : s_i >= T}`` is exactly the BH rejection set at level ``alpha_bh``, but ``#{c >= T}`` is
    counted at the stopping value, which may lie below the smallest rejected test score.

    This is what makes the e-values valid: ``T`` is a stopping time of the upward scan, so
    ``sum_{i null} E[e_i] <= n`` under exchangeability (``proofs/main.tex``, Lemmas 4-5), the
    condition e-BH needs (Wang and Ramdas, 2022). Counting at the smallest rejected test score
    instead (proposal v3.1, eq. 3) gives the same selections but larger e-values, and simulation
    shows the null mass can then exceed ``n`` by 20% with few calibration points.
    """
    if not 0.0 < alpha_bh < 1.0:
        raise ValueError("alpha_bh must lie in (0, 1)")
    test = _as_1d(test_scores, "test_scores")
    cal = _as_1d(cal_scores, "cal_scores")
    m, n = cal.size, test.size
    pvals = conformal_pvalues(test, cal)
    rejected = benjamini_hochberg(pvals, alpha_bh)
    n_rej = int(rejected.sum())
    if n_rej == 0:
        return ConformalEValues(np.zeros_like(test), float("inf"), 0)

    s_min = float(test[rejected].min())  # smallest rejected test score; #{test >= s_min} == n_rej
    below = test[test < s_min]
    s_prev = float(below.max()) if below.size else -np.inf
    # No test score lies in (s_prev, s_min), so #{test >= t} == n_rej there, while #{c >= t} falls
    # as t rises: the first crossing of the BH condition is at a calibration score in that gap or
    # at s_min itself (where BH guarantees it holds).
    cal_sorted = np.sort(cal)
    candidates = np.concatenate([cal_sorted[(cal_sorted > s_prev) & (cal_sorted < s_min)], [s_min]])
    n_cal_above = m - np.searchsorted(cal_sorted, candidates, side="left")
    holds = (1.0 + n_cal_above) / (1.0 + m) <= alpha_bh * n_rej / n  # same arithmetic as BH
    first = int(np.argmax(holds))  # holds[-1] is True by construction
    threshold = float(candidates[first])
    e_value = (1.0 + m) / (1.0 + float(n_cal_above[first]))
    evalues = np.where(test >= threshold, e_value, 0.0)
    return ConformalEValues(evalues, threshold, n_rej)
