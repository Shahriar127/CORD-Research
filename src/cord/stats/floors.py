"""Detection floors: selection floor (Prop. 2, eq. 5) and presence threshold (Prop. 3, eq. 6)."""

from __future__ import annotations

import math

from cord.stats.presence import exceedance_pvalue


def selection_floor(m_cal: int, alpha: float) -> float:
    """``rho_sel = 1 / (alpha (m_cal + 1))``: below this selected fraction BH/e-BH select nothing."""
    return 1.0 / (alpha * (m_cal + 1))


def presence_threshold_approx(m_cal: int, alpha: float, n_folds: int = 2) -> float:
    """Large-fold approximation ``rho_pres ≈ ln(J / alpha) / m_cal`` (regime ``N_j >> m_cal``)."""
    return math.log(n_folds / alpha) / m_cal


def presence_min_exceedances(n_fold: int, m_cal: int, alpha: float, n_folds: int = 2, k: int = 1) -> int:
    """Smallest exceedance count ``t`` for which the fold p-value is ``<= alpha / J`` (exact)."""
    level = alpha / n_folds
    lo, hi = 1, n_fold
    if exceedance_pvalue(hi, n_fold, m_cal, k) > level:
        raise ValueError("presence test cannot reject at this level for any exceedance count")
    while lo < hi:  # p-value is non-increasing in t
        mid = (lo + hi) // 2
        if exceedance_pvalue(mid, n_fold, m_cal, k) <= level:
            hi = mid
        else:
            lo = mid + 1
    return lo


def presence_threshold_exact(n_fold: int, m_cal: int, alpha: float, n_folds: int = 2, k: int = 1) -> float:
    """Exact presence threshold as a rate of high-scoring poison within a fold."""
    return presence_min_exceedances(n_fold, m_cal, alpha, n_folds, k) / n_fold
