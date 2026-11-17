"""Level 1 output: exact presence test of whether a dataset was attacked (Section 4.2, Prop. 3).

Exceedance statistic: ``T_j^{(k)}`` = number of fold scores strictly above the k-th largest
calibration score. If fold and calibration scores are exchangeable (no poison in either), then
``T^{(k)} >= t`` iff at most ``k - 1`` calibration points lie among the top ``t + k - 1`` pooled
scores, so

    P(T^{(k)} >= t) = HypergeomCDF(k - 1; N + m, m, t + k - 1),

which for ``k = 1`` reduces to ``C(N, t) / C(N + m, t)`` (eq. 4). The k > 1 variant tolerates up to
``k - 1`` contaminated calibration points at some loss of power (Stoimenova and Balakrishnan).

Baselines (Fisher combination, higher criticism) are calibrated by an exact Monte-Carlo
permutation test over the pooled scores, which is valid under the same exchangeability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from cord.stats.conformal import conformal_pvalues


def exceedance_count(fold_scores: ArrayLike, cal_scores: ArrayLike, k: int = 1) -> int:
    """Number of fold scores strictly greater than the k-th largest calibration score."""
    cal = np.asarray(cal_scores, dtype=np.float64).reshape(-1)
    fold = np.asarray(fold_scores, dtype=np.float64).reshape(-1)
    if not 1 <= k <= cal.size:
        raise ValueError(f"k must lie in [1, m_cal={cal.size}]")
    kth_largest = np.partition(cal, cal.size - k)[cal.size - k]
    return int((fold > kth_largest).sum())


def exceedance_pvalue(t: int, n_fold: int, n_cal: int, k: int = 1) -> float:
    """Exact null tail probability ``P(T^{(k)} >= t)``."""
    if t <= 0:
        return 1.0
    if t > n_fold:
        return 0.0
    log_p = stats.hypergeom.logcdf(k - 1, n_fold + n_cal, n_cal, t + k - 1)
    return float(min(1.0, np.exp(log_p)))


@dataclass(frozen=True)
class FoldPresence:
    n_fold: int
    n_cal: int
    exceedances: int
    pvalue: float


@dataclass(frozen=True)
class PresenceResult:
    """Bonferroni combination over folds: reject when ``min_j pi_j <= alpha / J``."""

    folds: tuple[FoldPresence, ...]
    alpha: float
    k: int
    reject: bool
    pvalue: float  # Bonferroni-adjusted, min(1, J * min_j pi_j)
    details: dict[str, float] = field(default_factory=dict)


def presence_test(
    fold_scores: Sequence[ArrayLike],
    cal_scores: Sequence[ArrayLike],
    alpha: float = 0.05,
    k: int = 1,
) -> PresenceResult:
    """Exact cross-fitted presence test.

    Args:
        fold_scores: fused scores of each fold's points (scored by that fold's model).
        cal_scores: calibration scores for each fold (each fold scores C_cal with its own model,
            and the audit may differ by fold).
        alpha: family-wise level of the dataset-level decision.
        k: use the k-th largest calibration score (k=1 is the classical exceedance test).
    """
    if len(fold_scores) != len(cal_scores) or not fold_scores:
        raise ValueError("need one calibration score vector per fold")
    J = len(fold_scores)
    folds = []
    for fs, cs in zip(fold_scores, cal_scores, strict=True):
        fs_arr = np.asarray(fs, dtype=np.float64).reshape(-1)
        cs_arr = np.asarray(cs, dtype=np.float64).reshape(-1)
        t = exceedance_count(fs_arr, cs_arr, k)
        folds.append(FoldPresence(fs_arr.size, cs_arr.size, t, exceedance_pvalue(t, fs_arr.size, cs_arr.size, k)))
    min_p = min(f.pvalue for f in folds)
    return PresenceResult(
        folds=tuple(folds),
        alpha=alpha,
        k=k,
        reject=bool(min_p <= alpha / J),
        pvalue=float(min(1.0, J * min_p)),
    )


# ---------------------------------------------------------------------------------------------
# Permutation-calibrated global-null baselines (Fisher, higher criticism)
# ---------------------------------------------------------------------------------------------


def fisher_statistic(pvalues: ArrayLike) -> float:
    p = np.asarray(pvalues, dtype=np.float64)
    return float(-2.0 * np.log(p).sum())


def higher_criticism_statistic(pvalues: ArrayLike, alpha0: float = 0.5) -> float:
    """Donoho–Jin HC* over the smallest ``alpha0`` fraction of p-values."""
    p = np.sort(np.asarray(pvalues, dtype=np.float64))
    n = p.size
    i = np.arange(1, n + 1)
    keep = (i <= max(1, int(alpha0 * n))) & (p < 1.0)
    if not keep.any():
        return 0.0
    pk, ik = p[keep], i[keep]
    hc = np.sqrt(n) * (ik / n - pk) / np.sqrt(pk * (1.0 - pk))
    return float(hc.max())


def permutation_global_test(
    fold_scores: ArrayLike,
    cal_scores: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Exact Monte-Carlo permutation p-value of ``statistic(conformal p-values of fold)``.

    Under the global null all pooled scores are exchangeable, so randomly re-splitting the
    pool into fold/calibration parts yields draws from the null distribution of the statistic.
    """
    fold = np.asarray(fold_scores, dtype=np.float64).reshape(-1)
    cal = np.asarray(cal_scores, dtype=np.float64).reshape(-1)
    observed = statistic(conformal_pvalues(fold, cal))
    pooled = np.concatenate([fold, cal])
    n = fold.size
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled)
        if statistic(conformal_pvalues(perm[:n], perm[n:])) >= observed:
            exceed += 1
    return (1.0 + exceed) / (1.0 + n_permutations)
