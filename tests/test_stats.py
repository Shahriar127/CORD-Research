"""Statistical core: exactness against naive implementations and Monte-Carlo validity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cord.stats import (
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_evalues,
    conformal_pvalues,
    e_benjamini_hochberg,
    exceedance_count,
    exceedance_pvalue,
    fisher_statistic,
    permutation_global_test,
    presence_min_exceedances,
    presence_test,
    presence_threshold_approx,
    selection_floor,
)

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


def naive_bh(p: np.ndarray, alpha: float) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    k = 0
    for i in range(n):
        if p[order[i]] <= alpha * (i + 1) / n:
            k = i + 1
    sel = np.zeros(n, bool)
    sel[order[:k]] = True
    return sel


@given(st.lists(finite, min_size=1, max_size=50), st.lists(finite, min_size=1, max_size=50))
def test_conformal_pvalues_match_definition(test: list[float], cal: list[float]) -> None:
    p = conformal_pvalues(test, cal)
    expected = [(1 + sum(c >= t for c in cal)) / (1 + len(cal)) for t in test]
    np.testing.assert_allclose(p, expected)


@given(st.lists(st.floats(0, 1), min_size=1, max_size=200), st.floats(0.01, 0.5))
@settings(max_examples=200)
def test_bh_matches_naive(p: list[float], alpha: float) -> None:
    arr = np.asarray(p)
    sel = benjamini_hochberg(arr, alpha)
    naive = naive_bh(arr, alpha)
    # Ties at the cut can only add hypotheses with p equal to an accepted one.
    assert sel.sum() == naive.sum() or np.all(arr[sel & ~naive] <= alpha * sel.sum() / len(arr))
    assert np.all(naive <= sel)


@given(st.lists(st.floats(0, 100), min_size=1, max_size=100), st.floats(0.01, 0.5))
def test_ebh_equals_bh_on_reciprocal(e: list[float], alpha: float) -> None:
    """e-BH on e equals BH on p = 1/e (Wang and Ramdas, 2022)."""
    arr = np.asarray(e)
    p = np.where(arr > 0, 1.0 / np.maximum(arr, 1e-300), np.inf)
    np.testing.assert_array_equal(e_benjamini_hochberg(arr, alpha), benjamini_hochberg(np.minimum(p, 1e300), alpha))


def test_by_is_more_conservative() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=900), rng.uniform(0, 1e-3, size=100)])
    assert benjamini_yekutieli(p, 0.1).sum() <= benjamini_hochberg(p, 0.1).sum()


def test_pvalues_superuniform_under_exchangeability() -> None:
    rng = np.random.default_rng(1)
    m, t = 200, np.array([0.01, 0.05, 0.1, 0.3])
    hits = np.zeros_like(t)
    reps = 4000
    for _ in range(reps):
        p = conformal_pvalues(rng.standard_normal(1), rng.standard_normal(m))[0]
        hits += p <= t
    assert np.all(hits / reps <= t + 3 * np.sqrt(t * (1 - t) / reps))


def test_evalues_null_mass_and_structure() -> None:
    """sum over nulls of E[e_i] <= n, and all nonzero e-values of a fold are equal."""
    rng = np.random.default_rng(2)
    n, m, reps = 300, 200, 3000
    total = 0.0
    for _ in range(reps):
        test = rng.standard_normal(n)
        test[:30] += 4.0
        res = conformal_evalues(test, rng.standard_normal(m), 0.1)
        nonzero = res.evalues[res.evalues > 0]
        assert nonzero.size == 0 or np.allclose(nonzero, nonzero[0])
        total += res.evalues[30:].sum()
    assert total / reps <= n


def test_ebh_fdr_control_monte_carlo() -> None:
    rng = np.random.default_rng(3)
    n, m, alpha, reps = 1000, 500, 0.1, 400
    fdps = []
    for _ in range(reps):
        test = rng.standard_normal(n)
        test[:80] += 3.5
        e = conformal_evalues(test, rng.standard_normal(m), 0.09).evalues
        sel = e_benjamini_hochberg(e, alpha)
        fdps.append(sel[80:].sum() / max(sel.sum(), 1))
    assert np.mean(fdps) <= alpha + 3 * np.std(fdps) / np.sqrt(reps)


def test_exceedance_pvalue_closed_form() -> None:
    N, m = 500, 100
    for t in range(20):
        expected = math.comb(N, t) / math.comb(N + m, t)
        assert exceedance_pvalue(t, N, m) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("k", [1, 3])
def test_presence_test_level_under_null(k: int) -> None:
    rng = np.random.default_rng(4 + k)
    alpha, reps = 0.1, 3000
    rejections = 0
    for _ in range(reps):
        folds = [rng.standard_normal(400) for _ in range(2)]
        cal = rng.standard_normal(150)
        rejections += presence_test(folds, [cal, cal], alpha, k=k).reject
    assert rejections / reps <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)


def test_exceedance_pvalue_exact_distribution() -> None:
    """P(T >= t) from simulation agrees with the hypergeometric formula (k = 2)."""
    rng = np.random.default_rng(6)
    N, m, k, reps = 60, 20, 2, 20000
    counts = np.array([exceedance_count(rng.standard_normal(N), rng.standard_normal(m), k) for _ in range(reps)])
    for t in (1, 3, 6):
        assert (counts >= t).mean() == pytest.approx(exceedance_pvalue(t, N, m, k), abs=0.015)


def test_floors_reproduce_table5() -> None:
    expected_sel = {500: 0.040, 1000: 0.020, 2000: 0.010, 5000: 0.0040, 20000: 0.0010}
    expected_pres = {500: 0.0074, 1000: 0.0037, 2000: 0.0018, 5000: 0.0007, 20000: 0.0002}
    for m, v in expected_sel.items():
        assert selection_floor(m, 0.05) == pytest.approx(v, rel=0.01)
        assert presence_threshold_approx(m, 0.05, 2) == pytest.approx(expected_pres[m], abs=5e-5)
    # The exact threshold approaches ln(J / alpha) / m_cal as N_j grows.
    big = 10**7
    assert presence_min_exceedances(big, 1000, 0.05, 2) / big == pytest.approx(math.log(40) / 1000, rel=0.01)


def test_permutation_global_test_is_valid() -> None:
    rng = np.random.default_rng(7)
    ps = [
        permutation_global_test(rng.standard_normal(100), rng.standard_normal(50), fisher_statistic, 50, rng)
        for _ in range(300)
    ]
    assert np.mean(np.asarray(ps) <= 0.1) <= 0.1 + 3 * np.sqrt(0.09 / 300)


def test_random_tiebreak_keeps_order_and_breaks_ties() -> None:
    from cord.pipeline import random_tiebreak

    rng = np.random.default_rng(0)
    fold, cal = np.array([3.0, 1.0, 2.0, 2.0]), np.array([2.0, 0.0, 5.0])
    f, c = random_tiebreak(fold, cal, rng)
    pooled_old, pooled_new = np.concatenate([fold, cal]), np.concatenate([f, c])
    assert np.unique(pooled_new).size == pooled_new.size  # all distinct
    # strict order of distinct original scores is preserved
    for a in range(pooled_old.size):
        for b in range(pooled_old.size):
            if pooled_old[a] < pooled_old[b]:
                assert pooled_new[a] < pooled_new[b]


def test_tiebreak_keeps_fdr_control_on_discrete_scores() -> None:
    """Discrete scores with heavy ties: fold-wise BH after random tie-breaking keeps FDR <= alpha * pi0."""
    from cord.config import CalibrationConfig
    from cord.pipeline import run_cord

    rng = np.random.default_rng(1)
    cfg = CalibrationConfig(alpha=0.2, alpha_bh=0.2, presence_k=1, n_resamples=1)
    n0, n1, m, reps = 150, 50, 60, 2000
    fdp = np.empty(reps)
    for r in range(reps):
        fold = np.concatenate([rng.integers(0, 6, n0), rng.integers(4, 6, n1)]).astype(float)
        cal = rng.integers(0, 6, m).astype(float)
        sel = run_cord([fold], [cal], cfg, rng).foldwise_bh
        fdp[r] = sel[:n0].sum() / max(sel.sum(), 1)
    assert fdp.mean() <= 0.2 * n0 / (n0 + n1) + 3 * fdp.std() / np.sqrt(reps)
