"""Assumption A2 as an executable property: every signal and the fusion are permutation-
equivariant on the scored set, and do not depend on the C_tr / C_cal membership of a row."""

from __future__ import annotations

import numpy as np
import pytest

from cord.config import FusionConfig, SignalsConfig
from cord.fusion import fused_scores, pu_fusion
from cord.signals import ScoredSet, compute_signals


def _scored_set(n: int = 400, k: int = 5, d: int = 24, seed: int = 0) -> ScoredSet:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)).astype(np.float32)
    cache = {
        "feat_layer3": rng.standard_normal((n, d)).astype(np.float16),
        "feat_penultimate": np.abs(rng.standard_normal((n, d))).astype(np.float16),
        "logits": logits,
        "pred": logits.argmax(1).astype(np.int16),
        "scaleup_pred": rng.integers(0, k, (5, n)).astype(np.int16),
        "strip_entropy": rng.random(n).astype(np.float32),
        "head_weight": rng.standard_normal((k, d)).astype(np.float32),
        "head_bias": np.zeros(k, np.float32),
    }
    traj = {"p_given": rng.random((6, n)).astype(np.float16)}
    return ScoredSet(cache, traj, labels, k, ssl=rng.standard_normal((n, 16)).astype(np.float16))


def _permute(s: ScoredSet, perm: np.ndarray) -> ScoredSet:
    cache = {}
    for key, v in s.cache.items():
        if key == "scaleup_pred":
            cache[key] = v[:, perm]
        elif key.startswith("head_"):
            cache[key] = v
        else:
            cache[key] = v[perm]
    return ScoredSet(
        cache,
        {"p_given": s.trajectory["p_given"][:, perm]},
        s.labels[perm],
        s.num_classes,
        None if s.ssl is None else s.ssl[perm],
    )


CFG = SignalsConfig(latent_pca_dim=8, knn_k=5)


def test_signals_are_permutation_equivariant() -> None:
    s = _scored_set()
    perm = np.random.default_rng(1).permutation(len(s))
    a = compute_signals(s, CFG)
    b = compute_signals(_permute(s, perm), CFG)
    assert set(a) == set(b) and len(a) >= 10
    for name in a:
        np.testing.assert_allclose(a[name][perm], b[name], rtol=1e-6, atol=1e-8, err_msg=name)


@pytest.mark.parametrize("method", ["hgb", "logistic"])
def test_pu_fusion_is_symmetric_in_the_unlabeled_pool(method: str) -> None:
    """Permuting rows inside D_j ∪ C_cal (keeping C_tr fixed) permutes the fused scores."""
    s = _scored_set(n=500)
    signals = compute_signals(s, CFG)
    ref = np.zeros(len(s), bool)
    ref[-100:] = True  # C_tr rows
    perm = np.concatenate([np.random.default_rng(2).permutation(400), np.arange(400, 500)])
    cfg = FusionConfig(method=method, hgb_max_iter=30)
    a = pu_fusion(signals, ref, cfg, seed=0)
    b = pu_fusion({k: v[perm] for k, v in signals.items()}, ref[perm], cfg, seed=0)
    np.testing.assert_allclose(a[perm], b, rtol=1e-5, atol=1e-6)


def test_pu_fusion_separates_planted_signal() -> None:
    rng = np.random.default_rng(3)
    n_pool, n_ref = 3000, 1000
    poison = np.zeros(n_pool + n_ref, bool)
    poison[:150] = True
    sig = {"a": rng.standard_normal(n_pool + n_ref) + 4 * poison, "b": rng.standard_normal(n_pool + n_ref)}
    ref = np.zeros(n_pool + n_ref, bool)
    ref[n_pool:] = True
    scores = fused_scores(sig, ref, FusionConfig(hgb_max_iter=50), seed=0)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(poison[:n_pool], scores["cord_pu"][:n_pool]) > 0.95
    assert {"mean_rank", "max_rank", "single:a", "single:b"} <= set(scores)


def test_clean_reference_recovers_pu_power_and_guards_misuse() -> None:
    """A reference contaminated at the pool rate (mode T1) carries little PU signal; dropping its
    poison from the fit restores it. Only reference rows may be excluded."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    n_pool, n_ref, rate = 4000, 1000, 0.05
    poison = rng.random(n_pool + n_ref) < rate
    sig = {"a": rng.standard_normal(n_pool + n_ref) + 2.5 * poison, "b": rng.standard_normal(n_pool + n_ref)}
    ref = np.zeros(n_pool + n_ref, bool)
    ref[n_pool:] = True
    cfg = FusionConfig(method="logistic")
    contaminated = roc_auc_score(poison[:n_pool], pu_fusion(sig, ref, cfg, 0)[:n_pool])
    clean = roc_auc_score(poison[:n_pool], pu_fusion(sig, ref, cfg, 0, exclude_from_fit=ref & poison)[:n_pool])
    assert clean > contaminated
    with pytest.raises(ValueError, match="reference"):
        pu_fusion(sig, ref, cfg, 0, exclude_from_fit=~ref & poison)


def test_ranking_score_options() -> None:
    from cord.fusion import mean_rank, ranking_score

    sig = {"a": np.arange(5.0), "b": np.arange(5.0)[::-1]}
    np.testing.assert_allclose(ranking_score(sig, "mean_rank"), mean_rank(sig))
    np.testing.assert_allclose(ranking_score(sig, "single:a"), sig["a"])
    with pytest.raises(KeyError):
        ranking_score(sig, "single:missing")


def test_orientation_recovers_reversed_signal_and_keeps_a2() -> None:
    """A poisoned minority in the *lower* tail of a signal is oriented upwards, and the oriented
    mean rank is permutation-equivariant on the scored set (A2)."""
    from sklearn.metrics import roc_auc_score

    from cord.fusion import orientation_signs, oriented_mean_rank

    rng = np.random.default_rng(5)
    n = 3000
    poison = rng.random(n) < 0.05
    sig = {
        "up": rng.standard_normal(n) + 3.0 * poison,  # designed direction
        "down": rng.standard_normal(n) - 3.0 * poison,  # reversed (e.g. confidence, backdoor not learned)
    }
    assert orientation_signs(sig) == {"up": 1.0, "down": -1.0}
    score = oriented_mean_rank(sig)
    assert roc_auc_score(poison, score) > 0.95
    perm = rng.permutation(n)
    np.testing.assert_allclose(oriented_mean_rank({k: v[perm] for k, v in sig.items()}), score[perm])
