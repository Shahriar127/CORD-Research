"""CORD Stage A: from cross-fitted fused scores to the three calibrated outputs (Section 4.2).

Inputs per fold ``j``: fused scores of ``D_j`` and of the (audited) calibration set ``C_cal``,
both produced by fold model ``f_{-j}``. Outputs:

1. presence test, exact, Bonferroni over folds (Prop. 3);
2. removal set ``R = {i : p_i <= t}``, expected clean share at most ``t`` (Prop. 1(i));
3. proxy set ``F`` = e-BH over the pooled conformal e-values, FDR <= alpha under any
   dependence between folds (Prop. 1(iii)); fold-wise BH and pooled BH are reported alongside.

This module is independent of how scores are produced, so it is shared by the synthetic
studies and the deep-learning pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from cord.config import CalibrationConfig
from cord.stats import (
    benjamini_hochberg,
    conformal_evalues,
    conformal_pvalues,
    e_benjamini_hochberg,
    estimate_poison_rate,
    false_discovery_proportion,
    fisher_statistic,
    higher_criticism_statistic,
    per_class_counts,
    permutation_global_test,
    presence_test,
    selection_floor,
    true_positive_proportion,
)
from cord.stats.presence import PresenceResult


@dataclass(frozen=True)
class CordOutputs:
    """Concatenated over folds in fold order. Masks index the points of ``D``."""

    pvalues: np.ndarray
    evalues: np.ndarray
    fold_id: np.ndarray
    presence: PresenceResult
    removal: np.ndarray
    proxy: np.ndarray
    foldwise_bh: np.ndarray
    pooled_bh: np.ndarray
    rate_estimate: float
    bh_thresholds: tuple[float, ...]


def run_cord(
    fold_scores: Sequence[np.ndarray], cal_scores: Sequence[np.ndarray], cfg: CalibrationConfig
) -> CordOutputs:
    pvals, evals, fold_id, fbh, thresholds = [], [], [], [], []
    for j, (fs, cs) in enumerate(zip(fold_scores, cal_scores, strict=True)):
        p = conformal_pvalues(fs, cs)
        e = conformal_evalues(fs, cs, cfg.alpha_bh)
        pvals.append(p)
        evals.append(e.evalues)
        thresholds.append(e.threshold)
        fbh.append(benjamini_hochberg(p, cfg.alpha))
        fold_id.append(np.full(p.size, j, dtype=np.int16))
    p_all, e_all = np.concatenate(pvals), np.concatenate(evals)
    presence = presence_test(fold_scores, cal_scores, cfg.alpha, cfg.presence_k)
    return CordOutputs(
        pvalues=p_all,
        evalues=e_all,
        fold_id=np.concatenate(fold_id),
        presence=presence,
        removal=p_all <= cfg.removal_level,
        proxy=e_benjamini_hochberg(e_all, cfg.alpha),
        foldwise_bh=np.concatenate(fbh),
        pooled_bh=benjamini_hochberg(p_all, cfg.alpha),
        rate_estimate=estimate_poison_rate(p_all, cfg.storey_lambda),
        bh_thresholds=tuple(thresholds),
    )


def presence_baselines(
    fold_scores: Sequence[np.ndarray],
    cal_scores: Sequence[np.ndarray],
    cfg: CalibrationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fisher and higher-criticism global tests, permutation-calibrated, Bonferroni over folds."""
    out: dict[str, Any] = {}
    J = len(fold_scores)
    for name, stat in (("fisher", fisher_statistic), ("hc", higher_criticism_statistic)):
        ps = [
            permutation_global_test(f, c, stat, cfg.presence_permutations, rng)
            for f, c in zip(fold_scores, cal_scores, strict=True)
        ]
        out[f"presence_{name}_pvalue"] = float(min(1.0, J * min(ps)))
        out[f"presence_{name}_reject"] = bool(min(ps) <= cfg.alpha / J)
    return out


def evaluate_outputs(
    out: CordOutputs,
    is_poison: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    target: int | None,
    fold_scores: Sequence[np.ndarray],
    m_cal_effective: Sequence[int],
    alpha: float,
) -> dict[str, Any]:
    """Ground-truth metrics for one replication (experiments only; never used by the method)."""
    null = ~is_poison
    n_poison = int(is_poison.sum())
    scores = np.concatenate(fold_scores)
    res: dict[str, Any] = {
        "n": int(is_poison.size),
        "n_poison": n_poison,
        "true_rate": float(is_poison.mean()),
        "rate_estimate": out.rate_estimate,
        "auroc": float(roc_auc_score(is_poison, scores)) if 0 < n_poison < is_poison.size else float("nan"),
        "presence_reject": out.presence.reject,
        "presence_pvalue": out.presence.pvalue,
        "presence_exceedances": [f.exceedances for f in out.presence.folds],
        "selection_floor": selection_floor(min(m_cal_effective), alpha),
        "removal_size": int(out.removal.sum()),
        "removal_clean_share": float((out.removal & null).sum() / max(int(null.sum()), 1)),
        "removal_recall": true_positive_proportion(out.removal, null),
    }
    for name, mask in (("proxy_ebh", out.proxy), ("foldwise_bh", out.foldwise_bh), ("pooled_bh", out.pooled_bh)):
        res[f"{name}_n"] = int(mask.sum())
        res[f"{name}_fdp"] = false_discovery_proportion(mask, null)
        res[f"{name}_power"] = true_positive_proportion(mask, null)
    counts = per_class_counts(out.removal, labels, num_classes)
    # Normalise by class size so large classes are not favoured.
    sizes = np.maximum(np.bincount(labels, minlength=num_classes), 1)
    guess = int(np.argmax(counts / sizes))
    res["target_guess"] = guess
    res["target_correct"] = None if target is None else bool(guess == target)
    return res
