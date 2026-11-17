"""Score fusion.

* :func:`pu_fusion`: AdaDetect-style positive-unlabeled fusion (Section 4.1). In fold ``j`` a
  classifier separates ``C_tr`` (label 0) from the unordered pool ``D_j ∪ C_cal`` (label 1) and
  its output on ``D_j ∪ C_cal`` is the fused score. The fit is a symmetric function of
  ``D_j ∪ C_cal``, so A2 holds whatever the classifier; its quality affects power only.
* Naive baselines (mean / max of within-set ranks) and single signals, for RQ3.
* :func:`exchangeability_diagnostic`: cross-validated AUROC of ``C_tr`` vs ``C_cal`` with a
  permutation p-value. A small p-value flags a hold-out that is not internally exchangeable.

Tree ensembles produce tied scores; ties are broken by the mean-rank fusion (also symmetric),
which preserves validity and recovers resolution at the top of the ranking where the
presence test and BH operate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cord.config import FusionConfig


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Average ranks scaled to (0, 1]; a symmetric function of the set."""
    return rankdata(x, method="average") / x.size


def mean_rank(signals: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.mean([rank_normalise(v) for v in signals.values()], axis=0)


def max_rank(signals: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.max([rank_normalise(v) for v in signals.values()], axis=0)


def tail_skewness(x: np.ndarray, q: float = 0.01) -> float:
    """Quantile tail skewness ``((x_{1-q} - med) - (med - x_q)) / (x_{1-q} - x_q)`` in [-1, 1]."""
    lo, med, hi = np.quantile(x, [q, 0.5, 1.0 - q])
    return float(((hi - med) - (med - lo)) / max(hi - lo, 1e-12))


def orientation_signs(signals: Mapping[str, np.ndarray], q: float = 0.01) -> dict[str, float]:
    """Orient each signal towards its heavier tail.

    A poisoned minority shifted in one direction makes that tail heavy, whatever the designed
    direction of the signal (e.g. confidence is *low* on held-out poison when the backdoor was
    not learned). The sign is a symmetric function of the scored set, so A2 is kept. Signals
    with no tail asymmetry keep their designed direction (+1).
    """
    return {k: (-1.0 if tail_skewness(np.asarray(v, np.float64), q) < 0 else 1.0) for k, v in signals.items()}


def oriented_mean_rank(signals: Mapping[str, np.ndarray]) -> np.ndarray:
    """Mean of within-set ranks after orienting every signal by :func:`orientation_signs`."""
    signs = orientation_signs(signals)
    return np.mean([rank_normalise(signs[k] * np.asarray(v, np.float64)) for k, v in signals.items()], axis=0)


def ranking_score(signals: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """A fusion-free score (symmetric in the scored set) used to rank C_tr for its audit."""
    if name == "oriented_mean_rank":
        return oriented_mean_rank(signals)
    if name == "mean_rank":
        return mean_rank(signals)
    if name == "max_rank":
        return max_rank(signals)
    if name.startswith("single:") and name[len("single:") :] in signals:
        return np.asarray(signals[name[len("single:") :]], dtype=np.float64)
    raise KeyError(
        f"unknown C_tr ranking {name!r}; use oriented_mean_rank, mean_rank, max_rank or single:<{', '.join(signals)}>"
    )


def break_ties(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Lexicographic score: order by ``primary``, then by ``secondary`` within ties."""
    dense = rankdata(primary, method="dense").astype(np.float64)
    sec = rank_normalise(secondary)  # in (0, 1]
    return dense + 0.999 * (sec - sec.min()) / max(1.0 - sec.min(), 1e-12) if sec.size else dense


def _classifier(cfg: FusionConfig, seed: int) -> object:
    if cfg.method == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(
        max_iter=cfg.hgb_max_iter,
        learning_rate=cfg.hgb_learning_rate,
        max_leaf_nodes=cfg.hgb_max_leaf_nodes,
        min_samples_leaf=cfg.hgb_min_samples_leaf,
        l2_regularization=cfg.hgb_l2,
        early_stopping=False,
        random_state=seed % (2**32),
    )


def pu_fusion(
    signals: Mapping[str, np.ndarray],
    reference_mask: np.ndarray,
    cfg: FusionConfig,
    seed: int,
    exclude_from_fit: np.ndarray | None = None,
) -> np.ndarray:
    """Fused scores for every row; rows with ``reference_mask`` (C_tr) are the negative class.

    Classes are re-weighted to equal total weight. Only scores of non-reference rows are used
    downstream. ``exclude_from_fit`` drops reference rows from the fit (e.g. poison found by an
    audit of C_tr); since C_tr is independent of the scored rows, this keeps A2.
    """
    X = np.column_stack(list(signals.values()))
    y = (~reference_mask).astype(np.int64)
    fit = np.ones(len(y), bool) if exclude_from_fit is None else ~exclude_from_fit
    if exclude_from_fit is not None and np.any(exclude_from_fit & ~reference_mask):
        raise ValueError("only reference (C_tr) rows may be excluded from the fit")
    n1, n0 = int(y[fit].sum()), int((1 - y[fit]).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("PU fusion needs both reference and unlabeled rows")
    weights = np.where(y[fit] == 1, 0.5 / n1, 0.5 / n0) * int(fit.sum())
    clf = _classifier(cfg, seed)
    if cfg.method == "logistic":
        clf.fit(X[fit], y[fit], logisticregression__sample_weight=weights)  # type: ignore[attr-defined]
    else:
        clf.fit(X[fit], y[fit], sample_weight=weights)  # type: ignore[attr-defined]
    raw = clf.decision_function(X)  # type: ignore[attr-defined]
    return break_ties(raw, mean_rank(signals))


def fused_scores(
    signals: Mapping[str, np.ndarray],
    reference_mask: np.ndarray,
    cfg: FusionConfig,
    seed: int,
    exclude_from_fit: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """``{method: scores}`` for CORD's PU fusion and the configured baselines."""
    out = {"cord_pu": pu_fusion(signals, reference_mask, cfg, seed, exclude_from_fit)}
    if "oriented_mean_rank" in cfg.baselines:
        out["oriented_mean_rank"] = oriented_mean_rank(signals)
    if "mean_rank" in cfg.baselines:
        out["mean_rank"] = mean_rank(signals)
    if "max_rank" in cfg.baselines:
        out["max_rank"] = break_ties(max_rank(signals), mean_rank(signals))
    if "single" in cfg.baselines:
        out.update({f"single:{k}": v for k, v in signals.items()})
    return out


def exchangeability_diagnostic(
    signals: Mapping[str, np.ndarray],
    tr_mask: np.ndarray,
    cal_mask: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """CV-AUROC of a classifier separating C_tr from C_cal, with a permutation p-value."""
    X = np.column_stack(list(signals.values()))
    rows = tr_mask | cal_mask
    Xh, yh = X[rows], cal_mask[rows].astype(np.int64)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))

    def cv_auc(labels: np.ndarray) -> float:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng.integers(2**31)))
        pred = cross_val_predict(model, Xh, labels, cv=cv, method="decision_function")
        return float(roc_auc_score(labels, pred))

    observed = cv_auc(yh)
    null = [cv_auc(rng.permutation(yh)) for _ in range(n_permutations)]
    pval = (1 + sum(v >= observed for v in null)) / (1 + n_permutations)
    return {"auroc": observed, "pvalue": float(pval)}
