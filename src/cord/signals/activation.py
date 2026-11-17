"""Activation dominance (lineage: AGPD, reverse backdoor distillation).

* ``act_topk_share``: share of the penultimate activation mass carried by the top ``k`` units
  (a trigger often drives a few dominant neurons).
* ``grad_concentration``: cosine similarity between the sample's loss gradient w.r.t. the
  penultimate features, ``W^T (softmax - onehot(y))``, and the robust mean gradient direction of
  its given class (poison of a target class shares one gradient direction; AGPD's dispersion idea).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cord.config import SignalsConfig

if TYPE_CHECKING:
    from cord.signals import ScoredSet


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def compute(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    feats = scored.cache["feat_penultimate"].astype(np.float64)
    k = max(1, int(np.ceil(cfg.activation_topk * feats.shape[1])))
    a = np.abs(feats)
    top = np.partition(a, a.shape[1] - k, axis=1)[:, -k:].sum(1)
    share = top / np.maximum(a.sum(1), 1e-12)

    probs = _softmax(scored.cache["logits"].astype(np.float64))
    probs[np.arange(len(scored)), scored.labels] -= 1.0
    grads = probs @ scored.cache["head_weight"].astype(np.float64)  # (n, d)
    unit = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-12)
    concentration = np.zeros(len(scored))
    for c in range(scored.num_classes):
        idx = np.flatnonzero(scored.labels == c)
        if idx.size == 0:
            continue
        direction = np.median(unit[idx], axis=0)  # coordinate-wise median: robust to minority poison
        direction /= max(np.linalg.norm(direction), 1e-12)
        concentration[idx] = unit[idx] @ direction
    return {"act_topk_share": share, "grad_concentration": concentration}
