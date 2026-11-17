"""Label-content disagreement (lineage: DBD). In the feature space of a frozen self-supervised
encoder, a dirty-label poison sits among images of its source class, so few of its nearest
neighbours share its given (target) label. The kNN graph is built on the whole scored set,
excluding the point itself, which keeps the signal symmetric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from cord.config import SignalsConfig

if TYPE_CHECKING:
    from cord.signals import ScoredSet


def knn_label_disagreement(features: np.ndarray, labels: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    f = torch.from_numpy(features.astype(np.float32))
    f = torch.nn.functional.normalize(f, dim=1)
    lab = torch.from_numpy(labels.astype(np.int64))
    n = f.shape[0]
    k = min(k, n - 1)
    out = np.empty(n)
    for start in range(0, n, chunk):
        sim = f[start : start + chunk] @ f.T
        rows = torch.arange(start, min(start + chunk, n))
        sim[torch.arange(len(rows)), rows] = -float("inf")  # exclude self
        nn_idx = sim.topk(k, dim=1).indices
        agree = (lab[nn_idx] == lab[rows, None]).float().mean(1)
        out[start : start + len(rows)] = 1.0 - agree.numpy()
    return out


def compute(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    if scored.ssl is None:
        raise ValueError("label_content needs SSL features; run the 'ssl' stage or disable the signal")
    return {"label_disagreement": knn_label_disagreement(scored.ssl, scored.labels, cfg.knn_k)}
