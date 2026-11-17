"""Latent geometry (lineage: Spectral Signatures, SCAn, SPC).

Per layer and per given class: a robust class-conditional Mahalanobis distance and a
spectral-signature projection, both estimated on the scored set with iterative trimming so
that poison inside a class does not define its own reference. A shared PCA (fit on the whole
scored set) reduces dimension first; all estimates are symmetric functions of the scored set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.covariance import ledoit_wolf

from cord.config import SignalsConfig

if TYPE_CHECKING:
    from cord.signals import ScoredSet


def _pca(x: np.ndarray, dim: int) -> np.ndarray:
    xc = x - x.mean(0, keepdims=True)
    dim = min(dim, xc.shape[1], max(1, xc.shape[0] - 1))
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:dim].T


def _robust_class_scores(z: np.ndarray, trim: float, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (squared Mahalanobis distance, squared top-direction projection) for one class."""
    n = z.shape[0]
    if n <= 2:
        return np.zeros(n), np.zeros(n)
    keep = np.ones(n, dtype=bool)
    mean = z.mean(0)
    for _ in range(max(1, iters)):
        mean = z[keep].mean(0)
        cov, _ = ledoit_wolf(z[keep] - mean, assume_centered=True)
        diff = z - mean
        d2 = np.einsum("ij,ij->i", diff, np.linalg.solve(cov, diff.T).T)
        if trim > 0:
            keep = d2 <= np.quantile(d2, 1.0 - trim)
    centred = z[keep] - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj2 = ((z - mean) @ vt[0]) ** 2
    return d2, proj2


def compute(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    labels = scored.labels
    layers = [k.removeprefix("feat_") for k in scored.cache if k.startswith("feat_")]
    for layer in layers:
        z = _pca(scored.cache[f"feat_{layer}"].astype(np.float64), cfg.latent_pca_dim)
        maha = np.zeros(len(labels))
        spec = np.zeros(len(labels))
        for c in range(scored.num_classes):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                continue
            m, s = _robust_class_scores(z[idx], cfg.latent_trim, cfg.latent_iters)
            # normalise within class so classes with different spread are comparable
            maha[idx] = m / max(np.median(m), 1e-12)
            spec[idx] = s / max(np.median(s), 1e-12)
        out[f"latent_maha_{layer}"] = maha
        out[f"latent_spectral_{layer}"] = spec
    return out
