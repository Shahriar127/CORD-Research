"""Confidence trajectory (lineage: ABL). Held-out poison activates the backdoor learned from the
other fold, so the fold model predicts its (target) label early and confidently across
checkpoints; clean held-out points are predicted less confidently, mislabelled ones not at all."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cord.config import SignalsConfig

if TYPE_CHECKING:
    from cord.signals import ScoredSet


def compute(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    p = scored.trajectory["p_given"].astype(np.float64)  # (n_checkpoints, n)
    logit = np.log(np.clip(p, 1e-6, 1.0)) - np.log(np.clip(1.0 - p, 1e-6, 1.0))
    return {
        "conf_mean": logit.mean(0),
        "conf_early": logit[: max(1, p.shape[0] // 4)].mean(0),
    }
