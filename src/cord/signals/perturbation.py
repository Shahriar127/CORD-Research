"""Perturbation consistency (lineage: STRIP, SCALE-UP).

* ``scaleup_consistency``: share of amplification factors under which the prediction is unchanged
  (triggered inputs keep the target prediction when pixel values are amplified).
* ``strip_neg_entropy``: negative mean prediction entropy under blending with fixed partners
  (triggered inputs stay confidently predicted).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cord.config import SignalsConfig

if TYPE_CHECKING:
    from cord.signals import ScoredSet


def compute(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    pred = scored.cache["pred"].astype(np.int64)
    scaled = scored.cache["scaleup_pred"].astype(np.int64)
    return {
        "scaleup_consistency": (scaled == pred[None, :]).mean(0),
        "strip_neg_entropy": -scored.cache["strip_entropy"].astype(np.float64),
    }
