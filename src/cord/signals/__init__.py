"""Detection signals (Table 4).

Contract (assumption A2). A signal maps the cached quantities of fold ``j`` to one score per
scored row (larger = more suspicious). It may use

* the row itself and the fold model ``f_{-j}`` (trained on ``D \\ D_j``, independent of the rows),
* frozen public encoders,
* and *symmetric* functions of the whole scored set ``D_j ∪ C`` (class statistics, kNN graphs,
  rank normalisation).

It must never use the row's membership in ``D_j``, ``C_tr`` or ``C_cal``. Every signal is
therefore permutation-equivariant on the scored set, which the test suite checks, and its
values do not change when the ``C_tr / C_cal`` partition is redrawn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cord.config import SignalsConfig
from cord.signals import activation, confidence, label_content, latent, perturbation


@dataclass(frozen=True)
class ScoredSet:
    """Inputs available to signals for one fold (rows in ``Split.scored_indices(j)`` order)."""

    cache: dict[str, np.ndarray]
    trajectory: dict[str, np.ndarray]
    labels: np.ndarray
    num_classes: int
    ssl: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.labels.size)


SignalFn = Callable[[ScoredSet, SignalsConfig], dict[str, np.ndarray]]

SIGNALS: dict[str, SignalFn] = {
    "latent": latent.compute,
    "confidence": confidence.compute,
    "perturbation": perturbation.compute,
    "activation": activation.compute,
    "label_content": label_content.compute,
}


def compute_signals(scored: ScoredSet, cfg: SignalsConfig) -> dict[str, np.ndarray]:
    """All enabled signals as ``{column_name: scores}``; columns are finite float64."""
    out: dict[str, np.ndarray] = {}
    for name in cfg.enabled:
        try:
            fn = SIGNALS[name]
        except KeyError as err:
            raise KeyError(f"unknown signal {name!r}; available: {sorted(SIGNALS)}") from err
        for col, values in fn(scored, cfg).items():
            v = np.asarray(values, dtype=np.float64)
            if v.shape != (len(scored),):
                raise ValueError(f"signal column {col} has shape {v.shape}")
            big = float(np.finfo(np.float64).max)
            out[col] = np.nan_to_num(v, nan=0.0, posinf=big, neginf=-big)
    return out


__all__ = ["SIGNALS", "ScoredSet", "compute_signals"]
