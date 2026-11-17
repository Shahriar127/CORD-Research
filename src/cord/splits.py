"""Random hold-out (mode T1) and cross-fitting folds.

Order of events (Section 3.2): the attacker poisons the pool first; only then does the defender
draw the hold-out ``C = C_tr ∪ C_cal`` uniformly at random and partition the rest into folds.
The split generator is seeded independently of the attack, so the attacker cannot anticipate it.

Index conventions (all indices refer to positions in the poisoned pool):

* ``folds[j]``           D_j, scored by the model ``f_{-j}`` trained on ``D \\ D_j``
* ``holdout``            C, never used to train any fold model
* ``scored_indices(j)``  D_j ∪ C, every point fold model ``j`` scores
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cord.utils.io import load_npz, save_npz


@dataclass(frozen=True)
class Split:
    holdout_tr: np.ndarray
    holdout_cal: np.ndarray
    folds: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        parts = [self.holdout_tr, self.holdout_cal, *self.folds]
        allidx = np.concatenate(parts)
        if np.unique(allidx).size != allidx.size:
            raise ValueError("split parts overlap")

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def holdout(self) -> np.ndarray:
        return np.concatenate([self.holdout_tr, self.holdout_cal])

    def train_indices(self, j: int) -> np.ndarray:
        """Training set of model ``f_{-j}``: all folds except ``j``; never the hold-out."""
        return np.sort(np.concatenate([f for i, f in enumerate(self.folds) if i != j]))

    def scored_indices(self, j: int) -> np.ndarray:
        """Points scored by ``f_{-j}``: fold ``j`` followed by the hold-out (fixed order)."""
        return np.concatenate([self.folds[j], self.holdout])

    def save(self, path: Path) -> None:
        arrays = {"holdout_tr": self.holdout_tr, "holdout_cal": self.holdout_cal}
        arrays.update({f"fold_{j}": f for j, f in enumerate(self.folds)})
        save_npz(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Split:
        d = load_npz(path)
        n_folds = sum(1 for k in d if k.startswith("fold_"))
        return cls(d["holdout_tr"], d["holdout_cal"], tuple(d[f"fold_{j}"] for j in range(n_folds)))


def draw_split(n_pool: int, m_tr: int, m_cal: int, n_folds: int, rng: np.random.Generator) -> Split:
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    if m_tr + m_cal >= n_pool:
        raise ValueError("hold-out larger than the pool")
    perm = rng.permutation(n_pool)
    holdout_tr = np.sort(perm[:m_tr])
    holdout_cal = np.sort(perm[m_tr : m_tr + m_cal])
    rest = perm[m_tr + m_cal :]
    folds = tuple(np.sort(f) for f in np.array_split(rest, n_folds))
    return Split(holdout_tr, holdout_cal, folds)


def resample_holdout_partition(
    holdout: np.ndarray, m_tr: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Redraw the C_tr / C_cal partition inside a fixed hold-out (Section 7.3, B resamples).

    Valid after training because no fold model ever saw any hold-out point.
    """
    perm = rng.permutation(holdout)
    return np.sort(perm[:m_tr]), np.sort(perm[m_tr:])
