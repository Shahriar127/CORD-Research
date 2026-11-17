"""Targeted audit of the calibration set (mode T1, Section 3.3).

The defender inspects only the ``b`` highest-scoring calibration points of each fold and removes
those with *positive evidence of tampering*. In experiments the auditor is an oracle on the
ground-truth poison mask, degraded in the two ways RQ5 / experiment E5 study:

* ``miss_rate``: each poisoned point shown to the auditor is missed with this probability
  (e.g. imperceptible triggers);
* ``wrong_removals``: clean points removed by mistake, either the highest-scoring clean ones
  (``top``, worst case) or drawn at random from the inspected set (``random``). Study S2 shows
  that this breaks validity, which is why the rule forbids removal on a wrong label alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cord.config import AuditConfig


@dataclass(frozen=True)
class AuditResult:
    keep: np.ndarray  # bool mask over calibration points
    n_inspected: int
    n_poison_inspected: int
    n_poison_removed: int
    n_clean_removed: int


def audit_calibration(
    cal_scores: np.ndarray, cal_is_poison: np.ndarray, cfg: AuditConfig, rng: np.random.Generator
) -> AuditResult:
    m = cal_scores.size
    keep = np.ones(m, dtype=bool)
    if cfg.budget == 0:
        return AuditResult(keep, 0, 0, 0, 0)
    top = np.argsort(-cal_scores, kind="stable")[: min(cfg.budget, m)]
    poison_top = top[cal_is_poison[top]]
    found = poison_top[rng.random(poison_top.size) >= cfg.miss_rate]
    clean_top = top[~cal_is_poison[top]]  # already in decreasing score order
    n_wrong = min(cfg.wrong_removals, clean_top.size)
    if cfg.wrong_mode == "top":
        wrong = clean_top[:n_wrong]
    else:
        wrong = rng.choice(clean_top, size=n_wrong, replace=False) if n_wrong else clean_top[:0]
    keep[found] = False
    keep[wrong] = False
    return AuditResult(keep, int(top.size), int(poison_top.size), int(found.size), int(n_wrong))
