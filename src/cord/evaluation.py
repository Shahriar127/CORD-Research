"""Cross-run aggregation and the replication statistics of Section 7.3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def collect_summaries(root: Path) -> pd.DataFrame:
    """One row per (run, method) from every ``results/summary.json`` below ``root``."""
    rows = []
    for path in sorted(Path(root).rglob("results*/summary.json")):
        s = json.loads(path.read_text())
        train = s.get("train", {})
        base = {
            "run": s["name"],
            "tag": s.get("results_tag", ""),
            "fusion_reference": s.get("fusion_reference", "random"),
            "attack": s["attack"],
            "rate": s["rate"],
            "seed": s["seed"],
            "m_cal": s["m_cal"],
            "alpha": s["alpha"],
            "clean_acc": np.mean([t["clean_acc"] for t in train.values()]) if train else np.nan,
            "asr": np.mean([t["asr"] for t in train.values() if t.get("asr") is not None]) if train else np.nan,
        }
        for method, metrics in s["methods"].items():
            rows.append({**base, "method": method, **metrics})
    return pd.DataFrame(rows)


def bootstrap_upper_bound(values: np.ndarray, level: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> float:
    """One-sided percentile-bootstrap upper bound of the mean over *seeds* (one value per seed),
    which respects the dependence of resamples within a seed."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if v.size == 1:
        return float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(1)
    return float(np.quantile(means, level))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval (presence-test false-alarm rate at the seed level)."""
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi
