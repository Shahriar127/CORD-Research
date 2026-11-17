"""Shared helpers for the synthetic studies: figure style, output paths and reporting.

Every setting draws from its own generator ``make_rng(BASE_SEED, study, *setting)``, so results
do not depend on the order in which settings run (the single global generator of the original
``audit_sim.py`` did).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

BASE_SEED = 20261117
# Validated categorical order (fixed, never cycled); markers give a second, non-colour channel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
        }
    )


def series_style(i: int) -> dict[str, object]:
    return {"color": SERIES[i], "marker": MARKERS[i], "markeredgecolor": "white", "markeredgewidth": 0.8}


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every repetition count (0.1 for a quick check)")
    return ap.parse_args()


def reps(n: int, scale: float) -> int:
    return max(10, round(n * scale))


def save(df: pd.DataFrame, fig: plt.Figure | None, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    if fig is not None:
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.png")
        plt.close(fig)
