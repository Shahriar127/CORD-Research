"""Study S1 (Section 6.1): validity and power of the selection rules, and the selection floor.

Setup: n = 10,000 points in two folds; clean features ~ N(0, I_2); poison shifted by ``shift`` in
each coordinate; each fold scores with its own linear direction; m_cal = 1,000 clean calibration
points; alpha = 0.05, alpha_bh = 0.045. Selection rules: fold-wise BH (union), pooled BH, e-BH on
the pooled conformal e-values, and FPR mode (p <= 0.01).

The selections run through :func:`cord.pipeline.run_cord`, the same code the deep pipeline uses.

    python sim/s1_selections.py            # full study (~1-2 min)
    python sim/s1_selections.py --scale 0.1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import BASE_SEED, parse_args, reps, save, series_style, setup_style
from cord.config import CalibrationConfig
from cord.pipeline import run_cord
from cord.stats import false_discovery_proportion, selection_floor, true_positive_proportion
from cord.utils.seeding import make_rng

N, M_CAL, ALPHA, ALPHA_BH, FPR_LEVEL = 10_000, 1_000, 0.05, 0.045, 0.01
# Fold directions: unit vectors at 15 and 75 degrees (projected poison shift 3.18 in both folds).
# The proposal only says "a different linear direction" per fold; the choice affects power,
# never validity.
DIRECTIONS = tuple(np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in (15.0, 75.0))


def one_rep(rho: float, shift: float, m_cal: int, rng: np.random.Generator) -> dict[str, float]:
    n_poison = round(rho * N)
    x = rng.standard_normal((N, 2))
    poison = np.zeros(N, dtype=bool)
    poison[rng.choice(N, n_poison, replace=False)] = True
    x[poison] += shift
    cal = rng.standard_normal((m_cal, 2))
    folds = np.array_split(rng.permutation(N), 2)
    cfg = CalibrationConfig(alpha=ALPHA, alpha_bh=ALPHA_BH, removal_level=FPR_LEVEL, n_resamples=1)
    out = run_cord([x[f] @ w for f, w in zip(folds, DIRECTIONS, strict=True)], [cal @ w for w in DIRECTIONS], cfg)
    null = ~np.concatenate([poison[f] for f in folds])
    res: dict[str, float] = {}
    for name, mask in (
        ("foldwise_bh", out.foldwise_bh),
        ("pooled_bh", out.pooled_bh),
        ("ebh", out.proxy),
        ("fpr_mode", out.removal),
    ):
        res[f"{name}_fdp"] = false_discovery_proportion(mask, null)
        res[f"{name}_power"] = true_positive_proportion(mask, null)
    res["fpr_mode_clean_share"] = float((out.removal & null).sum() / null.sum())
    return res


def run_grid(rates: list[float], shift: float, m_cal: int, n_reps: int, tag: str) -> pd.DataFrame:
    rows = []
    for rho in rates:
        rng = make_rng(BASE_SEED, "s1", tag, rho, shift, m_cal)
        for r in range(n_reps):
            rows.append({"rho": rho, "shift": shift, "m_cal": m_cal, "rep": r, **one_rep(rho, shift, m_cal, rng)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__ or "")
    setup_style()
    import matplotlib.pyplot as plt

    rates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
    ab = run_grid(rates, 2.6, M_CAL, reps(400, args.scale), "ab")
    floors_rates = sorted({*np.round(np.geomspace(0.001, 0.1, 15), 5)})
    c = pd.concat([run_grid(floors_rates, 4.5, m, reps(150, args.scale), "c") for m in (500, 1000, 2000, 5000)])

    g = ab.groupby("rho").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    names = {"foldwise_bh": "fold-wise BH", "pooled_bh": "pooled BH", "ebh": "e-BH"}
    for i, (k, label) in enumerate(names.items()):
        axes[0].plot(g.index, g[f"{k}_fdp"], label=label, **series_style(i))
    axes[0].axhline(ALPHA, color="#6b6a64", ls="--", lw=1)
    axes[0].set(title="(a) empirical FDR", xlabel="poisoning rate ρ", ylabel="FDR", xscale="log")
    for i, (k, label) in enumerate([*names.items(), ("fpr_mode", "FPR mode (p ≤ 0.01)")]):
        axes[1].plot(g.index, g[f"{k}_power"], label=label, **series_style(i))
    axes[1].set(title="(b) power", xlabel="poisoning rate ρ", ylabel="power", xscale="log")
    axes[1].legend(loc="center left", bbox_to_anchor=(0.0, 0.55))
    gc = c.groupby(["m_cal", "rho"]).foldwise_bh_power.mean()
    for i, m in enumerate((500, 1000, 2000, 5000)):
        s = gc.loc[m]
        axes[2].plot(s.index, s.values, label=f"m_cal = {m}", **series_style(i))
        axes[2].axvline(selection_floor(m, ALPHA), color=series_style(i)["color"], ls=":", lw=1)
    axes[2].set(
        title="(c) fold-wise BH power vs floor (dotted)", xlabel="poisoning rate ρ", ylabel="power", xscale="log"
    )
    axes[2].legend(loc="lower right")
    fig.tight_layout()
    save(ab, fig, args.out, "s1_selections")
    save(c, None, args.out, "s1_floor")

    q90 = ab[ab.rho == 0.03].foldwise_bh_fdp.quantile(0.9)
    p5 = g.loc[0.05]
    print("S1 findings (proposal v3.1 values in brackets)")
    print(
        f" (1) max FDR  fold-wise {g.foldwise_bh_fdp.max():.3f} [0.043]  pooled {g.pooled_bh_fdp.max():.3f} "
        f"[0.041]  e-BH {g.ebh_fdp.max():.3f} [0.033]"
    )
    print(f" (2) 90th pct FDP fold-wise BH at rho=3%: {q90:.3f} [0.126]")
    print(f" (3) power at rho=5%: e-BH {p5.ebh_power:.2f} [0.27]  fold-wise BH {p5.foldwise_bh_power:.2f} [0.44]")
    print(
        f" (5) FPR mode recall {g.fpr_mode_power.min():.2f}-{g.fpr_mode_power.max():.2f} [0.81], "
        f"clean share {g.fpr_mode_clean_share.mean():.4f} [0.01]"
    )


if __name__ == "__main__":
    main()
