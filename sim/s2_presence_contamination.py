"""Study S2 (Section 6.2): presence test, calibration contamination and the audit.

This script merges the former supplementary ``audit_sim.py`` (findings 6 and 7). Every setting
has its own seeded generator, so results no longer depend on call order.

Parts:
  (a) presence test vs BH against poisoning rate, shifts 3.0 and 4.5 (two folds of 5,000);
  (b) contamination: fold-wise BH power/FDR with eps*m_cal poisoned calibration points,
      unaudited vs audit of the top-100 calibration scores;
  (c) wrong removals by the auditor (finding 6): attack present (FDR, power) and no attack
      (presence false alarm, BH false selection);
  (d) presence power with an unaudited random hold-out (finding 7).

All statistics come from :mod:`cord.stats` and the audit from :func:`cord.audit.audit_calibration`.

    python sim/s2_presence_contamination.py              # full study (a few minutes)
    python sim/s2_presence_contamination.py --scale 0.1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import BASE_SEED, MUTED, parse_args, reps, save, series_style, setup_style
from cord.audit import audit_calibration
from cord.config import AuditConfig
from cord.stats import (
    benjamini_hochberg,
    conformal_pvalues,
    exceedance_count,
    exceedance_pvalue,
    false_discovery_proportion,
    presence_test,
    presence_threshold_exact,
    selection_floor,
    true_positive_proportion,
)
from cord.utils.seeding import make_rng

N_FOLD, M_CAL, ALPHA, J = 5_000, 1_000, 0.05, 2


def _scores(n: int, n_poison: int, shift: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal(n)
    poison = np.zeros(n, dtype=bool)
    poison[:n_poison] = True
    x[:n_poison] += shift
    return x, poison


# --------------------------------------------------------------------------------------- (a)
def part_a(n_reps: int) -> pd.DataFrame:
    """``cal = independent``: each fold has its own calibration draw (reproduces v3.1, Fig. 3a).
    ``cal = shared``: both folds use the same calibration points, as in the real pipeline where
    every fold model scores the same C_cal; folds are then positively dependent and the
    Bonferroni combination gains less from the second fold."""
    rows = []
    for cal_mode in ("independent", "shared"):
        for shift in (3.0, 4.5):
            for rho in (0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
                rng = make_rng(BASE_SEED, "s2a", cal_mode, shift, rho)
                for _ in range(n_reps):
                    folds = [_scores(N_FOLD, round(rho * N_FOLD), shift, rng)[0] for _ in range(J)]
                    shared = rng.standard_normal(M_CAL)
                    cals = (
                        [shared] * J
                        if cal_mode == "shared"
                        else [shared, *(rng.standard_normal(M_CAL) for _ in range(J - 1))]
                    )
                    pres = presence_test(folds, cals, ALPHA)
                    any_bh = any(
                        benjamini_hochberg(conformal_pvalues(f, c), ALPHA).any()
                        for f, c in zip(folds, cals, strict=True)
                    )
                    rows.append(
                        {
                            "cal": cal_mode,
                            "shift": shift,
                            "rho": rho,
                            "presence": float(pres.reject),
                            "bh_any": float(any_bh),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- (b)
def part_b(n_reps: int) -> pd.DataFrame:
    rows, rho = [], 0.05
    for shift in (3.0, 4.5):
        for n_cal_poison in (0, 1, 2, 5, 10, 20, 50):
            for audited in (False, True):
                rng = make_rng(BASE_SEED, "s2b", shift, n_cal_poison, audited)
                cfg = AuditConfig(budget=100 if audited else 0)
                for _ in range(n_reps):
                    fdps, pws = [], []
                    cal, cal_poison = _scores(M_CAL, n_cal_poison, shift, rng)
                    for _ in range(J):
                        x, poison = _scores(N_FOLD, round(rho * N_FOLD), shift, rng)
                        keep = audit_calibration(cal, cal_poison, cfg, rng).keep
                        sel = benjamini_hochberg(conformal_pvalues(x, cal[keep]), ALPHA)
                        fdps.append(false_discovery_proportion(sel, ~poison))
                        pws.append(true_positive_proportion(sel, ~poison))
                    rows.append(
                        {
                            "shift": shift,
                            "n_cal_poison": n_cal_poison,
                            "audited": audited,
                            "fdr": np.mean(fdps),
                            "power": np.mean(pws),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- (c)
def part_c(n_reps_attack: int, n_reps_null: int) -> pd.DataFrame:
    """Finding (6): one fold of 5,000, shift 4.5, top-100 audit that also removes k clean points."""
    rows = []
    for mode in ("random", "top"):
        for k in (0, 5, 10, 20, 40):  # attack present: rho = eps = 5%
            rng = make_rng(BASE_SEED, "s2c-attack", mode, k)
            cfg = AuditConfig(budget=100, wrong_removals=k, wrong_mode=mode)
            for _ in range(n_reps_attack):
                x, poison = _scores(N_FOLD, round(0.05 * N_FOLD), 4.5, rng)
                cal, cal_poison = _scores(M_CAL, round(0.05 * M_CAL), 4.5, rng)
                keep = audit_calibration(cal, cal_poison, cfg, rng).keep
                sel = benjamini_hochberg(conformal_pvalues(x, cal[keep]), ALPHA)
                rows.append(
                    {
                        "setting": "attack",
                        "mode": mode,
                        "k": k,
                        "fdr": false_discovery_proportion(sel, ~poison),
                        "power": true_positive_proportion(sel, ~poison),
                    }
                )
        for k in (0, 1, 2, 5, 10, 20):  # no attack at all
            rng = make_rng(BASE_SEED, "s2c-null", mode, k)
            cfg = AuditConfig(budget=100, wrong_removals=k, wrong_mode=mode)
            for _ in range(n_reps_null):
                x = rng.standard_normal(N_FOLD)
                cal = rng.standard_normal(M_CAL)
                kept = cal[audit_calibration(cal, np.zeros(M_CAL, bool), cfg, rng).keep]
                t = exceedance_count(x, kept)
                rows.append(
                    {
                        "setting": "null",
                        "mode": mode,
                        "k": k,
                        "presence_false_alarm": float(exceedance_pvalue(t, N_FOLD, kept.size) <= ALPHA / J),
                        "bh_false_selection": float(benjamini_hochberg(conformal_pvalues(x, kept), ALPHA).any()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- (d)
def part_d(n_reps: int) -> pd.DataFrame:
    """Finding (7): per-fold presence power, clean calibration vs unaudited random hold-out."""
    rows = []
    for rho in (0.004, 0.005, 0.01):
        for calibration in ("clean", "unaudited_random_holdout"):
            rng = make_rng(BASE_SEED, "s2d", rho, calibration)
            for _ in range(n_reps):
                x, _ = _scores(N_FOLD, round(rho * N_FOLD), 4.5, rng)
                n_cal_poison = 0 if calibration == "clean" else int(rng.binomial(M_CAL, rho))
                cal, _ = _scores(M_CAL, n_cal_poison, 4.5, rng)
                t = exceedance_count(x, cal)
                rows.append(
                    {
                        "rho": rho,
                        "calibration": calibration,
                        "detected": exceedance_pvalue(t, N_FOLD, M_CAL) <= ALPHA / J,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__ or "")
    setup_style()
    import matplotlib.pyplot as plt

    a = part_a(reps(1000, args.scale))
    b = part_b(reps(400, args.scale))
    c = part_c(reps(400, args.scale), reps(1000, args.scale))
    d = part_d(reps(1000, args.scale))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ga_all = a.groupby(["cal", "shift", "rho"]).mean(numeric_only=True)
    ga = ga_all.loc["independent"]
    for i, shift in enumerate((4.5, 3.0)):
        s = ga.loc[shift]
        axes[0].plot(s.index, s.presence, label=f"presence, shift {shift}", **series_style(2 * i))
        axes[0].plot(s.index, s.bh_any, ls="--", label=f"BH selects, shift {shift}", **series_style(2 * i + 1))
    axes[0].axvline(presence_threshold_exact(N_FOLD, M_CAL, ALPHA, J), color=MUTED, ls=":", lw=1)
    axes[0].axvline(selection_floor(M_CAL, ALPHA), color=MUTED, ls=":", lw=1)
    axes[0].axhline(ALPHA, color=MUTED, ls="--", lw=0.8)
    axes[0].set(title="(a) presence test vs BH", xlabel="poisoning rate ρ", ylabel="rejection rate")
    axes[0].legend(loc="center right")
    gb = b.groupby(["shift", "audited", "n_cal_poison"]).mean(numeric_only=True)
    i = 0
    for shift in (4.5, 3.0):
        for audited in (False, True):
            s = gb.loc[(shift, audited)]
            xs = s.index.to_numpy() + 0.5  # log axis: shift by 0.5 so that 0 is visible
            lbl = f"shift {shift}, {'audit top-100' if audited else 'no audit'}"
            axes[1].plot(xs, s.power, label=lbl, **series_style(i))
            axes[2].plot(xs, s.fdr, label=lbl, **series_style(i))
            i += 1
    axes[1].set(
        title="(b) fold-wise BH power, ρ = 5%",
        xlabel="poisoned calibration points (+0.5, log)",
        ylabel="power",
        xscale="log",
    )
    axes[1].legend(loc="center left")
    axes[2].axhline(ALPHA, color=MUTED, ls="--", lw=0.8)
    axes[2].set(
        title="(c) FDR under contamination",
        xlabel="poisoned calibration points (+0.5, log)",
        ylabel="FDR",
        xscale="log",
    )
    fig.tight_layout()
    for name, df in (("s2a_presence", a), ("s2b_contamination", b), ("s2c_wrong_removal", c), ("s2d_unaudited", d)):
        save(df, fig if name == "s2a_presence" else None, args.out, name)

    print("S2 findings (proposal v3.1 values in brackets)")
    print(
        f" (1) presence false alarm, no poison: shift 4.5 {ga.loc[(4.5, 0.0)].presence:.3f}, "
        f"shift 3.0 {ga.loc[(3.0, 0.0)].presence:.3f} [0.044, 0.053]"
    )
    s45 = ga.loc[4.5]
    print(
        f" (2) shift 4.5 presence at 0.3/0.4/0.5%: {s45.presence[0.003]:.2f}/{s45.presence[0.004]:.2f}/"
        f"{s45.presence[0.005]:.3f} [0.46/0.83/0.985]; BH any at 1.5/2%: {s45.bh_any[0.015]:.3f}/"
        f"{s45.bh_any[0.02]:.2f} [0.014/0.48]"
    )
    print(
        f"     shift 3.0 presence at 1%: {ga.loc[(3.0, 0.01)].presence:.2f} [0.87]; BH any at 2%: "
        f"{ga.loc[(3.0, 0.02)].bh_any:.3f} [0.003]"
    )
    sh = ga_all.loc["shared"]
    print(
        f"     shared calibration (realistic): false alarm {sh.loc[(4.5, 0.0)].presence:.3f}/"
        f"{sh.loc[(3.0, 0.0)].presence:.3f}; shift 4.5 presence at 0.3/0.4/0.5%: {sh.loc[(4.5, 0.003)].presence:.2f}/"
        f"{sh.loc[(4.5, 0.004)].presence:.2f}/{sh.loc[(4.5, 0.005)].presence:.3f}"
    )
    p = gb.power
    print(
        f" (3) shift 4.5 unaudited power with 0/2/5/10 poisoned cal: {p[(4.5, False, 0)]:.2f}/"
        f"{p[(4.5, False, 2)]:.2f}/{p[(4.5, False, 5)]:.2f}/{p[(4.5, False, 10)]:.3f} [0.93/0.39/0.04/0.003]"
    )
    print(
        f" (5) audited power at 50 poisoned cal: shift 4.5 {p[(4.5, True, 50)]:.2f} [0.93], shift 3.0 "
        f"{p[(3.0, True, 50)]:.2f} [0.29-0.30]; max audited FDR {gb.xs(True, level='audited').fdr.max():.3f} [<=0.047]"
    )
    gc = c.groupby(["setting", "mode", "k"]).mean(numeric_only=True)
    top = gc.loc[("attack", "top")].fdr
    print(
        f" (6) top wrong removals FDR k=0/5/10/20: {top[0]:.2f}/{top[5]:.2f}/{top[10]:.2f}/{top[20]:.2f} "
        f"[0.04/0.12/0.20/0.31]; random k=20/40: {gc.loc[('attack', 'random')].fdr[20]:.2f}/"
        f"{gc.loc[('attack', 'random')].fdr[40]:.2f} [0.06/0.18]"
    )
    fa = gc.loc[("null", "top")].presence_false_alarm
    print(
        f"     no attack, presence false alarm k=0/1/2/5/10 (top): {fa[0]:.3f}/{fa[1]:.2f}/{fa[2]:.2f}/"
        f"{fa[5]:.2f}/{fa[10]:.3f} [0.025/0.10/0.23/0.75/>0.99]"
    )
    gd = d.groupby(["rho", "calibration"]).detected.mean()
    print(
        f" (7) presence power clean vs unaudited at 0.5%: {gd[(0.005, 'clean')]:.2f} vs "
        f"{gd[(0.005, 'unaudited_random_holdout')]:.2f} [0.87 vs 0.02]; at 1%: {gd[(0.01, 'clean')]:.2f} vs "
        f"{gd[(0.01, 'unaudited_random_holdout')]:.2f} [1.00 vs 0.02]"
    )


if __name__ == "__main__":
    main()
