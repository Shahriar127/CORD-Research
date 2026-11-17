"""Gate G1 (Table 7): evaluate the pilot against its pre-registered numeric criteria.

    python experiments/pilot/check_gate_g1.py runs/pilot

Criteria (alpha = 0.05):
  1. upper 95% bound on FDR (e-BH proxy set) <= 1.1 alpha on every attack;
  2. presence-test false-alarm rate on clean data <= 0.07;
  3. FDR-mode power >= 50% at rho = 5% on at least two attacks;
  4. presence detected in >= 80% of runs at rho = 1% on at least two attacks;
  5. PU fusion within 5 points of the best single signal (power at rho = 5%), every attack.

Note: with only the pilot's seeds, criterion 2 has a wide interval; Section 7.3 asks for >= 30
no-attack seeds for the headline false-alarm estimate. The script prints the exact interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cord.evaluation import bootstrap_upper_bound, clopper_pearson, collect_summaries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-metric", default="proxy_ebh_power")
    ap.add_argument("--tag", default="", help="judge results-<tag>/ (default: the primary results/)")
    args = ap.parse_args()

    df = collect_summaries(args.runs)
    df = df[df.tag == args.tag] if not df.empty else df
    if df.empty:
        print("no finished runs found", file=sys.stderr)
        return 1
    cord = df[df.method == "cord_pu"]
    attacked = cord[cord.attack != "none"]
    results: list[tuple[str, bool, str]] = []

    worst = max((bootstrap_upper_bound(g.proxy_ebh_fdp.to_numpy()), a) for a, g in attacked.groupby("attack"))
    results.append(
        (
            "FDR upper bound <= 1.1 alpha (all attacks)",
            worst[0] <= 1.1 * args.alpha,
            f"worst {worst[1]}: {worst[0]:.4f}",
        )
    )

    clean = cord[cord.attack == "none"]
    # Seed level (Section 7.3): one binary decision per training seed, on the primary split.
    k, n = int(clean.presence_reject_primary.sum()), len(clean)
    lo, hi = clopper_pearson(k, n) if n else (np.nan, np.nan)
    rate = k / n if n else np.nan
    results.append(
        ("presence false alarm <= 0.07", bool(n) and rate <= 0.07, f"{k}/{n} = {rate:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
    )

    p5 = attacked[np.isclose(attacked.rate, 0.05)].groupby("attack")[args.power_metric].mean()
    results.append(("FDR-mode power >= 0.5 at 5% on >= 2 attacks", int((p5 >= 0.5).sum()) >= 2, p5.round(3).to_dict()))

    pr1 = attacked[np.isclose(attacked.rate, 0.01)].groupby("attack").presence_reject.mean()
    results.append(("presence >= 0.8 at 1% on >= 2 attacks", int((pr1 >= 0.8).sum()) >= 2, pr1.round(3).to_dict()))

    singles = df[df.method.str.startswith("single:") & (df.attack != "none") & np.isclose(df.rate, 0.05)]
    best_single = singles.groupby(["attack", "method"])[args.power_metric].mean().groupby("attack").max()
    gap = (best_single - p5).dropna()
    results.append(
        (
            "fusion within 0.05 of best single signal",
            bool((gap <= 0.05).all()) and not gap.empty,
            gap.round(3).to_dict(),
        )
    )

    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    return 0 if all(r[1] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
