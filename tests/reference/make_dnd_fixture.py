"""Build the reference fixture for ``tests/test_reference_evalues.py``.

Runs the procedures of Bashari et al. (2023), ``Meshiba/derandomized-novelty-detection``
(``algo.py``), on seeded synthetic scores and stores inputs and outputs as JSON, so the test
suite can compare ``cord.stats`` against them without the external repository. The upstream
repository has no license file, so none of its code is copied here; it is only executed.

    git clone https://github.com/Meshiba/derandomized-novelty-detection /path/to/dnd
    python tests/reference/make_dnd_fixture.py --dnd /path/to/dnd

Reference functions used: ``compute_all_pvalue``, ``BH``, ``EmpBH``, ``compute_threshold``,
``compute_evalue``, ``eBH``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cord.stats import conformal_evalues

OUT = Path(__file__).with_name("dnd_reference.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnd", type=Path, required=True, help="clone of Meshiba/derandomized-novelty-detection")
    args = ap.parse_args()
    sys.path.insert(0, str(args.dnd.resolve()))
    import algo  # type: ignore[import-not-found]

    commit = subprocess.run(
        ["git", "-C", str(args.dnd), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    rng = np.random.default_rng(20261117)
    cases = []
    for n_test, n_cal, n_alt, shift in (
        (200, 100, 30, 3.5),
        (500, 300, 20, 4.0),
        (300, 200, 0, 0.0),
        (400, 150, 60, 2.5),
    ):
        test = rng.standard_normal(n_test)
        test[:n_alt] += shift
        cal = rng.standard_normal(n_cal)
        case: dict[str, object] = {"test": test.tolist(), "cal": cal.tolist(), "levels": {}}
        pvalues = algo.compute_all_pvalue(test, cal)
        case["pvalues"] = pvalues.tolist()
        for level in (0.05, 0.1, 0.2):
            bh_rej, _ = algo.BH(pvalues, level)
            emp_rej = algo.EmpBH(cal, test, level)
            t_ref = float(algo.compute_threshold(test, cal, level))  # their FDP estimate: V m / (n K)
            # Their e-value formula evaluated at CORD's stopping threshold (first pooled value at
            # which the BH condition holds), and at the smallest EmpBH-rejected test score.
            t_cord = conformal_evalues(test, cal, level).threshold
            e_at_cord = algo.compute_evalue(test, cal, t_cord) if np.isfinite(t_cord) else np.zeros_like(test)
            t_min_rej = float(test[emp_rej].min()) if len(emp_rej) else float("inf")
            e_at_ref = algo.compute_evalue(test, cal, t_ref)
            ebh = {str(a): sorted(int(i) for i in algo.eBH(e_at_ref, a)[0]) for a in (0.05, 0.1, 0.2)}
            case["levels"][str(level)] = {
                "bh": sorted(int(i) for i in bh_rej),
                "empbh": sorted(int(i) for i in emp_rej),
                "threshold_cord": t_cord,
                "min_rejected_score": t_min_rej,
                "threshold_ref": t_ref,
                "evalues_at_cord_threshold": e_at_cord.tolist(),
                "evalues_at_ref": e_at_ref.tolist(),
                "ebh_on_ref_evalues": ebh,
            }
        cases.append(case)
    OUT.write_text(json.dumps({"upstream": "Meshiba/derandomized-novelty-detection", "commit": commit, "cases": cases}))
    print(f"wrote {OUT} ({len(cases)} cases, upstream {commit[:12]})")


if __name__ == "__main__":
    main()
