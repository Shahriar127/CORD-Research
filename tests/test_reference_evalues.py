"""Cross-check of ``cord.stats`` against the reference implementation of Bashari et al. (2023).

The fixture ``tests/reference/dnd_reference.json`` holds outputs of
``Meshiba/derandomized-novelty-detection`` (``algo.py``) on seeded synthetic scores; regenerate it
with ``tests/reference/make_dnd_fixture.py``.

Agreement checked: conformal p-values, BH, the reference e-value formula at CORD's threshold,
the e-value support (= BH rejections), and e-BH. One documented difference: the reference code
stops its upward scan with the FDP estimate ``V m / (n K)``, CORD with BH's
``(1 + V) m / ((n + 1) K)``. Both are stopping times of the upward scan over the pooled scores, so
both give valid e-values (``proofs/main.tex``, Lemmas 4-5); CORD's is never lower, so it never
selects more.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cord.stats import benjamini_hochberg, conformal_evalues, conformal_pvalues, e_benjamini_hochberg

FIXTURE = json.loads((Path(__file__).parent / "reference" / "dnd_reference.json").read_text())
CASES = FIXTURE["cases"]
LEVELS = ["0.05", "0.1", "0.2"]


def _mask(indices: list[int], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[indices] = True
    return m


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{len(c['test'])}-m{len(c['cal'])}")
def test_pvalues_and_bh_match_reference(case: dict) -> None:
    test, cal = np.asarray(case["test"]), np.asarray(case["cal"])
    p = conformal_pvalues(test, cal)
    np.testing.assert_allclose(p, case["pvalues"], rtol=0, atol=1e-15)
    for level in LEVELS:
        ref = case["levels"][level]
        np.testing.assert_array_equal(benjamini_hochberg(p, float(level)), _mask(ref["bh"], test.size))
        # EmpBH (Roquain and Mary's direct algorithm) is BH on conformal p-values
        np.testing.assert_array_equal(benjamini_hochberg(p, float(level)), _mask(ref["empbh"], test.size))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{len(c['test'])}-m{len(c['cal'])}")
def test_evalues_match_reference(case: dict) -> None:
    test, cal = np.asarray(case["test"]), np.asarray(case["cal"])
    for level in LEVELS:
        ref = case["levels"][level]
        ours = conformal_evalues(test, cal, float(level))
        # their e-value formula at our threshold gives our e-values
        assert ours.threshold == ref["threshold_cord"]
        np.testing.assert_allclose(ours.evalues, ref["evalues_at_cord_threshold"], rtol=0, atol=1e-12)
        # the nonzero e-values are exactly the BH (EmpBH) rejections
        np.testing.assert_array_equal(ours.evalues > 0, _mask(ref["empbh"], test.size))
        # stopping threshold: at or below the smallest rejected score; the reference's
        # (FDP estimate without the +1 terms) never lies above it
        assert ours.threshold <= ref["min_rejected_score"]
        assert ref["threshold_ref"] <= ours.threshold


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{len(c['test'])}-m{len(c['cal'])}")
def test_ebh_matches_reference(case: dict) -> None:
    n = len(case["test"])
    for level in LEVELS:
        ref = case["levels"][level]
        e = np.asarray(ref["evalues_at_ref"])
        for alpha, expected in ref["ebh_on_ref_evalues"].items():
            np.testing.assert_array_equal(e_benjamini_hochberg(e, float(alpha)), _mask(expected, n))


def test_fixture_is_not_degenerate() -> None:
    """The cases must exercise both empty and non-empty selections."""
    sizes = [len(c["levels"][lv]["bh"]) for c in CASES for lv in LEVELS]
    ebh = [len(s) for c in CASES for lv in LEVELS for s in c["levels"][lv]["ebh_on_ref_evalues"].values()]
    assert min(sizes) == 0 and max(sizes) > 10
    assert min(ebh) == 0 and max(ebh) > 10


def test_evalue_null_mass_with_strong_nonnulls() -> None:
    """Regression: with strong non-nulls and few calibration points, e-values that count the
    calibration set at the smallest rejected score (proposal v3.1, eq. 3) had null mass 1.22 N0.
    The stopping-time threshold keeps it at most N0 (Lemma 5)."""
    rng = np.random.default_rng(11)
    n0, n1, m, alpha, reps = 5, 40, 2, 0.9, 20_000
    mass = np.empty(reps)
    for r in range(reps):
        test = np.concatenate([rng.standard_normal(n0), rng.standard_normal(n1) + 10.0])
        mass[r] = conformal_evalues(test, rng.standard_normal(m), alpha).evalues[:n0].sum() / n0
    assert mass.mean() <= 1.0 + 3 * mass.std() / np.sqrt(reps)
