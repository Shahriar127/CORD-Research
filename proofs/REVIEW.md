# Internal statistical review of `proofs/main.tex`

Date: 2026-11-18. Reviewer: Claude, acting as the statistician for Gate G0 at the author's
request. This is a self-review of proofs Claude also drafted. It does not replace the independent
check G0 requires: an outside statistician must still sign off. Every claim below that changed the
proofs was checked by simulation or by a test.

## Verdict

Propositions 1–3 are correct as revised. Proposition 4 is now proved for contamination that does
not depend on the clean calibration points, which settles the proposal's conjecture
FDR ≤ α/(1−ε) for that case with no O(1/m) term. One defect affected both the proposal and the
code (the e-value threshold). It is fixed, and the fix is checked against the reference
implementation. Two items remain open (below).

## Findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | **Major** (proposal and code) | Eq. (3) took the e-value threshold as the smallest rejected fold score. That is not a stopping time, and the null e-value mass exceeded its bound (1.22 N⁰ in simulation) | Threshold = first pooled value at which the BH condition holds. Same selections; null mass 1.00 N⁰. Code, proof (Lemma 5, Remark 1), regression test and v3.2 docx updated |
| 2 | **Major** (code vs theorem) | Prop. 1(ii)–(iii) assumed distinct scores (A3), but discrete baselines (SCALE-UP takes 6 values) and the tree fusion produce ties | The implementation now breaks ties by independent uniforms before calibration (`cord.pipeline.random_tiebreak`, `calibration.tie_break: random`). A3 holds by construction; FDR on heavily tied scores is tested |
| 3 | **Major** (new result) | Prop. 4 was a conjecture | Lemma 4 is generalised to calibration sets with a fixed set of poisoned points. This gives Prop. 4(b)–(c): FDR ≤ α(1+m)/(1+m⁰) ≤ α/(1−ε) for fold-wise BH and e-BH, for any scores of the poisoned points. Prop. 4(a) holds for every audit that removes only poison. Simulation: bounds hold; the e-value bound is nearly attained by poison scoring lowest (1.976 vs 1.976) |
| 4 | Moderate | Prop. 1(i)–(iii) treated N_j⁰ and \|H₀∩D\| as constants; with a random hold-out they are random | Bounds restated with expectations; the clean-share argument conditions fold by fold |
| 5 | Moderate | Prop. 2 overstated: "nothing is selected" below the floor; per-class calibration "does not lower" it | Corollaries: below the floor P(selection) ≤ α/(1−ρ/ρ_sel), and every selection is mostly clean; per-class calibration lowers the floor by (m+1)/(m+K), 0.45% at m = 2,000, K = 10 |
| 6 | Moderate | No statement that tuning on evaluated data voids the guarantees (the proposal sweeps α_bh) | Remark on pre-specification: α, α_bh, t, k, signals, fusion and budgets are fixed before evaluation; picking the best hold-out redraw is not allowed |
| 7 | Minor | A presence-test rejection was read as "fold j is attacked" | Remark: the null is "no poison in D_j ∪ C_cal"; low-scoring calibration poison makes π_j smaller. A rejection says the dataset is attacked. The k > 1 variant is about power, not validity |
| 8 | Minor | Singular vectors are defined up to sign; a numerical SVD may set the sign by row order | Lemma 2(iv): the signals use only sign-invariant functions (squared projections, Mahalanobis distances) |
| 9 | Minor | A2 was called "equivalent" to its informal reading | Stated as a sufficient condition |

## Still open

1. **Prop. 4(b)–(c) for the default audit.** The audit inspects the top-b calibration points
   *by score*, so which poisoned points it reaches depends on the clean calibration points. Then
   the set of remaining poison is not fixed given 𝒢_j, and the stopping-time argument does not
   apply. Prop. 4(a) still holds. Study S2 finding (5) observed FDR ≤ 0.047 at α = 0.05. Options:
   prove a maximal-inequality version, or audit by a rule that does not look at calibration scores
   (e.g. random inspection), which the theorem covers at a cost in power.
2. **Residual-check lemma** (unchanged).

## For the outside statistician

The checklist at the end of `main.tex` lists what this review could not settle: completeness of
𝒢_j in A1, exact order-invariance of the fitted classifier, the stopping-time argument with
interleaved poisoned scores, and the two open items above. The simulations behind findings 1–3
are reproducible with the scripts in `tests/` (findings 1–2) and the table in `main.tex` §8
(finding 3).
