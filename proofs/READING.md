# Reading guide for Gate G0

Read in this order. For each paper: what to take from it, and where it is used in
`proofs/main.tex`. The proofs are self-contained, so the point of reading is to check them and to
cite correctly, not to import theorems.

| # | Paper (proposal ref.) | Take from it | Used in |
|---|---|---|---|
| 1 | Bates, Candès, Lei, Romano, Sesia, *Testing for outliers with conformal p-values*, Ann. Statist. 2023 [22] | Marginal validity of conformal p-values (the rank argument); why conformal p-values sharing one calibration set are dependent, and the PRDS result that makes BH valid for i.i.d. test points | Prop. 1(i); contrast with our proof of 1(ii), which avoids PRDS because our test and calibration points are drawn without replacement from one pool |
| 2 | Marandon, Lei, Mary, Roquain, *Adaptive novelty detection with FDR guarantee*, Ann. Statist. 2024 [23] | Learning the score on the pooled, unordered sample keeps exchangeability; the optimal score is a monotone function of the density ratio | Lemma 2 (A2 for the PU fusion, item viii); the power motivation in §4.1 |
| 3 | Mary and Roquain, *Semi-supervised multiple testing*, Electron. J. Statist. 2022 [44] | BH on empirical (conformal) p-values as an upward scan; the discreteness that creates the selection floor | Lemmas 3–4, Prop. 2 |
| 4 | Bashari, Epstein, Romano, Sesia, *Derandomized novelty detection with FDR control via conformal e-values*, NeurIPS 2023 [24] | Conformal e-values from a BH-type threshold; why the threshold must be a stopping value of the scan | Lemma 5 and Remark 1 (the correction to proposal eq. 3); code checked against their repository in `tests/test_reference_evalues.py` |
| 5 | Wang and Ramdas, *False discovery rate control with e-values*, JRSS-B 2022 [25] | e-BH, and its FDR bound under arbitrary dependence using only expectations of single e-values | Prop. 1(iii) |
| 6 | Bashari, Sesia, Romano, *Robust conformal outlier detection under contaminated reference data*, ICML 2025 [28] | Effect of contamination in the calibration set; Label-Trim's labelling budget on the most suspicious reference points | Prop. 4 (open); the audits of §3.3 |
| 7 | Qi, Xie, Li, Mahloujifar, Mittal, *Revisiting the assumption of latent separability for backdoor defenses*, ICLR 2023 [7] | How adaptive attacks suppress latent separation (Adap-Blend, Adap-Patch) | Threat model C2; the pilot's attack list |

## What to check while reading

1. **Paper 1:** compare their PRDS argument with Lemma 3 (supermartingale) of `main.tex`. Ours
   needs only that the split of clean points between fold and calibration is uniform (A1), which
   is what a random hold-out gives.
2. **Paper 2:** the exact condition under which a score learned on the pooled sample keeps
   validity. Compare it with Lemma 2(viii): our classifier is fitted on rows labelled
   `C_tr` = 0 and `D_j ∪ C_cal` = 1.
3. **Paper 4:** how their threshold is defined. In their code (`algo.compute_threshold`) the scan
   runs over the pooled scores, which agrees with our corrected definition. Proposal v3.1's eq. (3)
   ("the smallest score among its rejections") does not.
4. **Paper 5:** the proof of e-BH's FDR bound. Prop. 1(iii) reproduces it in three lines; check
   it uses only `Σ_null E[e_i] ≤ n`.
5. **Paper 6:** whether their contaminated-reference bound gives Prop. 4(a)–(b) directly, or only
   the non-adversarial case.

## Open questions for the statistician

The checklist at the end of `main.tex` lists them: completeness of the σ-field in A1, exact
order-invariance of the fitted classifier, the stopping-time argument when poisoned fold scores
interleave with clean ones, ties, and Proposition 4.
