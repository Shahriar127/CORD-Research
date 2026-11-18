# Proposal changes since v3.1, with their evidence

Current version: **v3.3** (`docs/CORD_Research_Proposal_v3.3.docx`) = v3.2 (Changes 1–5) plus
Change 7 below. Change 6 is not yet in the Word file.

Drafted 2026-11-17 from what the pipeline, the synthetic studies and two CPU runs showed. Each
item gives the reason, the evidence, and replacement text to paste into the proposal.

Evidence comes from two laptop runs on a 10k-image CIFAR-10 subset (J = 2, m_tr = m_cal = 1,000):

* **Run A** (`runs/cpu-sanity`): BadNets 5%, width 32, 15 epochs, standard augmentation. The fold
  models did **not** learn the backdoor (ASR ≈ 6%).
* **Run B** (`configs/cpu_minipilot.yaml`): BadNets 10%, width 16, 30 epochs, augmentation off.
  The fold models learned the backdoor (ASR 98.9% and 99.1%; clean accuracy 53–54%).

All numbers are means over 5 hold-out resamples of one pair of fold models: indicative, not
results.

---

## Change 1 — Audit the fusion reference `C_tr` (Sections 3.3, 4.1, Table 2, RQ5)

**Problem.** In mode T1, `C_tr` is a random hold-out, so it contains poison at the pool rate ρ.
The positive-unlabeled classifier separates `C_tr` from `D_j ∪ C_cal`, but both are samples of the
same mixture, so the estimated density ratio is close to 1 and the fused score loses most of the
signals' information. Validity is unaffected; power collapses.

**Evidence** (Run B; budgets are per fold; the clean share in R stayed at 0.10–0.11 throughout):

| Fusion reference | `C_tr` poison removed per fold | Fused AUROC | R recall | Presence |
|---|---|---|---|---|
| audited, b_tr = 100 | 36 of 97 | 0.76 | 0.46 | 0.6 |
| audited, b_tr = 300 (and b_cal = 300) | 70 of 97 | 0.90 | 0.78 | 1.0 |
| poison-free (upper bound), b_cal = 300 | 97 of 97 | 0.98 | 0.96 | 1.0 |

The best single signal reached AUROC 0.95 (SCALE-UP consistency): with a clean reference, PU
fusion exceeds it, as RQ3 hypothesises. In Run A, the unaudited reference gave AUROC 0.53 against
0.76 for a poison-free one.

**Replacement text, Section 3.3 (after the paragraph on mode T1).**

> The hold-out serves two purposes and is audited for both. The calibration part `C_cal` is
> audited as described above. The fusion part `C_tr` is the negative class of the
> positive-unlabeled classifier (Section 4.1); if it contains poison at the pool rate, the
> classifier compares two samples of the same mixture and the fused score loses its power,
> although not its validity. For each fold, the defender therefore ranks `C_tr` by the mean of
> within-set signal ranks, which does not depend on the fusion, inspects its `b_tr`
> highest-ranked points under the same positive-evidence rule, and removes confirmed poison
> from the fusion's training data. Because `C_tr` is independent of the scored points, this
> audit, including any mistakes it makes, cannot affect validity; it changes power only.

**Replacement text, Section 4.1 (positive-unlabeled fusion, after "No attack labels are used.").**

> The negative class is the audited `C_tr` (Section 3.3). The ranking used to select `C_tr`
> points for audit is a symmetric function of `D_j ∪ C`, so A2 holds.

**Table 2, row C4.** Replace "Poison lands in the hold-out and is not recognised by the audit"
with "Poison lands in `C_cal` or `C_tr` and is not recognised by the audit", and add to the power
column: "audit misses in `C_tr` reduce fusion power only".

**Section 7.2, baselines → fusion.** Add "mean rank of signals (reference-free)".

---

## Change 2 — Audit budgets scale with the expected poison count (Section 3.3, RQ5, Gate G2)

**Problem.** v3.1 states that "b in the low hundreds" suffices and RQ5 hypothesises recovery with
`b ≤ 200`. A top-b audit removes at most the poison it is shown. At ρ = 10% and m = 1,000, each
hold-out part contains about 100 poisoned points; a 100-point audit removed only 30–60 of them
per fold (the rest did not rank in the top 100), and every FDR selection and most presence
decisions were blocked.

**Evidence** (Run B, fused score): fold-wise BH power was 0 at b_cal = b_tr = 100 and 0.16
(FDR 0.051) at b_cal = b_tr = 300; with a poison-free `C_tr` and b_cal = 300 it was 0.56
(FDR 0.058), and e-BH power 0.31 (FDR 0.016).

**Replacement text, Section 3.3 (replace "With b in the low hundreds this replaces the audit of
thousands of samples assumed in earlier designs.").**

> The audit must be able to see the poison that matters: since a top-b audit removes at most the
> poison ranked in its top b, b has to exceed the expected number of high-scoring poisoned points,
> about ρ·m. In our runs, b ≈ 3ρm restored every output; for the rates and calibration sizes
> studied here this is a few hundred points, still far fewer than auditing the hold-out in full.

**RQ5 (Table 3), hypothesis.** Replace "auditing b ≤ 200 points restores at least 80% of
clean-calibration power" with "auditing b ≈ 3ρm points of each hold-out part restores at least
80% of the power obtained with poison-free hold-out parts, for audit-visible attacks". Gate G2:
same replacement.

**Experiment E5.** Sweep b as a multiple of ρm (1×, 2×, 3×, 5×) rather than at fixed values, for
both `C_cal` and `C_tr`.

---

## Change 3 — Presence test with a shared calibration set (Section 6.2, Figure 3a)

**Problem.** The v3.1 numbers in Section 6.2, finding (2), and Figure 3a come from a
simulation that drew an independent calibration set for each fold. In the method, both fold
models score the same `C_cal`. The fold p-values are then positively dependent, and the
Bonferroni combination gains less from the second fold. Validity is unaffected (Bonferroni holds
under any dependence; the false-alarm rate falls to about 0.03 at α = 0.05).

**Evidence** (Study S2, shift 4.5, two folds of 5,000, m_cal = 1,000, 1,000 repetitions):

| ρ | 0.3% | 0.4% | 0.5% | false alarm (ρ = 0) |
|---|---|---|---|---|
| independent `C_cal` per fold (v3.1) | 0.47 | 0.80 | 0.97 | 0.05 |
| shared `C_cal` (the method) | 0.34 | 0.66 | 0.91 | 0.03 |

**Replacement text, Section 6.2, Setup (append).**

> Both folds are scored against the same calibration set, as in the method; Figure 3a also shows,
> dotted, a variant with an independent calibration set per fold.

**Replacement text, Section 6.2, finding (1).**

> (1) With no poison the presence test rejected in 2.7% and 2.6% of runs in the two
> configurations, below the level 0.05: with a shared calibration set the fold p-values are
> positively dependent and the Bonferroni combination is conservative.

**Replacement text, Section 6.2, finding (2), first sentence.**

> (2) At shift 4.5 it detected the attack in 34% of runs at ρ = 0.3%, 66% at 0.4% and 91% at
> 0.5% (47%, 80% and 97% with an independent calibration set per fold), against a predicted
> threshold of 0.42% from the exact form of (6) with N_j = 5,000.

**Figure 3a caption (append).** "Solid: shared calibration set (the method). Dotted: independent
calibration set per fold."

**Note for Section 4.2.** The statement "needs no assumption on dependence between folds" stays
true. Do not replace Bonferroni with a combination that assumes independent folds (Fisher,
Šidák): with a shared `C_cal` that would be invalid.

---

## Change 4 — Signals under cross-fitting, and when they flip (Sections 4.1, 7.1, Table 4)

**Evidence.** With a learned backdoor (Run B), held-out poison was separated by
SCALE-UP consistency (AUROC 0.95), confidence trajectory (0.92) and STRIP (0.82); latent
geometry, activation dominance and label-content (pixel encoder) were near chance in this small
model. With an unlearned backdoor (Run A), the confidence and gradient signals reversed
direction (AUROC 0.18 and 0.09): held-out poison then looks like mislabelled data, and its given
(target) label is predicted with *low* confidence.

**Text for Section 4.1, after the description of signals.**

> A signal's direction depends on whether the fold model has learned the backdoor: when it has
> not, held-out poison behaves like mislabelled data and confidence-based signals reverse. The
> positive-unlabeled fusion learns directions from the data and is unaffected; fixed-direction
> combinations (mean rank) and single-signal baselines are not, which the pilot reports
> separately for runs with low attack success.

**Table 4.** Replace the blind-spot entries with measured per-signal AUROCs after the pilot.

**Implementation note.** A data-driven orientation (tail skewness) was tried for the `C_tr`
ranking. It fixed the reversed signals in Run A but mis-oriented bounded signals in Run B
(AUROC 0.72 against 0.84 for the designed orientation), so the designed orientation is the
default; the oriented variant remains available as `audit.tr_ranking=oriented_mean_rank`.

**Open items for the GPU pilot.** Confirm Run B with augmentation, full data and width 64;
measure how often fold models fail to learn low-rate backdoors (ρ = 1%), which is where the
reversal matters.

---

## Change 5 — Threshold of the conformal e-values (Section 4.1, eq. 3) and the selection floor (Section 5)

**Problem.** v3.1 defines the e-value threshold as "the smallest score among [BH's] rejections".
The validity of the e-values (Σ over null points of E[e_i] ≤ N_j, which e-BH needs) holds when the
threshold is a *stopping time* of an upward scan over the pooled fold and calibration scores. The
smallest rejected fold score is not one: whether the next point up is a fold point is not yet
known when the scan must decide. Selections are the same either way, but the v3.1 threshold counts
fewer calibration points and gives larger e-values.

**Evidence.** Simulation with 5 clean and 40 strongly shifted fold points, m = 2, α_bh = 0.9,
40,000 repetitions: null mass 1.22 N_j⁰ with the v3.1 threshold, 1.00 N_j⁰ with the stopping value
(standard error 0.003). At m ≥ 1,000 the difference is negligible (Study S1 is unchanged).
Fixed in `cord.stats.conformal_evalues`, checked against the reference code of Bashari et al. and
a brute-force implementation, with a regression test; proof in `proofs/main.tex`, Lemma 5 and
Remark 1.

**Replacement text, Section 4.1.** "Run BH at level α_bh on the p-values of fold j and let T_j be
the first value, scanning the pooled scores of the fold and of the calibration set upwards, at
which the BH condition holds; it may be a calibration score, and the points scoring at least this
value are exactly the BH rejections."

**Selection floor (Section 5).** Writing out Proposition 2 made two statements precise:
- below the floor a selection is not impossible but rare and mostly clean: every non-empty
  selection has FDP ≥ 1 − ρ/ρ_sel, so under FDR control its probability is at most
  α/(1 − ρ/ρ_sel);
- per-class calibration lowers the floor by the factor (m+1)/(m+K), not "not at all"; negligible
  at the sizes studied (0.4% at m = 2,000, K = 10).

Both are in the v3.2 document (Section 4.1, Section 5 and Appendix C).

---

## Change 6 — From the statistical review (Proposition 4, ties, pre-specification)

Source: `proofs/REVIEW.md` (internal review, 2026-11-18) and `proofs/main.tex`.

- **Proposition 4 is proved** for contamination that does not depend on the clean calibration
  points (no audit, a full audit, or an audit that does not look at calibration scores):
  fold-wise BH and e-BH have FDR ≤ α(1+m)/(1+m⁰) ≤ α/(1−ε) for *any* scores of the poisoned
  calibration points, with no O(1/m) term; the p-value bound (a) holds for every audit that
  removes only poison. **Text:** Table 2, row C4, validity column: replace "FDR inflated by at
  most 1/(1−ε) (Proposition 4, to be proved)" with "FDR inflated by at most a factor 1/(1−ε)
  (Proposition 4; proved unless the audit selects points by score)". Section 5, Proposition 4:
  replace "to be proved" and "we conjecture" by the statement above, and keep step (b) for the
  score-ranked audit as open, supported by Study S2 finding (5).
- **Ties.** Scores are made distinct by random tie-breaking before calibration; Section 4.1 should
  say so in one sentence ("ties between scores are broken by independent uniform variables").
- **Pre-specification.** Section 7.3, add: "α, α_bh, the removal level, k, the signal set, the
  fusion method and the audit budgets are fixed before the evaluated data are seen; the α_bh sweep
  of Section 6.1 is run on development conditions only."

---

## Change 7 — RBD [16] (made directly in v3.3)

Edits made in another session to produce v3.3; recorded here so the project stays consistent.

- **Introduction, "No threshold is principled":** STRIP [14] and RBD [16] fix a false-rejection
  rate on clean hold-out data, the empirical form of the removal-set bound; they say nothing about
  the purity of the flagged set or about whether an attack happened.
- **Section 2.1:** [16] is described as "agreement with a backdoor-only shadow model" (it was
  "activation dominance").
- **Table 4, activation row:** adds shadow-model agreement as a second form of the signal, and
  "tasks with few classes" as a blind spot.
- **Section 7.2:** RBD [16] joins the detection baselines.

**Consequences for the project.** (1) Shadow-model agreement is not implemented; it must be built
before the pilot or removed from Table 4, because the signal set is frozen after the pilot.
(2) A shadow model satisfies A2 only if it is derived from the fold model with data outside
`D_j ∪ C_cal` (`proofs/main.tex`, Lemma 2(x)). (3) RBD is a baseline to implement or obtain, with
the same trusted hold-out as CORD. (4) Check the volume and pages of [16] on IEEE Xplore
(DOI 10.1109/TDSC.2024.3369751) before submission.

