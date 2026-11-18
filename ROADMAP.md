# CORD roadmap

Updated **2026-11-18** (week 0). `[x]` = done, `[ ]` = to do, **next** = the actions to take now.
Gate criteria are the proposal's (Table 7, as changed in v3.2). Week 0 started on Tuesday
2026-11-17; dates below are the last day of each period.

## Where we are

Everything that can be done without a GPU or another person is done or under way: the pipeline is
built and tested, the simulations reproduce the proposal, Propositions 1–3 are proved in full and
Proposition 4 mostly, and the proposal is at v3.2. Nothing has run on a GPU yet, and nobody outside
has checked the proofs. Those two are now the critical path.

| Gate | Due | Done | Blocking item |
|---|---|---|---|
| **G0** theory | 2026-12-21 | 6 of 7 | Outside statistician |
| **G1** CIFAR-10 pilot | 2027-01-25 | 0 of 7 (code ready) | First GPU run |
| **G2** contamination | 2027-03-01 | 1 of 5 (Prop. 4 mostly proved) | Pilot scores |
| **G3** core tier, adaptive attack | 2027-04-26 | 0 of 6 | GPU compute beyond free notebooks |
| **G4** downstream | 2027-06-07 | 0 of 2 | — |

## Next actions, in order

1. **next** Email for the two open roles: a statistician to check `proofs/main.tex` (send
   `proofs/REVIEW.md` with it), and a supervisor or collaborator with GPU access. Slowest items;
   both are preconditions in the proposal.
2. **next** First Kaggle session (`experiments/kaggle/README.md`): one condition, BadNets 5%,
   standard augmentation. Check: ASR above 90% in both fold models, seconds per epoch, peak memory,
   DINOv2 download works. Then re-score it without retraining:
   `--set fusion.reference=random results_tag=random` and `fusion.reference=oracle_clean
   results_tag=cleanref`.
3. **next** Read the seven papers with `proofs/READING.md` beside you.
4. Put Change 6 (Proposition 4, ties, pre-specification) into the v3.2 Word file; Changes 1–5 are in.
5. Check our attacks against an independent implementation (`backdoor-toolbox` or BackdoorBench):
   one BadNets and one Adap-Blend model, compare ASR and clean accuracy.
6. From the timed run: write the pilot budget (54 fold trainings + DINOv2 features) and start the
   grid.

## Timeline

| Weeks | Ends | Gate | Main work |
|---|---|---|---|
| 0 | 2026-11-23 | — | Pipeline, simulations, proofs, v3.2 (**mostly done**) |
| 1–4 | 2026-12-21 | **G0** | Outside proof check, first GPU run, attack check |
| 5–9 | 2027-01-25 | **G1** | CIFAR-10 pilot |
| 10–14 | 2027-03-01 | **G2** | Contamination and audit study |
| 15–22 | 2027-04-26 | **G3** | GTSRB, Tiny-ImageNet, adaptive attack |
| 23–28 | 2027-06-07 | **G4** | Downstream study |
| 29–32 | 2027-07-05 | — | Writing, novelty check, submission |

**Compute decision on 2027-01-26 (start of week 10).** Without university or cloud GPU by then, cut scope to
CIFAR-10 + GTSRB with fewer baselines; theory, the pilot and the contamination study still make a
paper.

## Done so far

**Code** (`src/cord`, 67 tests, lint and type checks clean, CI on GitHub)
- [x] Repository `github.com/Shahriar127/CORD-Research`, MIT license, layered package, YAML configs.
- [x] Data: CIFAR-10 loader with MD5 check and safe unpickling; verified downloader.
- [x] Attacks: BadNets, Blend, SIG, WaNet, Adaptive-Blend.
- [x] Cross-fitted PreAct-ResNet-18 trainer: data on the GPU, mixed precision, resumes mid-fold.
- [x] Feature cache per fold; five signal families; frozen DINOv2 encoder.
- [x] PU fusion with audited `C_tr`; naive fusions and single-signal baselines.
- [x] The three outputs; audits of both hold-out parts; random tie-breaking; B hold-out redraws.
- [x] Resumable stages; re-scoring from saved artifacts alone (no dataset needed, settings saved
      per scoring; added 2026-11-18).
- [x] Pilot grid runner (shards, time budget) and gate G1 checker.
- [x] e-values and e-BH checked against the reference code of Bashari et al.

**Statistics and theory**
- [x] Study S1 and S2 reproduce Section 6 of the proposal; `audit_sim.py` merged, one seed per setting.
- [x] Propositions 1–3 proved in full; Proposition 4 proved except for the score-ranked audit
      (`proofs/main.tex`, 10 pages).
- [x] Internal statistical review: 9 findings, all resolved (`proofs/REVIEW.md`).
- [x] Reading guide (`proofs/READING.md`).

**Experiments on a laptop CPU**
- [x] Smoke run (all stages, about 1 minute).
- [x] Sanity run: backdoor not learned (ASR 6%); signals then reverse direction.
- [x] Mini-pilot: backdoor learned (ASR 99%); held-out poison separated (SCALE-UP AUROC 0.95,
      confidence 0.92, STRIP 0.82); with audits of 300 points, fused AUROC 0.90 and removal recall
      0.78.

**Proposal**
- [x] v3.2 (`~/Downloads/CORD_Research_Proposal_v3.2.docx`), with evidence for every change in
      `docs/proposal_v3.2_changes.md`:
      1. audit of the fusion reference `C_tr`;
      2. audit budgets of about 3ρm;
      3. shared calibration set in Study S2;
      4. signal direction depends on whether the backdoor was learned;
      5. corrected e-value threshold (eq. 3) and a precise selection floor.
- [ ] Change 6 (Proposition 4, ties, pre-specification) still to go into the Word file.

## Gate G0 — theory (due 2026-12-21)

- [x] Propositions 1–3 written in full.
- [x] Internal review of the proofs.
- [x] e-values and e-BH checked against the reference implementation.
- [x] Reading guide written.
- [x] S1 and S2 reproduce Section 6.
- [x] S1 and S2 reproduce from a clean clone of the pushed repository (commit `7446b99`,
      2026-11-18): all six result files byte-identical.
- [ ] **next** Outside statistician checks `proofs/main.tex`.

**Passes when** Propositions 1–3 are checked by an outside statistician and S1/S2 reproduce from
a clean clone. **Fallback:** claim only the single-fold (split-sample) guarantee.

## Gate G1 — CIFAR-10 pilot (due 2027-01-25)

- [ ] **next** One timed GPU condition (see Next actions, item 2).
- [ ] Attack check against `backdoor-toolbox` / BackdoorBench.
- [ ] Run the grid: 4 attacks × 2 rates × 3 seeds + 3 no-attack seeds = 27 conditions, 54 fold
      trainings (`python experiments/pilot/run_grid.py --max-hours 11`; rerun to resume).
- [ ] Per signal: held-out AUROC and rank correlation; keep a signal only if it adds separation on
      an attack the others miss, then **freeze** the signal set.
- [ ] Report runs with low ASR separately (signals reverse there).
- [ ] At least 30 no-attack seeds for the false-alarm estimate.
- [ ] `python experiments/pilot/check_gate_g1.py runs/pilot`.

**Passes when** FDR upper bound ≤ 1.1α on all four attacks, presence false-alarm rate ≤ 0.07,
FDR-mode power ≥ 50% at 5% on ≥ 2 attacks, presence ≥ 80% at 1% on ≥ 2 attacks, fusion within 5
points of the best single signal. **Fallback:** validity fails → stop and find the broken
assumption; only power fails → lead with the presence test, the floors and contamination.

## Gate G2 — contamination (due 2027-03-01)

- [x] Proposition 4: FDR ≤ α/(1−ε) for any placement of poisoned calibration points, when the
      audit does not select points by score.
- [ ] Proposition 4 for the default score-ranked audit: prove it, or compare with a
      random-inspection audit (covered by the theorem) in E5 and choose.
- [ ] E5 on cached scores: audit budgets at 1, 2, 3 and 5 × ρm for both hold-out parts, auditor
      miss rate, wrong removals (`audit.*` with `results_tag`).
- [ ] Measure audit cost: distinct points inspected per dataset, time per inspection.
- [ ] Residual-check lemma.

**Passes when** Proposition 4 holds (or is verified) and audits of about 3ρm points per hold-out
part restore at least 80% of poison-free-hold-out power on dirty-label attacks. **Fallback:** make
mode T2 the default and report T1 as a limit result.

## Gate G3 — core tier and adaptive attack (due 2027-04-26)

- [ ] GTSRB loader, config and test (README, "Adding a dataset").
- [ ] Tiny-ImageNet loader; VGG-19-BN model.
- [ ] More attacks: ISSBA, Label-Consistent, Narcissus, Adap-Patch.
- [ ] Baselines: check which have public code first (unchecked: SPC, RPP, CASCADE, ASSET, Beatrix,
      FLARE; BackdoorBench has AGPD, SPECTRE, Spectral Signatures, STRIP).
- [ ] Defence-aware attack (C2) and the evasion frontier.
- [ ] Validity on the core tier for attacker classes C1–C3.

## Gate G4 — downstream (due 2027-06-07), then writing

- [ ] Retraining without the removal set; exact purification with the proxy set; durability under
      retuning and reactivation.
- [ ] Novelty check for new preprints; preprint after G3; submit (target IEEE TIFS).

## Risks

| Risk | Status | What to do |
|---|---|---|
| No GPU compute beyond free notebooks | **open** | Ask now; decide on 2027-01-26 |
| No outside statistician for G0 | **open** | Ask now; `proofs/REVIEW.md` is the starting point |
| Signals weaken under cross-fitting | Partly answered (strong when the backdoor is learned, CPU) | First GPU run |
| Fold models miss low-rate backdoors | Seen on CPU | Measure at ρ = 1%; report separately |
| Default audit not covered by Proposition 4 | New (review) | Prove, or switch to random inspection after E5 |
| Audit cost at 3ρm per hold-out part | Open | Measure in G2 |
| Our attacks differ from published ones | Open | Attack check in G1 |
| A competing paper appears | Open | Preprint after G3; novelty check before submission |

## Datasets

| Dataset | Tier | How to get it | Status |
|---|---|---|---|
| CIFAR-10 | Pilot + core | `cord download cifar10` (MD5-checked) | **Done** |
| GTSRB | Core | `torchvision.datasets.GTSRB(download=True)` | Loader to write |
| Tiny-ImageNet | Core | Stanford's cs231n server, `tiny-imagenet-200.zip` (about 248 MB) | Loader to write |
| Long-tailed CIFAR-10 | Imbalance | Subsample CIFAR-10 with a fixed ratio and seed | To write |
| ISIC 2019 | Imbalance | challenge2019.isic-archive.com (CC-BY-NC 4.0; about 12.7 GB) | To write |
| ImageNet-100 | Scale (optional) | 100-class subset of ImageNet-1k; state the class list | Only with compute |

## Code to build on or check against

- **`vtu81/backdoor-toolbox`**: Qi et al.'s adaptive attacks; for checking our attacks.
- **`SCLBD/BackdoorBench`**: datasets, models and baselines matching Table 6.
- **`Meshiba/derandomized-novelty-detection`**: reference for e-values and e-BH (**checked**).

Sources (from the first roadmap; not re-checked):
[BackdoorBench](https://github.com/SCLBD/BackdoorBench) ·
[Circumventing Backdoor Defenses](https://github.com/Unispac/Circumventing-Backdoor-Defenses) ·
[Derandomized novelty detection](https://github.com/Meshiba/derandomized-novelty-detection) ·
[Tiny ImageNet record, SJTU](https://scidata.sjtu.edu.cn/records/swpcp-01e87) ·
[Fed-ISIC 2019 licence notes](https://owkin.github.io/FLamby/fed_isic.html) ·
[Processed ISIC2019, IEEE DataPort](https://ieee-dataport.org/node/15393) ·
[Kaggle free GPU quota comparison](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/)
