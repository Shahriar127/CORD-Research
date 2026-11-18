# CORD roadmap

Updated 2026-11-17 (week 0). This revises the first roadmap after the pipeline was built and
tested. Status marks: **done**, **next**, **open**. Gate criteria are the proposal's (Table 7, as
changed in v3.2).

## What changed from the first roadmap

| First roadmap said | Now | Why |
|---|---|---|
| Use `backdoor-toolbox` for attacks and the pilot | CORD has its own pipeline (`src/cord`) with five attacks, cross-fitted training and all signals; `backdoor-toolbox` and BackdoorBench are used to **check our attacks** and for **baselines** | Cross-fitting, the feature cache and the conformal outputs need code built around the hold-out and folds |
| Build the pilot code in weeks 5–9, one signal at a time | Built and tested on CPU (66 tests). Weeks 5–9 are now **running** the pilot | Done ahead of plan |
| "Cross-fitting may weaken the signals" | Partly answered: once a backdoor is learned, held-out poison is separated (SCALE-UP AUROC 0.95, confidence 0.92, STRIP 0.82; CPU mini-pilot) | Needs confirming on GPU with augmentation and full data |
| Audit the top 100–200 calibration points | Audit **both** parts of the hold-out (C_tr and C_cal), with budgets about **3ρm** each (300 in the configs) | Mini-pilot: 100-point audits blocked every FDR selection; proposal v3.2, Changes 1–2 |
| Figure 3a as in v3.1 | Shared calibration set is the main result; presence power is lower (91% vs 97% at ρ = 0.5%) | Proposal v3.2, Change 3 |
| Proposal v3.1 | **v3.2** written (`~/Downloads/CORD_Research_Proposal_v3.2.docx`, evidence in `docs/proposal_v3.2_changes.md`) | — |
| Time the pilot from one BadNets run | Still true: nothing has run on a GPU yet | The cost table must use measured numbers |

New risk found: **audit cost**. Auditing 300 points of each hold-out part per fold means up to
1,200 inspections per dataset (2 parts × 2 folds × 300; less where folds pick the same points).
This is still far fewer than auditing the hold-out in full, but it has to be argued in the paper and
measured (how many distinct points, how long one inspection takes).

## Timeline

| Week | Date (Monday) | Gate | Main work |
|---|---|---|---|
| 0 | 2026-11-17 | — | Pipeline, simulations, v3.2 (**done**) |
| 1–4 | → 2026-12-15 | **G0** | Proofs, first GPU run, attack check |
| 5–9 | → 2027-01-19 | **G1** | CIFAR-10 pilot |
| 10–14 | → 2027-02-23 | **G2** | Contamination and audit study |
| 15–22 | → 2027-04-20 | **G3** | Core tier (GTSRB, Tiny-ImageNet), adaptive attack |
| 23–28 | → 2027-06-01 | **G4** | Downstream study |
| 29–32 | → 2027-06-29 | — | Writing, novelty check, submission |

Compute decision point: **week 10 (2027-01-26)**. Without university or cloud GPU by then, cut
scope to CIFAR-10 + GTSRB with fewer baselines (see the end).

## Week 0 (laptop) — done

- [x] Git repository with `src/cord`, `sim/`, `proofs/`, `experiments/`, `configs/`, `tests/`;
      pushed to `github.com/Shahriar127/CORD-Research`. MIT license.
- [x] `audit_sim.py` merged into S2 (`sim/s2_presence_contamination.py`), one seed per setting.
- [x] S1 and S2 reproduce Section 6 of the proposal within Monte-Carlo error.
- [x] Full pipeline: attacks, cross-fitted trainer, feature cache, five signals, PU fusion, audits,
      three outputs, B hold-out redraws, gate G1 checker, Kaggle guide.
- [x] CPU checks: smoke run, sanity run (backdoor not learned), mini-pilot (backdoor learned).
- [x] Proposal v3.2 with the three design changes.

## Weeks 1–4 — Gate G0 (theory + first GPU run)

**Theory (laptop)**
- [ ] **next** Read in this order: Bates et al. (conformal p-values), Marandon et al. (AdaDetect),
      Bashari et al. 2023 (e-values), Wang and Ramdas (e-BH), Bashari et al. 2025 (Label-Trim),
      Qi et al. (adaptive attacks). What to take from each and where it is used:
      `proofs/READING.md`.
- [x] Write Propositions 1–3 in full in `proofs/main.tex` (2026-11-18; 8 pages, self-contained
      supermartingale proofs; open items listed at its end). Writing them found a bug in eq. (3),
      now fixed (`docs/proposal_v3.2_changes.md`, Change 5). A2 is verified component by component
      in Lemma 2, including the two items new in v3.2 (C_tr audit, tie-breaking of the fused score).
- [x] Internal statistical review (`proofs/REVIEW.md`, 2026-11-18): 9 findings fixed, including
      enforced tie-breaking and a proof of Proposition 4 for audits that do not use calibration
      scores.
- [ ] **next** Get `proofs/main.tex` checked by an outside statistician; start from
      `proofs/REVIEW.md` (still a precondition for G0).
- [x] Check `cord.stats` e-values and e-BH against `Meshiba/derandomized-novelty-detection` on the
      same inputs (one small script in `tests/`).
- [ ] **open** Email for the two roles: a statistician to check the proofs, and a supervisor or
      collaborator with GPU access. Slowest items; preconditions in the proposal.

**First GPU session (Kaggle)** — follow `experiments/kaggle/README.md`
- [ ] **next** One condition, BadNets 5%, standard augmentation. Record seconds per epoch and peak
      memory. **Check ASR is above 90%** in both fold models.
- [ ] Score it three ways without retraining: audited (default), `fusion.reference=random`,
      `fusion.reference=oracle_clean`. Confirm the mini-pilot's ordering holds.
- [ ] Check the attacks against an independent implementation: train one BadNets and one
      Adap-Blend model with `backdoor-toolbox` or BackdoorBench and compare ASR and clean
      accuracy with ours. Our Blend and Adap-Blend use a seeded noise pattern; pass
      `attack.params.trigger_image=<hello_kitty.png>` for the original trigger.
- [ ] Turn the measured run time into the pilot budget (54 fold trainings + DINOv2 features).

**G0 passes when** Propositions 1–3 are written and checked by an outside statistician, and
S1/S2 reproduce from a clean clone. **Fallback:** claim only the single-fold (split-sample)
guarantee.

## Weeks 5–9 — Gate G1 (CIFAR-10 pilot)

- [ ] Run the grid: 4 attacks × 2 rates × 3 seeds + 3 no-attack seeds = 27 conditions, 54 fold
      trainings. `python experiments/pilot/run_grid.py --max-hours 11`, rerun to resume, `--shard`
      to split across sessions.
- [ ] Per signal: held-out AUROC and rank correlation between signals. Keep a signal only if it adds
      separation on an attack the others miss, then **freeze** the signal set.
- [ ] Report runs where a fold model did not learn the backdoor (low ASR) separately: there,
      confidence-type signals reverse direction (seen in the CPU sanity run).
- [ ] Sweep audit budgets on cached scores (no retraining): `audit.budget`, `audit.budget_tr`
      ∈ {1, 2, 3, 5} × ρm.
- [ ] At least 30 no-attack seeds for the false-alarm estimate (Section 7.3). Can run on a small
      budget after the main grid.
- [ ] `python experiments/pilot/check_gate_g1.py runs/pilot`.

**G1 passes when** the FDR upper bound ≤ 1.1α on all four attacks, the presence false-alarm rate
≤ 0.07, FDR-mode power ≥ 50% at 5% on ≥ 2 attacks, presence ≥ 80% at 1% on ≥ 2 attacks, and the
fusion is within 5 points of the best single signal. **Fallback:** validity fails → stop and find
the broken assumption; only power fails → lead with the presence test, the floors and the
contamination analysis.

## Weeks 10–14 — Gate G2 (contamination)

- [x] Proposition 4 proved for audits that do not use calibration scores (FDR ≤ α/(1−ε)).
- [ ] Proposition 4(b)–(c) for the score-ranked top-b audit: prove, or compare with a random-
      inspection audit (covered by the theorem) in E5.
- [ ] Experiment E5 on cached scores: audit budgets as multiples of ρm for both hold-out parts,
      auditor miss rate, wrong removals (`audit.*` settings with `results_tag`).
- [ ] Measure audit cost: distinct points inspected per dataset and time per inspection.

**G2 passes when** Proposition 4 holds (or is verified), and audits of about 3ρm points per
hold-out part restore at least 80% of poison-free-hold-out power on dirty-label attacks.
**Fallback:** make mode T2 (external trusted set) the default and report T1 as a limit result.

## Weeks 15–32 — core tier, adaptive attack, downstream, writing

- [ ] GTSRB: loader in `src/cord/data/`, config `configs/core/gtsrb.yaml`, test (README, "Adding a
      dataset"). Then Tiny-ImageNet and VGG-19-BN.
- [ ] More attacks: ISSBA, Label-Consistent, Narcissus, Adap-Patch.
- [ ] Baselines: check which have public code before promising them. Unchecked so far: SPC, RPP,
      CASCADE, ASSET, Beatrix, FLARE. BackdoorBench already has AGPD, SPECTRE, Spectral Signatures
      and STRIP.
- [ ] Defence-aware attack (C2) and the evasion frontier (G3, week 22).
- [ ] Downstream study: retraining and exact purification (G4, week 28).
- [ ] Novelty check for new preprints on conformal backdoor detection, then submit (target TIFS).

The proposal's 1,000–1,500 runs would take well over a year on free notebooks. If no compute
arrives by week 10, cut scope to **CIFAR-10 + GTSRB with fewer baselines**: theory, the pilot and
the contamination study still make a paper.

## Biggest risks

| Risk | Status | What to do |
|---|---|---|
| No GPU compute for the core tier | **open** | Ask from week 1; decide at week 10 |
| No statistician for G0 | **open** | Ask from week 1 |
| Signals weaken under cross-fitting | Partly answered (CPU mini-pilot: strong when the backdoor is learned) | Confirm on the first GPU run |
| Fold models fail to learn low-rate backdoors | New | Measure in the pilot (ρ = 1%); report those runs separately |
| Audit too costly or not credible at 3ρm per part | New | Measure in G2; argue against auditing in full |
| Our attacks differ from published ones | New | Compare ASR with `backdoor-toolbox` / BackdoorBench in weeks 1–4 |
| A competing paper appears | **open** | Preprint after G3; novelty check before submission |

## Datasets

| Dataset | Tier | How to get it | Status in CORD |
|---|---|---|---|
| CIFAR-10 | Pilot + core | `cord download cifar10` (MD5-checked) | **Loader done** |
| GTSRB | Core | `torchvision.datasets.GTSRB(download=True)` | Loader to write (week 15) |
| Tiny-ImageNet | Core | Stanford's cs231n server, `tiny-imagenet-200.zip` (about 248 MB) | Loader to write |
| Long-tailed CIFAR-10 | Imbalance | Subsample CIFAR-10 with a fixed imbalance ratio and seed | To write |
| ISIC 2019 | Imbalance | challenge2019.isic-archive.com (CC-BY-NC 4.0; training zip about 12.7 GB) | To write |
| ImageNet-100 | Scale (optional) | 100-class subset of ImageNet-1k (free research registration); state the class list used | Only if compute allows |

## Code to build on or check against

- **`vtu81/backdoor-toolbox`**: Qi et al.'s adaptive attacks (Adap-Blend, Adap-Patch); use it to
  check our attack implementations.
- **`SCLBD/BackdoorBench`**: CIFAR-10/100, GTSRB, Tiny-ImageNet with PreAct-ResNet-18 and
  VGG-19-BN (matches Table 6); baselines AGPD, SPECTRE, Spectral Signatures, STRIP.
- **`Meshiba/derandomized-novelty-detection`**: Bashari et al.'s E-AdaDetect; use it to check our
  e-values and e-BH.

Sources (from the first roadmap; not re-checked here):
[BackdoorBench](https://github.com/SCLBD/BackdoorBench) ·
[Circumventing Backdoor Defenses](https://github.com/Unispac/Circumventing-Backdoor-Defenses) ·
[Derandomized novelty detection](https://github.com/Meshiba/derandomized-novelty-detection) ·
[Tiny ImageNet record, SJTU](https://scidata.sjtu.edu.cn/records/swpcp-01e87) ·
[Fed-ISIC 2019 licence notes](https://owkin.github.io/FLamby/fed_isic.html) ·
[Processed ISIC2019, IEEE DataPort](https://ieee-dataport.org/node/15393) ·
[Kaggle free GPU quota comparison](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/)
