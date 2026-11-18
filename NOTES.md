# CORD — project notes

Last updated: 2026-11-17

The week-by-week plan with gates and dates is in [`ROADMAP.md`](ROADMAP.md).

## 1. What has been done

### Pipeline (`src/cord/`)
- **Statistics core** (`stats/`): conformal p-values and e-values, BH, BY and e-BH, the exact
  presence test (with the k-th-largest variant), Fisher and higher-criticism baselines, the
  detection floors and the Storey rate estimate. `cord floors` reproduces Table 5.
- **Data** (`data/`): CIFAR-10 loader with MD5 checks and a restricted unpickler, a dataset
  registry for GTSRB and later datasets, and `cord download cifar10`.
- **Attacks** (`attacks/`): BadNets, Blend, SIG, WaNet and Adaptive-Blend. A seeded trigger lets
  each run regenerate its poisoned pool, which is verified by a SHA-256 digest.
- **Splits** (`splits.py`): order of events (poison → hold-out → folds), C_tr / C_cal, J folds,
  and the B redraws of the hold-out.
- **Model and training**: PreAct-ResNet-18 with multi-layer `embed()`. Data and augmentation sit
  on the GPU, with mixed precision and resumable state, and the confidence trajectory is
  recorded during training.
- **Feature cache** (`features/`): one pass per fold (layers, logits, SCALE-UP, STRIP), plus a
  frozen DINOv2 encoder (or `pixels` for offline tests).
- **Signals** (`signals/`): latent geometry, confidence trajectory, perturbation consistency,
  activation dominance and label-content disagreement. All are permutation-equivariant (A2).
- **Fusion** (`fusion/`): AdaDetect-style PU fusion, mean/max-rank and single-signal baselines,
  and the exchangeability diagnostic. `fusion.reference: random | oracle_clean`.
- **Audit** (`audit.py`): oracle auditor with a miss rate and wrong removals (RQ5 / E5).
- **Outputs** (`pipeline.py`): presence test, removal set R, proxy set F (e-BH), fold-wise and
  pooled BH, and ground-truth metrics.
- **Runner and CLI**: stages prepare → train → extract → ssl → score, each resumable;
  `results_tag` writes extra scorings into `results-<tag>/` without retraining.

### Experiments and studies
- `experiments/pilot/`: grid of 27 conditions (shardable, time-budgeted) and the gate G1 checker.
- `experiments/kaggle/README.md`: step-by-step GPU instructions.
- `sim/`: Studies S1 and S2, with `audit_sim.py` merged into S2 and one seed per setting.
  Figures and CSVs are in `sim/results/`.
- `proofs/main.tex`: G0 skeleton with assumptions, Propositions 1–4 and TODOs.

### Quality
- 47 tests pass (exactness, Monte-Carlo validity, A2 equivariance, end to end, resume,
  downloader safety). Ruff and mypy are clean. CI is in `.github/workflows/ci.yml`.
- A git repository is initialised. **Nothing is committed yet**, and there is no LICENSE.

### Results so far
- **S1 and S2 reproduce Section 6** within Monte-Carlo error. For example, S1 FDR is
  0.044/0.040/0.033 against 0.043/0.041/0.033, and S2 finding 6 is 0.04/0.13/0.20/0.31 against
  0.04/0.12/0.20/0.31.
- **CPU sanity run** (`runs/cpu-sanity`: 10k CIFAR images, BadNets 5%, 15 epochs, width 32):
  validity holds, with the clean share in R at about 0.10. The backdoor was **not** learned
  (ASR about 6%), so the run says nothing about power on real attacks.

## 2. Findings that affect the proposal

1. **PU fusion in mode T1.** C_tr is a random hold-out, so it holds poison at the pool rate and
   the PU classifier learns almost nothing.
   - On the same models, fused AUROC was 0.53 with a random C_tr and 0.76 with a clean C_tr.
   - Removal recall rose from 0.12 to 0.46, while the clean share stayed at 0.10.
   - **Decided (2026-11-17): audit C_tr.** For each fold, rank C_tr by `audit.tr_ranking`
     (default `mean_rank`), audit the top `audit.budget_tr` (default 100), and drop confirmed
     poison from the fusion fit. A2 and validity are unchanged.
   - In the CPU run the audit removed only about 3 of 47 C_tr poison per fold with `mean_rank`
     (AUROC 0.56), and about 13 with the latent signal (0.63). The ranking is now the bottleneck.
     Measure it on the GPU pilot with learned backdoors.
2. **Figure 3a assumes independent calibration per fold.** The pipeline uses a shared C_cal,
   and with it presence power is lower: 0.91 instead of 0.97 at ρ = 0.5% and shift 4.5. S2
   reports both; update the text or the figure.
3. **S1 fold directions** are unspecified in the proposal. 15° and 75° reproduce your numbers;
   state this in the paper.

### Problem tracker (updated 2026-11-17, after the CPU mini-pilot)

| # | Problem | Solution | Status |
|---|---|---|---|
| 1 | Fusion reference C_tr contains poison (T1) | Audit C_tr per fold, ranked by the mean of signal ranks (`fusion.reference: audited`) | Done. Mini-pilot: fused AUROC 0.76 → 0.90 (clean-reference bound 0.98) |
| 2 | Audits of 100 points cannot clean a hold-out part holding ~100 poison | Budgets ≈ 3ρm: `audit.budget` = `audit.budget_tr` = 300 in `configs/base.yaml` | Done. Mini-pilot: presence 0.6 → 1.0, R recall 0.46 → 0.78. Contradicts RQ5's "b ≤ 200": see the changes doc |
| 3 | Fig. 3a assumed independent C_cal per fold | Shared C_cal is the primary curve in S2; v3.1 setup dotted | Done |
| 4 | Do signals separate poison once the backdoor is learned? | CPU mini-pilot, `configs/cpu_minipilot.yaml` (augmentation off) | Yes: ASR 99%; SCALE-UP 0.95, confidence 0.92, STRIP 0.82; PU fusion with a clean reference 0.98. Repeat on GPU with augmentation |

| 5 | e-value threshold of proposal eq. (3) is not a stopping time | Threshold = first pooled value where BH's condition holds | Done (2026-11-18): null e-value mass 1.22 → 1.00 in the worst simulated case; code, proof, test and v3.2 docx updated |

| 6 | Ties broke the theorems' A3 for discrete scores | Random tie-breaking before calibration (`calibration.tie_break: random`) | Done (2026-11-18) |
| 7 | Proposition 4 was a conjecture | Proved FDR ≤ α/(1−ε) for audits that do not use calibration scores; open for the score-ranked audit | Done in part (2026-11-18), `proofs/REVIEW.md` |

| 8 | Proposal v3.3 adds RBD [16] (made in another session) | Shadow-model agreement in Table 4 needs code before the pilot; RBD baseline; check [16] on IEEE Xplore | Open; tracked in `ROADMAP.md` |

Proposal wording for all of these: `docs/proposal_v3.2_changes.md` (Changes 1–7). The current
proposal is `docs/CORD_Research_Proposal_v3.3.docx` (v3.2 plus the RBD edits; Change 6 not yet in). A data-driven orientation of
signals (`oriented_mean_rank`) was tried and is **not** the default: it helped only when the
backdoor was not learned.

## 3. What to do next

### Immediately (laptop)
- [x] Review the code and make the first commit (2026-11-17).
- [x] License: MIT (`LICENSE`).
- [ ] Push to the private repository `github.com/Shahriar127/CORD-Research`; the Kaggle notebook clones
      from it (see §4).
- [x] Finding 1 decided: audit C_tr (`fusion.reference: audited`, now the default).
- [ ] Update the proposal text for the C_tr audit: §3.3 (two audits, `budget_tr`), §4.1 (the
      fusion fit drops audited C_tr poison) and RQ5 (audit of C_tr as well as C_cal).
- [ ] Email for the two open roles: a statistician for G0 and a supervisor with GPU access.

### Gate G0 (weeks 1–4, no GPU)
- [ ] Write Propositions 1–3 in full in `proofs/main.tex`, including the A2 lemmas per component.
- [ ] Get the proofs checked by a statistician.
- [ ] Rerun `sim/s1_selections.py` and `sim/s2_presence_contamination.py` from a clean clone.
- [ ] Update Section 6.2 and Figure 3a for the shared-calibration case.

### First GPU session (Kaggle)
- [ ] `cord env` must show CUDA; then `cord download cifar10 --root /kaggle/working/data`.
- [ ] Run one condition (BadNets 5%) and record the time per epoch and the peak memory from
      `fold*/train_log.jsonl` and `train_summary.json`. **Check that ASR is above 90%.**
- [ ] Check that DINOv2 loads through `torch.hub` (needs internet on).
- [ ] Score the same run with `fusion.reference=oracle_clean results_tag=cleanref` and
      `fusion.reference=random results_tag=random`, and compare them with the audited default.
- [ ] Check which `audit.tr_ranking` surfaces the most C_tr poison
      (`audit_tr_poison_removed` in `results*/metrics.csv`), and fix it before the full pilot.

### Gate G1 (weeks 5–9): CIFAR-10 pilot
- [ ] `python experiments/pilot/run_grid.py --max-hours 11` (rerun to resume, or use `--shard`).
- [ ] Look at each signal's separation and the rank correlation between signals, and drop the
      signals that add nothing. Freeze the signal set.
- [ ] `python experiments/pilot/check_gate_g1.py runs/pilot` (plus `--tag cleanref`).
- [ ] Run at least 30 no-attack seeds for the false-alarm estimate (Section 7.3).
- [ ] Fill the cost table (E8) with measured numbers.

### Gate G2 (weeks 10–14): contamination
- [ ] Prove Proposition 4, or state it as a conjecture backed by evidence.
- [ ] Run E5 on cached scores: `audit.budget` ∈ {0, 50, 100, 200}, `audit.miss_rate`, and
      `audit.wrong_removals` (only the score stage, with `results_tag`).

### Core tier (weeks 15+)
- [ ] GTSRB: add `data/gtsrb.py`, register it, add `configs/core/gtsrb.yaml` and a test (README:
      "Adding a dataset").
- [ ] Tiny-ImageNet, then VGG-19-BN (register in `cord.models`).
- [ ] More attacks: ISSBA, Label-Consistent, Narcissus, Adap-Patch.
- [ ] Baselines from BackdoorBench and backdoor-toolbox; check which have public code first.
- [ ] The defence-aware attack (C2) and the downstream study (E7).
- [ ] Check for new competing preprints before submission.

## 4. Pushing to GitHub (one time)

GitHub account: <https://github.com/Shahriar127>. Repository (created 2026-11-17):
**`git@github.com:Shahriar127/CORD-Research.git`** (private).
The `origin` remote is already set in this folder.

1. Push from this folder over SSH:

   ```bash
   cd ~/"Documents/CORD Research"
   git push -u origin main
   ```

   SSH needs a key registered on GitHub. If the push says *Permission denied (publickey)*, run
   `ssh-keygen -t ed25519 -C "Shahriar127"`, add `~/.ssh/id_ed25519.pub` under github.com →
   Settings → SSH and GPG keys, and check with `ssh -T git@github.com`.
   Without SSH, switch to HTTPS with a personal access token as the password:
   `git remote set-url origin https://github.com/Shahriar127/CORD-Research.git`
   (fine-grained token, access to `CORD-Research`, *Contents: Read and write*).
2. If GitHub created the repository with a README or license, the first push is rejected. Run
   `git pull --rebase origin main` once, then push again.
3. Later pushes: `git push`.

**On Kaggle** (private repository): create a second fine-grained token with *Contents: Read-only*
on `CORD-Research`, store it as a Kaggle secret named `GITHUB_TOKEN` (*Add-ons → Secrets*), then:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/Shahriar127/CORD-Research.git && cd CORD-Research && pip install -q -e .
```

Never print the token or commit it. If it leaks, revoke it on GitHub.

## 5. Useful commands

```bash
source .venv/bin/activate
pytest -q                                   # tests
ruff check src tests sim experiments && mypy
cord run configs/smoke.yaml                 # CPU end-to-end check (~1 min)
cord floors --m-cal 2000
cord aggregate runs/pilot                   # -> runs/pilot/aggregate.csv
```
