# CORD

**Conformal evidence fusion for detecting backdoor poison, with presence testing, false
discovery control and detection floors.** Research code for the CORD proposal (v3.1).

Given an untrusted training pool, CORD answers three questions with finite-sample error
statements under an explicit exchangeability condition:

| Output | Question | Guarantee | Code |
|---|---|---|---|
| Presence test | Was the dataset attacked? | false-alarm probability ≤ α (exact) | `cord.stats.presence` |
| Removal set `R` | What to drop before retraining? | expected clean share discarded ≤ t | `cord.pipeline.run_cord` |
| Proxy set `F` | Which samples are surely poison? | FDR ≤ α (e-BH, any fold dependence) | `cord.stats.multiple_testing` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # Python ≥ 3.10, PyTorch ≥ 2.2
cord env                           # versions and the selected device
```

CIFAR-10: either `cord download cifar10 --root data` (MD5-verified) or point `data.root` at an
existing `cifar-10-batches-py/` folder. Pickled batches are loaded with a restricted unpickler,
so a tampered file cannot execute code.

## Quickstart

```bash
pytest -q                                      # 50 tests: exactness, validity, A2, end-to-end
cord floors                                    # Table 5 (selection floor, presence threshold)
python sim/s1_selections.py                    # Study S1  -> sim/results/
python sim/s2_presence_contamination.py        # Study S2 (includes the former audit_sim.py)
cord run configs/smoke.yaml                    # whole deep pipeline on CPU in ~1 minute
```

On a GPU the same commands run the real experiments (see
[`experiments/kaggle/README.md`](experiments/kaggle/README.md)):

```bash
cord run configs/pilot/cifar10.yaml --set attack.name=blend attack.rate=0.01 seed=2
python experiments/pilot/run_grid.py --max-hours 11      # the 27-condition pilot, resumable
python experiments/pilot/check_gate_g1.py runs/pilot     # gate G1 against its numeric criteria
```

## Pipeline

```
prepare ─► train (J fold models) ─► extract (feature cache) ─► ssl (frozen encoder) ─► score
  │           │                         │                                              │
  attack,     f_{-j} on D \ D_j,        one forward pass per fold:                     signals → PU fusion
  hold-out,   held-out confidence       features at several layers, logits,            → audit → p/e-values
  folds       trajectory recorded       SCALE-UP and STRIP predictions                 → three outputs
                                                                                       × B hold-out resamples
```

Each stage writes an artifact that doubles as its completion marker, so rerunning a command
resumes; training itself resumes mid-fold. The poisoned pool is regenerated from its seed and
checked against a stored SHA-256 digest instead of being saved.

## Mapping the proposal to the code

| Proposal | Where |
|---|---|
| §3.2 order of events (poison → hold-out → folds) | `runner.ExperimentRun.prepare`, `splits.draw_split` |
| §3.3 mode T1 random hold-out, targeted audit | `splits`, `audit.audit_calibration` |
| A2 score invariance | `signals/__init__.py` contract, `tests/test_signals_fusion.py` |
| §4.1 cross-fitting | `training.train_fold_model`, `Split.train_indices` |
| Table 4 signals | `signals/{latent,confidence,perturbation,activation,label_content}.py` |
| §4.1 PU fusion (AdaDetect) + diagnostic | `fusion.pu_fusion`, `fusion.exchangeability_diagnostic` |
| Eq. 2–3 conformal p-/e-values | `stats.conformal` |
| §4.2 presence test (k-th largest variant) | `stats.presence.presence_test` |
| §4.2 removal set, proxy set, per-class breakdown, Storey rate | `pipeline.run_cord`, `stats.estimation` |
| Prop. 2–3 floors (Table 5) | `stats.floors`, `cord floors` |
| §7.2 baselines: naive fusions, single signals, Fisher, HC | `fusion.fused_scores`, `pipeline.presence_baselines` |
| §7.3 B resamples, seed-level bootstrap, exact binomial | `runner.score`, `evaluation` |
| §6 Studies S1, S2 | `sim/` |
| Table 7 gate G1 | `experiments/pilot/check_gate_g1.py` |
| Propositions (G0) | `proofs/main.tex` |

## Layout

```
src/cord/
  stats/        conformal p/e-values, BH/BY/e-BH, presence test, floors   (NumPy only)
  data/         ImageDataset container, CIFAR-10 loader, safe unpickler, verified download
  attacks/      BadNets, Blend, SIG, WaNet, Adaptive-Blend (registry)
  models/       PreAct-ResNet-18 with multi-layer embed() (registry)
  training/     GPU-resident augmentation, AMP trainer with resumable state
  features/     per-fold feature cache, frozen DINOv2 / pixel encoders
  signals/      the five signal families of Table 4
  fusion/       PU fusion, naive fusions, exchangeability diagnostic
  audit.py      oracle auditor with miss rate and wrong removals (RQ5)
  pipeline.py   the three outputs and their ground-truth evaluation
  runner.py     resumable stages;  cli.py  the `cord` command
configs/        base, pilot and smoke configs (YAML, validated by pydantic)
experiments/    pilot grid, gate checker, Kaggle guide
sim/            synthetic studies S1 and S2
proofs/         LaTeX skeleton of Propositions 1–4
docs/           proposal v3.1 → v3.2 changes with evidence
tests/          pytest suite
```

## Adding a dataset (e.g. GTSRB)

1. Write `src/cord/data/gtsrb.py` with `load_gtsrb(root) -> ImageDataset` (uint8 NHWC, resize to
   32×32 to match BackdoorBench, set `mean`/`std`, `class_names`).
2. Register it in `src/cord/data/__init__.py` (`_REGISTRY["gtsrb"] = load_gtsrb`).
3. Add `configs/core/gtsrb.yaml` with `defaults: [../base.yaml]` and `data.name: gtsrb`.
4. Add a loader test to `tests/` (shape, label range, class count 43).

Attacks, splits, models, signals and statistics are size- and class-count-agnostic. For
Tiny-ImageNet (64×64) adjust `attack.params` (e.g. a larger BadNets patch). For a new
architecture, register a builder in `cord.models` whose module exposes `embed()` and `head`.

## Reproducibility

* Each component (attack, split, training, scoring) has its own seed, derived from the master
  `seed` by a stable hash, so results do not depend on execution order.
* The resolved config is written to every run folder; the pool digest makes silent data or
  code drift fail loudly.
* `train.deterministic: true` switches on deterministic CUDA kernels (slower).

## Design decision: audit both parts of the hold-out

In mode T1 the random hold-out contains poison at the pool rate, in both of its parts:

* `C_tr`, the negative class of the PU fusion: poison there makes the classifier compare two
  samples of the same mixture, and the fused score loses power (not validity);
* `C_cal`, the calibration set: poison there blocks FDR selections and the presence test
  (Study S2).

**Decision (2026-11-17): audit both, with budgets that scale with the expected poison count.**
For each fold, `C_tr` is ranked by the mean of signal ranks (`audit.tr_ranking`, a symmetric
function of the scored set, so A2 holds), its top `audit.budget_tr` are audited and confirmed
poison is dropped from the fusion fit; `C_cal` is audited as before (`audit.budget`). Defaults:
300 each, about 3ρm for the pilot's settings. `fusion.reference` also offers `random` (v3.1 as
written) and `oracle_clean` (all poison removed, an upper bound).

CPU mini-pilot (`configs/cpu_minipilot.yaml`: 10k-image CIFAR-10 subset, BadNets 10%, ASR 99%;
means over 5 hold-out resamples of one pair of fold models):

| Audits (C_tr / C_cal per fold) | fused AUROC | removal recall | presence | fold-wise BH power (FDR) |
|---|---|---|---|---|
| 100 / 100 | 0.76 | 0.46 | 0.6 | 0 |
| 300 / 300 | 0.90 | 0.78 | 1.0 | 0.16 (0.051) |
| poison-free C_tr / 300 | 0.98 | 0.96 | 1.0 | 0.56 (0.058) |

The best single signal reached 0.95 (SCALE-UP), so with a clean reference the fusion beats it.
Alternative scorings reuse the trained models (no retraining):

```bash
cord run runs/pilot/<run>/config.yaml --stages score \
    --set fusion.reference=oracle_clean results_tag=cleanref
cord run runs/pilot/<run>/config.yaml --stages score \
    --set audit.budget=100 audit.budget_tr=100 results_tag=b100
python experiments/pilot/check_gate_g1.py runs/pilot --tag cleanref
```

Full evidence and proposal wording: [`docs/proposal_v3.2_changes.md`](docs/proposal_v3.2_changes.md).

## Notes on the synthetic studies

S1 and S2 are reimplemented on the package's own statistics code. The proposal does not fix
two details, and the choices here are documented in the scripts:

* S1 fold directions: unit vectors at 15° and 75°. This reproduces v3.1 to within Monte-Carlo
  error (e.g. FDR 0.044 / 0.041 / 0.034 against 0.043 / 0.041 / 0.033; recall in FPR mode 0.81).
* S2(a): v3.1's numbers match an independent calibration draw per fold. The script reports
  both that and the shared-calibration case, which is how the pipeline works (every fold model
  scores the same `C_cal`). With shared calibration, presence power is lower (at shift 4.5 and
  ρ = 0.5%, about 0.93 instead of 0.98), because the folds are positively dependent.
* The former `audit_sim.py` (findings 6 and 7) is merged into S2 parts (c) and (d) and
  reproduces its numbers.
