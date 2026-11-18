# Kaggle codes — first GPU run

Copy each cell into a Kaggle notebook, in order. Runs the CORD pipeline on a Kaggle **GPU**
(T4 x2 or P100). CORD does not support TPUs: on a TPU machine it would silently run on CPU.

What the first run must tell us (ROADMAP, next action 2): the backdoor is learned (ASR above 90%
in both fold models), seconds per epoch, peak GPU memory, and whether the audited fusion beats the
unaudited one on a real GPU-trained model.

---

## Before you start (once)

1. **Notebook settings** (right-hand panel):
   - *Accelerator*: **GPU T4 x2** (or GPU P100)
   - *Internet*: **On** (pip, CIFAR-10, DINOv2 weights)
   - *Persistence*: **Files only** (keeps `/kaggle/working` between sessions)
2. **GitHub token** (the repository is private):
   - On GitHub: Settings → Developer settings → Personal access tokens → Fine-grained →
     *Repository access*: only `CORD-Research` → *Permissions*: Contents **Read-only**.
   - In the notebook: *Add-ons → Secrets* → add a secret named `GITHUB_TOKEN` with the token, and
     tick it for this notebook.
   - Never print the token or put it in a cell. If it leaks, revoke it on GitHub.

---

## Cell 1 — get the code (and update it on later sessions)

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "/kaggle/working/CORD-Research"
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
url = f"https://{token}@github.com/Shahriar127/CORD-Research.git"


def git(*args, cwd=None):
    # Never echo the command: it contains the token.
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git " + args[0] + " failed: " + r.stderr.replace(token, "***"))
    return r.stdout.strip()


if not os.path.exists(REPO):
    git("clone", "--quiet", url, REPO)
    git("remote", "set-url", "origin", "https://github.com/Shahriar127/CORD-Research.git", cwd=REPO)  # token not stored
else:
    git("pull", "--quiet", url, "main", cwd=REPO)
del token, url
os.chdir(REPO)
print(git("log", "--oneline", "-1", cwd=REPO))
assert os.path.exists("src/cord/data/cifar10.py"), "old commit: src/cord/data missing, pull again"
```

## Cell 2 — install and check the GPU

```python
!pip install -q -e /kaggle/working/CORD-Research
!cord env
```

Expected: `device: cuda (Tesla T4)` (or `Tesla P100-PCIE-16GB`). If it says `cpu`, the accelerator
is not set to GPU: fix the setting, restart the session and run from Cell 1.
If you updated the code in Cell 1 after `cord` was already imported, restart the kernel first.

## Cell 3 — CIFAR-10 (MD5-verified)

```python
!cord download cifar10 --root /kaggle/working/data
```

## Cell 4 — GPU smoke test (about 1–2 minutes)

Checks every stage on the GPU before spending quota on the real run.

```python
!cord run configs/smoke.yaml --set device=cuda train.amp=true data.root=/kaggle/working/data output_dir=/kaggle/working/runs
```

Expected: the last line starts with `score:`. Its numbers mean nothing (2 epochs, tiny network).

## Cell 5 — common settings for the real run

```python
DATA = "/kaggle/working/data"
OUT = "/kaggle/working/runs/pilot"
CFG = "configs/pilot/cifar10.yaml"  # BadNets 5%, seed 0, 100 epochs, PreAct-ResNet-18
RUN = f"{OUT}/pilot-cifar10-badnets-r0.05-s0"  # the run folder this config writes
COMMON = f"data.root={DATA} output_dir={OUT}"
```

## Cell 6 — prepare (poison the pool, draw the hold-out and folds)

```python
!cord run {CFG} --set {COMMON} --stages prepare
```

## Cell 7a — train both folds in parallel (GPU T4 x2: one fold per GPU)

Skip this and use 7b if you have a single GPU (P100).

```python
import subprocess, time

procs = []
for fold in (0, 1):
    log = open(f"/kaggle/working/train_fold{fold}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(fold)}
    procs.append(
        subprocess.Popen(
            ["cord", "run", CFG, "--set", *COMMON.split(), "--stages", "train", "--folds", str(fold)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    )
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for fold in (0, 1):
        lines = [l for l in open(f"/kaggle/working/train_fold{fold}.log") if "epoch" in l]
        print(f"fold {fold}: {lines[-1].strip() if lines else 'starting'}")
    print(f"--- {(time.time() - t0) / 60:.0f} min")
print("exit codes:", [p.returncode for p in procs])  # both must be 0
```

## Cell 7b — train both folds one after the other (single GPU)

```python
!cord run {CFG} --set {COMMON} --stages train
```

If the session stops during training, run the same cell again: each fold resumes from its last
saved epoch (every 5 epochs).

## Cell 8 — feature cache, DINOv2 features, scoring

```python
!cord run {CFG} --set {COMMON} --stages extract ssl score
```

The `ssl` stage downloads DINOv2 the first time (internet on).

## Cell 9 — did it work? (ASR, accuracy, speed, memory, detection)

```python
import json, pandas as pd

for fold in (0, 1):
    s = json.load(open(f"{RUN}/fold{fold}/train_summary.json"))
    log = pd.read_json(f"{RUN}/fold{fold}/train_log.jsonl", lines=True)
    sec_per_epoch = log["elapsed_s"].diff().median()
    print(
        f"fold {fold}: ASR {s['asr']:.3f} | clean acc {s['clean_acc']:.3f} | "
        f"{sec_per_epoch:.1f} s/epoch | peak {s['peak_gpu_mem_gb']:.2f} GB | {s['gpu']} | {s['precision']}"
    )

m = json.load(open(f"{RUN}/results/summary.json"))["methods"]
rows = ["cord_pu", "mean_rank", "single:scaleup_consistency", "single:conf_mean", "single:strip_neg_entropy"]
cols = [
    "auroc",
    "presence_reject",
    "removal_recall",
    "removal_clean_share",
    "proxy_ebh_power",
    "proxy_ebh_fdp",
    "foldwise_bh_power",
    "foldwise_bh_fdp",
]
pd.DataFrame({r: {c: m[r][c] for c in cols} for r in rows if r in m}).T.round(3)
```

What to look for:
- **ASR above 0.9 in both folds.** If not, the backdoor was not learned; stop and report it
  (ROADMAP risk "fold models miss low-rate backdoors").
- **precision `fp16`** on T4/P100 (`bf16` only on A100/L4/H100).
- `removal_clean_share` near 0.10 or below, FDR columns at or below 0.05: validity.
- Note s/epoch and peak memory for the cost table.

## Cell 10 — the same models, scored two other ways (no retraining, ~minutes)

```python
!cord run {CFG} --set {COMMON} fusion.reference=random results_tag=random --stages score
!cord run {CFG} --set {COMMON} fusion.reference=oracle_clean results_tag=cleanref --stages score
```

## Cell 11 — compare the three fusion references

```python
table = {}
for tag, folder in [
    ("audited (default)", "results"),
    ("random C_tr (v3.1)", "results-random"),
    ("poison-free C_tr (bound)", "results-cleanref"),
]:
    r = json.load(open(f"{RUN}/{folder}/summary.json"))["methods"]["cord_pu"]
    table[tag] = {
        c: r[c]
        for c in [
            "auroc",
            "removal_recall",
            "presence_reject",
            "foldwise_bh_power",
            "proxy_ebh_power",
            "audit_tr_poison_removed",
            "audit_tr_poison",
        ]
    }
pd.DataFrame(table).T.round(3)
```

Expected order (from the CPU mini-pilot): random < audited < poison-free.

## Cell 12 — download the results

```python
!cd /kaggle/working/runs && zip -qr /kaggle/working/first_run.zip pilot -x "*.pt"
!ls -lh /kaggle/working/first_run.zip
```

Download `first_run.zip` from the *Output* panel. It leaves out the model weights (`*.pt`) but
keeps everything needed to re-score on the laptop, without the dataset: unzip it into `runs/` in
the project, then (the saved config points at `/kaggle/...`, so override `output_dir`):

```bash
cord run runs/pilot/pilot-cifar10-badnets-r0.05-s0/config.yaml --stages score \
    --set output_dir=runs/pilot audit.budget=100 audit.budget_tr=100 results_tag=b100
```

---

## Later — the full pilot grid (27 conditions, 54 fold trainings)

Only after the first run passes Cell 9. Each session stops cleanly before Kaggle's 12-hour limit;
run the same cell again in a new session to continue where it stopped.

```python
!cd /kaggle/working/CORD-Research && python experiments/pilot/run_grid.py --max-hours 11 \
    --set data.root=/kaggle/working/data output_dir=/kaggle/working/runs/pilot
```

To use both T4 GPUs, run two shards at the same time, one per GPU:

```python
import subprocess

procs = [
    subprocess.Popen(
        [
            "python",
            "experiments/pilot/run_grid.py",
            "--shard",
            f"{g}/2",
            "--max-hours",
            "11",
            "--set",
            "data.root=/kaggle/working/data",
            "output_dir=/kaggle/working/runs/pilot",
        ],
        stdout=open(f"/kaggle/working/grid_shard{g}.log", "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
    )
    for g in (0, 1)
]
```

Progress, aggregation and gate G1:

```python
!python experiments/pilot/run_grid.py --list --set output_dir=/kaggle/working/runs/pilot
!cord aggregate /kaggle/working/runs/pilot
!python experiments/pilot/check_gate_g1.py /kaggle/working/runs/pilot
```

## If something goes wrong

| Symptom | Fix |
|---|---|
| `cord env` says `device: cpu` | Accelerator not set to GPU; change it, restart, run from Cell 1 |
| `No module named cord.data` / `src/cord/data` missing | Old commit; run Cell 1 again (it pulls), restart kernel, run Cell 2 |
| `regenerated poisoned pool differs from the prepared one` | Code changed after `prepare`; rerun with `--force` on the `prepare` stage, or start a new `output_dir` |
| Out of GPU memory | `--set train.batch_size=64 extract.batch_size=256` |
| Session ended mid-run | Run the same cell again; finished stages are skipped and training resumes |
| DINOv2 download fails | Internet is off; turn it on, or `--set signals.ssl_encoder=pixels` for a test (not for results) |
