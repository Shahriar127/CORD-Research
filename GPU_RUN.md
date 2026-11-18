# Running CORD on a lab GPU machine

Step-by-step for the first real run on a machine with an NVIDIA GPU (lab workstation or server).
Copy the commands in order. Nothing in the code changes between laptop and GPU: `device: auto`
finds CUDA, uses fp16 on older GPUs (T4, P100, RTX 20xx) and bf16 on newer ones (RTX 30xx/40xx,
A100, L4, H100).

What this first run must answer (ROADMAP, next action 2): **is the backdoor learned** (ASR above
0.9 in both fold models), **how long** one run takes, **how much GPU memory** it uses, and **does
the audited fusion** beat the unaudited one.

---

## 0. Check the machine (1 minute)

```bash
nvidia-smi          # GPU model, free memory, and "CUDA Version" (the driver's maximum)
python3 --version   # needs 3.10 or newer
```

You need at least one GPU with **8 GB free**. On a shared machine, note which GPUs are idle
(low `Memory-Usage`, no processes listed): you will pick them with `CUDA_VISIBLE_DEVICES`.

## 1. Get the code (once)

The repository is private. Either use an SSH key already added to your GitHub account:

```bash
git clone git@github.com:Shahriar127/CORD-Research.git
```

or HTTPS with a personal access token (GitHub → Settings → Developer settings → Fine-grained
token, *Contents: Read-only* on `CORD-Research`) typed as the password:

```bash
git clone https://github.com/Shahriar127/CORD-Research.git    # username: Shahriar127, password: the token
```

Then:

```bash
cd CORD-Research
git log --oneline -1          # latest commit
ls src/cord/data              # must list cifar10.py etc. (older clones missed this folder)
```

Later sessions: `git pull` inside the folder.

## 2. Python environment (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install a **CUDA build of PyTorch** that the driver supports. Take the exact command from
<https://pytorch.org/get-started/locally/> (Linux, Pip, Python, the CUDA version not above
`nvidia-smi`'s "CUDA Version"). It looks like:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121    # example; use the site's command
```

Then CORD itself:

```bash
pip install -e ".[dev]"
cord env
```

`cord env` **must** print `device: cuda (<your GPU name>)`. If it prints `cpu`, PyTorch was
installed without CUDA (repeat the PyTorch step) or no GPU is visible.

## 3. Tests and data (once, about 2 minutes)

```bash
pytest -q                               # 67 tests (1 is skipped until CIFAR-10 is downloaded)
cord download cifar10 --root data       # about 170 MB, MD5-verified
```

No internet on the lab machine? Copy `data/cifar-10-batches-py/` from the laptop
(`scp -r data/cifar-10-batches-py <lab>:CORD-Research/data/`).

## 4. Work inside tmux

Long runs must survive a closed laptop or a dropped SSH connection:

```bash
tmux new -s cord          # start (later: tmux attach -t cord)
source .venv/bin/activate
```

Detach with **Ctrl-b, then d**. The run keeps going.

## 5. GPU smoke test (1–2 minutes)

```bash
cord run configs/smoke.yaml --set device=cuda train.amp=true
```

The last line starts with `score:`. Its numbers mean nothing (tiny network, 2 epochs); it only
proves every stage works on this GPU.

## 6. The first real run (BadNets 5%, 100 epochs, PreAct-ResNet-18)

```bash
CFG=configs/pilot/cifar10.yaml
RUN=runs/pilot/pilot-cifar10-badnets-r0.05-s0

cord run $CFG --stages prepare
```

**Train**, pick one:

```bash
# (a) two idle GPUs (here 0 and 1): one fold each, about half the time
CUDA_VISIBLE_DEVICES=0 cord run $CFG --stages train --folds 0 > fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 cord run $CFG --stages train --folds 1 > fold1.log 2>&1 &
tail -f fold0.log fold1.log          # Ctrl-c stops watching, not training
wait

# (b) one GPU (here GPU 0): both folds in turn
CUDA_VISIBLE_DEVICES=0 cord run $CFG --stages train
```

Every 5 epochs a line like `epoch 5/100 loss … clean … asr …` appears. ASR should climb above 0.9
within the first 10–30 epochs.

**Features and scoring** (the `ssl` stage downloads DINOv2 once; needs internet):

```bash
CUDA_VISIBLE_DEVICES=0 cord run $CFG --stages extract ssl score
```

**Score the same models two other ways** (no retraining, a few minutes each):

```bash
cord run $CFG --stages score --set fusion.reference=random results_tag=random
cord run $CFG --stages score --set fusion.reference=oracle_clean results_tag=cleanref
```

If anything stops the run (reboot, crash, someone else's job), **run the same command again**:
finished stages are skipped and training resumes from its last saved epoch.

## 7. Check the results

```bash
python experiments/pilot/check_first_run.py $RUN
```

It prints, per fold, ASR, clean accuracy, seconds per epoch, peak GPU memory and precision; the
detection metrics of CORD and its main baselines; the comparison of the three fusion references;
and PASS / CHECK lines. Look for:

- **ASR ≥ 0.9 in both folds.** If not, stop: the backdoor was not learned, and nothing else in
  the run is meaningful. Send the output.
- **Precision `fp16` or `bf16`**, not `fp32`.
- **Clean share of the removal set near 0.10** and FDR columns near or below 0.05 (validity).
- **Fusion references in the order** random < audited < poison-free (as on the CPU mini-pilot).
- Write down **s/epoch** and **peak memory**: they become the cost table and the pilot budget.

## 8. Bring the results to the laptop

On the laptop, from the project folder (weights left out; everything needed to re-score stays):

```bash
rsync -av --exclude '*.pt' <lab>:CORD-Research/runs/ runs/
python experiments/pilot/check_first_run.py runs/pilot/pilot-cifar10-badnets-r0.05-s0
```

Re-scoring on the laptop needs no dataset, for example with smaller audits:

```bash
cord run runs/pilot/pilot-cifar10-badnets-r0.05-s0/config.yaml --stages score \
    --set audit.budget=100 audit.budget_tr=100 results_tag=b100
```

## 9. Afterwards: the full pilot (27 conditions, 54 fold trainings)

Only once the first run passes step 7.

```bash
# one GPU
CUDA_VISIBLE_DEVICES=0 python experiments/pilot/run_grid.py

# two GPUs: two tmux windows (Ctrl-b c opens a new one), one shard each
CUDA_VISIBLE_DEVICES=0 python experiments/pilot/run_grid.py --shard 0/2
CUDA_VISIBLE_DEVICES=1 python experiments/pilot/run_grid.py --shard 1/2

# progress, aggregate, gate G1
python experiments/pilot/run_grid.py --list
cord aggregate runs/pilot
python experiments/pilot/check_gate_g1.py runs/pilot
```

Interrupted? Run the same command again; finished conditions are skipped.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cord env` prints `device: cpu` | CPU-only PyTorch: reinstall with the CUDA command from pytorch.org; check `nvidia-smi` works |
| `CUDA out of memory` | Another job holds the GPU (check `nvidia-smi`), or add `--set train.batch_size=64 extract.batch_size=256` |
| `No module named cord.data` | Old clone: `git pull`, then `pip install -e ".[dev]"` again |
| `regenerated poisoned pool differs from the prepared one` | Code changed after `prepare`: add `--force` to the `prepare` command, or delete the run folder |
| DINOv2 download fails | No internet: copy `~/.cache/torch/hub` from a machine that has it, or `--set signals.ssl_encoder=pixels` for a test only |
| ASR stays low | Report it with `check_first_run.py` output; do not continue to the grid |
| SSH dropped | `tmux attach -t cord`; if the session is gone, rerun the last command (it resumes) |
