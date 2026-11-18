# Running CORD on a GPU (Kaggle, Colab or a server)

**Ready-to-paste notebook cells for the first Kaggle run: [`kaggle_codes.md`](kaggle_codes.md).**
GPU only (T4 or P100); CORD does not support TPUs.

Nothing in the code changes between CPU and GPU: `device: auto` picks CUDA when it exists,
mixed precision switches on (bf16 on Ampere and newer, fp16 with loss scaling on T4/P100),
and the dataset is moved to GPU memory, so the data loader is not a bottleneck.

## Kaggle notebook (free tier)

Settings: *Accelerator* → GPU (T4 or P100), *Internet* → on (needed once for `pip`,
CIFAR-10 and the DINOv2 weights), *Persistence* → files.

```bash
# cell 1: code and environment
# private repo: clone with a token stored as a Kaggle secret (see NOTES.md §4)
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/Shahriar127/CORD-Research.git && cd CORD-Research && pip install -q -e .
%cd CORD-Research
!cord env                                   # should print: device: cuda (Tesla T4)

# cell 2: data (verified MD5, safe extraction)
!cord download cifar10 --root /kaggle/working/data

# cell 3: one condition end to end, to time it before committing quota
!cord run configs/pilot/cifar10.yaml --set data.root=/kaggle/working/data output_dir=/kaggle/working/runs/pilot

# cell 4: the pilot grid; stops cleanly before the 12 h session limit, rerun to resume
!python experiments/pilot/run_grid.py --max-hours 11 \
    --set data.root=/kaggle/working/data output_dir=/kaggle/working/runs/pilot

# cell 5: progress, aggregation and gate G1
!python experiments/pilot/run_grid.py --list --set output_dir=/kaggle/working/runs/pilot
!cord aggregate /kaggle/working/runs/pilot
!python experiments/pilot/check_gate_g1.py /kaggle/working/runs/pilot
```

Splitting the grid across sessions or accounts: `--shard 0/3`, `--shard 1/3`, `--shard 2/3`.
Each condition writes into its own folder, so shards never collide; copy the folders back
together and run `cord aggregate` on the union.

## What gets resumed

| Interruption | What happens on rerun |
|---|---|
| Session killed mid-epoch | the fold restarts from its last `state.pt` (every `train.checkpoint_every` epochs) |
| Session killed between folds | finished folds (`model.pt`) are skipped |
| Session killed during scoring | scoring restarts; cached `signals.npz` are reused |
| Code changed after `prepare` | the regenerated pool fails its SHA-256 check and the run stops: rerun with `--force` |

## Storage per condition (CIFAR-10, J = 2)

Two models (~45 MB each), two feature caches (~50 MB each), DINOv2 features (~40 MB) and
results: about 250 MB. The 27-condition pilot fits in Kaggle's 20 GB output limit.

## Cost

Not measured yet. Time the first condition (cell 3); `fold*/train_log.jsonl` records
seconds per epoch and `fold*/train_summary.json` the peak GPU memory. Put these numbers in
the cost table (E8) instead of the proposal's estimate.
