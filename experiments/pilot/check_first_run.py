"""Check one finished run: did the backdoor train, how fast, how much memory, and how well CORD did.

    python experiments/pilot/check_first_run.py runs/pilot/pilot-cifar10-badnets-r0.05-s0

Prints per-fold training facts, the detection metrics of CORD and its main baselines, and, when
the run was also scored with other fusion references (``results-random``, ``results-cleanref``),
a comparison of the three. Ends with PASS / CHECK lines for the first-run criteria (ROADMAP).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

METRICS = [
    "auroc",
    "presence_reject",
    "removal_recall",
    "removal_clean_share",
    "foldwise_bh_power",
    "foldwise_bh_fdp",
    "proxy_ebh_power",
    "proxy_ebh_fdp",
]
METHODS = ["cord_pu", "mean_rank", "single:scaleup_consistency", "single:conf_mean", "single:strip_neg_entropy"]
REFERENCES = [
    ("audited (default)", "results"),
    ("random C_tr", "results-random"),
    ("poison-free C_tr", "results-cleanref"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run folder, e.g. runs/pilot/pilot-cifar10-badnets-r0.05-s0")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    run: Path = args.run
    checks: list[tuple[bool, str]] = []

    print(f"== training ({run.name})")
    for fold in sorted(run.glob("fold*")):
        summary_path = fold / "train_summary.json"
        if not summary_path.exists():
            print(f"{fold.name}: not trained yet")
            checks.append((False, f"{fold.name} trained"))
            continue
        s = json.loads(summary_path.read_text())
        log = pd.read_json(fold / "train_log.jsonl", lines=True)
        sec = log["elapsed_s"].diff().median()
        mem = s.get("peak_gpu_mem_gb")
        print(
            f"{fold.name}: ASR {s['asr']:.3f} | clean acc {s['clean_acc']:.3f} | {sec:.1f} s/epoch | "
            f"{len(log)} epochs | peak {mem:.2f} GB | {s.get('gpu')} | {s.get('precision')}"
            if mem is not None
            else f"{fold.name}: ASR {s['asr']:.3f} | clean acc {s['clean_acc']:.3f} | {sec:.1f} s/epoch | "
            f"{len(log)} epochs | {s.get('device')} | {s.get('precision')}"
        )
        checks.append((s["asr"] >= 0.9, f"{fold.name} learned the backdoor (ASR {s['asr']:.3f} >= 0.9)"))
        if s.get("device", "").startswith("cuda"):
            checks.append((s.get("precision") in ("fp16", "bf16"), f"{fold.name} used mixed precision on the GPU"))

    summary = run / "results" / "summary.json"
    if not summary.exists():
        print("\nno results/summary.json yet: run the extract, ssl and score stages")
        return 1
    methods = json.loads(summary.read_text())["methods"]
    print("\n== detection (means over hold-out redraws)")
    table = pd.DataFrame({m: {c: methods[m][c] for c in METRICS} for m in METHODS if m in methods}).T
    print(table.round(3).to_string())
    cord = methods["cord_pu"]
    share = cord["removal_clean_share"]
    checks.append((share <= 0.12, f"clean share of removal set {share:.3f} (nominal 0.10; allows noise up to 0.12)"))
    for col in ("foldwise_bh_fdp", "proxy_ebh_fdp"):
        text = f"{col} {cord[col]:.3f} (nominal {args.alpha}; allows noise up to {1.5 * args.alpha:.3f})"
        checks.append((cord[col] <= 1.5 * args.alpha, text))

    available = [(label, run / folder / "summary.json") for label, folder in REFERENCES]
    available = [(label, p) for label, p in available if p.exists()]
    if len(available) > 1:
        print("\n== fusion reference (same trained models)")
        cols = ["auroc", "removal_recall", "presence_reject", "foldwise_bh_power", "proxy_ebh_power"]
        ref = {}
        for label, p in available:
            r = json.loads(p.read_text())["methods"]["cord_pu"]
            ref[label] = {c: r[c] for c in cols} | {
                "C_tr poison removed/fold": r.get("audit_tr_poison_removed"),
                "C_tr poison": r.get("audit_tr_poison"),
            }
        print(pd.DataFrame(ref).T.round(3).to_string())

    print("\n== first-run checks")
    for ok, text in checks:
        print(f"[{'PASS' if ok else 'CHECK'}] {text}")
    return 0 if all(ok for ok, _ in checks) else 2


if __name__ == "__main__":
    sys.exit(main())
