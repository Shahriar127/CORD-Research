"""Run (or resume) the pilot grid.

Resumable at three levels: finished conditions are skipped, finished folds are skipped, and an
interrupted fold resumes from its last saved epoch. ``--shard i/n`` splits the grid across
sessions or machines; ``--max-hours`` stops cleanly before a notebook session limit.

    python experiments/pilot/run_grid.py --grid experiments/pilot/grid.yaml
    python experiments/pilot/run_grid.py --shard 0/3 --max-hours 11 --set data.root=/kaggle/input/cifar10
    python experiments/pilot/run_grid.py --list
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from cord.config import load_config
from cord.runner import ExperimentRun
from cord.utils.runtime import configure_logging, get_logger

log = get_logger("grid")


def conditions(grid: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for attack, params in grid["attacks"].items():
        for rate in grid["rates"]:
            for seed in grid["seeds"]:
                out.append({"attack": attack, "params": params or {}, "rate": rate, "seed": seed})
    for seed in grid.get("no_attack_seeds", []):
        out.append({"attack": "none", "params": {}, "rate": 0.0, "seed": seed})
    return out


def run_name(c: dict[str, Any], prefix: str) -> str:
    return f"{prefix}-{c['attack']}-r{c['rate']:g}-s{c['seed']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, default=Path("experiments/pilot/grid.yaml"))
    ap.add_argument("--prefix", default="pilot-cifar10")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th condition starting at i")
    ap.add_argument("--max-hours", type=float, default=None, help="stop before starting work past this budget")
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides applied to every condition")
    ap.add_argument("--list", action="store_true", help="print the conditions and their status, then exit")
    args = ap.parse_args()

    grid = yaml.safe_load(args.grid.read_text())
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    todo = [c for k, c in enumerate(conditions(grid)) if k % shard_n == shard_i]
    configure_logging(logging.INFO)
    start = time.time()

    for c in todo:
        overrides = [
            f"name={run_name(c, args.prefix)}",
            f"attack.name={c['attack']}",
            f"attack.rate={c['rate']}",
            f"seed={c['seed']}",
            *[f"attack.params.{k}={v}" for k, v in c["params"].items()],
            *args.set,
        ]
        cfg = load_config(Path(grid["base_config"]), overrides)
        done = (cfg.run_dir / "results" / "summary.json").exists()
        if args.list:
            print(f"{'done' if done else 'todo'}  {cfg.name}")
            continue
        if done:
            continue
        if args.max_hours is not None and (time.time() - start) / 3600 > args.max_hours:
            log.warning("time budget reached; stopping before %s (rerun the same command to resume)", cfg.name)
            break
        log.info("### condition %s", cfg.name)
        ExperimentRun(cfg).run(tuple(args.stages) if args.stages else ("prepare", "train", "extract", "ssl", "score"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
