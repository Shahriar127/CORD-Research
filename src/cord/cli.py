"""Command-line interface.

cord run configs/pilot/cifar10_badnets.yaml --set attack.rate=0.01 seed=2
cord run configs/smoke.yaml --stages prepare train --folds 0
cord floors --m-cal 500 1000 2000 5000 20000
cord aggregate runs/pilot -o runs/pilot/summary.csv
cord env
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from cord.config import load_config
    from cord.runner import STAGES, ExperimentRun
    from cord.utils.runtime import configure_logging

    cfg = load_config(args.config, args.set)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO, str(cfg.run_dir / "run.log"))
    ExperimentRun(cfg).run(tuple(args.stages or STAGES), args.folds, args.force)
    return 0


def _cmd_floors(args: argparse.Namespace) -> int:
    from cord.stats import presence_threshold_approx, presence_threshold_exact, selection_floor

    cols = ("rho_sel a=.05", "rho_sel a=.10", "rho_pres approx", "rho_pres exact")
    header = f"{'m_cal':>8} | " + " | ".join(f"{c:>15}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in args.m_cal:
        exact = presence_threshold_exact(args.n_fold, m, args.alpha, args.folds)
        print(
            f"{m:>8} | {selection_floor(m, 0.05):>13.4%} | {selection_floor(m, 0.10):>13.4%} | "
            f"{presence_threshold_approx(m, args.alpha, args.folds):>15.4%} | {exact:>14.4%}"
        )
    print(f"(presence: alpha={args.alpha}, J={args.folds}, exact form uses N_j={args.n_fold})")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from cord.evaluation import collect_summaries

    df = collect_summaries(Path(args.runs))
    if df.empty:
        print(f"no results/summary.json under {args.runs}", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else Path(args.runs) / "aggregate.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from cord.data.download import download_dataset
    from cord.utils.runtime import configure_logging

    configure_logging()
    print(download_dataset(args.dataset, args.root))
    return 0


def _cmd_env(_: argparse.Namespace) -> int:
    import platform

    import numpy
    import sklearn
    import torch

    import cord
    from cord.utils.runtime import resolve_device

    print(
        f"cord {cord.__version__} | python {platform.python_version()} | torch {torch.__version__} | "
        f"numpy {numpy.__version__} | sklearn {sklearn.__version__}"
    )
    dev = resolve_device("auto")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cord", description="CORD backdoor-poison detection pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run (or resume) an experiment")
    r.add_argument("config", type=Path)
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    r.add_argument("--stages", nargs="*", help="subset of: prepare train extract ssl score")
    r.add_argument("--folds", nargs="*", type=int, help="restrict train/extract to these folds")
    r.add_argument("--force", action="store_true", help="recompute the selected stages")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_cmd_run)

    f = sub.add_parser("floors", help="print the detection floors (Table 5)")
    f.add_argument("--m-cal", nargs="*", type=int, default=[500, 1000, 2000, 5000, 20000])
    f.add_argument("--alpha", type=float, default=0.05)
    f.add_argument("--folds", type=int, default=2)
    f.add_argument("--n-fold", type=int, default=24000, help="fold size N_j for the exact threshold")
    f.set_defaults(func=_cmd_floors)

    a = sub.add_parser("aggregate", help="collect results/summary.json of many runs into one CSV")
    a.add_argument("runs", type=Path)
    a.add_argument("-o", "--output")
    a.set_defaults(func=_cmd_aggregate)

    d = sub.add_parser("download", help="download and verify a dataset (e.g. on a fresh GPU machine)")
    d.add_argument("dataset", choices=["cifar10"])
    d.add_argument("--root", type=Path, default=Path("data"))
    d.set_defaults(func=_cmd_download)

    e = sub.add_parser("env", help="print versions and the selected device")
    e.set_defaults(func=_cmd_env)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
