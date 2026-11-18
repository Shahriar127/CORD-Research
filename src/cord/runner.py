"""Resumable orchestration of one experiment.

Stages and their artifacts (inside ``<output_dir>/<name>/``):

========  =====================================================  ==========================
stage     work                                                   artifact (= done marker)
========  =====================================================  ==========================
prepare   poison the pool, draw hold-out and folds               ``prepare.json``
train     train ``f_{-j}`` per fold (GPU), held-out trajectory    ``fold{j}/model.pt``
extract   one-pass feature cache per fold (GPU)                  ``fold{j}/features.npz``
ssl       frozen-encoder features of the pool (GPU, optional)    ``ssl_<encoder>.npy``
score     signals, fusion, audit, three outputs, B resamples     ``results/summary.json``
========  =====================================================  ==========================

Each stage skips work whose artifact exists, so re-running the same command after a crash or a
notebook timeout continues where it stopped. The poisoned pool is regenerated
deterministically from the attack seed and verified against a stored SHA-256 digest instead
of being stored (150 MB per CIFAR run).
"""

from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cord.attacks import Attack, PoisonedPool, build_attack
from cord.audit import audit_calibration
from cord.config import ExperimentConfig, dump_config
from cord.data import ImageDataset, load_dataset
from cord.features import encode_images, extract_fold_features
from cord.fusion import exchangeability_diagnostic, fused_scores, ranking_score
from cord.models import build_model
from cord.pipeline import evaluate_outputs, presence_baselines, run_cord
from cord.signals import ScoredSet, compute_signals
from cord.splits import Split, draw_split, resample_holdout_partition
from cord.training import FoldData, train_fold_model
from cord.utils.io import array_digest, load_npz, read_json, save_npz, write_json
from cord.utils.runtime import get_logger, resolve_device
from cord.utils.seeding import derive_seed, make_rng, seed_everything

log = get_logger("runner")

STAGES = ("prepare", "train", "extract", "ssl", "score")


class ExperimentRun:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.dir = cfg.run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------ lazily built objects
    @cached_property
    def device(self) -> str:
        return resolve_device(self.cfg.device)

    @cached_property
    def dataset(self) -> ImageDataset:
        d = self.cfg.data
        ds = load_dataset(d.name, d.root)
        if d.subset_train is not None or d.subset_test is not None:
            ds = ds.subset(d.subset_train, d.subset_test, make_rng(0, "subset", d.name))
        return ds

    @cached_property
    def attack(self) -> Attack:
        a = self.cfg.attack
        return build_attack(
            a.name,
            self.dataset.image_shape,
            self.dataset.num_classes,
            target=a.target,
            mapping=a.mapping,
            trigger_rng=make_rng(self.cfg.component_seed("attack"), "trigger"),
            params=a.params,
        )

    @cached_property
    def pool(self) -> PoisonedPool:
        ds = self.dataset
        pool = self.attack.poison(
            ds.x_train, ds.y_train, self.cfg.attack.rate, make_rng(self.cfg.component_seed("attack"), "poison")
        )
        marker = self.dir / "prepare.json"
        if marker.exists():
            expected = read_json(marker)["pool_digest"]
            if array_digest(pool.x, pool.y, pool.kind) != expected:
                raise RuntimeError(
                    "regenerated poisoned pool differs from the prepared one; "
                    "code or data changed since 'prepare' -- rerun with --force"
                )
        return pool

    @cached_property
    def split(self) -> Split:
        path = self.dir / "split.npz"
        if path.exists():
            return Split.load(path)
        s = self.cfg.split
        return draw_split(
            len(self.pool.y), s.m_tr, s.m_cal, s.n_folds, make_rng(self.cfg.component_seed("split"), "split")
        )

    @property
    def results_dir(self) -> Path:
        tag = self.cfg.results_tag
        return self.dir / (f"results-{tag}" if tag else "results")

    def fold_dir(self, j: int) -> Path:
        return self.dir / f"fold{j}"

    # ------------------------------------------------------------------ stages
    def prepare(self, force: bool = False) -> None:
        marker = self.dir / "prepare.json"
        if marker.exists() and not force:
            log.info("prepare: done")
            return
        marker.unlink(missing_ok=True)
        dump_config(self.cfg, self.dir / "config.yaml")
        pool, split = self.pool, self.split
        split.save(self.dir / "split.npz")
        pm = pool.poison_mask
        info = {
            "pool_digest": array_digest(pool.x, pool.y, pool.kind),
            "n_pool": len(pool.y),
            "true_rate": pool.rate,
            "attack_info": pool.info,
            "poison_in_holdout_tr": int(pm[split.holdout_tr].sum()),
            "poison_in_holdout_cal": int(pm[split.holdout_cal].sum()),
            "poison_per_fold": [int(pm[f].sum()) for f in split.folds],
            "fold_sizes": [len(f) for f in split.folds],
        }
        write_json(marker, info)
        log.info("prepare: %s", info)

    def train(self, folds: list[int] | None = None, force: bool = False) -> None:
        cfg, split, pool, ds = self.cfg, self.split, self.pool, self.dataset
        x_bd, y_bd = self.attack.backdoor_test_set(ds.x_test, ds.y_test)
        for j in folds if folds is not None else range(split.n_folds):
            out = self.fold_dir(j)
            if (out / "model.pt").exists() and not force:
                log.info("train fold %d: done", j)
                continue
            seed = derive_seed(cfg.component_seed("train"), "fold", j)
            seed_everything(seed, cfg.train.deterministic)
            tr, sc = split.train_indices(j), split.scored_indices(j)
            data = FoldData(pool.x[tr], pool.y[tr], pool.x[sc], pool.y[sc], ds.x_test, ds.y_test, x_bd, y_bd)
            model = build_model(cfg.model.name, ds.num_classes, ds.mean, ds.std, **cfg.model.kwargs)
            log.info("train fold %d on %s: %d train / %d scored", j, self.device, len(tr), len(sc))
            t0 = time.perf_counter()
            train_fold_model(model, data, cfg.train, self.device, seed, out)
            self.timings[f"train_fold{j}"] = time.perf_counter() - t0

    def extract(self, folds: list[int] | None = None, force: bool = False) -> None:
        import torch

        cfg, split, pool, ds = self.cfg, self.split, self.pool, self.dataset
        for j in folds if folds is not None else range(split.n_folds):
            out = self.fold_dir(j)
            if (out / "features.npz").exists() and not force:
                log.info("extract fold %d: done", j)
                continue
            if not (out / "model.pt").exists():
                raise FileNotFoundError(f"fold {j} has no trained model; run the train stage first")
            model = build_model(cfg.model.name, ds.num_classes, ds.mean, ds.std, **cfg.model.kwargs)
            model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
            rng = make_rng(cfg.component_seed("train"), "strip_partners", j)
            tr = split.train_indices(j)
            partners = pool.x[rng.choice(tr, size=cfg.extract.strip_blends, replace=False)]
            t0 = time.perf_counter()
            feats = extract_fold_features(
                model, pool.x[split.scored_indices(j)], partners, cfg.extract, self.device, cfg.train.amp
            )
            save_npz(out / "features.npz", **feats)
            self.timings[f"extract_fold{j}"] = time.perf_counter() - t0
            log.info("extract fold %d: %s", j, {k: v.shape for k, v in feats.items()})

    @property
    def ssl_path(self) -> Path:
        return self.dir / f"ssl_{self.cfg.signals.ssl_encoder}.npy"

    def ssl(self, force: bool = False) -> None:
        if "label_content" not in self.cfg.signals.enabled:
            log.info("ssl: label_content disabled, skipping")
            return
        if self.ssl_path.exists() and not force:
            log.info("ssl: done")
            return
        t0 = time.perf_counter()
        emb = encode_images(
            self.pool.x, self.cfg.signals.ssl_encoder, self.device, self.cfg.signals.ssl_batch_size, self.cfg.train.amp
        )
        self.ssl_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.ssl_path, emb)
        self.timings["ssl"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ scoring
    def _fold_signals(self, j: int, ssl: np.ndarray | None, force: bool) -> dict[str, np.ndarray]:
        path = self.fold_dir(j) / "signals.npz"
        if path.exists() and not force:
            return load_npz(path)
        rows = self.split.scored_indices(j)
        scored = ScoredSet(
            cache=load_npz(self.fold_dir(j) / "features.npz"),
            trajectory=load_npz(self.fold_dir(j) / "trajectory.npz"),
            labels=self.pool.y[rows].astype(np.int64),
            num_classes=self.dataset.num_classes,
            ssl=None if ssl is None else ssl[rows],
        )
        signals = compute_signals(scored, self.cfg.signals)
        save_npz(path, **signals)
        return signals

    def _reference_exclusions(
        self, signals: dict[str, np.ndarray], ref_mask: np.ndarray, j: int, b: int, stats: dict[str, float]
    ) -> np.ndarray | None:
        """C_tr rows to drop from the PU fit, according to ``fusion.reference``.

        The ranking that selects C_tr points for audit is a symmetric function of the scored set,
        and only C_tr rows are dropped, so the fit stays a symmetric function of D_j ∪ C_cal (A2).
        """
        cfg = self.cfg
        poison_rows = self.pool.poison_mask[self.split.scored_indices(j)]
        if cfg.fusion.reference == "random":
            return None
        if cfg.fusion.reference == "oracle_clean":
            exclude = ref_mask & poison_rows
        else:
            tr_rows = np.flatnonzero(ref_mask)
            score = ranking_score(signals, cfg.audit.tr_ranking)[tr_rows]
            tr_audit = cfg.audit.model_copy(update={"budget": cfg.audit.budget_tr, "wrong_removals": 0})
            a = audit_calibration(
                score, poison_rows[tr_rows], tr_audit, make_rng(cfg.component_seed("score"), "audit_tr", b, j)
            )
            exclude = np.zeros(ref_mask.size, dtype=bool)
            exclude[tr_rows[~a.keep]] = True
        stats["audit_tr_poison_removed"] += exclude.sum() / self.split.n_folds  # mean per fold
        return exclude

    def score(self, force: bool = False) -> dict[str, Any]:
        cfg, split, pool = self.cfg, self.split, self.pool
        summary_path = self.results_dir / "summary.json"
        if summary_path.exists() and not force:
            log.info("score: done")
            return dict(read_json(summary_path))
        t0 = time.perf_counter()
        ssl = np.load(self.ssl_path) if "label_content" in cfg.signals.enabled else None
        signals = [self._fold_signals(j, ssl, force) for j in range(split.n_folds)]
        holdout = split.holdout
        n_fold = [len(f) for f in split.folds]
        is_poison_D = np.concatenate([pool.poison_mask[f] for f in split.folds])
        labels_D = np.concatenate([pool.y[f] for f in split.folds]).astype(np.int64)
        target = cfg.attack.target if (cfg.attack.name != "none" and cfg.attack.mapping == "all_to_one") else None
        seed = cfg.component_seed("score")

        rows: list[dict[str, Any]] = []
        extra: dict[str, Any] = {}
        for b in range(cfg.calibration.n_resamples):
            rng_b = make_rng(seed, "resample", b)
            tr_set = split.holdout_tr if b == 0 else resample_holdout_partition(holdout, cfg.split.m_tr, rng_b)[0]
            tr_mask_h = np.isin(holdout, tr_set)
            cal_idx = holdout[~tr_mask_h]
            cal_poison = pool.poison_mask[cal_idx]
            per_method: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
            tr_stats = {
                "audit_tr_poison": int(pool.poison_mask[holdout[tr_mask_h]].sum()),
                "audit_tr_poison_removed": 0.0,
            }
            for j in range(split.n_folds):
                ref_mask = np.concatenate([np.zeros(n_fold[j], bool), tr_mask_h])
                exclude = self._reference_exclusions(signals[j], ref_mask, j, b, tr_stats)
                fused = fused_scores(
                    signals[j], ref_mask, cfg.fusion, derive_seed(seed, "fusion", b, j), exclude_from_fit=exclude
                )
                for method, s in fused.items():
                    per_method.setdefault(method, ([], []))
                    per_method[method][0].append(s[: n_fold[j]])
                    per_method[method][1].append(s[n_fold[j] :][~tr_mask_h])
                if b == 0 and cfg.fusion.exchangeability_permutations > 0:
                    sig_h = {k: v[n_fold[j] :] for k, v in signals[j].items()}
                    extra[f"exchangeability_fold{j}"] = exchangeability_diagnostic(
                        sig_h, tr_mask_h, ~tr_mask_h, cfg.fusion.exchangeability_permutations, make_rng(seed, "exch", j)
                    )

            for method, (fold_s, cal_s) in per_method.items():
                audited, audit_stats = (
                    [],
                    {"audit_poison_in_cal": int(cal_poison.sum()), "audit_poison_removed": 0, "audit_clean_removed": 0},
                )
                for j, cs in enumerate(cal_s):
                    a = audit_calibration(cs, cal_poison, cfg.audit, make_rng(seed, "audit", b, method, j))
                    audited.append(cs[a.keep])
                    audit_stats["audit_poison_removed"] += a.n_poison_removed
                    audit_stats["audit_clean_removed"] += a.n_clean_removed
                outputs = run_cord(fold_s, audited, cfg.calibration, make_rng(seed, "ties", b, method))
                row = {
                    "resample": b,
                    "method": method,
                    **audit_stats,
                    **tr_stats,
                    **evaluate_outputs(
                        outputs,
                        is_poison_D,
                        labels_D,
                        self.dataset.num_classes,
                        target,
                        fold_s,
                        [len(c) for c in audited],
                        cfg.calibration.alpha,
                    ),
                }
                if b == 0 and method == "cord_pu":
                    if cfg.calibration.presence_baselines:
                        row.update(presence_baselines(fold_s, audited, cfg.calibration, make_rng(seed, "presence")))
                    save_npz(
                        self.results_dir / "cord_outputs_b0.npz",
                        scores=np.concatenate(fold_s),
                        pvalues=outputs.pvalues,
                        evalues=outputs.evalues,
                        fold_id=outputs.fold_id,
                        removal=outputs.removal,
                        proxy=outputs.proxy,
                        is_poison=is_poison_D,
                        labels=labels_D,
                        pool_index=np.concatenate(split.folds),
                    )
                rows.append(row)

        df = pd.DataFrame(rows)
        (self.results_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(self.results_dir / "metrics.csv", index=False)
        summary = self._summarise(df, extra)
        self.timings["score"] = time.perf_counter() - t0
        summary["timings_s_this_session"] = self.timings
        write_json(summary_path, summary)
        log.info(
            "score: %s",
            {
                k: summary["methods"]["cord_pu"][k]
                for k in ("presence_reject", "proxy_ebh_fdp", "proxy_ebh_power", "removal_recall")
            },
        )
        return summary

    def _summarise(self, df: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
        drop = ["presence_exceedances", "resample", "target_guess"]
        numeric = df.drop(columns=drop).select_dtypes(include=["number", "bool"])
        by_method = numeric.groupby(df["method"]).mean(numeric_only=True)
        # Primary split (b = 0): the binary presence decision a defender would actually report;
        # the seed-level false-alarm rate and its exact binomial interval use this (Section 7.3).
        primary = df[df["resample"] == 0].set_index("method")["presence_reject"].astype(float)
        by_method["presence_reject_primary"] = primary
        train = {
            f"fold{j}": read_json(self.fold_dir(j) / "train_summary.json")
            for j in range(self.split.n_folds)
            if (self.fold_dir(j) / "train_summary.json").exists()
        }
        return {
            "name": self.cfg.name,
            "results_tag": self.cfg.results_tag,
            "fusion_reference": self.cfg.fusion.reference,
            "attack": self.cfg.attack.name,
            "rate": self.cfg.attack.rate,
            "seed": self.cfg.seed,
            "m_cal": self.cfg.split.m_cal,
            "alpha": self.cfg.calibration.alpha,
            "n_resamples": self.cfg.calibration.n_resamples,
            "prepare": read_json(self.dir / "prepare.json"),
            "train": train,
            "diagnostics": extra,
            "methods": {
                m: {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for m, r in by_method.iterrows()
            },
        }

    # ------------------------------------------------------------------ driver
    def run(self, stages: tuple[str, ...] = STAGES, folds: list[int] | None = None, force: bool = False) -> None:
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
        for stage in STAGES:
            if stage not in stages:
                continue
            log.info("=== %s / %s ===", self.cfg.name, stage)
            if stage == "prepare":
                self.prepare(force)
            elif stage == "train":
                self.train(folds, force)
            elif stage == "extract":
                self.extract(folds, force)
            elif stage == "ssl":
                self.ssl(force)
            else:
                self.score(force)
