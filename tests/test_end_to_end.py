"""Full pipeline on a synthetic dataset: every stage, artifacts, resume and determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cord.data as data_mod
from cord.config import load_config
from cord.data import ImageDataset
from cord.runner import ExperimentRun
from cord.utils.io import read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_registered(tiny_dataset: ImageDataset, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(data_mod._REGISTRY, "tiny", lambda _root: tiny_dataset)
    return "tiny"


def _cfg(tmp_path: Path, name: str, *extra: str):  # type: ignore[no-untyped-def]
    return load_config(
        ROOT / "configs/smoke.yaml",
        [
            f"name={name}",
            f"output_dir={tmp_path}",
            "data.name=tiny",
            "data.subset_train=null",
            "data.subset_test=null",
            "split.m_tr=60",
            "split.m_cal=60",
            "train.epochs=1",
            "audit.budget=10",
            "extract.strip_blends=1",
            "signals.latent_pca_dim=4",
            *extra,
        ],
    )


@pytest.mark.slow
def test_pipeline_runs_and_resumes(tmp_path: Path, synthetic_registered: str) -> None:
    cfg = _cfg(tmp_path, "e2e")
    ExperimentRun(cfg).run()
    run = tmp_path / "e2e"
    for rel in (
        "prepare.json",
        "split.npz",
        "fold0/model.pt",
        "fold1/features.npz",
        "fold1/trajectory.npz",
        "ssl_pixels.npy",
        "results/metrics.csv",
        "results/summary.json",
        "results/cord_outputs_b0.npz",
    ):
        assert (run / rel).exists(), rel
    summary = read_json(run / "results/summary.json")
    assert "cord_pu" in summary["methods"] and 0.0 <= summary["methods"]["cord_pu"]["proxy_ebh_fdp"] <= 1.0

    mtime = (run / "fold0/model.pt").stat().st_mtime
    ExperimentRun(cfg).run()  # resume: nothing is recomputed
    assert (run / "fold0/model.pt").stat().st_mtime == mtime

    out = np.load(run / "results/cord_outputs_b0.npz")
    assert out["pvalues"].min() >= 1.0 / (cfg.split.m_cal + 1) - 1e-12


@pytest.mark.slow
def test_no_attack_condition(tmp_path: Path, synthetic_registered: str) -> None:
    cfg = _cfg(tmp_path, "clean", "attack.name=none", "attack.rate=0.0")
    ExperimentRun(cfg).run()
    s = read_json(tmp_path / "clean/results/summary.json")
    assert s["methods"]["cord_pu"]["n_poison"] == 0


@pytest.mark.slow
def test_rescore_without_dataset(tmp_path: Path, synthetic_registered: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run trained elsewhere can be re-scored from its artifacts alone: no dataset, no attack
    code, and the scoring's own settings are saved next to its results."""
    ExperimentRun(_cfg(tmp_path, "moved")).run()

    def unavailable(_root: Path) -> ImageDataset:
        raise FileNotFoundError("dataset not on this machine")

    monkeypatch.setitem(data_mod._REGISTRY, "tiny", unavailable)
    cfg = _cfg(tmp_path, "moved", "results_tag=b5", "audit.budget=5")
    ExperimentRun(cfg).run(stages=("score",))
    out = tmp_path / "moved" / "results-b5"
    assert (out / "summary.json").exists()
    assert "budget: 5" in (out / "config.yaml").read_text()
