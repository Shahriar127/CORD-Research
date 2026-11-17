from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from cord.config import load_config
from cord.data.safe_pickle import safe_load_pickle
from cord.models import build_model
from cord.training.augment import random_crop_flip

ROOT = Path(__file__).resolve().parents[1]


def test_model_forward_matches_embed() -> None:
    model = build_model("preact_resnet18", 10, (0.5,) * 3, (0.25,) * 3, width=8).eval()
    x = torch.rand(4, 3, 32, 32)
    emb = model.embed(x)
    assert emb["penultimate"].shape == (4, 64) and emb["layer3"].shape == (4, 32)
    torch.testing.assert_close(model(x), emb["logits"])


def test_augment_identity_and_shapes() -> None:
    x = torch.rand(8, 3, 16, 16)
    torch.testing.assert_close(random_crop_flip(x, 0, None, flip=False), x)
    g = torch.Generator().manual_seed(0)
    y = random_crop_flip(x, 4, g, flip=True)
    assert y.shape == x.shape
    # Each crop is a translated window of the zero-padded image: total mass cannot increase.
    assert torch.all(y.sum((1, 2, 3)) <= x.sum((1, 2, 3)) + 1e-4)


@pytest.mark.parametrize("cfg", ["configs/smoke.yaml", "configs/pilot/cifar10.yaml"])
def test_shipped_configs_validate(cfg: str) -> None:
    c = load_config(ROOT / cfg, ["attack.rate=0.01", "seed=3"])
    assert c.attack.rate == 0.01 and c.seed == 3 and c.split.n_folds == 2


def test_config_rejects_unknown_keys_and_inconsistency() -> None:
    with pytest.raises(ValidationError):
        load_config(ROOT / "configs/smoke.yaml", ["attack.ratee=0.1"])
    with pytest.raises(ValidationError):
        load_config(ROOT / "configs/smoke.yaml", ["attack.name=none", "attack.rate=0.1"])


def test_component_seeds_are_independent() -> None:
    c = load_config(ROOT / "configs/smoke.yaml")
    seeds = {c.component_seed(k) for k in ("attack", "split", "train", "score")}
    assert len(seeds) == 4


def test_safe_unpickler_blocks_code(tmp_path: Path) -> None:
    class Evil:
        def __reduce__(self):  # type: ignore[no-untyped-def]
            import os

            return (os.system, ("echo pwned",))

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        safe_load_pickle(bad)
    good = tmp_path / "good.pkl"
    good.write_bytes(pickle.dumps({b"data": np.zeros((2, 3), np.uint8), b"labels": [1, 2]}))
    assert safe_load_pickle(good)[b"labels"] == [1, 2]


@pytest.mark.skipif(not (ROOT / "data/cifar-10-batches-py/data_batch_1").exists(), reason="CIFAR-10 not present")
def test_cifar10_loads() -> None:
    from cord.data import load_dataset

    ds = load_dataset("cifar10", ROOT / "data")
    assert ds.x_train.shape == (50000, 32, 32, 3) and ds.x_test.shape == (10000, 32, 32, 3)
    assert np.bincount(ds.y_train).tolist() == [5000] * 10
