from __future__ import annotations

import numpy as np
import pytest

from cord.attacks import SampleKind, available_attacks, build_attack
from cord.audit import audit_calibration
from cord.config import AuditConfig
from cord.data import ImageDataset
from cord.splits import draw_split, resample_holdout_partition
from cord.utils.io import array_digest

DIRTY = ["badnets", "blend", "wanet", "adaptive_blend"]


def _make(name: str, ds: ImageDataset, seed: int = 0, **params: object):
    return build_attack(name, ds.image_shape, ds.num_classes, trigger_rng=np.random.default_rng(seed), params=params)


@pytest.mark.parametrize("name", DIRTY)
def test_dirty_label_attack_contract(name: str, tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    pool = _make(name, ds).poison(ds.x_train, ds.y_train, 0.05, np.random.default_rng(1))
    poison = pool.kind == SampleKind.POISON
    assert pool.x.dtype == np.uint8 and pool.x.shape == ds.x_train.shape
    assert poison.sum() == round(0.05 * len(ds.y_train))
    assert np.all(pool.y[poison] == 0) and np.all(pool.y_clean[poison] != 0)
    clean = pool.kind == SampleKind.CLEAN
    np.testing.assert_array_equal(pool.x[clean], ds.x_train[clean])
    np.testing.assert_array_equal(pool.y[clean], ds.y_train[clean])
    assert np.any(pool.x[poison] != ds.x_train[poison])


@pytest.mark.parametrize("name", [*DIRTY, "sig"])
def test_attack_is_deterministic(name: str, tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    rate = 0.01 if name == "sig" else 0.05
    digests = {
        array_digest(
            *(lambda p: (p.x, p.y, p.kind))(
                _make(name, ds, 3).poison(ds.x_train, ds.y_train, rate, np.random.default_rng(9))
            )
        )
        for _ in range(2)
    }
    assert len(digests) == 1


def test_sig_is_clean_label(tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    pool = _make("sig", ds).poison(ds.x_train, ds.y_train, 0.02, np.random.default_rng(0))
    np.testing.assert_array_equal(pool.y, ds.y_train)
    assert np.all(ds.y_train[pool.poison_mask] == 0)


def test_adaptive_blend_has_cover_samples(tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    pool = _make("adaptive_blend", ds).poison(ds.x_train, ds.y_train, 0.05, np.random.default_rng(0))
    cover = pool.kind == SampleKind.COVER
    assert cover.sum() == round(0.05 * len(ds.y_train))
    np.testing.assert_array_equal(pool.y[cover], ds.y_train[cover])
    assert pool.poison_mask.sum() == 2 * round(0.05 * len(ds.y_train))


def test_backdoor_test_set_excludes_target(tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    x_bd, y_bd = _make("badnets", ds).backdoor_test_set(ds.x_test, ds.y_test)
    assert len(x_bd) == int((ds.y_test != 0).sum()) and np.all(y_bd == 0)


def test_no_attack_is_identity(tiny_dataset: ImageDataset) -> None:
    ds = tiny_dataset
    pool = _make("none", ds).poison(ds.x_train, ds.y_train, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(pool.x, ds.x_train)
    assert not pool.poison_mask.any()
    assert "none" in available_attacks()


def test_split_contract() -> None:
    s = draw_split(1000, 100, 150, 2, np.random.default_rng(0))
    allidx = np.concatenate([s.holdout_tr, s.holdout_cal, *s.folds])
    assert np.array_equal(np.sort(allidx), np.arange(1000))
    for j in range(2):
        assert not np.intersect1d(s.train_indices(j), s.holdout).size
        assert not np.intersect1d(s.train_indices(j), s.folds[j]).size
        np.testing.assert_array_equal(s.scored_indices(j)[: len(s.folds[j])], s.folds[j])
    tr, cal = resample_holdout_partition(s.holdout, 100, np.random.default_rng(1))
    assert len(tr) == 100 and np.array_equal(np.sort(np.concatenate([tr, cal])), np.sort(s.holdout))


def test_split_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = draw_split(500, 50, 50, 3, np.random.default_rng(0))
    s.save(tmp_path / "split.npz")
    t = type(s).load(tmp_path / "split.npz")
    assert all(np.array_equal(a, b) for a, b in zip(s.folds, t.folds, strict=True)) and t.n_folds == 3


def test_audit_oracle_and_failure_modes() -> None:
    rng = np.random.default_rng(0)
    scores = np.arange(1000, dtype=float)[::-1]  # point 0 has the highest score
    poison = np.zeros(1000, bool)
    poison[[0, 5, 50, 500]] = True  # 500 lies outside the top-100
    a = audit_calibration(scores, poison, AuditConfig(budget=100), rng)
    assert set(np.flatnonzero(~a.keep)) == {0, 5, 50} and a.n_clean_removed == 0
    assert audit_calibration(scores, poison, AuditConfig(budget=100, miss_rate=1.0), rng).keep.all()
    w = audit_calibration(scores, poison, AuditConfig(budget=100, wrong_removals=2, wrong_mode="top"), rng)
    assert set(np.flatnonzero(~w.keep)) == {0, 5, 50, 1, 2}
    assert audit_calibration(scores, poison, AuditConfig(budget=0), rng).keep.all()
