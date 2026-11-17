"""Poisoning attack interface.

An attack modifies the untrusted pool before the defender sees it. Following Section 3.1, a
sample is *poisoned* iff the attacker modified it (image and/or label); this includes cover /
noise samples that keep their true label (Adap-Blend, WaNet noise mode, clean-label attacks).
Naturally mislabelled samples are clean.

Triggers are drawn once from a dedicated trigger generator, so the same attack object
reproduces the identical poisoned pool and the matching test-time trigger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar, Literal

import numpy as np

Mapping = Literal["all_to_one", "all_to_all"]


class SampleKind(IntEnum):
    CLEAN = 0
    POISON = 1  # carries the trigger (dirty-label: relabelled to target)
    COVER = 2  # modified by the attacker but keeps its true label (regularisation / noise samples)


@dataclass(frozen=True)
class PoisonedPool:
    x: np.ndarray  # uint8 NHWC after poisoning
    y: np.ndarray  # given labels after poisoning
    y_clean: np.ndarray  # original labels
    kind: np.ndarray  # int8 SampleKind per sample
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def poison_mask(self) -> np.ndarray:
        """Ground-truth non-null hypotheses: every sample the attacker modified."""
        return self.kind != SampleKind.CLEAN

    @property
    def rate(self) -> float:
        return float(self.poison_mask.mean())


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


class Attack(ABC):
    """Base class. Subclasses implement :meth:`_trigger` and may override selection/labels."""

    name: ClassVar[str]
    clean_label: ClassVar[bool] = False

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        num_classes: int,
        target: int = 0,
        mapping: Mapping = "all_to_one",
        trigger_rng: np.random.Generator | None = None,
    ) -> None:
        if not 0 <= target < num_classes:
            raise ValueError("target class out of range")
        if self.clean_label and mapping != "all_to_one":
            raise ValueError("clean-label attacks are all-to-one")
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.target = target
        self.mapping: Mapping = mapping
        self.trigger_rng = trigger_rng if trigger_rng is not None else np.random.default_rng(0)

    # -- trigger ---------------------------------------------------------------------------
    @abstractmethod
    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        """Apply the trigger to a uint8 NHWC batch."""

    def apply_trigger(self, x: np.ndarray, train: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
        if x.size == 0:
            return x.copy()
        return self._trigger(x, train, rng if rng is not None else np.random.default_rng(0))

    # -- labels and selection --------------------------------------------------------------
    def target_labels(self, y: np.ndarray) -> np.ndarray:
        if self.mapping == "all_to_one":
            return np.full_like(y, self.target)
        return (y + 1) % self.num_classes

    def candidates(self, y: np.ndarray) -> np.ndarray:
        if self.clean_label:
            return np.flatnonzero(y == self.target)
        if self.mapping == "all_to_one":
            return np.flatnonzero(y != self.target)
        return np.arange(y.size)

    def n_cover(self, n_pool: int, rate: float) -> int:
        return 0

    # -- poisoning -------------------------------------------------------------------------
    def poison(self, x: np.ndarray, y: np.ndarray, rate: float, rng: np.random.Generator) -> PoisonedPool:
        """Poison ``round(rate * n)`` samples of the pool chosen uniformly among candidates."""
        n = y.size
        n_poison = round(rate * n)
        cand = self.candidates(y)
        if n_poison > cand.size:
            raise ValueError(f"{self.name}: rate {rate} needs {n_poison} candidates, have {cand.size}")
        poison_idx = np.sort(rng.choice(cand, size=n_poison, replace=False))
        kind = np.zeros(n, dtype=np.int8)
        kind[poison_idx] = SampleKind.POISON

        x_out, y_out = x.copy(), y.copy()
        x_out[poison_idx] = self.apply_trigger(x[poison_idx], train=True, rng=rng)
        if not self.clean_label:
            y_out[poison_idx] = self.target_labels(y[poison_idx])

        n_cov = self.n_cover(n, rate)
        if n_cov:
            rest = np.setdiff1d(np.arange(n), poison_idx)
            cover_idx = np.sort(rng.choice(rest, size=n_cov, replace=False))
            x_out[cover_idx] = self._cover(x[cover_idx], rng)
            kind[cover_idx] = SampleKind.COVER
        return PoisonedPool(
            x_out,
            y_out,
            y.copy(),
            kind,
            {"attack": self.name, "n_poison": int(n_poison), "n_cover": int(n_cov), "target": self.target},
        )

    def _cover(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def backdoor_test_set(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Triggered test inputs (excluding the target class) and the attacker's desired labels."""
        keep = y != self.target if self.mapping == "all_to_one" else np.ones_like(y, dtype=bool)
        return self.apply_trigger(x[keep], train=False), self.target_labels(y[keep])


class NoAttack(Attack):
    """The no-attack condition: the pool is returned untouched."""

    name = "none"

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        return x.copy()

    def poison(self, x: np.ndarray, y: np.ndarray, rate: float, rng: np.random.Generator) -> PoisonedPool:
        return PoisonedPool(x.copy(), y.copy(), y.copy(), np.zeros(y.size, np.int8), {"attack": "none"})

    def backdoor_test_set(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:0], y[:0]
