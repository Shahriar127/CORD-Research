"""Attack registry. ``build_attack(name, ...)`` constructs any registered attack by name."""

from __future__ import annotations

from typing import Any

import numpy as np

from cord.attacks.base import Attack, NoAttack, PoisonedPool, SampleKind
from cord.attacks.triggers import ATTACKS

_REGISTRY: dict[str, type[Attack]] = {"none": NoAttack, **ATTACKS}


def register_attack(cls: type[Attack]) -> type[Attack]:
    if cls.name in _REGISTRY:
        raise KeyError(f"attack {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def build_attack(
    name: str,
    image_shape: tuple[int, int, int],
    num_classes: int,
    *,
    target: int = 0,
    mapping: str = "all_to_one",
    trigger_rng: np.random.Generator | None = None,
    params: dict[str, Any] | None = None,
) -> Attack:
    try:
        cls = _REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"unknown attack {name!r}; available: {sorted(_REGISTRY)}") from err
    return cls(image_shape, num_classes, target, mapping, trigger_rng, **(params or {}))  # type: ignore[arg-type]


def available_attacks() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Attack", "PoisonedPool", "SampleKind", "available_attacks", "build_attack", "register_attack"]
