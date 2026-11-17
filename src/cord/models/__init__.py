"""Model registry. Every model maps [0, 1] images to logits, and ``embed(x)`` returns a dict
containing at least ``"penultimate"`` and ``"logits"``; ``head`` is the final linear layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from cord.models.preact_resnet import PreActResNet, preact_resnet18

Builder = Callable[..., nn.Module]

_REGISTRY: dict[str, Builder] = {"preact_resnet18": preact_resnet18}


def register_model(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        _REGISTRY[name] = fn
        return fn

    return deco


def build_model(
    name: str, num_classes: int, mean: tuple[float, ...], std: tuple[float, ...], **kwargs: Any
) -> nn.Module:
    try:
        builder = _REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}") from err
    return builder(num_classes=num_classes, mean=mean, std=std, **kwargs)


__all__ = ["PreActResNet", "build_model", "register_model"]
