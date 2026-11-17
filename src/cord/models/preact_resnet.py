"""PreAct-ResNet-18 (He et al., 2016), the BackdoorBench reference architecture for 32x32 inputs.

Every CORD model takes inputs in [0, 1] (normalisation is a module inside the network, so
attacks, perturbation signals and blending all operate in pixel space) and exposes
:meth:`embed`, returning pooled representations at several depths in one forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Normalize(nn.Module):
    mean: Tensor
    std: Tensor

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut: nn.Module | None = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    """``embed`` returns {"layer2", "layer3", "penultimate"} pooled features and ``logits``."""

    def __init__(
        self,
        num_blocks: tuple[int, int, int, int],
        num_classes: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        width: int = 64,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.in_planes = width
        self.conv1 = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(width, num_blocks[0], 1)
        self.layer2 = self._make_layer(2 * width, num_blocks[1], 2)
        self.layer3 = self._make_layer(4 * width, num_blocks[2], 2)
        self.layer4 = self._make_layer(8 * width, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(8 * width)
        self.linear = nn.Linear(8 * width, num_classes)
        self.feature_dim = 8 * width

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    @property
    def head(self) -> nn.Linear:
        return self.linear

    def embed(self, x: Tensor) -> dict[str, Tensor]:
        out = self.conv1(self.normalize(x))
        out = self.layer1(out)
        l2 = self.layer2(out)
        l3 = self.layer3(l2)
        l4 = F.relu(self.bn(self.layer4(l3)))
        pen = F.adaptive_avg_pool2d(l4, 1).flatten(1)
        return {
            "layer2": F.adaptive_avg_pool2d(l2, 1).flatten(1),
            "layer3": F.adaptive_avg_pool2d(l3, 1).flatten(1),
            "penultimate": pen,
            "logits": self.linear(pen),
        }

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.normalize(x))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1).flatten(1)
        return self.linear(out)


def preact_resnet18(num_classes: int, mean: tuple[float, ...], std: tuple[float, ...], width: int = 64) -> PreActResNet:
    return PreActResNet((2, 2, 2, 2), num_classes, mean, std, width=width, in_channels=len(mean))
