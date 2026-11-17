"""Device-resident batching and augmentation.

The whole uint8 dataset lives on the accelerator; batches are sliced, converted and augmented
there (random crop with zero padding and horizontal flip, the BackdoorBench CIFAR recipe). This
removes the CPU data-loader bottleneck that dominates small-image training on notebook GPUs.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor


def to_device_images(x: np.ndarray, device: str | torch.device) -> Tensor:
    """uint8 NHWC numpy -> uint8 NCHW tensor on ``device`` (kept uint8 to save memory)."""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous().to(device)


def as_float(x_u8: Tensor) -> Tensor:
    return x_u8.float().div_(255.0)


def random_crop_flip(x: Tensor, padding: int, generator: torch.Generator | None, flip: bool = True) -> Tensor:
    """Batched random crop (zero padding) and horizontal flip on a float NCHW tensor."""
    n, _, h, w = x.shape
    if padding > 0:
        padded = torch.nn.functional.pad(x, (padding, padding, padding, padding))
        dev = x.device
        oy = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
        ox = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
        rows = (oy[:, None] + torch.arange(h, device=dev)[None, :])[:, None, :, None]
        cols = (ox[:, None] + torch.arange(w, device=dev)[None, :])[:, None, None, :]
        batch = torch.arange(n, device=dev)[:, None, None, None]
        chans = torch.arange(x.shape[1], device=dev)[None, :, None, None]
        x = padded[batch, chans, rows, cols]
    if flip:
        mask = torch.rand(n, device=x.device, generator=generator) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(3), x)
    return x


def iterate_batches(
    n: int,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
    device: str | torch.device,
    drop_last: bool = False,
) -> Iterator[Tensor]:
    order = torch.randperm(n, device=device, generator=generator) if shuffle else torch.arange(n, device=device)
    stop = n - (n % batch_size) if drop_last and n >= batch_size else n
    for start in range(0, stop, batch_size):
        yield order[start : start + batch_size]
