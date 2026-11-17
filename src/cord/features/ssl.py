"""Frozen public encoders for the label-content signal.

A frozen public model does not depend on the pool, so any feature it produces satisfies A2
trivially and is computed once per (poisoned) pool, not per fold.

Encoders:
* ``dinov2_vits14`` / ``dinov2_vitb14``: DINOv2 (Oquab et al., 2024) via ``torch.hub`` (needs
  internet on first use; weights are cached by torch.hub). Images are resized to 224 px.
* ``pixels``: down-sampled raw pixels. Data-independent and offline; meant for tests and smoke
  runs, not for results.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cord.training.augment import as_float, to_device_images
from cord.training.trainer import autocast_context

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}


def available_encoders() -> list[str]:
    return sorted(_DINO | {"pixels"})


@torch.no_grad()
def encode_images(x: np.ndarray, encoder: str, device: str, batch_size: int = 256, amp: bool = True) -> np.ndarray:
    """Encode uint8 NHWC images; returns float16 (N, d) embeddings."""
    if encoder == "pixels":
        chunks = []
        for start in range(0, len(x), 4096):
            t = as_float(torch.from_numpy(np.ascontiguousarray(x[start : start + 4096])).permute(0, 3, 1, 2))
            small = F.adaptive_avg_pool2d(t, 8).flatten(1)
            chunks.append(small - small.mean(1, keepdim=True))
        return torch.cat(chunks).numpy().astype(np.float16)
    if encoder not in _DINO:
        raise KeyError(f"unknown encoder {encoder!r}; available: {available_encoders()}")

    # trust_repo=True: official repository; avoids an interactive trust prompt in batch jobs.
    model = torch.hub.load("facebookresearch/dinov2", encoder, trust_repo=True).to(device).eval()
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    xs = to_device_images(x, device)
    out = []
    for start in range(0, xs.shape[0], batch_size):
        b = as_float(xs[start : start + batch_size])
        b = F.interpolate(b, size=224, mode="bicubic", align_corners=False).clamp_(0, 1)
        with autocast_context(device, amp):
            emb = model((b - mean) / std)
        out.append(emb.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)
