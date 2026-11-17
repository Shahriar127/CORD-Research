"""One-pass feature cache for a fold model.

Everything the signals need from ``f_{-j}`` is computed here, once, for the scored points
``D_j ∪ C``; signals then run on CPU from the cache. Stored arrays:

* ``feat_<layer>``       pooled representations (float16), several depths
* ``logits``             (n, K) float32
* ``head_weight/bias``   final linear layer, for closed-form gradient statistics
* ``scaleup_pred``       (n_factors, n) predictions under pixel amplification (SCALE-UP)
* ``pred``               (n,) clean prediction
* ``strip_entropy``      (n,) mean prediction entropy under blending with fixed partners (STRIP)

STRIP partners are drawn from ``D \\ D_j`` (the fold model's own training data), which is
independent of the scored set, so assumption A2 holds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from cord.config import ExtractConfig
from cord.training.augment import as_float, to_device_images
from cord.training.trainer import autocast_context


@torch.no_grad()
def extract_fold_features(
    model: nn.Module,
    x_scored: np.ndarray,
    x_partners: np.ndarray,
    cfg: ExtractConfig,
    device: str,
    amp: bool,
) -> dict[str, np.ndarray]:
    model = model.to(device).eval()
    x = to_device_images(x_scored, device)
    partners = as_float(to_device_images(x_partners, device))
    layers = tuple(cfg.layers)
    feats: dict[str, list[Tensor]] = {name: [] for name in layers}
    logits_all: list[Tensor] = []
    scale_preds: list[Tensor] = []
    strip_ent: list[Tensor] = []
    w = cfg.strip_weight

    def run(batch: Tensor) -> Any:
        with autocast_context(device, amp):
            return model(batch)

    for start in range(0, x.shape[0], cfg.batch_size):
        xb = as_float(x[start : start + cfg.batch_size])
        with autocast_context(device, amp):
            emb = model.embed(xb)  # type: ignore[operator]
        for name in layers:
            feats[name].append(emb[name].float().cpu())
        logits = emb["logits"].float()
        logits_all.append(logits.cpu())

        scale_preds.append(
            torch.stack([run(torch.clamp(xb * f, 0.0, 1.0)).argmax(1) for f in cfg.scaleup_factors]).cpu()
        )

        ent = torch.zeros(xb.shape[0], device=xb.device)
        for p in partners:
            probs = run(w * xb + (1.0 - w) * p[None]).float().softmax(1)
            ent += -(probs * torch.log(probs.clamp_min(1e-12))).sum(1)
        strip_ent.append((ent / max(1, len(partners))).cpu())

    head: nn.Linear = model.head  # type: ignore[assignment]
    logits_np = torch.cat(logits_all).numpy()
    out: dict[str, np.ndarray] = {f"feat_{name}": torch.cat(feats[name]).numpy().astype(np.float16) for name in layers}
    out.update(
        logits=logits_np.astype(np.float32),
        pred=logits_np.argmax(1).astype(np.int16),
        scaleup_pred=torch.cat(scale_preds, dim=1).numpy().astype(np.int16),
        scaleup_factors=np.asarray(cfg.scaleup_factors, np.float32),
        strip_entropy=torch.cat(strip_ent).numpy().astype(np.float32),
        head_weight=head.weight.detach().float().cpu().numpy(),
        head_bias=head.bias.detach().float().cpu().numpy(),
    )
    return out
