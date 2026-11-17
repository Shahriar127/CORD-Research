"""Concrete attacks: BadNets, Blend, SIG, WaNet and Adaptive-Blend.

Defaults follow BackdoorBench / the original papers where they are image-free. Where the
original uses a specific image (the Hello-Kitty blend of Chen et al.), a seeded random pattern
is the default and ``trigger_image`` lets you supply the original for exact comparability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from cord.attacks.base import Attack, Mapping, to_float, to_uint8


def _load_pattern(path: str | None, shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Float pattern in [0, 1] with the image shape: from a file, else seeded uniform noise."""
    h, w, c = shape
    if path is None:
        return rng.random((h, w, c), dtype=np.float32)
    from PIL import Image

    img = Image.open(Path(path)).convert("RGB" if c == 3 else "L").resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(h, w, c)


class BadNets(Attack):
    """Gu et al.: a small checkerboard patch in the bottom-right corner (dirty label)."""

    name = "badnets"

    def __init__(self, *args: object, patch_size: int = 3, margin: int = 1, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        h, w, c = self.image_shape
        if patch_size + margin > min(h, w):
            raise ValueError("patch does not fit the image")
        ii, jj = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="ij")
        self.patch = np.repeat((((ii + jj) % 2 == 0) * 255).astype(np.uint8)[..., None], c, axis=2)
        self.r0 = h - margin - patch_size
        self.c0 = w - margin - patch_size

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        out = x.copy()
        p = self.patch.shape[0]
        out[:, self.r0 : self.r0 + p, self.c0 : self.c0 + p, :] = self.patch
        return out


class Blend(Attack):
    """Chen et al.: blend a global pattern with weight ``alpha`` (dirty label)."""

    name = "blend"

    def __init__(
        self,
        *args: object,
        alpha: float = 0.2,
        alpha_test: float | None = None,
        trigger_image: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.alpha = alpha
        self.alpha_test = alpha if alpha_test is None else alpha_test
        self.pattern = _load_pattern(trigger_image, self.image_shape, self.trigger_rng)

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        a = self.alpha if train else self.alpha_test
        return to_uint8((1.0 - a) * to_float(x) + a * self.pattern)


class SIG(Attack):
    """Barni et al.: additive horizontal sinusoid, clean label (poison drawn from the target class)."""

    name = "sig"
    clean_label = True

    def __init__(self, *args: object, delta: float = 20.0, frequency: float = 6.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        h, w, c = self.image_shape
        cols = np.arange(w, dtype=np.float32)
        signal = delta * np.sin(2.0 * np.pi * cols * frequency / w) / 255.0
        self.signal = np.broadcast_to(signal[None, :, None], (h, w, c)).astype(np.float32)

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        return to_uint8(to_float(x) + self.signal)


class WaNet(Attack):
    """Nguyen and Tran: smooth warping field (k control points, strength s), dirty label.

    ``cross_ratio`` > 0 adds noise-mode samples (random extra warp, true label kept), counted
    as COVER samples. Defaults follow the paper: k=4, s=0.5, grid_rescale=1.
    """

    name = "wanet"

    def __init__(
        self,
        *args: object,
        k: int = 4,
        s: float = 0.5,
        grid_rescale: float = 1.0,
        cross_ratio: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        import torch
        import torch.nn.functional as F

        h, w, _ = self.image_shape
        if h != w:
            raise ValueError("WaNet implementation assumes square images")
        self.h = h
        ins = torch.from_numpy(self.trigger_rng.random((1, 2, k, k), dtype=np.float32)) * 2 - 1
        ins = ins / ins.abs().mean()
        noise_grid = F.interpolate(ins, size=h, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        lin = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(lin, lin, indexing="ij")
        identity = torch.stack((yy, xx), dim=2)[None]
        grid = (identity + s * noise_grid / h) * grid_rescale
        self.grid = torch.clamp(grid, -1, 1)
        self.cross_ratio = cross_ratio

    def _warp(self, x: np.ndarray, grid: torch.Tensor) -> np.ndarray:
        """Warp a uint8 batch with a shared grid (1, H, W, 2) or per-sample grids (n, H, W, 2)."""
        import torch
        import torch.nn.functional as F

        out = np.empty_like(x)
        chunk = 1024
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            xb = torch.from_numpy(to_float(x[start:stop])).permute(0, 3, 1, 2)
            g = grid.expand(stop - start, -1, -1, -1) if grid.shape[0] == 1 else grid[start:stop]
            yb = F.grid_sample(xb, g, align_corners=True).permute(0, 2, 3, 1).numpy()
            out[start:stop] = to_uint8(yb)
        return out

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        return self._warp(x, self.grid)

    def n_cover(self, n_pool: int, rate: float) -> int:
        return round(self.cross_ratio * rate * n_pool)

    def _cover(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import torch

        noise = torch.from_numpy(rng.random((len(x), self.h, self.h, 2), dtype=np.float32)) * 2 - 1
        grid2 = torch.clamp(self.grid + noise / self.h, -1, 1)
        return self._warp(x, grid2)


class AdaptiveBlend(Attack):
    """Qi et al. (ICLR 2023) Adap-Blend: the blend pattern is split into ``pieces`` patches and each
    poisoned or cover sample carries a random subset (``1 - mask_rate`` of the pieces) at
    ``alpha``; test inputs carry the full pattern at ``alpha_test``. Cover samples keep their
    true label, which suppresses latent separability.
    """

    name = "adaptive_blend"

    def __init__(
        self,
        *args: object,
        alpha: float = 0.15,
        alpha_test: float = 0.2,
        pieces: int = 16,
        mask_rate: float = 0.5,
        cover_rate: float | None = None,
        trigger_image: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        side = round(np.sqrt(pieces))
        if side * side != pieces:
            raise ValueError("pieces must be a perfect square")
        h, w, _ = self.image_shape
        self.side, self.alpha, self.alpha_test, self.mask_rate = side, alpha, alpha_test, mask_rate
        self.cover_rate = cover_rate
        self.pattern = _load_pattern(trigger_image, self.image_shape, self.trigger_rng)
        self.row_edges = np.linspace(0, h, side + 1).round().astype(int)
        self.col_edges = np.linspace(0, w, side + 1).round().astype(int)

    def _piece_masks(self, n: int, rng: np.random.Generator) -> np.ndarray:
        h, w, _ = self.image_shape
        n_keep = self.side**2 - round(self.mask_rate * self.side**2)
        masks = np.zeros((n, h, w, 1), dtype=np.float32)
        for i in range(n):
            for p in rng.choice(self.side**2, size=n_keep, replace=False):
                r, c = divmod(int(p), self.side)
                masks[i, self.row_edges[r] : self.row_edges[r + 1], self.col_edges[c] : self.col_edges[c + 1]] = 1.0
        return masks

    def _blend(self, x: np.ndarray, alpha: float, mask: np.ndarray | float) -> np.ndarray:
        a = alpha * mask
        return to_uint8((1.0 - a) * to_float(x) + a * self.pattern)

    def _trigger(self, x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
        if train:
            return self._blend(x, self.alpha, self._piece_masks(len(x), rng))
        return self._blend(x, self.alpha_test, 1.0)

    def n_cover(self, n_pool: int, rate: float) -> int:
        return round((rate if self.cover_rate is None else self.cover_rate) * n_pool)

    def _cover(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._blend(x, self.alpha, self._piece_masks(len(x), rng))


ATTACKS: dict[str, type[Attack]] = {cls.name: cls for cls in (BadNets, Blend, SIG, WaNet, AdaptiveBlend)}

__all__ = ["ATTACKS", "SIG", "AdaptiveBlend", "BadNets", "Blend", "Mapping", "WaNet"]
