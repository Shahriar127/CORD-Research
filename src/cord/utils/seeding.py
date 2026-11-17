"""Reproducible, order-independent seeding.

Every random component gets its own generator derived from ``(master_seed, *keys)`` through a
stable hash. Results therefore do not depend on the order in which components draw numbers,
which a single global generator cannot guarantee.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np


def stable_hash(*keys: object) -> int:
    """Platform- and process-stable 32-bit hash (``hash()`` is salted per process)."""
    return zlib.crc32(repr(keys).encode("utf-8")) & 0xFFFFFFFF


def derive_seed(master: int, *keys: object) -> int:
    """A 63-bit seed derived from a master seed and string/int keys."""
    seq = np.random.SeedSequence([master & 0xFFFFFFFF, stable_hash(*keys)])
    return int(seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def make_rng(master: int, *keys: object) -> np.random.Generator:
    """Independent NumPy generator for the component named by ``keys``."""
    return np.random.default_rng(np.random.SeedSequence([master & 0xFFFFFFFF, stable_hash(*keys)]))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch global state (for libraries that use global RNGs)."""
    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)  # noqa: NPY002 - seeds third-party code using the legacy global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed & 0xFFFFFFFF))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
