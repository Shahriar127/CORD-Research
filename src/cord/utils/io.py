"""Artifact I/O: atomic writes so an interrupted session (e.g. a notebook timeout) never leaves
half-written files that a resumed run would mistake for finished work."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_target(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return fd, Path(tmp)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path, data: Any) -> None:
    fd, tmp = _atomic_target(path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_jsonable, sort_keys=False)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_to_jsonable) + "\n")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, tmp = _atomic_target(path)
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **arrays)  # type: ignore[arg-type]
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_torch(path: Path, obj: Any) -> None:
    import torch

    fd, tmp = _atomic_target(path)
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def array_digest(*arrays: np.ndarray) -> str:
    """SHA-256 over the raw bytes of arrays (used to verify regenerated poisoned pools)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
