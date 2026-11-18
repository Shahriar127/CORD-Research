"""Restricted unpickler for dataset files distributed as pickles (e.g. CIFAR "python" batches).

``pickle.load`` can execute arbitrary code. Dataset pickles only need a handful of NumPy
reconstruction helpers, so everything else is refused.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),  # protocol-5 arrays (NumPy >= 1.x / 2.x)
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def safe_load_pickle(path: Path, encoding: str = "bytes") -> Any:
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f, encoding=encoding).load()
