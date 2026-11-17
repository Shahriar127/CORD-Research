"""Device selection, logging and lightweight timing."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER_NAME = "cord"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False


def resolve_device(spec: str = "auto") -> str:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    import torch

    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        get_logger("timing").info("%s took %.1fs", label, elapsed)
        if sink is not None:
            sink[label] = sink.get(label, 0.0) + elapsed
