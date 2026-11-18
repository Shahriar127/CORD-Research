"""Verified dataset download for fresh GPU machines (Kaggle, Colab, servers).

The archive is checked against its published MD5 before extraction, and extraction uses the
tarfile ``data`` filter, which rejects absolute paths, ``..`` components and special files.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from cord.utils.runtime import get_logger

log = get_logger("download")

ARCHIVES = {
    "cifar10": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "c58f30108f718f92721af3b95e74349a",
        "cifar-10-batches-py",
    ),
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(name: str, root: Path) -> Path:
    """Download and extract ``name`` under ``root``; returns the extracted directory."""
    try:
        url, md5, dirname = ARCHIVES[name]
    except KeyError as err:
        raise KeyError(f"no downloader for {name!r}; available: {sorted(ARCHIVES)}") from err
    root = Path(root)
    target = root / dirname
    if target.exists():
        log.info("%s already present at %s", name, target)
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(url).name
    if not archive.exists() or _md5(archive) != md5:
        log.info("downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as resp, open(archive, "wb") as f:
            shutil.copyfileobj(resp, f)
    got = _md5(archive)
    if got != md5:
        raise ValueError(f"checksum mismatch for {archive}: {got} != {md5}")
    with tarfile.open(archive, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, filter="data")
        else:  # Python < 3.10.12 / 3.11.4: validate members manually
            base = root.resolve()
            for m in tar.getmembers():
                dest = (root / m.name).resolve()
                if not (m.isfile() or m.isdir()) or base not in (dest, *dest.parents):
                    raise ValueError(f"unsafe archive member {m.name!r}")
            tar.extractall(root)  # members validated above
    archive.unlink()
    log.info("extracted to %s", target)
    return target
