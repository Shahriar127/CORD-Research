from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from cord.data import download as dl


def _archive(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, str]:
    path = tmp_path / "src" / "archive.tar.gz"
    path.parent.mkdir()
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path, hashlib.md5(path.read_bytes()).hexdigest()


def test_download_verifies_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, md5 = _archive(tmp_path, {"set/data_batch_1": b"x"})
    monkeypatch.setitem(dl.ARCHIVES, "fake", (src.as_uri(), md5, "set"))
    out = dl.download_dataset("fake", tmp_path / "data")
    assert (out / "data_batch_1").read_bytes() == b"x"
    assert dl.download_dataset("fake", tmp_path / "data") == out  # idempotent


def test_download_rejects_bad_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _archive(tmp_path, {"set/a": b"x"})
    monkeypatch.setitem(dl.ARCHIVES, "fake", (src.as_uri(), "0" * 32, "set"))
    with pytest.raises(ValueError, match="checksum"):
        dl.download_dataset("fake", tmp_path / "data")


def test_download_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, md5 = _archive(tmp_path, {"../escape": b"x"})
    monkeypatch.setitem(dl.ARCHIVES, "fake", (src.as_uri(), md5, "set"))
    with pytest.raises((ValueError, tarfile.TarError)):
        dl.download_dataset("fake", tmp_path / "data")
    assert not (tmp_path / "escape").exists()
