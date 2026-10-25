from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import download_policies


def test_download_is_pinned_verified_and_cached(tmp_path: Path, monkeypatch) -> None:
    payload = b"policy weights"
    manifest = tmp_path / "manifest.json"
    revision = "a" * 40
    manifest.write_text(
        json.dumps(
            {
                "revision": revision,
                "files": {
                    "walk.onnx": hashlib.sha256(payload).hexdigest(),
                },
            }
        )
    )
    calls = []

    def download(url, *, timeout):
        calls.append(url)
        assert f"/resolve/{revision}/" in url
        assert timeout > 0
        return io.BytesIO(payload)

    monkeypatch.setattr(download_policies, "MANIFEST", manifest)
    monkeypatch.setattr(download_policies, "POLICY_DIR", tmp_path)
    monkeypatch.setattr(download_policies.urllib.request, "urlopen", download)
    download_policies.main()
    download_policies.main()
    assert len(calls) == 1
    assert (tmp_path / "walk.onnx").read_bytes() == payload
    assert not list(tmp_path.glob("*.download"))


def test_corrupt_download_preserves_existing_file(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"revision": "a" * 40, "files": {"walk.onnx": "0" * 64}}))
    existing = tmp_path / "walk.onnx"
    existing.write_bytes(b"old file")
    monkeypatch.setattr(download_policies, "MANIFEST", manifest)
    monkeypatch.setattr(download_policies, "POLICY_DIR", tmp_path)
    monkeypatch.setattr(
        download_policies.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_policies.main()
    assert existing.read_bytes() == b"old file"
    assert not list(tmp_path.glob("*.download"))
