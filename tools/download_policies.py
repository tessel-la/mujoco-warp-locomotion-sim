#!/usr/bin/env python3
"""Download and verify Pollen's public Microduck deployment policies."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "policies"
MANIFEST = POLICY_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    base_url = (
        "https://huggingface.co/spaces/pollen-robotics/microduck-simulator/resolve/"
        f"{manifest['revision']}/app/public/policies"
    )
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected in manifest["files"].items():
        destination = POLICY_DIR / filename
        if destination.is_file() and sha256(destination) == expected:
            print(f"verified {filename}")
            continue
        with tempfile.NamedTemporaryFile(
            dir=POLICY_DIR, suffix=".download", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(
                f"{base_url}/{filename}?download=true", timeout=30
            ) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            actual = sha256(temporary)
            if actual != expected:
                raise RuntimeError(f"checksum mismatch for {filename}: {actual} != {expected}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"downloaded {filename}")


if __name__ == "__main__":
    main()
