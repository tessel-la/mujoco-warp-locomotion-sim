#!/usr/bin/env python3
"""Download and verify Pollen's public Microduck deployment policies."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "policies"
MANIFEST = POLICY_DIR / "manifest.json"
BASE_URL = (
    "https://huggingface.co/spaces/pollen-robotics/microduck-simulator/resolve/main/"
    "app/public/policies"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected in manifest["files"].items():
        destination = POLICY_DIR / filename
        if destination.is_file() and sha256(destination) == expected:
            print(f"verified {filename}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}?download=true", temporary)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {filename}: {actual} != {expected}")
        temporary.replace(destination)
        print(f"downloaded {filename}")


if __name__ == "__main__":
    main()
