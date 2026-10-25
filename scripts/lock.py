"""Update dependency locks and preserve release-please's project-version marker."""

import re
import subprocess
from pathlib import Path

import tomllib

root = Path(__file__).resolve().parents[1]
subprocess.run(["uv", "lock"], cwd=root, check=True)
name = tomllib.loads((root / "pyproject.toml").read_text())["project"]["name"]
lock = root / "uv.lock"
pattern = r'(\[\[package\]\]\nname = "' + re.escape(name) + r'"\nversion = "[^\"]+")[^\n]*'
text, count = re.subn(pattern, r"\1 # x-release-please-version", lock.read_text())
if count != 1:
    raise RuntimeError("Expected one root project version in uv.lock")
lock.write_text(text)
for options in [
    ["--extra", "mjwarp", "--output-file", "requirements.lock"],
    ["--output-file", "requirements-mock.lock"],
]:
    subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", *options],
        cwd=root,
        check=True,
    )
