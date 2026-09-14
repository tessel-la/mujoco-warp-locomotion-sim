from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_action_interface(path: Path) -> dict[str, list[str]]:
    """Field names of a ROS 2 `.action` file, by section.

    Small enough to do here, and it keeps the contract tests checking the interface
    files themselves rather than a hand-maintained copy of what they are meant to say.
    """
    sections = ("goal", "result", "feedback")
    parsed: dict[str, list[str]] = {name: [] for name in sections}
    index = 0
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped == "---":
            index += 1
            continue
        parsed[sections[index]].append(stripped.split()[1])
    return parsed
