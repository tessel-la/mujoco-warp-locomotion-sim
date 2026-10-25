from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.mujoco_warp_backend import MujocoWarpBackend

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_physics_uses_native_mujoco_and_preserves_policy_mouth_and_reset(monkeypatch) -> None:
    mujoco = pytest.importorskip("mujoco")
    # The CPU path must not import or initialize Warp, even when installed alongside MuJoCo.
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(sys.modules, "mujoco_warp", None)

    class Renderer:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    settings = load_settings(ROOT / "config/default.toml", device="cpu")
    backend = MujocoWarpBackend(settings)
    try:
        backend.initialize()
        start_x = backend.state()["base_pose"]["position"][0]
        backend.set_command(0.35, 0, 0)
        for _ in range(200):
            backend.step()
        state = backend.state()
        assert state["simulation_time"] == pytest.approx(1.0)
        assert np.isfinite(state["joint_positions"]).all()
        assert state["base_pose"]["position"][0] - start_x > 0.05
        backend.set_mouth(True)
        for _ in range(20):
            backend.step()
        assert backend.state()["mouth"]["position"] < -0.01
        backend.reset()
        assert backend.state()["simulation_time"] == 0
        assert backend.state()["command"]["linear_x"] == 0
        assert backend.state()["mouth"]["opened"] is False
    finally:
        backend.close()
