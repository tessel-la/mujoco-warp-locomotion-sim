from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.runtime import SimulationRuntime

ROOT = Path(__file__).resolve().parents[1]


def make_runtime(*, timeout: float = 0.08) -> SimulationRuntime:
    settings = load_settings(ROOT / "config/default.toml").with_overrides(backend="mock")
    settings = replace(
        settings,
        simulation=replace(
            settings.simulation,
            realtime=True,
            dt=0.005,
            state_hz=100.0,
            render_hz=30.0,
            command_timeout=timeout,
        ),
    )
    runtime = SimulationRuntime(settings)
    runtime.start()
    deadline = time.monotonic() + 2.0
    while not runtime.ready and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.ready, runtime.startup_error
    return runtime


def wait_until(runtime: SimulationRuntime, predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    sequence = 0
    while time.monotonic() < deadline:
        state = runtime.wait_for_state(sequence, 0.1)
        if state is not None:
            sequence = int(state["sequence"])
            if predicate(state):
                return state
    raise AssertionError("state predicate did not become true")


def test_command_moves_robot_and_watchdog_stops_it() -> None:
    runtime = make_runtime()
    try:
        accepted = runtime.command(9.0, -9.0, 9.0)
        assert accepted == {"linear_x": 0.45, "linear_y": -0.2, "angular_z": 1.2}
        moving = wait_until(runtime, lambda state: state["mode"] == "walking")
        assert moving["base_pose"]["position"][0] > 0.0
        stopped = wait_until(runtime, lambda state: state["command_stale"])
        assert stopped["command"] == {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        assert stopped["mode"] == "standing"
    finally:
        runtime.stop()


def test_disable_and_reset_preserve_safety_state() -> None:
    runtime = make_runtime(timeout=1.0)
    try:
        runtime.command(0.2, 0.0, 0.0)
        wait_until(runtime, lambda state: state["base_pose"]["position"][0] > 0.002)
        runtime.set_enabled(False)
        disabled = wait_until(runtime, lambda state: not state["enabled"])
        assert disabled["command"] == {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        runtime.request_reset()
        reset = wait_until(runtime, lambda state: state["base_pose"]["position"][0] == 0.0)
        assert reset["enabled"] is False
    finally:
        runtime.stop()


def test_mock_camera_is_a_jpeg() -> None:
    runtime = make_runtime()
    try:
        deadline = time.monotonic() + 2.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = runtime.wait_for_frame(0, 0.1)
        assert frame is not None
        assert frame.jpeg.startswith(b"\xff\xd8")
        assert frame.jpeg.endswith(b"\xff\xd9")
    finally:
        runtime.stop()


def test_mouth_and_jump_actions_are_visible_in_state() -> None:
    runtime = make_runtime(timeout=1.0)
    try:
        assert runtime.request_mouth(True) is True
        opened = wait_until(runtime, lambda state: state["mouth"]["opened"])
        assert opened["joint_names"][-1] == "mouth"
        assert opened["joint_positions"][-1] < 0.0

        runtime.request_jump()
        jumping = wait_until(runtime, lambda state: state["mode"] == "jumping")
        assert jumping["base_pose"]["position"][2] > 0.62
        landed = wait_until(
            runtime,
            lambda state: state["mode"] != "jumping"
            and state["base_pose"]["position"][2] == 0.62,
        )
        assert landed["base_twist"]["linear"][2] == 0.0
    finally:
        runtime.stop()


def test_mock_one_shot_policy_returns_to_walk() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        assert runtime.request_policy("kick_left") == "kick_left"
        wait_until(runtime, lambda state: state["policy"]["mode"] == "kick_left")
        returned = wait_until(runtime, lambda state: state["policy"]["mode"] == "walk")
        assert "kick_left" in returned["policy"]["available"]
    finally:
        runtime.stop()
