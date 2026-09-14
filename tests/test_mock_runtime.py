from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.motion import POLICY, RELATIVE_MOVE, MotionSpec
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


def wait_for_motion(runtime: SimulationRuntime, motion_id: int, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runtime.motion_status()
        if status["id"] == motion_id and status["state"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError(f"motion {motion_id} never finished: {runtime.motion_status()}")


def test_relative_move_walks_the_robot_and_reports_where_it_stopped() -> None:
    runtime = make_runtime(timeout=1.0)
    try:
        started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=0.4))
        assert started["state"] == "running"

        status = wait_for_motion(runtime, started["id"])

        assert status["outcome"] == "succeeded"
        assert status["remaining"] <= runtime.settings.motion.position_tolerance
        position = runtime.latest_state()["base_pose"]["position"]
        assert position[0] > 0.4 - runtime.settings.motion.position_tolerance
        # The motion drove through the ordinary command path, so the staleness
        # watchdog never zeroed it mid-walk.
        assert status["traveled"] > 0.3
    finally:
        runtime.stop()


def test_a_policy_motion_stays_active_for_the_whole_policy() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        started = runtime.start_motion(MotionSpec(kind=POLICY, motion="kick_left"))
        wait_until(runtime, lambda state: state["policy"]["mode"] == "kick_left")
        assert runtime.motion_status()["state"] == "running"

        status = wait_for_motion(runtime, started["id"])

        assert status["outcome"] == "succeeded"
        assert status["duration"] >= 0.4
        assert runtime.latest_state()["policy"]["mode"] == "walk"
    finally:
        runtime.stop()


def test_the_motion_slot_is_published_with_the_rest_of_the_state() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        assert runtime.latest_state() is None or "motion" in runtime.latest_state()
        started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=0.4))
        published = wait_until(runtime, lambda state: state["motion"]["id"] == started["id"])
        assert published["motion"]["kind"] == "relative_move"
    finally:
        runtime.stop()


def test_an_operator_velocity_command_preempts_a_motion_but_a_zero_one_does_not() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=2.0))
        wait_until(runtime, lambda state: state["base_pose"]["position"][0] > 0.02)

        # The bridge's watchdog posts a single zero after a teleoperation publisher
        # goes quiet. That must not cancel a motion that is already driving.
        runtime.command(0.0, 0.0, 0.0)
        assert runtime.motion_status()["state"] == "running"

        runtime.command(0.15, 0.0, 0.0)
        status = runtime.motion_status()
        assert status["id"] == started["id"]
        assert status["outcome"] == "preempted"
        assert status["message"] == "operator velocity command"
    finally:
        runtime.stop()


def test_disable_reset_and_a_direct_policy_change_all_end_a_motion() -> None:
    for act, outcome in (
        (lambda runtime: runtime.set_enabled(False), "failed"),
        (lambda runtime: runtime.request_reset(), "failed"),
        (lambda runtime: runtime.request_policy("sit"), "preempted"),
        (lambda runtime: runtime.request_jump(), "preempted"),
    ):
        runtime = make_runtime(timeout=5.0)
        try:
            started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=2.0))
            act(runtime)
            status = runtime.motion_status()
            assert status["id"] == started["id"]
            assert status["outcome"] == outcome, (outcome, status)
        finally:
            runtime.stop()


def test_a_motion_is_cancelable_and_stops_the_robot() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=2.0))
        wait_until(runtime, lambda state: state["base_pose"]["position"][0] > 0.02)

        assert runtime.cancel_motion(started["id"]) is True
        assert runtime.motion_status()["outcome"] == "canceled"
        stopped = wait_until(
            runtime,
            lambda state: state["command"] == {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0},
        )
        assert stopped["motion"]["outcome"] == "canceled"
    finally:
        runtime.stop()


def test_a_relative_move_asks_for_walking_before_it_drives() -> None:
    runtime = make_runtime(timeout=5.0)
    try:
        runtime.request_policy("sit")
        wait_until(runtime, lambda state: state["policy"]["mode"] == "sit")

        started = runtime.start_motion(MotionSpec(kind=RELATIVE_MOVE, x=0.3))
        status = wait_for_motion(runtime, started["id"])

        assert status["outcome"] == "succeeded"
        assert runtime.latest_state()["policy"]["mode"] == "walk"
    finally:
        runtime.stop()
