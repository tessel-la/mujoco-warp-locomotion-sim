from __future__ import annotations

import ast
from pathlib import Path

from conftest import parse_action_interface

ROOT = Path(__file__).resolve().parents[1]
INTERFACES = ROOT / "ros/microduck_interfaces"


def test_ros_bridge_is_valid_python_and_exposes_complete_contract() -> None:
    source = (ROOT / "ros/mujoco_warp_ros_bridge.py").read_text()
    ast.parse(source)
    for name in (
        '"/joint_states"',
        '"/cmd_vel"',
        '"/odom"',
        '"/imu/data"',
        '"/clock"',
        '"/emergency_stop"',
        '"/locomotion/policy_cmd"',
        '"/locomotion/reset"',
        '"/locomotion/enable"',
        '"/locomotion/mouth"',
        '"/locomotion/mouth/open"',
        '"/locomotion/mouth/close"',
        '"/locomotion/jump"',
        '"/locomotion/execute_motion"',
        '"/locomotion/move_relative"',
        '"/diagnostics"',
    ):
        assert name in source
    assert "ReliabilityPolicy.RELIABLE" in source
    assert "TransformBroadcaster" in source
    assert "ActionServer" in source
    assert "ExecuteMotion" in source


def test_ros_image_package_dependencies_are_installed() -> None:
    dockerfile = (ROOT / "Dockerfile.ros").read_text()
    for package in (
        "diagnostic-msgs",
        "geometry-msgs",
        "nav-msgs",
        "robot-state-publisher",
        "rosgraph-msgs",
        "sensor-msgs",
        "std-srvs",
        "rosidl-default-generators",
        "tf2-ros",
    ):
        assert package in dockerfile


def test_execute_motion_action_interface_lists_policy_modes_and_jump() -> None:
    action = (ROOT / "ros/microduck_interfaces/action/ExecuteMotion.action").read_text()
    assert "string motion" in action
    assert "bool success" in action
    assert "string active_mode" in action
    bridge = (ROOT / "ros/mujoco_warp_ros_bridge.py").read_text()
    for motion in (
        "walk",
        "stand",
        "sit",
        "ground_pick",
        "kick_left",
        "kick_right",
        "roulade",
        "recover",
        "jump",
    ):
        assert f'"{motion}"' in bridge


def test_move_relative_declares_the_fields_a_behavior_tree_binds_to() -> None:
    action = parse_action_interface(INTERFACES / "action/MoveRelative.action")
    # Flat float64 fields: Robo-Boy's action parameter editor renders those as
    # labelled number inputs, which a nested geometry message would not be.
    assert action["goal"] == [
        "x",
        "y",
        "yaw",
        "max_linear_speed",
        "max_angular_speed",
        "position_tolerance",
        "yaw_tolerance",
        "timeout",
    ]
    assert action["result"] == [
        "success",
        "outcome",
        "message",
        "traveled",
        "position_error",
        "yaw_error",
        "duration",
    ]
    assert action["feedback"] == [
        "phase",
        "progress",
        "traveled",
        "remaining",
        "yaw_remaining",
        "controller",
    ]


def test_execute_motion_keeps_its_old_fields_and_gains_a_real_outcome() -> None:
    action = parse_action_interface(INTERFACES / "action/ExecuteMotion.action")
    # Trees written against the previous interface only set `motion` and read
    # `success`/`message`/`final_mode`, so they keep working unchanged.
    assert action["goal"][0] == "motion"
    for field in ("success", "message", "final_mode"):
        assert field in action["result"]
    for field in ("phase", "active_mode"):
        assert field in action["feedback"]
    # A ROS action cannot report a timeout as its own terminal state.
    assert "outcome" in action["result"]
    assert "progress" in action["feedback"]


def test_both_actions_are_generated_into_the_interface_package() -> None:
    cmake = (INTERFACES / "CMakeLists.txt").read_text()
    for name in ("ExecuteMotion", "MoveRelative"):
        assert f'"action/{name}.action"' in cmake
    # Extending an action changes its type hash, so the package version has to move
    # with it or a stale overlay will silently fail to deserialize goals.
    assert "<version>0.1.0</version>" not in (INTERFACES / "package.xml").read_text()


def test_the_bridge_reads_completion_from_the_simulator_rather_than_guessing() -> None:
    bridge = (ROOT / "ros/mujoco_warp_ros_bridge.py").read_text()
    assert "_run_motion_goal" in bridge
    assert '"/api/v1/motion"' in bridge
    assert "/cancel" in bridge
    # Both terminal branches a ROS action needs beyond abort.
    assert "goal_handle.succeed()" in bridge
    assert "goal_handle.canceled()" in bridge
    assert "is_cancel_requested" in bridge
    # The old lifecycle inferred completion from mode strings and fixed deadlines.
    assert "one_shot" not in bridge
    assert "expected_modes" not in bridge
