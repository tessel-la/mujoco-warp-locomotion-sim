from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
