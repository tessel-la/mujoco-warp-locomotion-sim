from __future__ import annotations

import json
from pathlib import Path

from conftest import parse_action_interface

ROOT = Path(__file__).resolve().parents[1]
TREES = sorted((ROOT / "behavior_trees").glob("*.json"))
BRIDGE = (ROOT / "ros/mujoco_warp_ros_bridge.py").read_text()

# The action servers the bridge actually offers, and the interface each one speaks.
ACTIONS = {
    "/locomotion/execute_motion": "microduck_interfaces/action/ExecuteMotion",
    "/locomotion/move_relative": "microduck_interfaces/action/MoveRelative",
}

# Node types Robo-Boy's executor dispatches on (`BehaviorNodeType` in
# src/features/behaviorTree/types.ts).
NODE_TYPES = {
    "sequence",
    "selector",
    "parallel",
    "retry",
    "repeat",
    "timeout",
    "ifElse",
    "subtree",
    "action",
    "service",
    "topic",
    "subscriber",
    "condition",
}


def goal_fields(action_type: str) -> list[str]:
    name = action_type.rsplit("/", 1)[-1]
    return parse_action_interface(ROOT / f"ros/microduck_interfaces/action/{name}.action")["goal"]


def test_the_repository_ships_behavior_trees_for_every_action() -> None:
    assert TREES
    used = {
        node["data"]["actionName"]
        for tree in TREES
        for node in json.loads(tree.read_text())["tree"]["nodes"]
        if node["type"] == "action"
    }
    assert used == set(ACTIONS)


def test_every_tree_is_a_valid_robo_boy_export() -> None:
    for path in TREES:
        saved = json.loads(path.read_text())
        assert saved["version"], path
        tree = saved["tree"]
        assert tree["id"] == path.stem, path
        assert tree["name"] and tree["description"], path

        ids = {node["id"] for node in tree["nodes"]}
        assert len(ids) == len(tree["nodes"]), path
        for node in tree["nodes"]:
            assert node["type"] in NODE_TYPES, (path, node["type"])
            assert node["data"]["label"], (path, node["id"])
        for edge in tree["edges"]:
            assert edge["source"] in ids, (path, edge)
            assert edge["target"] in ids, (path, edge)
        # A tree the executor can run has exactly one node with no incoming edge.
        targets = {edge["target"] for edge in tree["edges"]}
        assert len(ids - targets) == 1, path


def test_every_action_node_matches_an_interface_the_bridge_serves() -> None:
    for path in TREES:
        for node in json.loads(path.read_text())["tree"]["nodes"]:
            if node["type"] != "action":
                continue
            data = node["data"]
            name = data["actionName"]
            assert name in ACTIONS, (path, name)
            assert data["actionType"] == ACTIONS[name], (path, name)
            assert f'"{name}"' in BRIDGE, name
            # Every goal field the tree sets has to exist in the interface, or
            # rosbridge silently drops it and the robot does something else.
            declared = goal_fields(data["actionType"])
            for field in data.get("parameters", {}):
                assert field in declared, (path, name, field)


def test_the_motion_names_used_in_trees_are_ones_the_bridge_accepts() -> None:
    for path in TREES:
        for node in json.loads(path.read_text())["tree"]["nodes"]:
            motion = node["data"].get("parameters", {}).get("motion")
            if motion is not None:
                assert f'"{motion}"' in BRIDGE, (path, motion)


def test_the_cancellation_tree_actually_exercises_cancellation() -> None:
    tree = json.loads((ROOT / "behavior_trees/microduck-failure-recovery.json").read_text())["tree"]
    nodes = {node["id"]: node for node in tree["nodes"]}
    budget = next(node for node in tree["nodes"] if node["type"] == "timeout")
    wrapped = next(edge["target"] for edge in tree["edges"] if edge["source"] == budget["id"])

    # The wrapped goal must be unable to finish inside the budget, or the tree would
    # demonstrate nothing: the timeout is what makes Robo-Boy cancel the goal.
    assert budget["data"]["timeout"] < nodes[wrapped]["data"]["parameters"]["timeout"] * 1000
    assert any(node["type"] == "selector" for node in tree["nodes"])
