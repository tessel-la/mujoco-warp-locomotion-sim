from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.policy_controller import (
    ACTION_SIZE,
    DEFAULT_POSE,
    OBSERVATION_SIZE,
    PolicyController,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_observation: np.ndarray | None = None

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, OBSERVATION_SIZE])]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, ACTION_SIZE])]

    def run(self, _outputs, feeds):
        self.last_observation = feeds["obs"].copy()
        return [np.full((1, ACTION_SIZE), 0.1, dtype=np.float32)]


def make_controller() -> tuple[PolicyController, SimpleNamespace]:
    settings = load_settings(ROOT / "config/default.toml")
    controller = PolicyController(
        settings.policy,
        joint_qpos=list(range(7, 21)),
        joint_dof=list(range(6, 20)),
        gyro_slice=slice(0, 3),
        body_id=1,
        session_factory=FakeSession,
    )
    controller.initialize()
    qpos = np.zeros(21, dtype=np.float64)
    qpos[7:21] = DEFAULT_POSE
    data = SimpleNamespace(
        qpos=qpos,
        qvel=np.zeros(20, dtype=np.float64),
        sensordata=np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        xquat=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.12]]),
    )
    return controller, data


def test_policy_controller_loads_all_official_modes_and_builds_61d_observation() -> None:
    controller, data = make_controller()
    controller.set_command(0.2, -0.1, 0.4)

    targets = controller.targets(data)

    assert sorted(controller.sessions) == [
        "ground_pick",
        "kick_left",
        "kick_right",
        "roulade",
        "sitstand",
        "stand",
        "walk",
    ]
    assert targets == pytest.approx(DEFAULT_POSE + 0.1)
    assert controller.last_observation.shape == (61,)
    assert controller.last_observation[:6] == pytest.approx([0.1, -0.2, 0.3, 0, 0, -1])
    assert controller.last_observation[48:51] == pytest.approx([0.2, -0.1, 0.4])


def test_policy_controller_switches_sit_and_one_shot_modes() -> None:
    controller, data = make_controller()

    assert controller.request_mode("sit") == "sitstand"
    for _ in range(40):
        controller.after_step(data)
    controller.targets(data)
    assert controller.last_observation[48] == 1.0

    assert controller.request_mode("ground_pick") == "ground_pick"
    controller.targets(data)
    assert controller.last_observation[48:51] == pytest.approx([1.0, 0.0, 0.0])
    for _ in range(140):
        controller.after_step(data)
    assert controller.mode == "walk"
