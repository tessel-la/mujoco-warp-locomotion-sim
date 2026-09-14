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
    POSTURE_SETTLE_STEPS,
    SIT_FLAG_DELAY_STEPS,
    STAND_UP_STEPS,
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


def test_a_persistent_mode_finishes_as_soon_as_it_is_active() -> None:
    controller, data = make_controller()
    before = controller.status()["episode"]

    controller.request_mode("walk")
    opened = controller.status()
    assert opened["episode"] == before + 1
    assert opened["kind"] == "persistent"
    assert opened["phase"] == "running"

    controller.after_step(data)
    # Walking is the controller's resting state, not a motion with an end, so it is
    # done once it is active rather than after an invented delay.
    assert controller.status()["phase"] == "complete"


def test_a_one_shot_mode_stays_running_until_it_hands_back_to_walking() -> None:
    controller, data = make_controller()
    controller.request_mode("kick_left")

    for _ in range(24):
        controller.after_step(data)
        assert controller.status()["phase"] == "running"
    progress = controller.status()["progress"]
    assert 0.9 <= progress < 1.0

    controller.after_step(data)
    assert controller.mode == "walk"
    assert controller.status()["phase"] == "complete"
    assert controller.status()["progress"] == 1.0


def steps_to_finish(controller, data, limit: int = 600) -> int:
    for step in range(1, limit + 1):
        controller.after_step(data)
        if controller.status()["phase"] != "running":
            return step
    raise AssertionError("the episode never finished")


def test_sit_and_stand_wait_for_the_posture_instead_of_matching_a_mode_name() -> None:
    controller, data = make_controller()

    # Both map onto the same `sitstand` session, so a check on the mode name alone
    # would report either one complete the instant it was requested.
    controller.request_mode("stand")
    assert controller.mode == "sitstand"
    assert controller.status()["kind"] == "transient"
    assert steps_to_finish(controller, data) == POSTURE_SETTLE_STEPS

    controller.request_mode("sit")
    # `sit` holds the stand pose before the flag flips, and only then settles.
    assert steps_to_finish(controller, data) == SIT_FLAG_DELAY_STEPS + POSTURE_SETTLE_STEPS
    assert controller.status()["phase"] == "complete"
    controller.targets(data)
    assert controller.last_observation[48] == 1.0


def test_walking_out_of_a_sit_waits_for_the_stand_up() -> None:
    controller, data = make_controller()
    controller.request_mode("sit")
    steps_to_finish(controller, data)

    controller.request_mode("walk")
    # Standing back up first is a real transition, so this episode is not persistent.
    assert controller.status()["kind"] == "transient"
    assert steps_to_finish(controller, data) == STAND_UP_STEPS
    assert controller.mode == "walk"
    assert controller.status()["phase"] == "complete"


def test_an_episode_abandoned_by_a_reset_is_reported_as_failed() -> None:
    controller, data = make_controller()
    controller.request_mode("roulade")
    controller.after_step(data)
    assert controller.status()["phase"] == "running"

    controller.reset()

    assert controller.status()["phase"] == "failed"
    assert controller.status()["request"] == "roulade"


def test_the_episode_counter_never_goes_backwards() -> None:
    controller, data = make_controller()
    controller.request_mode("kick_left")
    episode = controller.status()["episode"]

    controller.reset()

    assert controller.status()["episode"] == episode
    controller.request_mode("kick_right")
    assert controller.status()["episode"] == episode + 1
