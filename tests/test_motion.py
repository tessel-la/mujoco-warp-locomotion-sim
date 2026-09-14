from __future__ import annotations

import math
from pathlib import Path

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.motion import (
    CANCELED,
    FAILED,
    POLICY,
    PREEMPTED,
    RELATIVE_MOVE,
    SUCCEEDED,
    TIMEOUT,
    MotionController,
    MotionSpec,
    wrap_angle,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(ROOT / "config/default.toml")
DT = 0.02


def make_state(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    *,
    mode: str = "walk",
    enabled: bool = True,
    fallen: bool = False,
    recovery: str | None = None,
    policy_episode: int = 0,
    policy_phase: str = "complete",
    policy_progress: float = 1.0,
    jump_episode: int = 0,
    jump_phase: str = "complete",
) -> dict:
    half = yaw / 2.0
    return {
        "enabled": enabled,
        "fallen": fallen,
        "mode": "walking",
        "base_pose": {
            "position": [x, y, 0.12],
            "quaternion_wxyz": [math.cos(half), 0.0, 0.0, math.sin(half)],
        },
        "policy": {
            "mode": mode,
            "recovery": recovery,
            "episode": policy_episode,
            "request": mode,
            "kind": "persistent",
            "phase": policy_phase,
            "progress": policy_progress,
        },
        "jump": {"episode": jump_episode, "phase": jump_phase},
    }


def make_controller() -> MotionController:
    return MotionController(SETTINGS.motion, SETTINGS.control)


class Robot:
    """A robot that tracks the commanded velocity exactly.

    The point of these tests is the motion state machine, not the walking policy, so
    the plant is perfect and every assertion is deterministic.
    """

    def __init__(self) -> None:
        self.x = self.y = self.yaw = 0.0
        self.now = 0.0
        self.commands: list[list[float]] = []
        self.policy_requests: list[str] = []
        self.jumps = 0

    def run(self, controller: MotionController, limit: float = 60.0, **state_kwargs) -> dict:
        """Tick until the motion ends or `limit` seconds of simulated time pass."""
        for _ in range(int(limit / DT)):
            state = make_state(self.x, self.y, self.yaw, **state_kwargs)
            tick = controller.tick(state, self.now)
            if tick.policy_request is not None:
                self.policy_requests.append(tick.policy_request)
            if tick.jump_request:
                self.jumps += 1
            if tick.command is not None:
                self.commands.append(tick.command)
                self._integrate(tick.command)
            self.now += DT
            if not controller.active:
                break
        return controller.status(self.now)

    def _integrate(self, command: list[float]) -> None:
        linear_x, linear_y, angular_z = command
        self.yaw = wrap_angle(self.yaw + angular_z * DT)
        self.x += (math.cos(self.yaw) * linear_x - math.sin(self.yaw) * linear_y) * DT
        self.y += (math.sin(self.yaw) * linear_x + math.cos(self.yaw) * linear_y) * DT


def test_idle_slot_reports_nothing_running() -> None:
    status = make_controller().status(0.0)
    assert status["state"] == "idle"
    assert status["id"] == 0


def test_forward_move_arrives_within_tolerance_and_reports_what_it_covered() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.5), 0.0)
    robot = Robot()

    status = robot.run(controller)

    assert status["outcome"] == SUCCEEDED
    assert status["state"] == "succeeded"
    assert status["remaining"] <= SETTINGS.motion.position_tolerance
    assert robot.x > 0.5 - SETTINGS.motion.position_tolerance
    assert robot.y == 0.0
    # Traveled is the path actually walked, not a projection onto the goal vector.
    assert status["traveled"] > 0.0
    assert abs(status["traveled"] - robot.x) < 1e-6
    assert status["progress"] == 1.0


def test_rotation_and_lateral_offsets_use_the_same_primitive() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, yaw=math.pi / 2), 0.0)
    robot = Robot()
    assert robot.run(controller)["outcome"] == SUCCEEDED
    assert abs(wrap_angle(robot.yaw - math.pi / 2)) <= SETTINGS.motion.yaw_tolerance
    assert math.hypot(robot.x, robot.y) < 1e-9

    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, y=0.25), 0.0)
    sidestep = Robot()
    assert sidestep.run(controller)["outcome"] == SUCCEEDED
    assert sidestep.y > 0.25 - SETTINGS.motion.position_tolerance
    assert abs(sidestep.x) < 1e-9


def test_offsets_are_measured_in_the_base_frame_at_acceptance() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.4), 0.0)
    robot = Robot()
    robot.yaw = math.pi / 2  # already facing +y when the goal is accepted

    assert robot.run(controller)["outcome"] == SUCCEEDED
    assert robot.y > 0.4 - SETTINGS.motion.position_tolerance
    assert abs(robot.x) < 1e-6


def test_commanded_speed_respects_the_limits_and_the_gait_floor() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=1.0, yaw=1.0), 0.0)
    robot = Robot()
    robot.run(controller)

    moving = [command for command in robot.commands if any(command)]
    assert moving
    for linear_x, linear_y, angular_z in moving:
        assert abs(linear_x) <= SETTINGS.control.max_linear_x + 1e-9
        assert abs(linear_y) <= SETTINGS.control.max_linear_y + 1e-9
        assert abs(angular_z) <= SETTINGS.control.max_angular_z + 1e-9
        assert math.hypot(linear_x, linear_y) <= SETTINGS.motion.max_linear_speed + 1e-9
    # Slowing towards the target must never fall under the speed the walking policy
    # needs to take a step at all, or the move would creep and time out.
    approaching = [math.hypot(x, y) for x, y, _ in moving if math.hypot(x, y) > 0.0]
    assert min(approaching) >= SETTINGS.motion.min_linear_speed - 1e-9


def test_a_goal_may_lower_the_speed_but_not_raise_it() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.5, max_linear_speed=0.1), 0.0)
    robot = Robot()
    robot.run(controller)
    assert max(math.hypot(x, y) for x, y, _ in robot.commands) <= 0.1 + 1e-9

    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.5, max_linear_speed=99.0), 0.0)
    greedy = Robot()
    greedy.run(controller)
    assert max(math.hypot(x, y) for x, y, _ in greedy.commands) <= (
        SETTINGS.motion.max_linear_speed + 1e-9
    )


def test_progress_never_runs_ahead_of_the_slower_axis() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.6, yaw=math.pi / 2), 0.0)
    robot = Robot()
    seen: list[float] = []

    for _ in range(3000):
        state = make_state(robot.x, robot.y, robot.yaw)
        tick = controller.tick(state, robot.now)
        if tick.command is not None:
            robot._integrate(tick.command)
        status = controller.status(robot.now)
        if status["phase"] in {"moving", "settling"}:
            seen.append(status["progress"])
        robot.now += DT
        if not controller.active:
            break

    assert seen
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert seen == sorted(seen)
    assert controller.status(robot.now)["outcome"] == SUCCEEDED


def test_a_move_waits_for_walking_before_it_drives() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.3), 0.0)
    robot = Robot()

    # Sitting: velocity commands are ignored by the policy, so the move has to ask
    # for walking first and must not pretend it is making progress meanwhile.
    robot.run(controller, limit=0.5, mode="sit")
    assert robot.policy_requests == ["walk"]
    assert controller.status(robot.now)["phase"] == "aligning"
    assert all(command == [0.0, 0.0, 0.0] for command in robot.commands)

    assert robot.run(controller)["outcome"] == SUCCEEDED
    # Asked once for the mode it saw, not once per tick.
    assert robot.policy_requests == ["walk"]


def test_a_move_lets_automatic_recovery_finish_instead_of_fighting_it() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.3), 0.0)
    robot = Robot()

    robot.run(controller, limit=0.5, mode="recover", recovery="recovering")

    assert robot.policy_requests == []
    assert controller.status(robot.now)["phase"] == "aligning"


def test_the_reference_gait_drives_without_a_policy_switch() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.3), 0.0)
    robot = Robot()

    assert robot.run(controller, mode="reference_gait")["outcome"] == SUCCEEDED
    assert robot.policy_requests == []


def test_a_move_fails_when_the_robot_falls_or_locomotion_is_disabled() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.5), 0.0)
    status = Robot().run(controller, fallen=True)
    assert status["outcome"] == FAILED
    assert status["message"] == "robot fell over"

    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.5), 0.0)
    status = Robot().run(controller, enabled=False)
    assert status["outcome"] == FAILED
    assert status["state"] == "aborted"


def test_a_move_that_cannot_arrive_times_out_and_reports_how_far_it_got() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=50.0, timeout=2.0), 0.0)

    status = Robot().run(controller, limit=10.0)

    assert status["outcome"] == TIMEOUT
    assert status["state"] == "aborted"
    assert status["traveled"] > 0.0
    assert status["remaining"] > 1.0
    assert 0.0 < status["progress"] < 1.0


def test_cancel_only_applies_to_the_motion_the_caller_started() -> None:
    controller = make_controller()
    started = controller.request(MotionSpec(kind=RELATIVE_MOVE, x=1.0), 0.0)

    assert controller.cancel(started["id"] + 1, 0.5) is False
    assert controller.cancel(started["id"], 0.5) is True
    assert controller.active is False

    status = controller.status(0.5)
    assert status["outcome"] == CANCELED
    assert status["state"] == "canceled"
    # A cancel on an already finished motion is not a second cancellation.
    assert controller.cancel(started["id"], 0.6) is False


def test_a_new_request_preempts_the_one_holding_the_slot() -> None:
    controller = make_controller()
    first = controller.request(MotionSpec(kind=RELATIVE_MOVE, x=1.0), 0.0)
    second = controller.request(MotionSpec(kind=RELATIVE_MOVE, x=-1.0), 0.5)

    assert second["id"] == first["id"] + 1
    assert controller.status(0.5)["id"] == second["id"]
    assert controller.status(0.5)["state"] == "running"


def test_abort_ends_the_active_motion_with_the_given_reason() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=1.0), 0.0)

    assert controller.abort(PREEMPTED, "operator velocity command", 0.4) is True
    assert controller.abort(PREEMPTED, "nothing left to abort", 0.5) is False

    status = controller.status(0.5)
    assert status["outcome"] == PREEMPTED
    assert status["message"] == "operator velocity command"


def test_a_policy_motion_requests_the_mode_then_waits_for_the_episode_to_finish() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="kick_left"), 0.0)

    # First tick asks for the mode against the episode counter it observed.
    tick = controller.tick(make_state(policy_episode=7), 0.0)
    assert tick.policy_request == "kick_left"
    assert controller.status(0.0)["phase"] == "starting"

    # The request has not landed yet: still starting, still no completion claimed.
    controller.tick(make_state(policy_episode=7), 0.05)
    assert controller.active is True

    running = make_state(policy_episode=8, policy_phase="running", policy_progress=0.4)
    controller.tick(running, 0.1)
    status = controller.status(0.1)
    assert status["phase"] == "running"
    assert status["progress"] == 0.4

    done = make_state(policy_episode=8, policy_phase="complete", policy_progress=1.0)
    controller.tick(done, 0.6)
    status = controller.status(0.6)
    assert status["outcome"] == SUCCEEDED
    assert status["duration"] == 0.6


def test_a_policy_motion_that_the_controller_could_not_finish_fails() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="roulade"), 0.0)
    controller.tick(make_state(policy_episode=2), 0.0)
    controller.tick(make_state(policy_episode=3, policy_phase="failed"), 0.2)

    status = controller.status(0.2)
    assert status["outcome"] == FAILED
    assert status["state"] == "aborted"


def test_a_policy_motion_notices_when_something_else_takes_the_controller() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="sit"), 0.0)
    controller.tick(make_state(policy_episode=1), 0.0)
    # Two episodes have opened since ours was requested, so ours is not the one running.
    controller.tick(make_state(policy_episode=3, policy_phase="running"), 0.2)

    assert controller.status(0.2)["outcome"] == PREEMPTED


def test_a_policy_motion_stops_waiting_when_the_policy_never_starts() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="ground_pick", timeout=1.0), 0.0)
    for step in range(200):
        controller.tick(make_state(policy_episode=4), step * DT)
        if not controller.active:
            break

    status = controller.status(4.0)
    assert status["outcome"] == TIMEOUT


def test_jump_watches_its_own_record_rather_than_the_policy() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="jump"), 0.0)

    tick = controller.tick(make_state(jump_episode=0), 0.0)
    assert tick.jump_request is True
    assert tick.policy_request is None

    controller.tick(make_state(jump_episode=1, jump_phase="running"), 0.1)
    assert controller.active is True

    controller.tick(make_state(jump_episode=1, jump_phase="complete"), 0.4)
    assert controller.status(0.4)["outcome"] == SUCCEEDED


def test_a_refused_jump_fails_instead_of_waiting_for_a_takeoff() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="jump"), 0.0)
    controller.tick(make_state(jump_episode=0), 0.0)
    controller.tick(make_state(jump_episode=1, jump_phase="failed"), 0.1)

    assert controller.status(0.1)["outcome"] == FAILED


def test_a_policy_motion_fails_immediately_when_locomotion_is_disabled() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=POLICY, motion="stand"), 0.0)

    controller.tick(make_state(enabled=False), 0.0)

    assert controller.status(0.0)["outcome"] == FAILED
    assert controller.status(0.0)["message"] == "locomotion is disabled"


def test_feedback_reports_the_distance_left_even_before_the_robot_can_drive() -> None:
    controller = make_controller()
    controller.request(MotionSpec(kind=RELATIVE_MOVE, x=0.6), 0.0)

    # Sitting: nothing has moved yet, so "remaining" must be the whole trip rather
    # than the zero that would read as "arrived".
    controller.tick(make_state(mode="sit"), 0.0)

    status = controller.status(0.0)
    assert status["phase"] == "aligning"
    assert status["remaining"] == 0.6
    assert status["progress"] == 0.0
