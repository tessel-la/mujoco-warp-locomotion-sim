"""Long-running motion primitives with an observable execution lifecycle.

A velocity command or a policy switch is instantaneous: it is applied and it is done.
A *motion* is not. "Move forward half a metre" and "play the left kick" both occupy
the robot for a while, can be preempted, can time out, and finish at a moment only the
simulation knows. This module owns that distinction so a ROS action server can stay
active for exactly as long as the robot is busy and report the real outcome.

It deliberately does not implement any controller. A relative move computes a velocity
command and hands it to the same path teleoperation uses, so the trained walking policy
does the walking; a policy motion just asks for the mode and watches the episode record
the policy controller already keeps.

Only `tick` runs on the simulation thread. Everything else is called from API threads,
so every entry point takes the lock.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from .config import ControlSettings, MotionSettings

RELATIVE_MOVE = "relative_move"
POLICY = "policy"
KINDS = (RELATIVE_MOVE, POLICY)

# Terminal outcomes. A ROS action has no distinct "timed out" terminal state -- a
# timeout is just an abort -- so the outcome is what lets a caller tell a timeout from
# a failure, and an operator override from a cancelled goal.
SUCCEEDED = "succeeded"
TIMEOUT = "timeout"
CANCELED = "canceled"
PREEMPTED = "preempted"
FAILED = "failed"

# Goal states, matching the three terminal states a ROS action can report.
RUNNING = "running"
IDLE = "idle"
_GOAL_STATES = {
    SUCCEEDED: "succeeded",
    CANCELED: "canceled",
    TIMEOUT: "aborted",
    PREEMPTED: "aborted",
    FAILED: "aborted",
}

# Controller modes that honour a velocity command. Every other policy mode ignores it
# (see `PolicyController._command_observation`), so a relative move has to reach one of
# these before it can drive anywhere.
VELOCITY_MODES = frozenset({"walk", "reference_gait"})

# The non-RL jump primitive keeps its lifecycle record beside the policy's.
JUMP = "jump"


@dataclass(frozen=True)
class MotionSpec:
    """One requested motion. Zero means "use the configured default"."""

    kind: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    max_linear_speed: float = 0.0
    max_angular_speed: float = 0.0
    position_tolerance: float = 0.0
    yaw_tolerance: float = 0.0
    timeout: float = 0.0
    motion: str = ""


@dataclass(frozen=True)
class MotionTick:
    """What the simulation loop should do on behalf of the active motion."""

    command: list[float] | None = None
    policy_request: str | None = None
    jump_request: bool = False


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(wxyz: Any) -> float:
    w, x, y, z = (float(value) for value in wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class _Motion:
    """Shared lifecycle bookkeeping for one requested motion."""

    motion_id: int
    kind: str
    spec: MotionSpec
    settings: MotionSettings
    control: ControlSettings
    started_at: float
    timeout: float
    state: str = RUNNING
    outcome: str = ""
    message: str = ""
    phase: str = "starting"
    progress: float = 0.0
    traveled: float = 0.0
    remaining: float = 0.0
    yaw_remaining: float = 0.0
    controller: str = ""
    duration: float = 0.0

    def finish(self, outcome: str, message: str, now: float) -> None:
        if self.state != RUNNING:
            return
        self.state = _GOAL_STATES[outcome]
        self.outcome = outcome
        self.message = message
        self.phase = "done"
        self.duration = now - self.started_at
        if outcome == SUCCEEDED:
            self.progress = 1.0

    def elapsed(self, now: float) -> float:
        return self.duration if self.state != RUNNING else now - self.started_at

    def status(self, now: float) -> dict[str, Any]:
        return {
            "id": self.motion_id,
            "kind": self.kind,
            "state": self.state,
            "outcome": self.outcome,
            "message": self.message,
            "elapsed": self.elapsed(now),
            "phase": self.phase,
            "progress": self.progress,
            "traveled": self.traveled,
            "remaining": self.remaining,
            "yaw_remaining": self.yaw_remaining,
            "controller": self.controller,
            "duration": self.duration,
        }

    def tick(self, state: dict[str, Any], now: float) -> MotionTick:
        raise NotImplementedError


@dataclass
class _RelativeMove(_Motion):
    """Drive to a pose offset expressed in the base frame at goal acceptance.

    The robot is holonomic on the ground, so forward, lateral, and turn-in-place are
    the same controller with the other fields left at zero. There is no commandable
    vertical axis on a biped, and none is invented here.
    """

    _origin: tuple[float, float, float] | None = None
    _previous: tuple[float, float] | None = None
    _initial_distance: float = 0.0
    _initial_yaw_error: float = 0.0
    _settling_since: float | None = None
    _walk_requested_for: str | None = None

    def tick(self, state: dict[str, Any], now: float) -> MotionTick:
        position = state["base_pose"]["position"]
        current = (float(position[0]), float(position[1]))
        yaw = yaw_from_quaternion(state["base_pose"]["quaternion_wxyz"])
        mode = str(state.get("policy", {}).get("mode", "walk"))
        self.controller = mode

        if not state.get("enabled", True):
            self.finish(FAILED, "locomotion is disabled", now)
            return MotionTick(command=[0.0, 0.0, 0.0])
        if state.get("fallen"):
            self.finish(FAILED, "robot fell over", now)
            return MotionTick(command=[0.0, 0.0, 0.0])
        if now - self.started_at > self.timeout:
            self.finish(TIMEOUT, f"did not arrive within {self.timeout:.1f}s", now)
            return MotionTick(command=[0.0, 0.0, 0.0])

        if self._origin is None:
            self._origin = (current[0], current[1], yaw)
            self._previous = current
            self._initial_distance = math.hypot(self.spec.x, self.spec.y)
            self._initial_yaw_error = abs(wrap_angle(self.spec.yaw))
        assert self._previous is not None
        self.traveled += math.hypot(current[0] - self._previous[0], current[1] - self._previous[1])
        self._previous = current

        forward, lateral, yaw_error = self._errors(current, yaw)
        self.remaining = math.hypot(forward, lateral)
        self.yaw_remaining = abs(yaw_error)
        self.progress = self._progress()

        # A velocity command only reaches the legs while the controller is walking.
        if mode not in VELOCITY_MODES:
            self.phase = "aligning"
            if state.get("policy", {}).get("recovery") is not None:
                # Automatic recovery is putting the robot back on its feet and hands
                # control to walking itself. Asking for walk now would fight it.
                return MotionTick(command=[0.0, 0.0, 0.0])
            if self._walk_requested_for == mode:
                return MotionTick(command=[0.0, 0.0, 0.0])
            # Edge-triggered: asking once per mode change gets the controller back to
            # walking without opening a fresh policy episode on every tick.
            self._walk_requested_for = mode
            return MotionTick(command=[0.0, 0.0, 0.0], policy_request="walk")

        position_tolerance = self.spec.position_tolerance or self.settings.position_tolerance
        yaw_tolerance = self.spec.yaw_tolerance or self.settings.yaw_tolerance
        if self.remaining <= position_tolerance and self.yaw_remaining <= yaw_tolerance:
            return self._settle(now)

        self._settling_since = None
        self.phase = "moving"
        return MotionTick(
            command=self._velocity(forward, lateral, yaw_error, position_tolerance, yaw_tolerance)
        )

    def _errors(self, current: tuple[float, float], yaw: float) -> tuple[float, float, float]:
        assert self._origin is not None
        origin_x, origin_y, origin_yaw = self._origin
        cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
        target_x = origin_x + cosine * self.spec.x - sine * self.spec.y
        target_y = origin_y + sine * self.spec.x + cosine * self.spec.y
        delta_x, delta_y = target_x - current[0], target_y - current[1]
        forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
        lateral = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
        return forward, lateral, wrap_angle(origin_yaw + self.spec.yaw - yaw)

    def _progress(self) -> float:
        """The laggier of the two axes, so progress never runs ahead of the motion."""
        fractions = []
        if self._initial_distance > 0.0:
            fractions.append(1.0 - self.remaining / self._initial_distance)
        if self._initial_yaw_error > 0.0:
            fractions.append(1.0 - self.yaw_remaining / self._initial_yaw_error)
        if not fractions:
            return 1.0
        return min(1.0, max(0.0, min(fractions)))

    def _velocity(
        self,
        forward: float,
        lateral: float,
        yaw_error: float,
        position_tolerance: float,
        yaw_tolerance: float,
    ) -> list[float]:
        max_linear = min(
            self.spec.max_linear_speed or self.settings.max_linear_speed,
            self.settings.max_linear_speed,
        )
        max_angular = min(
            self.spec.max_angular_speed or self.settings.max_angular_speed,
            self.settings.max_angular_speed,
        )
        linear_x = linear_y = 0.0
        if self.remaining > position_tolerance:
            # Slow down as the target approaches, but never below the speed the
            # walking policy needs to take a step at all.
            speed = min(max_linear, self.settings.linear_gain * self.remaining)
            speed = max(speed, min(self.settings.min_linear_speed, max_linear))
            scale = speed / self.remaining
            linear_x = _clamp(forward * scale, self.control.max_linear_x)
            linear_y = _clamp(lateral * scale, self.control.max_linear_y)
        angular_z = 0.0
        if self.yaw_remaining > yaw_tolerance:
            rate = min(max_angular, self.settings.angular_gain * self.yaw_remaining)
            rate = max(rate, min(self.settings.min_angular_speed, max_angular))
            angular_z = _clamp(math.copysign(rate, yaw_error), self.control.max_angular_z)
        return [linear_x, linear_y, angular_z]

    def _settle(self, now: float) -> MotionTick:
        """Stop, let the gait come to rest, then report where it actually ended up.

        Settling is deliberately terminal. The walking policy keeps taking small steps
        for a moment after the command goes to zero, so re-opening the goal when it
        drifts back outside tolerance would never converge. The goal succeeded because
        it reached tolerance; the result carries the final error either way.
        """
        self.phase = "settling"
        if self._settling_since is None:
            self._settling_since = now
        elif now - self._settling_since >= self.settings.settle_time:
            self.finish(SUCCEEDED, "reached the requested offset", now)
        return MotionTick(command=[0.0, 0.0, 0.0])


@dataclass
class _PolicyMotion(_Motion):
    """Trigger an existing control policy and stay active until it completes.

    Completion is read from the episode record the policy controller keeps rather
    than inferred from mode strings, so this cannot miss a motion that started and
    finished between two observations.
    """

    _requested: bool = False
    _episode: int | None = None

    @property
    def _record_key(self) -> str:
        return JUMP if self.spec.motion == JUMP else "policy"

    def tick(self, state: dict[str, Any], now: float) -> MotionTick:
        record = state.get(self._record_key, {})
        self.controller = str(state.get("policy", {}).get("mode", state.get("mode", "")))

        if not state.get("enabled", True):
            self.finish(FAILED, "locomotion is disabled", now)
            return MotionTick()
        if now - self.started_at > self.timeout:
            self.finish(TIMEOUT, f"did not complete within {self.timeout:.1f}s", now)
            return MotionTick()

        if not self._requested:
            self._episode = int(record.get("episode", 0))
            self._requested = True
            self.phase = "starting"
            if self.spec.motion == JUMP:
                return MotionTick(jump_request=True)
            return MotionTick(policy_request=self.spec.motion)

        episode = int(record.get("episode", 0))
        assert self._episode is not None
        if episode <= self._episode:
            self.phase = "starting"
            return MotionTick()
        if episode > self._episode + 1:
            self.finish(PREEMPTED, "another request took over the controller", now)
            return MotionTick()

        phase = str(record.get("phase", RUNNING))
        self.progress = float(record.get("progress", 0.0 if phase == RUNNING else 1.0))
        if phase == RUNNING:
            self.phase = RUNNING
            return MotionTick()
        if phase == "complete":
            self.finish(SUCCEEDED, f"{self.spec.motion} completed", now)
        else:
            self.finish(FAILED, f"{self.spec.motion} did not complete", now)
        return MotionTick()


def idle_status() -> dict[str, Any]:
    """What the motion slot reports before anything has ever been requested."""
    return {
        "id": 0,
        "kind": "",
        "state": IDLE,
        "outcome": "",
        "message": "",
        "elapsed": 0.0,
        "phase": "",
        "progress": 0.0,
        "traveled": 0.0,
        "remaining": 0.0,
        "yaw_remaining": 0.0,
        "controller": "",
        "duration": 0.0,
    }


class MotionController:
    """The robot's single motion slot.

    A robot can only do one thing at a time, so there is one active motion and one
    monotonically increasing id. That id is how a caller detects it was preempted:
    if the slot no longer holds the id it started, someone else took the robot.
    """

    def __init__(self, settings: MotionSettings, control: ControlSettings) -> None:
        self._settings = settings
        self._control = control
        self._lock = threading.Lock()
        self._next_id = 0
        self._active: _Motion | None = None
        self._last: _Motion | None = None

    def request(self, spec: MotionSpec, now: float) -> dict[str, Any]:
        with self._lock:
            self._terminate(PREEMPTED, "replaced by a newer motion request", now)
            self._next_id += 1
            motion = self._build(spec, now)
            self._active = motion
            self._last = motion
            return motion.status(now)

    def _build(self, spec: MotionSpec, now: float) -> _Motion:
        if spec.kind == RELATIVE_MOVE:
            timeout = spec.timeout or self._settings.move_timeout
            return _RelativeMove(
                motion_id=self._next_id,
                kind=spec.kind,
                spec=spec,
                settings=self._settings,
                control=self._control,
                started_at=now,
                timeout=timeout,
            )
        return _PolicyMotion(
            motion_id=self._next_id,
            kind=spec.kind,
            spec=spec,
            settings=self._settings,
            control=self._control,
            started_at=now,
            timeout=spec.timeout or self._settings.policy_timeout,
        )

    def cancel(self, motion_id: int, now: float) -> bool:
        with self._lock:
            if self._active is None or self._active.motion_id != motion_id:
                return False
            self._terminate(CANCELED, "canceled by the caller", now)
            return True

    def abort(self, outcome: str, message: str, now: float) -> bool:
        """End the active motion because something else took the robot."""
        with self._lock:
            if self._active is None:
                return False
            self._terminate(outcome, message, now)
            return True

    def _terminate(self, outcome: str, message: str, now: float) -> None:
        if self._active is None:
            return
        self._active.finish(outcome, message, now)
        self._active = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    def status(self, now: float) -> dict[str, Any]:
        with self._lock:
            if self._last is None:
                return idle_status()
            return self._last.status(now)

    def tick(self, state: dict[str, Any], now: float) -> MotionTick:
        """Advance the active motion. Simulation thread only."""
        with self._lock:
            motion = self._active
            if motion is None:
                return MotionTick()
            result = motion.tick(state, now)
            if motion.state != RUNNING:
                self._active = None
            return result
