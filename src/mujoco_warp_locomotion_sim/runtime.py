from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from . import motion
from .backend import LocomotionBackend
from .config import Settings
from .mock_backend import MockBackend


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    timestamp: float
    jpeg: bytes


class SimulationRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: LocomotionBackend | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: str | None = None
        self._command_lock = threading.Lock()
        self._command = [0.0, 0.0, 0.0]
        self._command_time = 0.0
        self._enabled = True
        self._reset_requested = False
        self._policy_requested: str | None = None
        self._mouth_requested: bool | None = None
        self._jump_requested = False
        self._motion = motion.MotionController(settings.motion, settings.control)
        self._motion_command: list[float] | None = None
        self._state_condition = threading.Condition()
        self._state: dict[str, Any] | None = None
        self._state_sequence = 0
        self._frame_condition = threading.Condition()
        self._frame: FrameSnapshot | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._startup_error is None

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else self.settings.simulation.backend

    @property
    def device_name(self) -> str:
        return self._backend.device if self._backend else self.settings.simulation.device

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mjwarp-locomotion", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def command(self, linear_x: float, linear_y: float, angular_z: float) -> dict[str, float]:
        values = [
            self._clamp(linear_x, self.settings.control.max_linear_x),
            self._clamp(linear_y, self.settings.control.max_linear_y),
            self._clamp(angular_z, self.settings.control.max_angular_z),
        ]
        now = time.monotonic()
        if max(abs(value) for value in values) > 1e-9:
            # An operator reaching for the stick outranks a running motion. A zero
            # command is deliberately not an override: the bridge's watchdog posts one
            # after a teleoperation publisher goes quiet, and a motion that is already
            # in charge of stopping must not be cancelled by that.
            self._motion.abort(motion.PREEMPTED, "operator velocity command", now)
        with self._command_lock:
            self._command = values
            self._command_time = now
        return {"linear_x": values[0], "linear_y": values[1], "angular_z": values[2]}

    @staticmethod
    def _clamp(value: float, maximum: float) -> float:
        return max(-maximum, min(maximum, value))

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._motion.abort(motion.FAILED, "locomotion disabled", time.monotonic())
        with self._command_lock:
            self._enabled = enabled
            if not enabled:
                self._command = [0.0, 0.0, 0.0]
                self._command_time = time.monotonic()

    def request_reset(self) -> None:
        self._motion.abort(motion.FAILED, "simulation reset", time.monotonic())
        with self._command_lock:
            self._reset_requested = True
            self._command = [0.0, 0.0, 0.0]
            self._command_time = time.monotonic()

    def request_policy(self, mode: str) -> str:
        self._motion.abort(motion.PREEMPTED, "policy changed directly", time.monotonic())
        with self._command_lock:
            self._policy_requested = mode
        return mode

    def request_mouth(self, opened: bool) -> bool:
        with self._command_lock:
            self._mouth_requested = opened
        return opened

    def request_jump(self) -> None:
        self._motion.abort(motion.PREEMPTED, "jump requested directly", time.monotonic())
        with self._command_lock:
            self._jump_requested = True

    def start_motion(self, spec: motion.MotionSpec) -> dict[str, Any]:
        """Take the motion slot, preempting whatever held it."""
        return self._motion.request(spec, time.monotonic())

    def cancel_motion(self, motion_id: int) -> bool:
        return self._motion.cancel(motion_id, time.monotonic())

    def motion_status(self) -> dict[str, Any]:
        return self._motion.status(time.monotonic())

    def latest_state(self) -> dict[str, Any] | None:
        with self._state_condition:
            return dict(self._state) if self._state else None

    def wait_for_state(self, after_sequence: int, timeout: float = 5.0) -> dict[str, Any] | None:
        with self._state_condition:
            self._state_condition.wait_for(
                lambda: self._state_sequence > after_sequence or self._stop.is_set(),
                timeout=timeout,
            )
            return dict(self._state) if self._state else None

    def latest_frame(self) -> FrameSnapshot | None:
        with self._frame_condition:
            return self._frame

    def wait_for_frame(self, after_sequence: int, timeout: float = 5.0) -> FrameSnapshot | None:
        with self._frame_condition:
            self._frame_condition.wait_for(
                lambda: (
                    (self._frame is not None and self._frame.sequence > after_sequence)
                    or self._stop.is_set()
                ),
                timeout=timeout,
            )
            return self._frame

    def _create_backend(self) -> LocomotionBackend:
        if self.settings.simulation.backend == "mock":
            return MockBackend(self.settings)
        from .mujoco_warp_backend import MujocoWarpBackend

        return MujocoWarpBackend(self.settings)

    def _run(self) -> None:
        try:
            self._backend = self._create_backend()
            self._backend.initialize()
            self._ready.set()
            next_step = time.monotonic()
            next_state = next_step
            next_frame = next_step
            while not self._stop.is_set():
                now = time.monotonic()
                with self._command_lock:
                    command = self._command.copy()
                    stale = now - self._command_time > self.settings.simulation.command_timeout
                    enabled = self._enabled
                    reset = self._reset_requested
                    self._reset_requested = False
                    policy = self._policy_requested
                    self._policy_requested = None
                    mouth = self._mouth_requested
                    self._mouth_requested = None
                    jump = self._jump_requested
                    self._jump_requested = False
                    motion_command = self._motion_command
                if motion_command is not None:
                    command = list(motion_command)
                if stale or not enabled:
                    # Zeroing here outranks the motion override, which is the safe
                    # ordering: disabling aborts the motion outright, and a motion that
                    # is still driving refreshes the timestamp so `stale` cannot fire.
                    command = [0.0, 0.0, 0.0]
                if reset:
                    self._backend.reset()
                    next_state = now
                    next_frame = now
                if policy is not None:
                    self._apply_policy(policy)
                if mouth is not None:
                    self._backend.set_mouth(mouth)
                if jump:
                    self._backend.jump()
                self._backend.set_enabled(enabled)
                self._backend.set_command(*command)
                self._backend.step()

                if now >= next_state:
                    self._publish_state(stale, now)
                    next_state = now + 1.0 / self.settings.simulation.state_hz
                if now >= next_frame:
                    self._publish_frame()
                    next_frame = now + 1.0 / self.settings.simulation.render_hz
                if self.settings.simulation.realtime:
                    next_step += self.settings.simulation.dt
                    time.sleep(max(0.0, next_step - time.monotonic()))
                else:
                    time.sleep(0)
        except Exception as error:
            self._startup_error = f"{type(error).__name__}: {error}"
            self._ready.set()
        finally:
            if self._backend:
                self._backend.close()
            with self._state_condition:
                self._state_condition.notify_all()
            with self._frame_condition:
                self._frame_condition.notify_all()

    def _apply_policy(self, mode: str) -> bool:
        """Switch modes without letting a rejected name take the physics loop down."""
        assert self._backend is not None
        try:
            self._backend.set_policy(mode)
        except ValueError:
            return False
        return True

    def _drive_motion(self, state: dict[str, Any], now: float) -> None:
        """Give the active motion one turn at the robot.

        Runs at `state_hz`, on the state the loop just published, so closed-loop
        control costs no extra backend call. The command is written back through the
        same locked slot teleoperation uses, with the timestamp refreshed so the
        staleness watchdog does not zero a motion that is still driving.
        """
        assert self._backend is not None
        tick = self._motion.tick(state, now)
        if tick.policy_request is not None and not self._apply_policy(tick.policy_request):
            self._motion.abort(
                motion.FAILED, f"policy mode {tick.policy_request!r} is unavailable", now
            )
            return
        if tick.jump_request:
            self._backend.jump()
        with self._command_lock:
            self._motion_command = tick.command
            if tick.command is not None:
                self._command_time = now

    def _publish_state(self, command_stale: bool, now: float) -> None:
        assert self._backend is not None
        state = self._backend.state()
        self._drive_motion(state, now)
        state["motion"] = self._motion.status(now)
        with self._state_condition:
            self._state_sequence += 1
            state.update(
                {
                    "sequence": self._state_sequence,
                    "timestamp": time.time(),
                    "command_stale": command_stale,
                }
            )
            self._state = state
            self._state_condition.notify_all()

    def _publish_frame(self) -> None:
        assert self._backend is not None
        jpeg = self._backend.jpeg_frame()
        with self._frame_condition:
            sequence = 1 if self._frame is None else self._frame.sequence + 1
            self._frame = FrameSnapshot(sequence, time.time(), jpeg)
            self._frame_condition.notify_all()
