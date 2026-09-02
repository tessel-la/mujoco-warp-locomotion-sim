from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import PolicySettings

DEFAULT_POSE = np.asarray(
    [
        0.0,
        -0.08726646259971647,
        -0.457924,
        -0.004940,
        0.452984,
        0.3490658503988659,
        0.3490658503988659,
        0.0,
        0.0,
        0.0,
        0.08726646259971647,
        0.457924,
        0.004940,
        -0.452984,
    ],
    dtype=np.float32,
)

OBSERVATION_SIZE = 61
ACTION_SIZE = 14
COMMAND_SIZE = 13

SessionFactory = Callable[[Path], Any]


def _default_session_factory(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("onnxruntime is required when policy.enabled=true") from error
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class PolicyController:
    """Runtime adapter for Pollen's shared 61-observation/14-action policies."""

    def __init__(
        self,
        settings: PolicySettings,
        *,
        joint_qpos: list[int],
        joint_dof: list[int],
        gyro_slice: slice,
        body_id: int,
        session_factory: SessionFactory = _default_session_factory,
    ) -> None:
        self.settings = settings
        self.joint_qpos = joint_qpos
        self.joint_dof = joint_dof
        self.gyro_slice = gyro_slice
        self.body_id = body_id
        self._session_factory = session_factory
        self.sessions: dict[str, Any] = {}
        self._io_names: dict[str, tuple[str, str]] = {}
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.last_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        self.mode = "walk"
        self._sit_flag = 0.0
        self._mode_steps = 0
        self._phase = 0.0
        self._return_to_walk_steps = 0
        self._fall_debounce = 0
        self._recovery_state: str | None = None
        self._recovery_steps = 0
        self._upright_steps = 0

    def initialize(self) -> None:
        for name, path in self.settings.policy_paths().items():
            if path is None:
                continue
            session = self._session_factory(path)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(f"policy {name!r} must have exactly one input and one output")
            input_width = inputs[0].shape[-1]
            output_width = outputs[0].shape[-1]
            if input_width != OBSERVATION_SIZE or output_width != ACTION_SIZE:
                raise ValueError(
                    f"policy {name!r} has shape {inputs[0].shape} -> {outputs[0].shape}; "
                    f"expected [1,{OBSERVATION_SIZE}] -> [1,{ACTION_SIZE}]"
                )
            result = session.run(
                [outputs[0].name],
                {inputs[0].name: np.zeros((1, OBSERVATION_SIZE), dtype=np.float32)},
            )[0]
            if np.asarray(result).shape != (1, ACTION_SIZE) or not np.isfinite(result).all():
                raise ValueError(f"policy {name!r} failed warm-up inference")
            self.sessions[name] = session
            self._io_names[name] = (inputs[0].name, outputs[0].name)
        if "walk" not in self.sessions:
            raise ValueError("policy.enabled=true requires a valid walking policy")
        self.reset()

    def reset(self) -> None:
        self.last_action.fill(0)
        self.last_observation.fill(0)
        self.mode = "walk"
        self._sit_flag = 0.0
        self._mode_steps = 0
        self._phase = 0.0
        self._return_to_walk_steps = 0
        self._fall_debounce = 0
        self._recovery_state = None
        self._recovery_steps = 0
        self._upright_steps = 0

    def set_command(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        self.command[:] = (linear_x, linear_y, angular_z)

    def request_mode(self, requested: str) -> str:
        aliases = {"walking": "walk", "ground-pick": "ground_pick"}
        requested = aliases.get(requested.strip().lower(), requested.strip().lower())
        required = {
            "walk": "walk",
            "stand": "sitstand",
            "sit": "sitstand",
            "ground_pick": "ground_pick",
            "kick_left": "kick_left",
            "kick_right": "kick_right",
            "roulade": "roulade",
            "recover": "stand",
        }
        if requested not in required:
            raise ValueError(f"unknown policy mode {requested!r}")
        session_name = required[requested]
        if session_name not in self.sessions:
            raise ValueError(f"policy mode {requested!r} is unavailable")

        self._mode_steps = 0
        self._phase = 0.0
        self._return_to_walk_steps = 0
        self._recovery_state = None
        if requested == "walk" and self.mode == "sitstand" and self._sit_flag > 0:
            self._sit_flag = 0.0
            self._return_to_walk_steps = 100
        elif requested == "stand":
            self.mode = "sitstand"
            self._sit_flag = 0.0
        elif requested == "sit":
            self.mode = "sitstand"
            self._sit_flag = 0.0
            self._mode_steps = -40
        elif requested == "recover":
            self.mode = "recover"
            self.last_action.fill(0)
        else:
            self.mode = requested
            if requested == "walk":
                self.last_action.fill(0)
        return self.mode

    @staticmethod
    def _projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
        w = quaternion[0]
        xyz = quaternion[1:]
        gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        cross = np.cross(xyz, gravity) * 2.0
        return gravity - w * cross + np.cross(xyz, cross)

    def _command_observation(self) -> np.ndarray:
        command = np.zeros(COMMAND_SIZE, dtype=np.float32)
        if self.mode == "walk":
            command[:3] = self.command
        elif self.mode == "sitstand":
            command[0] = self._sit_flag
        elif self.mode == "ground_pick":
            angle = 2.0 * math.pi * self._phase
            command[0] = math.cos(angle)
            command[1] = math.sin(angle)
        return command

    def build_observation(self, data: Any) -> np.ndarray:
        gravity = self._projected_gravity(np.asarray(data.xquat[self.body_id]))
        joint_positions = np.asarray(
            [data.qpos[index] for index in self.joint_qpos], dtype=np.float32
        )
        joint_velocities = np.asarray(
            [data.qvel[index] for index in self.joint_dof], dtype=np.float32
        )
        observation = np.concatenate(
            (
                np.asarray(data.sensordata[self.gyro_slice], dtype=np.float32),
                gravity,
                joint_positions - DEFAULT_POSE,
                joint_velocities,
                self.last_action,
                self._command_observation(),
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise RuntimeError("policy observation is non-finite or has the wrong shape")
        self.last_observation = observation
        return observation

    def _session_name(self) -> str:
        return "stand" if self.mode == "recover" else self.mode

    def targets(self, data: Any) -> np.ndarray | None:
        if self._recovery_state == "fallen":
            return None
        name = self._session_name()
        session = self.sessions.get(name)
        if session is None:
            raise RuntimeError(f"no loaded session for active policy mode {self.mode!r}")
        input_name, output_name = self._io_names[name]
        observation = self.build_observation(data)
        action = np.asarray(
            session.run([output_name], {input_name: observation.reshape(1, -1)})[0],
            dtype=np.float32,
        ).reshape(-1)
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
            raise RuntimeError(f"policy {name!r} returned an invalid action")
        self.last_action = action
        return DEFAULT_POSE + action * self.settings.action_scale

    def after_step(self, data: Any) -> bool:
        """Advance one policy tick. Return True when the simulation should reset."""
        self._mode_steps += 1
        if self.mode == "sitstand" and self._mode_steps == 0:
            self._sit_flag = 1.0
        if self._return_to_walk_steps > 0:
            self._return_to_walk_steps -= 1
            if self._return_to_walk_steps == 0:
                self.mode = "walk"
                self.last_action.fill(0)
        elif self.mode == "ground_pick":
            self._phase += 0.02 / 4.0
            if self._phase >= 0.7:
                self.mode = "walk"
        elif self.mode in {"kick_left", "kick_right"} and self._mode_steps >= 25:
            self.mode = "walk"
        elif self.mode == "roulade" and self._mode_steps >= 150:
            self.mode = "walk"
            self.last_action.fill(0)

        gravity_z = float(self._projected_gravity(np.asarray(data.xquat[self.body_id]))[2])
        height = float(data.xpos[self.body_id, 2])
        dead = not math.isfinite(gravity_z) or not math.isfinite(height)
        fallen = gravity_z > -0.5 or height < 0.02
        if dead:
            return True
        if self._recovery_state == "fallen":
            self._recovery_steps += 1
            if self._recovery_steps >= 15:
                self._recovery_state = "recovering"
                self.mode = "recover"
                self._recovery_steps = 0
                self.last_action.fill(0)
            return False
        if self._recovery_state == "recovering" or self.mode == "recover":
            self._recovery_steps += 1
            self._upright_steps = self._upright_steps + 1 if gravity_z < -0.85 else 0
            if self._upright_steps >= 50:
                self._recovery_state = None
                self.mode = "walk"
                self.last_action.fill(0)
            return self._recovery_steps >= 300
        if not self.settings.automatic_recovery or self.mode != "walk":
            self._fall_debounce = 0
            return False
        if fallen:
            self._fall_debounce += 1
            if self._fall_debounce >= 5 and "stand" in self.sessions:
                self._recovery_state = "fallen"
                self._recovery_steps = 0
                self._fall_debounce = 0
            return False
        self._fall_debounce = 0
        return False

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "active": self._session_name(),
            "mode": self.mode,
            "available": sorted(self.sessions),
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "recovery": self._recovery_state,
        }
