from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image

from .config import Settings
from .policy_controller import PolicyController

MICRODUCK_STAND_TARGETS = np.asarray(
    [
        0.0,
        -0.0872664626,
        -0.457924,
        -0.004940,
        0.452984,
        0.3490658504,
        0.3490658504,
        0.0,
        0.0,
        0.0,
        0.0872664626,
        0.457924,
        0.004940,
        -0.452984,
    ],
    dtype=np.float32,
)

MOUTH_CLOSED_TARGET = 0.0
MOUTH_OPEN_TARGET = -0.48
JUMP_TAKEOFF_VELOCITY = 1.35


class MujocoWarpBackend:
    """Single-world MJWarp runtime with RL and reference-gait controllers."""

    name = "mujoco_warp"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.simulation.device
        self.enabled = True
        self.command = [0.0, 0.0, 0.0]
        self.phase = 0.0
        self._control_accumulator = 0.0
        self._current_targets = MICRODUCK_STAND_TARGETS.copy()
        self._mujoco: Any = None
        self._mjw: Any = None
        self._wp: Any = None
        self._model: Any = None
        self._data: Any = None
        self._host_data: Any = None
        self._renderer: Any = None
        self._graph: Any = None
        self._body_id = -1
        self._joint_qpos: list[int] = []
        self._joint_dof: list[int] = []
        self._mouth_qpos = -1
        self._mouth_dof = -1
        self._mouth_target = MOUTH_CLOSED_TARGET
        self._mouth_open = False
        self._jump_requested = False
        self._sensor_slices: dict[str, slice] = {}
        self._policy: PolicyController | None = None

    def initialize(self) -> None:
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as error:  # pragma: no cover - needs optional runtime
            raise RuntimeError(
                "install the 'mjwarp' extra to use the MuJoCo Warp backend"
            ) from error

        self._mujoco, self._mjw, self._wp = mujoco, mjw, wp
        wp.init()
        wp.set_device(self.device)
        self._model = mujoco.MjModel.from_xml_path(str(self.settings.simulation.model))
        self._model.opt.timestep = self.settings.simulation.dt
        self._host_data = mujoco.MjData(self._model)
        self._body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, self.settings.robot.base_body)
        self._joint_qpos = []
        self._joint_dof = []
        for name in self.settings.robot.joint_names:
            joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos.append(int(self._model.jnt_qposadr[joint_id]))
            self._joint_dof.append(int(self._model.jnt_dofadr[joint_id]))
        mouth_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "mouth")
        self._mouth_qpos = int(self._model.jnt_qposadr[mouth_id])
        self._mouth_dof = int(self._model.jnt_dofadr[mouth_id])
        for name in ("imu_ang_vel", "imu_accel", "orientation"):
            sensor_id = self._name_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            start = int(self._model.sensor_adr[sensor_id])
            self._sensor_slices[name] = slice(start, start + int(self._model.sensor_dim[sensor_id]))
        if self.settings.policy.enabled:
            self._policy = PolicyController(
                self.settings.policy,
                joint_qpos=self._joint_qpos,
                joint_dof=self._joint_dof,
                gyro_slice=self._sensor_slices["imu_ang_vel"],
                body_id=self._body_id,
            )
            self._policy.initialize()
        self._renderer = mujoco.Renderer(
            self._model,
            height=self.settings.camera.height,
            width=self.settings.camera.width,
        )
        self.reset()

    def _name_id(self, object_type: Any, name: str) -> int:
        object_id = int(self._mujoco.mj_name2id(self._model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo model has no object named {name!r}")
        return object_id

    def _make_device_data(self) -> None:
        self._data = self._mjw.put_data(
            self._model,
            self._host_data,
            nconmax=self.settings.simulation.nconmax,
            njmax=self.settings.simulation.njmax,
        )
        self._graph = None
        if self._wp.get_device().is_cuda:
            with self._wp.ScopedCapture() as capture:
                self._mjw.step(self._mjw_model, self._data)
            self._graph = capture.graph

    def reset(self) -> None:
        key_id = self._name_id(self._mujoco.mjtObj.mjOBJ_KEY, self.settings.robot.keyframe)
        self._mujoco.mj_resetDataKeyframe(self._model, self._host_data, key_id)
        self._mujoco.mj_forward(self._model, self._host_data)
        self.phase = 0.0
        self._control_accumulator = 1.0 / self.settings.control.control_hz
        self._current_targets = MICRODUCK_STAND_TARGETS.copy()
        self.command = [0.0, 0.0, 0.0]
        self._mouth_target = MOUTH_CLOSED_TARGET
        self._mouth_open = False
        self._jump_requested = False
        if self._policy is not None:
            self._policy.reset()
        self._mjw_model = self._mjw.put_model(self._model)
        self._make_device_data()

    def set_command(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        self.command = [linear_x, linear_y, angular_z]
        if self._policy is not None:
            self._policy.set_command(linear_x, linear_y, angular_z)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.command = [0.0, 0.0, 0.0]

    def set_policy(self, mode: str) -> str:
        if self._policy is None:
            raise ValueError("RL policies are disabled")
        return self._policy.request_mode(mode)

    def set_mouth(self, opened: bool) -> None:
        self._mouth_open = opened
        self._mouth_target = MOUTH_OPEN_TARGET if opened else MOUTH_CLOSED_TARGET

    def jump(self) -> None:
        if self.enabled:
            self._jump_requested = True

    def _targets(self) -> np.ndarray:
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        activity = min(1.0, math.hypot(vx, vy) / 0.08 + abs(wz) / 0.3)
        speed_scale = activity * min(
            1.0,
            max(0.25, abs(vx) / max(self.settings.control.max_linear_x, 1e-6)),
        )
        left_wave = math.sin(self.phase)
        right_wave = -left_wave

        def leg(side: float, wave: float) -> list[float]:
            lift = max(0.0, wave) * speed_scale
            return [
                side * wz * 0.12,
                -side * 0.0872664626
                + side * 0.05 * math.cos(self.phase) * activity
                - vy * 0.20,
                -side
                * (0.457924 + self.settings.control.hip_swing * wave * speed_scale),
                side * (-0.004940 + self.settings.control.knee_lift * lift),
                side
                * (
                    0.452984
                    - self.settings.control.ankle_swing * wave * speed_scale
                    - 0.20 * lift
                ),
            ]

        return np.asarray(
            leg(1.0, left_wave)
            + [0.3490658504, 0.3490658504, 0.25 * wz, 0.04 * left_wave * activity]
            + leg(-1.0, right_wave),
            dtype=np.float32,
        )

    def step(self) -> None:
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        activity = min(1.0, math.hypot(vx, vy) / 0.08 + abs(wz) / 0.3)
        self._control_accumulator += self.settings.simulation.dt
        control_period = 1.0 / self.settings.control.control_hz
        policy_tick = self._control_accumulator >= control_period
        if policy_tick:
            elapsed = self._control_accumulator
            self._control_accumulator %= control_period
            if self._policy is not None:
                targets = self._policy.targets(self._host_data)
                if targets is not None:
                    self._current_targets = targets
            else:
                self.phase += (
                    2.0 * math.pi * self.settings.control.gait_frequency * activity * elapsed
                )
                self._current_targets = self._targets()
        control_targets = np.concatenate(
            (self._current_targets, np.asarray([self._mouth_target], dtype=np.float32))
        )
        self._wp.copy(
            self._data.ctrl,
            self._wp.array([control_targets], dtype=self._wp.float32, device=self.device),
        )

        if self._jump_requested:
            qvel = np.asarray(self._host_data.qvel, dtype=np.float32).copy()
            # Jump is an operator-facing simulation primitive. It deliberately
            # leaves the 14-value learned action contract unchanged.
            if self._host_data.xpos[self._body_id, 2] <= 0.18 and abs(qvel[2]) < 0.25:
                qvel[2] = JUMP_TAKEOFF_VELOCITY
                self._wp.copy(
                    self._data.qvel,
                    self._wp.array([qvel], dtype=self._wp.float32, device=self.device),
                )
            self._jump_requested = False

        # RL policies act only through the 14 joint targets. The fallback
        # reference gait uses a bounded virtual boom to track cmd_vel.
        xfrc = np.zeros((self._model.nbody, 6), dtype=np.float32)
        if self._policy is None:
            self._apply_reference_assist(xfrc, vx, vy, wz)
        self._wp.copy(
            self._data.xfrc_applied,
            self._wp.array([xfrc], dtype=self._wp.spatial_vector, device=self.device),
        )
        if self._graph is None:
            self._mjw.step(self._mjw_model, self._data)
        else:
            self._wp.capture_launch(self._graph)
        self._wp.synchronize()
        self._mjw.get_data_into(self._host_data, self._model, self._data)
        if self._policy is not None and policy_tick and self._policy.after_step(self._host_data):
            self.reset()

    def _apply_reference_assist(
        self, xfrc: np.ndarray, vx: float, vy: float, wz: float
    ) -> None:
        """Apply the virtual boom used only by the non-RL reference gait."""
        yaw = self._yaw_from_wxyz(self._host_data.xquat[self._body_id])
        desired_world = np.asarray(
            [math.cos(yaw) * vx - math.sin(yaw) * vy, math.sin(yaw) * vx + math.cos(yaw) * vy]
        )
        error = desired_world - np.asarray(self._host_data.qvel[:2])
        xfrc[self._body_id, :2] = np.clip(
            self.settings.control.velocity_assist * error,
            -10.0,
            10.0,
        )
        xfrc[self._body_id, 5] = float(
            np.clip(
                self.settings.control.yaw_assist * (wz - self._host_data.qvel[5]),
                -1.0,
                1.0,
            )
        )
        if not self.enabled:
            return
        roll, pitch = self._roll_pitch_from_wxyz(self._host_data.xquat[self._body_id])
        xfrc[self._body_id, 2] = float(
            np.clip(
                self.settings.control.height_assist
                * (
                    self.settings.control.standing_height
                    - self._host_data.xpos[self._body_id, 2]
                )
                - self.settings.control.height_damping * self._host_data.qvel[2],
                -8.0,
                8.0,
            )
        )
        xfrc[self._body_id, 3] = float(
            np.clip(
                -self.settings.control.posture_assist * roll
                - self.settings.control.posture_damping * self._host_data.qvel[3],
                -1.0,
                1.0,
            )
        )
        xfrc[self._body_id, 4] = float(
            np.clip(
                -self.settings.control.posture_assist * pitch
                - self.settings.control.posture_damping * self._host_data.qvel[4],
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def _yaw_from_wxyz(quaternion: Any) -> float:
        w, x, y, z = (float(value) for value in quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _roll_pitch_from_wxyz(quaternion: Any) -> tuple[float, float]:
        w, x, y, z = (float(value) for value in quaternion)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return roll, pitch

    @staticmethod
    def _projected_gravity_z(quaternion: Any) -> float:
        values = np.asarray(quaternion, dtype=np.float64)
        values /= max(float(np.linalg.norm(values)), 1e-12)
        _, x, y, _ = values
        return -(1.0 - 2.0 * (x * x + y * y))

    def _sensor(self, name: str) -> list[float]:
        return [float(value) for value in self._host_data.sensordata[self._sensor_slices[name]]]

    def state(self) -> dict[str, Any]:
        position = [float(value) for value in self._host_data.xpos[self._body_id]]
        orientation = [float(value) for value in self._host_data.xquat[self._body_id]]
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        # Use the deployment policy's orientation criterion. A sit policy can
        # legitimately lower the trunk below 6 cm without being a fall.
        gravity_z = self._projected_gravity_z(orientation)
        fallen = gravity_z > -0.5 or position[2] < 0.02
        jumping = position[2] > 0.18 and abs(float(self._host_data.qvel[2])) > 0.1
        return {
            "backend": self.name,
            "device": self.device,
            "simulation_time": float(self._host_data.time),
            "enabled": self.enabled,
            "mode": "disabled"
            if not self.enabled
            else "jumping"
            if jumping
            else self._policy.mode
            if self._policy is not None
            else "fallen"
            if fallen
            else "walking"
            if max(abs(vx), abs(vy), abs(wz)) > 1e-4
            else "standing",
            "fallen": fallen,
            "command": {"linear_x": vx, "linear_y": vy, "angular_z": wz},
            "joint_names": [*self.settings.robot.joint_names, "mouth"],
            "joint_positions": [
                *[float(self._host_data.qpos[index]) for index in self._joint_qpos],
                float(self._host_data.qpos[self._mouth_qpos]),
            ],
            "joint_velocities": [
                *[float(self._host_data.qvel[index]) for index in self._joint_dof],
                float(self._host_data.qvel[self._mouth_dof]),
            ],
            "mouth": {
                "opened": self._mouth_open,
                "position": float(self._host_data.qpos[self._mouth_qpos]),
            },
            "base_pose": {"position": position, "quaternion_wxyz": orientation},
            "base_twist": {
                "linear": [float(value) for value in self._host_data.qvel[:3]],
                "angular": [float(value) for value in self._host_data.qvel[3:6]],
            },
            "imu": {
                "frame": "imu_link",
                "orientation_wxyz": self._sensor("orientation"),
                "angular_velocity": self._sensor("imu_ang_vel"),
                "linear_acceleration": self._sensor("imu_accel"),
            },
            "contacts": int(self._host_data.ncon),
            "policy": self._policy.status()
            if self._policy is not None
            else {
                "enabled": False,
                "active": "reference_gait",
                "mode": "reference_gait",
                "available": [],
                "observation_size": 61,
                "action_size": 14,
                "recovery": None,
            },
        }

    def jpeg_frame(self) -> bytes:
        self._renderer.update_scene(self._host_data, camera=self.settings.robot.camera)
        pixels = self._renderer.render()
        output = io.BytesIO()
        Image.fromarray(pixels).save(
            output,
            "JPEG",
            quality=self.settings.camera.jpeg_quality,
        )
        return output.getvalue()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
