from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageDraw

from .backend import JumpEpisode
from .config import Settings
from .policy_controller import MODE_KINDS

# Trunk height above which the mock robot counts as off the ground.
MOCK_GROUND_HEIGHT = 0.6201

# How long each mode occupies the mock controller. The one-shot durations match the
# real policies; `sit`/`stand` get the same settle window the RL sit/stand transition
# is given, so an action waiting on either backend sees the same lifecycle.
POLICY_DURATIONS = {
    "ground_pick": 2.8,
    "kick_left": 0.5,
    "kick_right": 0.5,
    "roulade": 3.0,
    "recover": 1.0,
    "sit": 1.5,
    "stand": 1.5,
}


class MockBackend:
    """Deterministic locomotion backend for API/ROS work without CUDA."""

    name = "mock"
    device = "cpu"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = True
        self.policy_mode = "walk"
        self._policy_time_remaining = 0.0
        self._policy_episode = 0
        self._policy_request = "walk"
        self._policy_kind = MODE_KINDS["walk"]
        self._policy_phase = "complete"
        self._jump = JumpEpisode()
        self.command = [0.0, 0.0, 0.0]
        self.sim_time = 0.0
        self.pose = [0.0, 0.0, 0.62]
        self.mouth_open = False
        self.mouth_position = 0.0
        self.vertical_velocity = 0.0
        self.yaw = 0.0
        self.phase = 0.0
        self.positions = self._gait_positions(0.0)
        self.velocities = [0.0] * len(self.positions)
        self._control_accumulator = 1.0 / self.settings.control.control_hz
        self._frame_index = 0

    def initialize(self) -> None:
        return

    def set_command(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        self.command = [linear_x, linear_y, angular_z]

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.command = [0.0, 0.0, 0.0]
            self._jump.abandon()

    def set_policy(self, mode: str) -> str:
        if mode not in MODE_KINDS:
            raise ValueError(f"unknown policy mode {mode!r}")
        self.policy_mode = mode
        self._policy_time_remaining = POLICY_DURATIONS.get(mode, 0.0)
        self._policy_episode += 1
        self._policy_request = mode
        self._policy_kind = MODE_KINDS[mode]
        self._policy_phase = "running" if self._policy_time_remaining > 0.0 else "complete"
        return self.policy_mode

    def set_mouth(self, opened: bool) -> None:
        self.mouth_open = opened

    def jump(self) -> None:
        if not self.enabled:
            return
        self._jump.request()
        if self.pose[2] <= MOCK_GROUND_HEIGHT:
            self.vertical_velocity = 1.35
        else:
            self._jump.refuse()

    def reset(self) -> None:
        enabled = self.enabled
        # Episode counters stay monotonic across a reset so a new episode is never
        # mistaken for the old one; a reset abandons whatever was still running.
        policy_episode = self._policy_episode
        jump_episode = self._jump
        jump_episode.abandon()
        self.__init__(self.settings)
        self.enabled = enabled
        self._policy_episode = policy_episode
        self._jump = jump_episode

    def step(self) -> None:
        dt = self.settings.simulation.dt
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        self.yaw += wz * dt
        self.pose[0] += (math.cos(self.yaw) * vx - math.sin(self.yaw) * vy) * dt
        self.pose[1] += (math.sin(self.yaw) * vx + math.cos(self.yaw) * vy) * dt
        if self.vertical_velocity != 0.0 or self.pose[2] > 0.62:
            self.vertical_velocity -= 9.81 * dt
            self.pose[2] += self.vertical_velocity * dt
            if self.pose[2] <= 0.62:
                self.pose[2] = 0.62
                self.vertical_velocity = 0.0
        target_mouth = -0.48 if self.mouth_open else 0.0
        self.mouth_position += (target_mouth - self.mouth_position) * min(1.0, dt * 12.0)
        activity = min(1.0, math.hypot(vx, vy) / 0.08 + abs(wz) / 0.3)
        self._control_accumulator += dt
        control_period = 1.0 / self.settings.control.control_hz
        if self._control_accumulator >= control_period:
            elapsed = self._control_accumulator
            self._control_accumulator %= control_period
            previous = self.positions
            self.phase += 2.0 * math.pi * self.settings.control.gait_frequency * activity * elapsed
            self.positions = self._gait_positions(activity)
            self.velocities = [
                (new - old) / elapsed for new, old in zip(self.positions, previous, strict=True)
            ]
        else:
            self.velocities = [0.0] * len(self.positions)
        self.sim_time += dt
        self._jump.observe(self.pose[2] > MOCK_GROUND_HEIGHT)
        if self._policy_time_remaining > 0.0:
            self._policy_time_remaining -= dt
            if self._policy_time_remaining <= 0.0:
                self._policy_time_remaining = 0.0
                self._policy_phase = "complete"
                if self._policy_kind == "oneshot":
                    self.policy_mode = "walk"

    def _gait_positions(self, activity: float) -> list[float]:
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        scale = activity * min(
            1.0, max(0.25, abs(vx) / max(self.settings.control.max_linear_x, 1e-6))
        )
        values: list[float] = []
        for side in (1.0, -1.0):
            wave = math.sin(self.phase + (0.0 if side > 0 else math.pi))
            lift = max(0.0, wave) * scale
            values.extend(
                [
                    side * wz * 0.18,
                    side * (0.08 * math.cos(self.phase) * activity) - vy * 0.25,
                    -0.22 + self.settings.control.hip_swing * wave * scale,
                    0.44 + self.settings.control.knee_lift * lift,
                    -0.22 - self.settings.control.ankle_swing * wave * scale - 0.25 * lift,
                    -side * 0.05 * math.cos(self.phase) * activity,
                ]
            )
            if side > 0:
                values.extend([-0.05 * vx, 0.25 * wz])
        return values

    def _policy_progress(self) -> float:
        if self._policy_phase != "running":
            return 1.0
        duration = POLICY_DURATIONS[self._policy_request]
        return min(1.0, max(0.0, 1.0 - self._policy_time_remaining / duration))

    def state(self) -> dict[str, Any]:
        vx, vy, wz = self.command if self.enabled else (0.0, 0.0, 0.0)
        half = self.yaw / 2.0
        quaternion = [math.cos(half), 0.0, 0.0, math.sin(half)]
        return {
            "backend": self.name,
            "device": self.device,
            "simulation_time": self.sim_time,
            "enabled": self.enabled,
            "mode": "jumping"
            if self.pose[2] > MOCK_GROUND_HEIGHT
            else "walking"
            if max(abs(vx), abs(vy), abs(wz)) > 1e-4
            else "standing",
            "fallen": False,
            "command": {"linear_x": vx, "linear_y": vy, "angular_z": wz},
            "joint_names": [*self.settings.robot.joint_names, "mouth"],
            "joint_positions": [*self.positions, self.mouth_position],
            "joint_velocities": [*self.velocities, 0.0],
            "mouth": {"opened": self.mouth_open, "position": self.mouth_position},
            "jump": self._jump.status(),
            "base_pose": {"position": self.pose.copy(), "quaternion_wxyz": quaternion},
            "base_twist": {
                "linear": [vx, vy, 0.0],
                "angular": [0.0, 0.0, wz],
            },
            "imu": {
                "frame": "imu_link",
                "orientation_wxyz": quaternion,
                "angular_velocity": [0.0, 0.0, wz],
                "linear_acceleration": [0.0, 0.0, 9.81],
            },
            "contacts": 2,
            "policy": {
                "enabled": False,
                "active": "mock",
                "mode": self.policy_mode,
                "available": [
                    "ground_pick",
                    "kick_left",
                    "kick_right",
                    "recover",
                    "roulade",
                    "sit",
                    "stand",
                    "walk",
                ],
                "observation_size": 61,
                "action_size": 14,
                "recovery": None,
                "episode": self._policy_episode,
                "request": self._policy_request,
                "kind": self._policy_kind,
                "phase": self._policy_phase,
                "progress": self._policy_progress(),
            },
        }

    def jpeg_frame(self) -> bytes:
        width, height = self.settings.camera.width, self.settings.camera.height
        image = Image.new("RGB", (width, height), (23, 29, 39))
        draw = ImageDraw.Draw(image)
        horizon = int(height * 0.48)
        draw.rectangle((0, horizon, width, height), fill=(66, 73, 78))
        for offset in range(-8, 9):
            x = int(width / 2 + offset * 65 - self.pose[1] * 100)
            draw.line((x, horizon, int(width / 2 + offset * 130), height), fill=(93, 101, 106))
        for row in range(1, 7):
            y = int(horizon + (height - horizon) * (row / 7) ** 1.8)
            draw.line((0, y, width, y), fill=(93, 101, 106))
        draw.text((18, 16), "Duckbot head camera (mock)", fill=(245, 190, 42))
        draw.text((18, 38), f"t={self.sim_time:6.2f}s  x={self.pose[0]:+.2f}m", fill="white")
        self._frame_index += 1
        output = io.BytesIO()
        image.save(output, "JPEG", quality=self.settings.camera.jpeg_quality)
        return output.getvalue()

    def close(self) -> None:
        return
