from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8010
    cors_origins: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationSettings:
    backend: str
    device: str
    model: Path
    urdf: Path
    dt: float
    realtime: bool
    state_hz: float
    render_hz: float
    command_timeout: float
    nconmax: int
    njmax: int
    show_viewer: bool = False


@dataclass(frozen=True)
class RobotSettings:
    name: str
    base_body: str
    camera: str
    keyframe: str
    joint_names: list[str]


@dataclass(frozen=True)
class ControlSettings:
    control_hz: float
    max_linear_x: float
    max_linear_y: float
    max_angular_z: float
    gait_frequency: float
    hip_swing: float
    knee_lift: float
    ankle_swing: float
    velocity_assist: float
    yaw_assist: float
    standing_height: float
    height_assist: float
    height_damping: float
    posture_assist: float
    posture_damping: float


@dataclass(frozen=True)
class PolicySettings:
    enabled: bool
    directory: Path
    walking: str
    standing: str | None
    sitstand: str | None
    ground_pick: str | None
    kick_left: str | None
    kick_right: str | None
    roulade: str | None
    action_scale: float
    automatic_recovery: bool

    def policy_paths(self) -> dict[str, Path | None]:
        return {
            "walk": self.directory / self.walking,
            "stand": self.directory / self.standing if self.standing else None,
            "sitstand": self.directory / self.sitstand if self.sitstand else None,
            "ground_pick": self.directory / self.ground_pick if self.ground_pick else None,
            "kick_left": self.directory / self.kick_left if self.kick_left else None,
            "kick_right": self.directory / self.kick_right if self.kick_right else None,
            "roulade": self.directory / self.roulade if self.roulade else None,
        }


@dataclass(frozen=True)
class CameraSettings:
    topic: str
    frame: str
    width: int
    height: int
    fov: float
    jpeg_quality: int


@dataclass(frozen=True)
class Settings:
    source: Path
    server: ServerSettings
    simulation: SimulationSettings
    robot: RobotSettings
    control: ControlSettings
    policy: PolicySettings
    camera: CameraSettings

    def with_overrides(
        self,
        *,
        backend: str | None = None,
        device: str | None = None,
        show_viewer: bool | None = None,
    ) -> Settings:
        return replace(
            self,
            simulation=replace(
                self.simulation,
                backend=backend or self.simulation.backend,
                device=device or self.simulation.device,
                show_viewer=(
                    self.simulation.show_viewer if show_viewer is None else show_viewer
                ),
            ),
        )


def default_config_path() -> Path:
    configured = os.environ.get("MJWARP_LOCOMOTION_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        Path.cwd() / "config" / "default.toml",
        Path(__file__).resolve().parents[2] / "config" / "default.toml",
        Path("/app/config/default.toml"),
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{key}] table")
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    source = Path(path).expanduser().resolve() if path else default_config_path()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    sim_data = dict(_table(data, "simulation"))
    for field_name in ("model", "urdf"):
        value = Path(sim_data[field_name])
        sim_data[field_name] = value if value.is_absolute() else (source.parent / value).resolve()
    policy_data = dict(_table(data, "policy"))
    policy_directory = Path(policy_data["directory"])
    policy_data["directory"] = (
        policy_directory
        if policy_directory.is_absolute()
        else (source.parent / policy_directory).resolve()
    )
    settings = Settings(
        source=source,
        server=ServerSettings(**_table(data, "server")),
        simulation=SimulationSettings(**sim_data),
        robot=RobotSettings(**_table(data, "robot")),
        control=ControlSettings(**_table(data, "control")),
        policy=PolicySettings(**policy_data),
        camera=CameraSettings(**_table(data, "camera")),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if settings.simulation.backend not in {"mujoco_warp", "mock"}:
        raise ValueError("simulation.backend must be 'mujoco_warp' or 'mock'")
    if settings.simulation.dt <= 0 or settings.simulation.state_hz <= 0:
        raise ValueError("simulation dt and state_hz must be positive")
    if settings.simulation.render_hz <= 0 or settings.simulation.command_timeout <= 0:
        raise ValueError("render_hz and command_timeout must be positive")
    if settings.simulation.nconmax <= 0 or settings.simulation.njmax <= 0:
        raise ValueError("nconmax and njmax must be positive")
    if settings.control.control_hz <= 0:
        raise ValueError("control.control_hz must be positive")
    if settings.policy.action_scale <= 0:
        raise ValueError("policy.action_scale must be positive")
    if settings.policy.enabled:
        missing = [
            str(path)
            for path in settings.policy.policy_paths().values()
            if path is not None and not path.is_file()
        ]
        if missing:
            raise ValueError(f"policy files do not exist: {', '.join(missing)}")
    if not settings.robot.joint_names or len(settings.robot.joint_names) != len(
        set(settings.robot.joint_names)
    ):
        raise ValueError("robot.joint_names must be non-empty and unique")
    if not settings.camera.topic.startswith("/"):
        raise ValueError("camera.topic must be absolute")
    if settings.camera.width <= 0 or settings.camera.height <= 0:
        raise ValueError("camera dimensions must be positive")
    for path in (settings.simulation.model, settings.simulation.urdf):
        if not path.is_file():
            raise ValueError(f"robot asset does not exist: {path}")
