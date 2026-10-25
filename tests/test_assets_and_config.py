from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.mujoco_warp_backend import MujocoWarpBackend
from tools.export_microduck_urdf import _rpy, export

ROOT = Path(__file__).resolve().parents[1]


def test_show_viewer_defaults_to_headless_and_can_be_overridden() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    assert settings.simulation.show_viewer is False

    with_viewer = settings.with_overrides(show_viewer=True)
    assert with_viewer.simulation.show_viewer is True

    # Omitting show_viewer preserves whatever the setting already was, the same
    # None-means-unchanged contract backend/device already follow.
    assert with_viewer.with_overrides().simulation.show_viewer is True


def test_fall_orientation_allows_an_upright_low_sit() -> None:
    assert MujocoWarpBackend._projected_gravity_z((1.0, 0.0, 0.0, 0.0)) == pytest.approx(
        -1.0
    )
    assert MujocoWarpBackend._projected_gravity_z(
        (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    ) == pytest.approx(0.0)


def test_fall_orientation_is_a_plain_float_so_the_state_stays_serializable() -> None:
    upside_down = (0.0, 1.0, 0.0, 0.0)
    gravity_z = MujocoWarpBackend._projected_gravity_z(upside_down)
    assert type(gravity_z) is float
    fallen = gravity_z > -0.5 or 0.1 < 0.02
    json.dumps({"fallen": fallen})


def _quaternion_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _included_robot_root(scene_path: Path) -> ET.Element:
    scene = ET.parse(scene_path).getroot()
    include = scene.find("./include")
    assert include is not None
    return ET.parse(scene_path.parent / include.attrib["file"]).getroot()


def test_default_config_resolves_assets_and_joint_contract() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    assert settings.simulation.backend == "mujoco_warp"
    assert settings.simulation.model.is_file()
    assert settings.simulation.urdf.is_file()
    assert len(settings.robot.joint_names) == 14
    assert settings.camera.topic == "/locomotion/camera/image_raw"
    assert settings.camera.frame == "head_camera_optical_frame"
    assert settings.robot.name == "microduck"
    assert settings.robot.base_body == "trunk_base"
    assert settings.simulation.model.name == "scene.xml"
    assert settings.simulation.nconmax == 256
    assert settings.simulation.njmax == 1024
    assert settings.policy.enabled is True
    assert len(settings.policy.policy_paths()) == 7


def test_official_policy_bundle_matches_manifest() -> None:
    manifest = json.loads((ROOT / "policies/manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        policy = ROOT / "policies" / filename
        assert policy.is_file()
        assert hashlib.sha256(policy.read_bytes()).hexdigest() == expected


def test_mjcf_urdf_and_config_use_the_same_actuated_joints() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    mjcf = _included_robot_root(settings.simulation.model)
    urdf = ET.parse(settings.simulation.urdf).getroot()
    mjcf_joints = {
        element.attrib["name"]
        for element in mjcf.findall(".//joint")
        if element.attrib.get("name") and element.attrib.get("type") != "free"
    }
    actuator_joints = {element.attrib["joint"] for element in mjcf.findall("./actuator/position")}
    urdf_moving_joints = {
        element.attrib["name"]
        for element in urdf.findall("./joint")
        if element.attrib.get("type") != "fixed"
    }
    expected = {*settings.robot.joint_names, "mouth"}
    assert mjcf_joints == expected
    assert actuator_joints == expected
    assert urdf_moving_joints == expected


def test_microduck_urdf_meshes_are_derived_from_vendored_assets() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    urdf = ET.parse(settings.simulation.urdf).getroot()
    package_prefix = "package://microduck/assets/"
    mesh_filenames = {
        element.attrib["filename"]
        for element in urdf.findall(".//mesh")
    }
    assert mesh_filenames
    assert all(filename.startswith(package_prefix) for filename in mesh_filenames)
    mesh_root = settings.simulation.urdf.parent / "assets"
    assert all(
        (mesh_root / filename.removeprefix(package_prefix)).is_file()
        for filename in mesh_filenames
    )
    assert (settings.simulation.urdf.parent / "LICENSE.rl").is_file()
    assert (settings.simulation.urdf.parent / "SOURCE.md").is_file()
    assert (ROOT / "tools/export_microduck_urdf.py").is_file()


@pytest.mark.parametrize(
    "quaternion",
    [
        (0.5, -0.5, -0.5, -0.5),  # left foot visual
        (0.5, -0.5, 0.5, 0.5),  # right foot visual
        (0.0, 0.707107, 0.0, 0.707107),  # ankle motor at Euler singularity
    ],
)
def test_urdf_export_preserves_singular_microduck_mesh_rotations(
    quaternion: tuple[float, float, float, float],
) -> None:
    exported_rpy = tuple(float(value) for value in _rpy(" ".join(map(str, quaternion))).split())
    round_trip = _quaternion_from_rpy(exported_rpy)
    norm = math.sqrt(sum(value * value for value in quaternion))
    normalized = tuple(value / norm for value in quaternion)
    dot = abs(sum(left * right for left, right in zip(normalized, round_trip, strict=True)))
    assert dot == pytest.approx(1.0, abs=1e-10)


def test_tracked_microduck_urdf_matches_exporter(tmp_path: Path) -> None:
    generated = tmp_path / "microduck.urdf"
    export(ROOT / "assets/microduck/robot_allcollisions.xml", generated)
    assert generated.read_text() == (ROOT / "assets/microduck/microduck.urdf").read_text()


def test_microduck_head_camera_faces_robot_forward() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    mjcf = _included_robot_root(settings.simulation.model)
    camera = mjcf.find(".//camera[@name='head_camera']")
    assert camera is not None
    # MuJoCo looks down local -Z with local +Y as image-up. This mounting
    # produces Microduck +X view direction and +Z image-up in the STAND pose.
    assert camera.attrib["quat"] == "0.707107 0 0 -0.707107"
    urdf = ET.parse(settings.simulation.urdf).getroot()
    optical_joint = urdf.find("./joint[@name='head_camera_link_to_head_camera_optical_frame']")
    assert optical_joint is not None
    assert optical_joint.find("./origin").attrib == {
        "xyz": "0 0 0",
        "rpy": "-1.57079632679 0 -1.57079632679",
    }


def test_standing_keyframe_contains_free_joint_and_all_servo_positions() -> None:
    settings = load_settings(ROOT / "config/default.toml")
    mjcf = ET.parse(settings.simulation.model).getroot()
    key = mjcf.find(f"./keyframe/key[@name='{settings.robot.keyframe}']")
    assert key is not None
    assert len(key.attrib["qpos"].split()) == 8 + len(settings.robot.joint_names)


def test_official_microduck_training_snapshot_and_isolated_container_are_present() -> None:
    training = ROOT / "training/microduck_rl"
    assert (training / "uv.lock").is_file()
    assert (training / "LICENSE").is_file()
    assert (training / "UPSTREAM.md").is_file()
    assert (training / "src/mjlab_microduck/tasks/__init__.py").is_file()
    assert (training / "src/mjlab_microduck/actuator/friction_dr_bam.py").is_file()
    pyproject = (training / "pyproject.toml").read_text()
    assert '"mjlab==1.3.0"' in pyproject
    assert '"warp-lang==1.12.0"' in pyproject
    compose = (ROOT / "docker-compose.training.yml").read_text()
    assert "microduck_rl_train:" in compose
    assert "runtime: nvidia" in compose
    assert "Dockerfile.training" in compose


def test_compose_exposes_both_runtime_services() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "mujoco_warp_locomotion_sim:" in compose
    assert "mujoco_warp_ros_bridge:" in compose
    assert "runtime: nvidia" in compose


def test_mock_configuration_does_not_require_policy_weights(tmp_path: Path) -> None:
    source = (ROOT / "config/default.toml").read_text()
    source = source.replace('directory = "../policies"', f'directory = "{tmp_path}/missing"')
    source = source.replace('"../assets/', f'"{ROOT}/assets/')
    config = tmp_path / "config.toml"
    config.write_text(source)
    settings = load_settings(config, backend="mock", device="cpu")
    assert settings.simulation.backend == "mock"
    assert settings.server.host == "127.0.0.1"
    with pytest.raises(ValueError, match="policy files do not exist"):
        load_settings(config, backend="mujoco_warp", device="cpu")


def test_invalid_backend_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="simulation.backend"):
        load_settings(ROOT / "config/default.toml", backend="invalid")
