#!/usr/bin/env python3
"""Export a ROS visualization URDF from Pollen's Microduck MJCF.

The MJCF remains the source of truth for physics and training.  This adapter
only translates its body/joint tree and visual mesh placements so
robot_state_publisher and browser-based ROS viewers can display the same CAD.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _values(value: str | None, default: str) -> list[float]:
    return [float(part) for part in (value or default).split()]


def _rpy(quaternion: str | None) -> str:
    w, x, y, z = _values(quaternion, "1 0 0 0")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))

    # Extract fixed-axis URDF roll-pitch-yaw from Rz(yaw) Ry(pitch) Rx(roll).
    # The common direct formula loses the coupled roll/yaw component at a
    # +/-90 degree pitch. Several official Microduck foot meshes sit exactly
    # at that singularity, so handle it explicitly instead.
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    cos_pitch = math.hypot(r00, r10)
    pitch = math.atan2(-r20, cos_pitch)
    if cos_pitch > 1e-9:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        yaw = 0.0
        roll = math.atan2(r01, r11) if r20 < 0 else math.atan2(-r01, r11)
    return f"{roll:.12g} {pitch:.12g} {yaw:.12g}"


def _xyz(value: str | None) -> str:
    return " ".join(f"{item:.12g}" for item in _values(value, "0 0 0"))


def _origin(parent: ET.Element, source: ET.Element) -> None:
    ET.SubElement(
        parent,
        "origin",
        {"xyz": _xyz(source.get("pos")), "rpy": _rpy(source.get("quat"))},
    )


def export(source: Path, destination: Path) -> None:
    mjcf = ET.parse(source).getroot()
    robot = ET.Element("robot", {"name": "pollen_microduck"})
    robot.append(
        ET.Comment(
            " Generated from Pollen Robotics microduck_rl robot_allcollisions.xml; "
            "physics and training continue to use the original MJCF. "
        )
    )

    rgba_by_material = {
        item.get("name", ""): item.get("rgba", "0.75 0.75 0.75 1")
        for item in mjcf.findall("./asset/material")
    }
    mesh_file_by_name: dict[str, str] = {}
    for item in mjcf.findall("./asset/mesh"):
        filename = item.get("file")
        if not filename:
            continue
        mesh_file_by_name[item.get("name") or Path(filename).stem] = filename

    ET.SubElement(robot, "link", {"name": "base_link"})

    root_body = mjcf.find("./worldbody/body[@name='trunk_base']")
    if root_body is None:
        raise ValueError("MJCF has no trunk_base root body")

    def add_body(body: ET.Element, parent_link: str) -> None:
        link_name = body.get("name")
        if not link_name:
            raise ValueError("every MJCF body must have a name")
        link = ET.SubElement(robot, "link", {"name": link_name})

        for geom in body.findall("./geom[@class='visual']"):
            mesh_name = geom.get("mesh")
            if geom.get("type") != "mesh" or not mesh_name:
                continue
            filename = mesh_file_by_name.get(mesh_name, f"{mesh_name}.stl")
            visual = ET.SubElement(link, "visual")
            _origin(visual, geom)
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(
                geometry,
                "mesh",
                {"filename": f"package://microduck/assets/{filename}"},
            )
            material_name = geom.get("material", f"{mesh_name}_material")
            material = ET.SubElement(visual, "material", {"name": material_name})
            ET.SubElement(
                material,
                "color",
                {"rgba": rgba_by_material.get(material_name, "0.75 0.75 0.75 1")},
            )

        joint_source = next(
            (joint for joint in body.findall("./joint") if joint.get("type") != "free"),
            None,
        )
        if parent_link == "base_link":
            joint_name = "base_link_to_trunk_base"
            joint_type = "fixed"
        elif joint_source is None:
            joint_name = f"{parent_link}_to_{link_name}"
            joint_type = "fixed"
        else:
            joint_name = joint_source.get("name", f"{parent_link}_to_{link_name}")
            joint_type = "revolute"

        joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": joint_type})
        if parent_link == "base_link":
            # trunk_base is a MuJoCo free body. Its world pose is supplied by
            # odom -> base_link, so its authored spawn height is not a URDF
            # kinematic offset.
            ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        else:
            _origin(joint, body)
        ET.SubElement(joint, "parent", {"link": parent_link})
        ET.SubElement(joint, "child", {"link": link_name})
        if joint_type == "revolute" and joint_source is not None:
            ET.SubElement(joint, "axis", {"xyz": _xyz(joint_source.get("axis"))})
            lower, upper = _values(joint_source.get("range"), "-3.14159265359 3.14159265359")
            ET.SubElement(
                joint,
                "limit",
                {
                    "lower": f"{lower:.12g}",
                    "upper": f"{upper:.12g}",
                    "effort": "1.0",
                    "velocity": "12.0",
                },
            )

        site_links = {"imu": "imu_link", "head_camera": "head_camera_link"}
        for site in body.findall("./site"):
            site_name = site.get("name", "")
            frame_link = site_links.get(site_name)
            if frame_link is None:
                continue
            ET.SubElement(robot, "link", {"name": frame_link})
            frame_joint = ET.SubElement(
                robot,
                "joint",
                {"name": f"{link_name}_to_{frame_link}", "type": "fixed"},
            )
            _origin(frame_joint, site)
            ET.SubElement(frame_joint, "parent", {"link": link_name})
            ET.SubElement(frame_joint, "child", {"link": frame_link})
            if site_name == "head_camera":
                # REP-103 camera_link is X-forward/Y-left/Z-up. Image headers
                # use an optical frame with Z-forward/X-right/Y-down.
                optical_link = "head_camera_optical_frame"
                ET.SubElement(robot, "link", {"name": optical_link})
                optical_joint = ET.SubElement(
                    robot,
                    "joint",
                    {"name": f"{frame_link}_to_{optical_link}", "type": "fixed"},
                )
                ET.SubElement(
                    optical_joint,
                    "origin",
                    {"xyz": "0 0 0", "rpy": f"{-math.pi / 2:.12g} 0 {-math.pi / 2:.12g}"},
                )
                ET.SubElement(optical_joint, "parent", {"link": frame_link})
                ET.SubElement(optical_joint, "child", {"link": optical_link})

        for child in body.findall("./body"):
            add_body(child, link_name)

    add_body(root_body, "base_link")
    ET.indent(robot, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        '<?xml version="1.0"?>\n'
        "<!-- ROS visualization adapter; regenerate with tools/export_microduck_urdf.py. -->\n"
        + ET.tostring(robot, encoding="unicode")
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)


if __name__ == "__main__":
    main()
