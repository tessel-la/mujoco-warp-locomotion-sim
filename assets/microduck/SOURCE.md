# Pollen Robotics Microduck assets

The MJCF files and CAD meshes in this directory are copied unchanged from
`pollen-robotics/microduck_rl` branch `develop`, commit
`c690e5599866893b11b3ab2764883a8b0800cbe7` (retrieved 2026-08-27):

<https://github.com/pollen-robotics/microduck_rl>

`scene.xml` plus `robot_allcollisions.xml` is the active full-collision model.
`microduck.urdf` is a generated ROS visualization adapter derived from that
MJCF by `tools/export_microduck_urdf.py`; it is not an upstream Pollen file.

The upstream code is Apache-2.0; see `LICENSE.rl`. The upstream README states
that the 3D model files are Creative Commons BY-SA-NC. Preserve both notices
when redistributing these assets.
