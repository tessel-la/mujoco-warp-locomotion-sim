# Open Duck Mini v2 assets

The `robot.urdf` description and files under `meshes/` are derived from
[Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini/tree/v2),
revision `b23317a485b3cec7d8417f352478778b3475173c`.

They are distributed under the Apache License 2.0 in `LICENSE`.

Local changes to `robot.urdf`:

- mesh references use `package://open_duck_mini_v2/meshes/`;
- integration frames use `base_link`, `head_link`, `head_camera_link`, and
  `imu_link`;
- upstream ankle joints are named `*_ankle_pitch` and the simulator's
  `*_ankle_roll` joints are inserted at the foot frames;
- head pitch, head roll, and antenna joints are fixed because the local
  fourteen-joint controller does not drive them.

The STL mesh files themselves are unmodified.
