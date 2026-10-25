# Pollen Microduck MuJoCo Warp Sim

A GPU-accelerated simulation of Pollen Robotics' Microduck built on
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with a ROS 2 Jazzy
sidecar that appears in [Robo Boy](../robo-boy) through the same DDS and
rosbridge path as the Genesis simulator.

The live runtime now loads Pollen's full-collision `robot_allcollisions.xml`
scene, CAD, 14-servo order, sensors, camera, and standing keyframe. The complete
[`microduck_rl`](https://github.com/pollen-robotics/microduck_rl) `develop`
snapshot is included under `training/microduck_rl` for its mjlab/MuJoCo Warp
PPO tasks, BAM actuator model, domain randomization, playback, and ONNX export.
Because upstream provides MJCF rather than URDF, Robo Boy uses a generated
visualization-only URDF derived from that exact MJCF hierarchy and mesh set.

## Included

- Current MuJoCo Warp device flow (`put_model`, `put_data`, GPU `step`, and
  host synchronization for ROS/rendering), with CUDA graph capture on NVIDIA
  devices.
- Seven official Pollen ONNX policies for walking, standing/sitting, ground
  pickup, left/right ball kicks, and roulade. The walking policy accepts forward,
  lateral, and turning velocity commands at 50 Hz and acts only through the 14
  simulated joints.
- Two ROS 2 actions with real asynchronous semantics, for Robo Boy Behavior Tree
  action nodes. `MoveRelative` walks a requested distance, sidestep, or turn by
  closing the loop on odometry through the trained walking policy. `ExecuteMotion`
  runs a learned mode and stays active until that mode reaches its own completion
  condition. Both report progress while running and distinguish success, failure,
  timeout, and cancellation. An explicitly non-RL jump primitive and an auxiliary
  articulated mouth are available for operator demos without changing the policies'
  fixed 14-action contract.
- A policy-free reference gait is retained as an explicit fallback by setting
  `policy.enabled = false`.
- A deterministic mock backend for API, ROS, and Robo Boy development without
  CUDA or MuJoCo.
- REST, WebSocket state, MJPEG, and immutable URDF endpoints.
- CORS-enabled Microduck CAD mesh resources on port 8000 for Robo Boy's URDF view.
- ROS 2 `/cmd_vel`, `/odom`, `/joint_states`, `/imu/data`, `/tf`, `/clock`,
  image, diagnostics, enable/e-stop, and reset interfaces.
- Host-networked CycloneDDS configuration aligned with Robo Boy, plus a
  Tessella Dashboard app entry.

## Start with Robo Boy

Choose one simulator profile below. The profiles are mutually exclusive; do
not combine the CUDA and CPU overrides.

### NVIDIA CUDA

The default profile requires an NVIDIA driver and NVIDIA Container Toolkit and
selects `cuda:0`:

```bash
docker compose --env-file ../robo-boy/.env up --build
```

### CPU

Use this profile for CPU-only machines and AMD hosts:

```bash
docker compose --env-file ../robo-boy/.env \
  -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

MuJoCo Warp has no ROCm/HIP backend. AMD systems therefore run this simulator
on the CPU; there is no ROCm profile for MuJoCo Warp.

### AMD ROCm

MuJoCo Warp does not support ROCm/HIP. On AMD, use the CPU command above. The
Robo Boy GUI and ROS interfaces are unchanged.

### Robo Boy base stack

Start Robo Boy with the same environment in its own directory:

```bash
cd ../robo-boy
docker compose up --build
```

Connect Robo Boy to the host/domain. Its resource discovery will find the
camera, URDF, odometry, TF, diagnostics, and command topics. The default fixed
frame `odom` already matches this simulator.

For a fast end-to-end integration check that installs no MuJoCo/CUDA runtime:

```bash
docker compose --env-file ../robo-boy/.env \
  -f docker-compose.yml -f docker-compose.mock.yml up --build
```

The simulator API listens on <http://localhost:8010>. Robo Boy's existing
`web_video_server` consumes the ROS image; the API also exposes a direct MJPEG
stream at:

```text
http://localhost:8010/video_stream/stream?topic=/locomotion/camera/image_raw&type=mjpeg
```

### Robo Boy graphical panels

The simulator runs headless by default. Use Robo Boy for the graphical
interface after starting the simulator above:

```bash
cd ../robo-boy
ROBOBOY_PANEL_SOURCES_FILE=../robo-boy-microduck-control-panel/examples/panel-sources.json \
  docker compose --env-file .env \
  -f docker-compose.yml \
  -f infra/compose/panels.yml \
  -f ../robo-boy-microduck-control-panel/examples/docker-compose.panel.yml \
  up --build
```

Open <https://localhost>, choose **Quick Connect**, then add:

- **Microduck Control** to drive the robot and select policies;
- **Camera** with `/locomotion/camera/image_raw` for the head camera;
- **3D Visualization** with fixed frame `odom` for the full robot view.

Start the simulator first because its ROS sidecar creates the
`microduck-interfaces` volume mounted by Robo Boy. The direct MJPEG URL above is
the lightest option when only the camera view is needed.

### Native interactive viewer

For local physics debugging on the machine actually running the simulator --
distinct from Robo Boy's panels above, which work over the network and need no
display on that machine -- start with `--viewer` or set `MJWARP_SHOW_VIEWER=1`
to open MuJoCo's own `mujoco.viewer` window (needs a display and, in Docker,
`docker-compose.gui.yml`: it mounts the host's X11 socket and forwards
`DISPLAY`). Same mechanism `genesis-manipulator-sim` uses for its `--viewer`,
and what the Tessella Dashboard's headless/GUI toggle switches for this card.

## Drive it

The walking RL policy is active after startup. Publish continuously while a
control is held. Both the ROS sidecar and simulator independently stop stale
commands after 500 ms.

```bash
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.20, y: 0.0}, angular: {z: 0.25}}"
```

Select a skill by publishing one of `walk`, `stand`, `sit`, `ground_pick`,
`kick_left`, `kick_right`, `roulade`, or `recover`:

```bash
ros2 topic pub --once /locomotion/policy_cmd std_msgs/msg/String '{data: sit}'
ros2 topic pub --once /locomotion/policy_cmd std_msgs/msg/String '{data: stand}'
ros2 topic pub --once /locomotion/policy_cmd std_msgs/msg/String '{data: walk}'
```

Ask for a distance, a sidestep, or a turn instead of holding a velocity. The move
runs the same walking policy, closing the loop on odometry, and finishes when the
robot has arrived:

```bash
ros2 action send_goal /locomotion/move_relative \
  microduck_interfaces/action/MoveRelative '{x: 0.5}' --feedback
ros2 action send_goal /locomotion/move_relative \
  microduck_interfaces/action/MoveRelative '{yaw: 1.5708}' --feedback
ros2 action send_goal /locomotion/move_relative \
  microduck_interfaces/action/MoveRelative '{x: 0.4, y: 0.2, yaw: -0.785}' --feedback
```

Reset after a fall, disable locomotion, or engage the shared emergency stop:

```bash
ros2 service call /locomotion/reset std_srvs/srv/Empty '{}'
ros2 service call /locomotion/enable std_srvs/srv/SetBool '{data: false}'
ros2 topic pub --once /emergency_stop std_msgs/msg/Bool '{data: true}'
```

Direct HTTP commands are useful for smoke tests:

```bash
curl -X POST http://127.0.0.1:8010/api/v1/command \
  -H 'content-type: application/json' \
  -d '{"linear_x":0.2,"linear_y":0.0,"angular_z":0.2}'
curl -X POST http://127.0.0.1:8010/api/v1/policy \
  -H 'content-type: application/json' -d '{"mode":"kick_left"}'
curl -X POST http://127.0.0.1:8010/api/v1/mouth \
  -H 'content-type: application/json' -d '{"opened":true}'
curl -X POST http://127.0.0.1:8010/api/v1/jump \
  -H 'content-type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8010/api/v1/motion \
  -H 'content-type: application/json' \
  -d '{"kind":"relative_move","x":0.5}'
curl -X POST http://127.0.0.1:8010/api/v1/motion \
  -H 'content-type: application/json' \
  -d '{"kind":"policy","motion":"kick_left"}'
curl http://127.0.0.1:8010/api/v1/motion
curl http://127.0.0.1:8010/api/v1/state
```

Run a learned motion (or the simulator-only jump) through the ROS action API. The
goal stays active for as long as the policy is actually running, so it composes in a
Behavior Tree Sequence without guesswork:

```bash
ros2 action send_goal /locomotion/execute_motion \
  microduck_interfaces/action/ExecuteMotion '{motion: kick_left}' --feedback
ros2 service call /locomotion/mouth std_srvs/srv/SetBool '{data: true}'
ros2 service call /locomotion/jump std_srvs/srv/Trigger '{}'
```

Five ready-to-import Behavior Trees live in [`behavior_trees/`](behavior_trees),
covering a single move, a move/turn/move sequence, a repeated patrol, a policy
followed by a move, and cancellation with a fallback branch. Import any of them from
Robo Boy's Behavior Tree toolbar.

An optional movement panel and two further trees live in
[`../robo-boy-microduck-control-panel`](../robo-boy-microduck-control-panel).
The ROS sidecar exports `microduck_interfaces` into the named
`microduck-interfaces` volume; the panel repository's opt-in Compose overlay
mounts it into Robo Boy's rosbridge container.

Changing an action's fields changes its type hash, so after pulling a new
`microduck_interfaces` version rebuild the sidecar image **and** restart Robo Boy's
rosbridge container, or the stale overlay will fail to deserialize goals.

The checked-in `policies/manifest.json` pins every policy checksum and its
upstream source. To restore or verify the bundle without reinstalling the
runtime, run `python tools/download_policies.py`.

## Train Pollen's RL policies

Training is a separate, pinned Python 3.12 image so its `mjlab==1.3.0`, Warp,
Torch, and BAM versions cannot disturb the dashboard runtime or host NVIDIA
driver. The first image build downloads several GB of CUDA-enabled Python
wheels and produces a large image (about 9.3 GB with the current lockfile); it
installs nothing on the host.

```bash
docker compose -f docker-compose.training.yml build
docker compose -f docker-compose.training.yml run --rm microduck_rl_train
```

The default command trains Pollen's main walking environment with 4096 parallel
worlds:

```text
train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

Every upstream task and CLI remains available. For example:

```bash
docker compose -f docker-compose.training.yml run --rm microduck_rl_train list-envs
docker compose -f docker-compose.training.yml run --rm microduck_rl_train \
  train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096
docker compose -f docker-compose.training.yml run --rm microduck_rl_train \
  scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path ENTITY/PROJECT/RUN --onnx-file /policies/walk.onnx
```

Training logs and W&B state use named Docker volumes; exported policies written
to `/policies` appear in this repository's `policies/` directory. Set
`WANDB_API_KEY` only if the chosen upstream workflow uses W&B.

## ROS contract

| Name | Type | Direction | Purpose |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Robo Boy → sim | planar velocity command |
| `/locomotion/policy_cmd` | `std_msgs/msg/String` | Robo Boy → sim | select an RL policy mode |
| `/emergency_stop` | `std_msgs/msg/Bool` | Robo Boy → sim | disable/enable controller |
| `/joint_states` | `sensor_msgs/msg/JointState` | sim → Robo Boy | fourteen servo states |
| `/odom` | `nav_msgs/msg/Odometry` | sim → Robo Boy | base pose and twist |
| `/imu/data` | `sensor_msgs/msg/Imu` | sim → Robo Boy | simulated body IMU |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | sim → Robo Boy | `odom → base_link → robot` |
| `/clock` | `rosgraph_msgs/msg/Clock` | sim → ROS | simulation time |
| `/locomotion/camera/image_raw` | `sensor_msgs/msg/Image` | sim → Robo Boy | reliable RGB camera |
| `/locomotion/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | sim → Robo Boy | JPEG camera |
| `/locomotion/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | sim → Robo Boy | calibrated pinhole metadata |
| `/locomotion/status` | `std_msgs/msg/String` | sim → Robo Boy | latched JSON mode/health |
| `/locomotion/execute_motion` | `microduck_interfaces/action/ExecuteMotion` | Robo Boy → sim | run an RL mode or the simulator jump to completion |
| `/locomotion/move_relative` | `microduck_interfaces/action/MoveRelative` | Robo Boy → sim | walk to a pose offset in the base frame |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | sim → Robo Boy | standard diagnostic state |
| `/locomotion/reset` | `std_srvs/srv/Empty` | Robo Boy → sim | reset to standing keyframe |
| `/locomotion/enable` | `std_srvs/srv/SetBool` | Robo Boy → sim | controller enable switch |
| `/locomotion/mouth` | `std_srvs/srv/SetBool` | Robo Boy → sim | open or close the auxiliary mouth joint |
| `/locomotion/jump` | `std_srvs/srv/Trigger` | Robo Boy → sim | apply one bounded simulator jump impulse |

Camera QoS is reliable/volatile with depth five because Robo Boy's
`web_video_server` uses a reliable subscription. `robot_state_publisher` owns
the URDF-derived transforms; the bridge publishes only `odom → base_link`, so
there are no duplicate TF authorities.

See [integration notes](docs/integration.md) and
[architecture](docs/architecture.md) for extension points.

## Native development and verification

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
mujoco-warp-locomotion-sim doctor --backend mock
pytest
ruff check .
python -m compileall -q src ros tests
```

Install `.[mjwarp,dev]` to run `doctor --backend mujoco_warp --device cpu` or
`--device cuda:0`. The pinned MJWarp release supports CPU for debugging, while
an NVIDIA GPU is the intended deployment path. This development extra is only
for the operator runtime; use `docker-compose.training.yml` for Pollen's pinned
RL environment.

The optional [camera WebRTC gateway](docs/camera-webrtc.md) publishes plain WHEP video for Robo-Boy and other standard clients.
