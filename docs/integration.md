# Robo Boy and ROS 2 integration

## DDS discovery

Both Compose projects use host networking. Load `../robo-boy/.env` when starting
the simulator so these settings agree:

- `ROS_DISTRO`
- `ROS_DOMAIN_ID`
- `ROS_LOCALHOST_ONLY`
- `ROS_AUTOMATIC_DISCOVERY_RANGE` and `ROS_STATIC_PEERS`
- `RMW_IMPLEMENTATION` and `CYCLONEDDS_URI`

The included CycloneDDS profile permits multicast for participant discovery,
keeps RTPS payloads under a normal Ethernet MTU, and supports many transient
host-networked participants. Point both projects at the same absolute
`CYCLONEDDS_CONFIG_PATH` for deployments with a fixed interface or peer list.

## What Robo Boy discovers

Robo Boy's rosapi discovery sees standard message and service packages already
installed in its ROS image. No custom workspace overlay is required. The
`robot_state_publisher` sidecar publishes the URDF description and joint TF;
the bridge adds the dynamic `odom → base_link` transform. This matches Robo
Boy's default `odom` fixed frame.

Add a URDF visualization using `/robot_description`, an odometry visualization
using `/odom`, and select `/locomotion/camera/image_raw` in the camera panel.
A custom gamepad can publish `geometry_msgs/msg/Twist` on `/cmd_vel` with axes
mapped to `linear.x`, `linear.y`, and `angular.z`.

The camera publisher is reliable rather than the usual best-effort sensor QoS.
That is required for endpoint compatibility with the reliable subscriber in
Robo Boy's `web_video_server`.

## Safety behavior

`/cmd_vel` is clamped to the configured planar limits. Commands time out at
both the bridge and runtime, and non-finite values are rejected. The shared
`/emergency_stop` Bool topic disables actuator commands. Use
`/locomotion/enable` to re-enable the controller explicitly; resetting the
physics does not silently clear an engaged stop.

## RL policy control

The runtime loads the official Microduck ONNX bundle at startup, checks its
`[1, 61] → [1, 14]` contract, warms each model, and applies the selected
policy's joint targets at 50 Hz. The observation contains angular velocity,
projected gravity, relative joint positions, joint velocities, the previous
action, and the policy command vector.

`/cmd_vel` feeds the walking policy. Publish a `std_msgs/msg/String` on
`/locomotion/policy_cmd` to select `walk`, `stand`, `sit`, `ground_pick`,
`kick_left`, `kick_right`, `roulade`, or `recover`. The same modes are available
through `POST /api/v1/policy`. Policy inference uses ONNX Runtime while MuJoCo
Warp keeps physics on the configured GPU. No virtual velocity, height, or
posture assistance is applied while RL is enabled.

Set `[policy].enabled = false` in the TOML configuration only when the simple
reference controller is useful for diagnostics.

For Behavior Trees, prefer the `/locomotion/execute_motion`
`microduck_interfaces/action/ExecuteMotion` action. Set its `motion` goal to
any policy name above. The same action accepts `jump`, but reports it as a
simulator primitive because no jump ONNX policy is present in the official
bundle. The action waits for one-shot motions to finish before succeeding, so
it composes cleanly in Sequence nodes.

The auxiliary mouth deliberately sits outside the policy observation/action
contract. Call `/locomotion/mouth` (`std_srvs/srv/SetBool`), or use the
no-argument `/locomotion/mouth/open` and `/locomotion/mouth/close` Trigger
services. `/locomotion/jump` provides the equivalent no-argument jump service.
