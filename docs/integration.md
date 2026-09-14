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

## Actions, and what stays a service

The split is by duration, not by importance. Anything instantaneous stays a service or
a topic; anything that occupies the robot for a while is an action, so Robo Boy's
Behavior Tree executor can hold a node Running for exactly that long.

| Interface | Kind | Why |
| --- | --- | --- |
| `/locomotion/reset`, `/locomotion/enable` | service | instantaneous state writes |
| `/locomotion/mouth`, `/mouth/open`, `/mouth/close` | service | a servo settling in about 0.1 s, with no policy behind it |
| `/locomotion/policy_cmd` | topic | fire-and-forget mode select for the teleoperation panel |
| `/locomotion/jump` | service | kept for the panel; the action form is `ExecuteMotion` with `motion: jump` |
| `/locomotion/execute_motion` | action | a control policy running to its own completion condition |
| `/locomotion/move_relative` | action | a closed-loop walk to a requested offset |

### `MoveRelative`

`x` (forward), `y` (left), and `yaw` (counter-clockwise) are a pose offset in the base
frame as it stood when the goal was accepted. The robot is holonomic on the ground, so
"forward 0.5 m", "sidestep 0.2 m", "turn 90 degrees", and "go to a relative X/Y facing
a new heading" are all this one goal with the unused fields left at zero. There is no
vertical axis: it is a biped walking on the floor.

The move drives the trained walking policy through the ordinary velocity path, closing
the loop on odometry at 50 Hz, and switches the controller to `walk` first when some
other policy holds it. `max_linear_speed`, `max_angular_speed`, `position_tolerance`,
`yaw_tolerance`, and `timeout` default to the `[motion]` configuration when left at
zero, and may lower the configured limits but never raise them. The move stops as soon
as it is inside the tolerance bands, so the tolerance is also the accuracy: ask for a
tighter one when a goal needs it. Feedback carries the phase, progress, distance
traveled, distance remaining, heading remaining, and the active controller; the result
carries what was actually achieved rather than claiming the offset was hit exactly.

### `ExecuteMotion`

Set `motion` to any policy name above, or to `jump`, which the action reports as a
simulator primitive because the official bundle contains no jump ONNX policy. The goal
finishes when the requested mode reaches its own completion condition, which differs by
mode: `walk` is the controller's resting state and is done once it is active; `sit` and
`stand` are done once the posture has settled; the one-shot motions are done when the
policy hands control back to walking. A motion that knocked the robot over and needed
automatic recovery reports failure rather than success.

### Outcomes and preemption

A ROS action has no distinct terminal state for a timeout, so both actions carry an
`outcome` string — `succeeded`, `timeout`, `canceled`, `preempted`, or `failed` —
alongside the usual succeeded/canceled/aborted goal state. Cancelling a goal from a
Behavior Tree (a Timeout node, or stopping the tree) cancels the motion in the
simulator rather than leaving it running.

The simulator holds one motion slot, because the robot can only do one thing at a time.
A newer motion, a direct policy change, or a non-zero operator `/cmd_vel` preempts the
motion holding it; disabling locomotion, resetting, or falling over fails it. A zero
`/cmd_vel` does not preempt, so the bridge's own stale-command watchdog cannot cancel a
motion that is already stopping.

The auxiliary mouth deliberately sits outside the policy observation/action
contract. Call `/locomotion/mouth` (`std_srvs/srv/SetBool`), or use the
no-argument `/locomotion/mouth/open` and `/locomotion/mouth/close` Trigger
services. `/locomotion/jump` provides the equivalent no-argument jump service.
