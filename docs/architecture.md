# Architecture

```text
Robo Boy gamepad / ROS client
            │
            ▼
  rosbridge ── /cmd_vel ──> ROS 2 sidecar ── HTTP command ──┐
                                                            ▼
                                                 SimulationRuntime thread
                                                            │
                                         ┌──────────────────┴──────────────┐
                                         ▼                                 ▼
                             RL policy adapter                     MuJoCo Warp step
                                                                         │
                                     ┌───────────────────────────────────┤
                                     ▼                                   ▼
                              JSON/WS state                       head-camera JPEG
                                     │                                   │
                                     └──────────── ROS sidecar ───────────┘
                                                        │
                     /joint_states, /odom, /imu, /tf, /clock, camera, diagnostics
                                                        │
                                                        ▼
                                                    Robo Boy
```

Only `SimulationRuntime`'s worker thread touches the physics backend or
renderer. API handlers update a locked latest-command value; readers consume
immutable state and camera snapshots. Slow clients skip frames rather than
growing a queue.

The command watchdog exists at two boundaries. The ROS sidecar sends zero after
an interrupted publisher, while the runtime independently substitutes zero for
stale HTTP commands. Disabling locomotion also clears the current command.

`MujocoWarpBackend` maintains one MJWarp world because it is the
operator-facing digital twin. Batched PPO training lives in the isolated
`microduck_rl_train` image and uses Pollen's unchanged mjlab environments,
thousands of worlds, BAM actuators, and domain randomization. This separation
also allows the live runtime and training stack to pin different compatible
MuJoCo Warp releases without changing the host NVIDIA installation.

The live runtime validates and warms Pollen's seven official ONNX policies,
builds their shared 61-value observation, and writes each 14-value action to
the simulated joint targets at 50 Hz. These policies receive no virtual base,
height, or posture assistance. A bounded PD reference gait remains available
only when RL is explicitly disabled for diagnostics.

Operator actions do not widen that learned contract. The mouth target is
appended as an auxiliary fifteenth actuator only when controls are copied into
MuJoCo Warp. Jump writes a single bounded takeoff velocity into the simulated
free base and is labeled as a simulator primitive by the HTTP, ROS action, and
panel documentation.
