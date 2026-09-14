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

## Motions and the action lifecycle

A velocity command or a policy switch is instantaneous: it is applied and it is done.
A *motion* is not. "Move forward half a metre" and "play the left kick" both occupy
the robot for a while, can be preempted, can time out, and finish at a moment only the
simulation knows. `motion.py` owns that distinction, and the ROS action servers are
thin adapters over it.

The motion layer lives in the simulator rather than in the ROS sidecar because only
the simulator can arbitrate. A closed loop in the sidecar would run over HTTP, would
have to out-publish the runtime's 500 ms staleness watchdog, and could not see
teleoperation, the enable switch, a reset, or a fall. `MotionController.tick` runs
from `_publish_state`, on the state the loop has just computed at `state_hz`, so
closed-loop control costs no extra backend call, and writes its command back through
the same locked slot teleoperation uses.

It implements no controller of its own. A relative move computes a velocity and hands
it to the trained walking policy through the ordinary command path; a policy motion
asks for the mode and then watches it. What made that second part possible is the
*episode record*: `PolicyController` already runs the state machine that knows when a
one-shot motion hands control back to walking, so it now publishes that as
`episode`/`request`/`kind`/`phase`/`progress`, and `JumpEpisode` does the same for the
non-RL jump. Completion is therefore read, never inferred from mode strings or fixed
delays, and a motion that started and finished between two observations cannot be
missed. `MockBackend` reports the same record, so both backends behave identically.

There is one motion slot and one monotonically increasing id, because the robot can
only do one thing at a time. An id that no longer matches is exactly how a caller
learns it was preempted. Arbitration is explicit: a non-zero operator velocity command,
a direct policy change, or a newer motion preempts the active one; disabling, resetting,
or falling fails it. A *zero* velocity command deliberately does not preempt, because
the sidecar's watchdog posts one after a teleoperation publisher goes quiet.

Operator actions do not widen that learned contract. The mouth target is
appended as an auxiliary fifteenth actuator only when controls are copied into
MuJoCo Warp. Jump writes a single bounded takeoff velocity into the simulated
free base and is labeled as a simulator primitive by the HTTP, ROS action, and
panel documentation.
