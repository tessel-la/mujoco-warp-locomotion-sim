from __future__ import annotations

from typing import Any, Protocol


class LocomotionBackend(Protocol):
    """A physics backend driving one Microduck.

    `state()` is the only window anything outside the simulation thread has into the
    robot, so long-running actions read their progress from it. Beyond the pose,
    twist, and joint values, every backend must report two lifecycle records so a
    caller can tell *when a requested behaviour has actually finished* rather than
    guessing from mode strings:

    ``state()["policy"]`` carries, alongside its existing fields:

    - ``episode``: increments on every accepted `set_policy`, never resets
    - ``request``: the mode name that episode asked for
    - ``kind``: ``persistent`` | ``transient`` | ``oneshot`` (see
      `policy_controller.MODE_KINDS`)
    - ``phase``: ``running`` | ``complete`` | ``failed``

    ``state()["jump"]`` carries ``episode`` and ``phase`` with the same meaning for
    the non-RL jump primitive. `JumpEpisode` implements that record.
    """

    name: str
    device: str

    def initialize(self) -> None: ...

    def set_command(self, linear_x: float, linear_y: float, angular_z: float) -> None: ...

    def set_enabled(self, enabled: bool) -> None: ...

    def set_policy(self, mode: str) -> str: ...

    def set_mouth(self, opened: bool) -> None: ...

    def jump(self) -> None: ...

    def reset(self) -> None: ...

    def step(self) -> None: ...

    def state(self) -> dict[str, Any]: ...

    def jpeg_frame(self) -> bytes: ...

    def close(self) -> None: ...


class JumpEpisode:
    """Lifecycle record for the simulator's jump primitive.

    Jump is not an RL policy and so has no policy episode, but it still takes time --
    a takeoff, a flight, and a landing. Anything waiting on it needs the same
    observable contract the policies expose, including the case where the takeoff is
    refused because the robot is already off the ground.
    """

    def __init__(self) -> None:
        self.episode = 0
        self.phase = "complete"
        self._airborne = False

    def request(self) -> None:
        self.episode += 1
        self.phase = "running"
        self._airborne = False

    def refuse(self) -> None:
        """The takeoff gate rejected this jump, so it will never leave the ground."""
        if self.phase == "running":
            self.phase = "failed"

    def abandon(self) -> None:
        """A reset or a disable dropped the jump before it landed."""
        if self.phase == "running":
            self.phase = "failed"

    def observe(self, airborne: bool) -> None:
        """Advance one step. The jump finishes on the first landing after a takeoff."""
        if self.phase != "running":
            return
        if airborne:
            self._airborne = True
        elif self._airborne:
            self.phase = "complete"

    def status(self) -> dict[str, Any]:
        return {"episode": self.episode, "phase": self.phase}
