from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .motion import JUMP, POLICY, RELATIVE_MOVE, MotionSpec
from .policy_controller import MODE_KINDS
from .runtime import SimulationRuntime

# The controller owns the list of modes it can run; this only exposes it.
POLICY_MODES = tuple(MODE_KINDS)
# `jump` is a simulator primitive rather than a trained policy, so it is only
# reachable as a motion, never as a policy mode.
MOTION_NAMES = (*POLICY_MODES, JUMP)


class VelocityCommand(BaseModel):
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0

    @field_validator("linear_x", "linear_y", "angular_z")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("command values must be finite")
        return value


class EnableRequest(BaseModel):
    enabled: bool


class MouthRequest(BaseModel):
    opened: bool


class PolicyRequest(BaseModel):
    mode: Literal[POLICY_MODES]  # type: ignore[valid-type]


class RelativeMoveRequest(BaseModel):
    """Drive to a pose offset in the base frame. Zero means "use the default"."""

    kind: Literal["relative_move"] = "relative_move"
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    max_linear_speed: float = 0.0
    max_angular_speed: float = 0.0
    position_tolerance: float = 0.0
    yaw_tolerance: float = 0.0
    timeout: float = 0.0

    @field_validator("x", "y", "yaw")
    @classmethod
    def finite_offset(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("offsets must be finite")
        return value

    @field_validator(
        "max_linear_speed",
        "max_angular_speed",
        "position_tolerance",
        "yaw_tolerance",
        "timeout",
    )
    @classmethod
    def finite_and_not_negative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("limits must be finite and not negative")
        return value


class PolicyMotionRequest(BaseModel):
    """Run one control policy, or the simulator jump, to its completion condition."""

    kind: Literal["policy"]
    motion: Literal[MOTION_NAMES]  # type: ignore[valid-type]
    timeout: float = 0.0

    @field_validator("timeout")
    @classmethod
    def finite_and_not_negative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("timeout must be finite and not negative")
        return value


MotionRequest = Annotated[
    RelativeMoveRequest | PolicyMotionRequest, Field(discriminator="kind")
]


def create_app(settings: Settings, runtime: SimulationRuntime | None = None) -> FastAPI:
    simulation = runtime or SimulationRuntime(settings)
    urdf = Path(settings.simulation.urdf).read_bytes()
    urdf_hash = hashlib.sha256(urdf).hexdigest()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        simulation.start()
        yield
        simulation.stop()

    app = FastAPI(title="MuJoCo Warp Locomotion Sim", version="0.1.0", lifespan=lifespan)
    app.state.runtime = simulation
    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok"
            if simulation.ready
            else "error"
            if simulation.startup_error
            else "starting",
            "ready": simulation.ready,
            "backend": simulation.backend_name,
            "device": simulation.device_name,
            "error": simulation.startup_error,
        }

    @app.get("/api/v1/robot/description")
    async def robot_description() -> dict[str, Any]:
        return {
            "name": settings.robot.name,
            "base_frame": settings.robot.base_body,
            "joint_names": [*settings.robot.joint_names, "mouth"],
            "policy_joint_names": settings.robot.joint_names,
            "urdf_sha256": urdf_hash,
            "camera_topic": settings.camera.topic,
            "camera_frame": settings.camera.frame,
            "command_topic": "/cmd_vel",
            "odometry_topic": "/odom",
            "policy_command_topic": "/locomotion/policy_cmd",
            "motion_action": "/locomotion/execute_motion",
            "move_relative_action": "/locomotion/move_relative",
            "mouth_service": "/locomotion/mouth",
            "jump_service": "/locomotion/jump",
            "policy_enabled": settings.policy.enabled,
        }

    @app.get("/api/v1/robot/urdf")
    @app.get("/robot_description")
    async def get_urdf(request: Request) -> Response:
        if request.headers.get("if-none-match", "").strip('"') == urdf_hash:
            return Response(status_code=304)
        return Response(urdf, media_type="application/xml", headers={"ETag": f'"{urdf_hash}"'})

    @app.get("/api/v1/state")
    async def state() -> dict[str, Any]:
        snapshot = simulation.latest_state()
        if snapshot is None:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        return snapshot

    @app.post("/api/v1/command")
    async def command(payload: VelocityCommand) -> dict[str, Any]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        return {"accepted": True, "command": simulation.command(**payload.model_dump())}

    @app.post("/api/v1/reset", status_code=202)
    async def reset() -> dict[str, bool]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        simulation.request_reset()
        return {"accepted": True}

    @app.post("/api/v1/enable")
    async def enable(payload: EnableRequest) -> dict[str, bool]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        simulation.set_enabled(payload.enabled)
        return {"enabled": payload.enabled}

    @app.post("/api/v1/policy", status_code=202)
    async def policy(payload: PolicyRequest) -> dict[str, Any]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        return {"accepted": True, "mode": simulation.request_policy(payload.mode)}

    @app.post("/api/v1/mouth", status_code=202)
    async def mouth(payload: MouthRequest) -> dict[str, Any]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        return {"accepted": True, "opened": simulation.request_mouth(payload.opened)}

    @app.post("/api/v1/jump", status_code=202)
    async def jump() -> dict[str, bool]:
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        simulation.request_jump()
        return {"accepted": True}

    @app.post("/api/v1/motion", status_code=202)
    async def start_motion(payload: MotionRequest) -> dict[str, Any]:
        """Take the motion slot, preempting whatever holds it.

        The robot can only do one thing at a time, so there is a single slot and a
        single monotonically increasing id. A caller that started id N and later reads
        a different id knows it was preempted rather than having to guess.
        """
        if not simulation.ready:
            raise HTTPException(503, simulation.startup_error or "simulation is starting")
        fields = payload.model_dump()
        fields["kind"] = RELATIVE_MOVE if fields["kind"] == "relative_move" else POLICY
        return {"accepted": True, "motion": simulation.start_motion(MotionSpec(**fields))}

    @app.get("/api/v1/motion")
    async def motion_status() -> dict[str, Any]:
        return simulation.motion_status()

    @app.post("/api/v1/motion/{motion_id}/cancel")
    async def cancel_motion(motion_id: int) -> dict[str, Any]:
        canceled = simulation.cancel_motion(motion_id)
        return {"canceled": canceled, "motion": simulation.motion_status()}

    @app.get("/api/v1/camera/frame.jpg")
    async def camera_frame() -> Response:
        frame = simulation.latest_frame()
        if frame is None:
            raise HTTPException(503, "camera frame is not ready")
        return Response(
            frame.jpeg, media_type="image/jpeg", headers={"X-Frame-Sequence": str(frame.sequence)}
        )

    async def mjpeg_stream() -> AsyncIterator[bytes]:
        sequence = 0
        while True:
            frame = await asyncio.to_thread(simulation.wait_for_frame, sequence, 5.0)
            if frame is None or frame.sequence <= sequence:
                continue
            sequence = frame.sequence
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame.jpeg)).encode()
                + b"\r\n\r\n"
                + frame.jpeg
                + b"\r\n"
            )

    @app.get("/video_stream/stream")
    async def video_stream(
        topic: Annotated[str, Query()] = "/locomotion/camera/image_raw",
        type: Annotated[str, Query()] = "mjpeg",
    ) -> StreamingResponse:
        if topic != settings.camera.topic or type != "mjpeg":
            raise HTTPException(404, "unknown camera stream")
        return StreamingResponse(
            mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/api/v1/streams/state")
    async def state_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        sequence = 0
        try:
            while True:
                snapshot = await asyncio.to_thread(simulation.wait_for_state, sequence, 5.0)
                if snapshot is None or int(snapshot["sequence"]) <= sequence:
                    continue
                sequence = int(snapshot["sequence"])
                await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
        except WebSocketDisconnect:
            return

    return app
