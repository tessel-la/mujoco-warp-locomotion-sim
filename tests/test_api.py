from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from mujoco_warp_locomotion_sim.api import create_app
from mujoco_warp_locomotion_sim.config import load_settings
from mujoco_warp_locomotion_sim.runtime import SimulationRuntime

ROOT = Path(__file__).resolve().parents[1]


def mock_settings():
    settings = load_settings(ROOT / "config/default.toml").with_overrides(backend="mock")
    return replace(
        settings,
        simulation=replace(settings.simulation, realtime=True, state_hz=100.0, render_hz=30.0),
    )


async def wait_ready(client: AsyncClient) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if (await client.get("/health")).json()["ready"]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError((await client.get("/health")).json())


def run_api_test(callback) -> None:
    async def run() -> None:
        settings = mock_settings()
        runtime = SimulationRuntime(settings)
        runtime.start()
        try:
            transport = ASGITransport(app=create_app(settings, runtime))
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await wait_ready(client)
                await callback(client)
        finally:
            runtime.stop()

    asyncio.run(run())


def test_api_command_state_reset_enable_and_description() -> None:
    async def assertions(client: AsyncClient) -> None:
        description = (await client.get("/api/v1/robot/description")).json()
        assert description["command_topic"] == "/cmd_vel"
        assert description["policy_command_topic"] == "/locomotion/policy_cmd"
        assert description["policy_enabled"] is True
        assert len(description["joint_names"]) == 15
        assert len(description["policy_joint_names"]) == 14
        assert description["motion_action"] == "/locomotion/execute_motion"

        response = await client.post(
            "/api/v1/command",
            json={"linear_x": 4.0, "linear_y": -4.0, "angular_z": 4.0},
        )
        assert response.status_code == 200
        assert response.json()["command"] == {
            "linear_x": 0.45,
            "linear_y": -0.2,
            "angular_z": 1.2,
        }
        assert (await client.get("/api/v1/state")).status_code == 200
        assert (await client.post("/api/v1/enable", json={"enabled": False})).json() == {
            "enabled": False
        }
        assert (await client.post("/api/v1/reset", json={})).status_code == 202
        policy = await client.post("/api/v1/policy", json={"mode": "sit"})
        assert policy.status_code == 202
        assert policy.json() == {"accepted": True, "mode": "sit"}
        mouth = await client.post("/api/v1/mouth", json={"opened": True})
        assert mouth.status_code == 202
        assert mouth.json() == {"accepted": True, "opened": True}
        jump = await client.post("/api/v1/jump", json={})
        assert jump.status_code == 202
        assert jump.json() == {"accepted": True}

    run_api_test(assertions)


def test_urdf_etag_camera_and_validation() -> None:
    async def assertions(client: AsyncClient) -> None:
        urdf = await client.get("/robot_description")
        assert urdf.status_code == 200
        assert urdf.headers["content-type"].startswith("application/xml")
        cached = await client.get(
            "/robot_description", headers={"If-None-Match": urdf.headers["etag"]}
        )
        assert cached.status_code == 304

        deadline = time.monotonic() + 2.0
        camera = await client.get("/api/v1/camera/frame.jpg")
        while camera.status_code == 503 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            camera = await client.get("/api/v1/camera/frame.jpg")
        assert camera.status_code == 200
        assert camera.content.startswith(b"\xff\xd8")
        wrong_stream = await client.get(
            "/video_stream/stream",
            params={"topic": "/wrong", "type": "mjpeg"},
        )
        assert wrong_stream.status_code == 404
        invalid = await client.post("/api/v1/command", json={"linear_x": "NaN"})
        assert invalid.status_code == 422

    run_api_test(assertions)
