from __future__ import annotations

import argparse
import json
import os
import sys
import time

import uvicorn

from .api import create_app
from .config import load_settings
from .runtime import SimulationRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo Warp biped locomotion simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="run the simulator API")
    serve.add_argument("--config")
    serve.add_argument("--backend", choices=("mujoco_warp", "mock"))
    serve.add_argument("--device")
    serve.add_argument("--viewer", action="store_true", help="show the interactive MuJoCo viewer")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    doctor = subparsers.add_parser("doctor", help="initialize and step a backend")
    doctor.add_argument("--config")
    doctor.add_argument("--backend", choices=("mujoco_warp", "mock"), default="mock")
    doctor.add_argument("--device")
    return parser


def main() -> None:
    args = _parser().parse_args()
    # MJWARP_SHOW_VIEWER lets a Compose overlay flip this on without restating `command:`, which
    # would fight with whichever device override is also active.
    show_viewer = getattr(args, "viewer", False) or os.environ.get("MJWARP_SHOW_VIEWER") == "1"
    settings = load_settings(
        args.config,
        backend=args.backend,
        device=args.device,
        show_viewer=True if show_viewer else None,
    )
    if args.command == "serve":
        uvicorn.run(
            create_app(settings),
            host=args.host or settings.server.host,
            port=args.port or settings.server.port,
        )
        if settings.simulation.show_viewer:
            # Verified by direct reproduction: closing mujoco.viewer's GLFW window and the
            # offscreen EGL Renderer cleanly (already done by SimulationRuntime.stop() above,
            # via the FastAPI lifespan) still segfaults during CPython's own native-library
            # teardown afterward. Headless shutdown -- no GLFW context ever created -- exits 0.
            # os._exit skips that teardown instead of fixing it upstream.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return

    runtime = SimulationRuntime(settings)
    runtime.start()
    # Cold CUDA kernel compilation can exceed 30 s; match the image startup grace.
    deadline = time.monotonic() + 180.0
    while not runtime.ready and not runtime.startup_error and time.monotonic() < deadline:
        time.sleep(0.05)
    if not runtime.ready:
        print(runtime.startup_error or "backend initialization timed out", file=sys.stderr)
        runtime.stop()
        raise SystemExit(1)
    runtime.command(0.1, 0.0, 0.0)
    time.sleep(0.1)
    print(json.dumps(runtime.latest_state(), indent=2))
    runtime.stop()
