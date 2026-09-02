from __future__ import annotations

import argparse
import json
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
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    doctor = subparsers.add_parser("doctor", help="initialize and step a backend")
    doctor.add_argument("--config")
    doctor.add_argument("--backend", choices=("mujoco_warp", "mock"), default="mock")
    doctor.add_argument("--device")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = load_settings(args.config).with_overrides(backend=args.backend, device=args.device)
    if args.command == "serve":
        uvicorn.run(
            create_app(settings),
            host=args.host or settings.server.host,
            port=args.port or settings.server.port,
        )
        return

    runtime = SimulationRuntime(settings)
    runtime.start()
    deadline = time.monotonic() + 30.0
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
