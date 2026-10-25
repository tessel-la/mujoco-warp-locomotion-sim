# mujoco-warp-locomotion-sim: agent guide

Read [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), and the guide for the area you change.
Maintainers with a sibling `../tessella-handbook/` checkout also read its `AGENTS.md` and routed pages.
Public contributions do not require access to that private handbook; the rules below apply independently.

## Workflow

- Integration branch: `dev`; release branch: `main`. Create topic branches from `dev` and target PRs there. Before the first remote `dev` exists, maintainers establish it from `main`.
- Use `feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `test/`, `perf/`, or `refactor/` prefixes. No `codex/` or other tool prefixes.
- Use Conventional Commits with lowercase subjects. Do not add tool-attribution footers or `Co-Authored-By` trailers.
- Do not commit, push, open/merge PRs, publish, or deploy unless the user authorized that action. Preserve existing work.
- Inspect types, runtime, tests, and configuration before editing. Update matching docs with behavior changes.
- Never commit secrets, weaken validation, or change a trust boundary to make a check pass.

## Boundaries

Keep ROS 2 Jazzy interfaces and shared DDS discovery compatible with Robo-Boy.
Use an isolated ROS domain for tests. Never run motion smoke tests against physical hardware.
Keep command watchdogs, loopback HTTP defaults, and upstream asset/policy notices intact.

## Checks

Run from the repository root; the README explains prerequisites and complete runtime setup.

```bash
uv sync --locked --extra dev --python 3.12
python3 tools/download_policies.py
uv run --frozen --extra dev pytest
uv run --frozen --extra dev ruff check .
```

Report what changed, exact checks and their results, and anything not verified. Passing unit tests
are not proof of live ROS, GPU, browser, or hardware behavior. Releases follow [docs/releasing.md](docs/releasing.md).
