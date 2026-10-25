# Contributing

Start with the [README](README.md) and [agent guide](AGENTS.md). Use `dev` for integration
and `main` for releases. Topic PRs target `dev` and are squash merged; promotion PRs
from `dev` to `main` keep a merge commit. Prefix a branch by its purpose, such as
`feat/` or `fix/`, and use a Conventional Commit PR title with a lowercase subject.

Describe the behavior you change, compatibility or deployment effects, limitations, and
checks you ran. Include related docs and focused regression tests in the same change.
Do not include secrets, build caches, dependency directories, or tool-attribution footers.

## Verify

```bash
uv sync --locked --extra dev --python 3.12
python3 tools/download_policies.py
uv run --frozen --extra dev pytest
uv run --frozen --extra dev ruff check .
```

See [releasing](docs/releasing.md) for dependency updates and publication. Use an isolated
simulator/domain for motion checks. Keep license notices for reused code, models, and policies.

## Limitations

Unit tests cannot establish browser accessibility, GPU compatibility, or physical robot safety.
State which environments you exercised and which you could not test.
