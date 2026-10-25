# Releasing

`dev` integrates topic PRs; `main` receives reviewed release promotions. release-please
owns version changes, tags, and `CHANGELOG.md`. The manifest file seeds the current source
version; it does not assert that a release with that version has already been published.

## First release setup

1. Review and merge the readiness topic through `dev`, preserving existing default-branch work. The prepared local `dev` starts at the current `main`; push it before enabling the workflows.
2. Enable GitHub Actions and allow Actions to create pull requests. The built-in token handles release PRs and tags; `RELEASE_PLEASE_TOKEN` is optional for deployments that require a separate bot token. Never commit one.
3. Configure protection/review for `dev` and `main`. Allow the sync workflow to merge `main` back into `dev`, or resolve a protected-branch rejection through review. Do not force-push.
4. Open a promotion from `dev` to `main` with a Conventional Commit title. Merge it with a merge commit. Review the release-please PR, then merge it to create the versioned tag and release.
5. Run the publication checks below before making any catalog entry live. Confirm repository history has been scanned for secrets and third-party notices are present before making source public.

While the project is on `0.x`, describe incompatible changes in the PR body and release notes
without `!` or `BREAKING CHANGE:` markers that would force a `1.0.0` bump.

## Dependency updates

Use uv 0.9.5 and Python 3.12. Update `uv.lock` intentionally, export the Docker requirements,
and commit both the lock and generated exports. Docker requires their hashes.

```bash
python3.12 scripts/lock.py
```

The helper preserves the root project version marker in `uv.lock`; release-please uses
it to bump the editable project metadata without changing dependency versions. CI checks
the lock and exports. Do not remove that marker.

Go2's PyTorch wheel channel/version is selected separately in its Dockerfile and GPU
overlays because CPU, CUDA, and ROCm have different wheel availability. It is deliberately
not replaced by the general PyPI resolution. GPU driver compatibility must be checked
on the intended host. Microduck training retains its independent vendored lockfile.

## Publication checks

Build the CPU or mock image from a clean checkout, check health and command expiry, then
connect the matching panel through Robo-Boy in an isolated domain. Exercise relevant
custom action overlays only for action examples. Verify existing deployment defaults
and stop/restart behavior before release. Never run motion checks against a live robot.

## Limitations

Source visibility changes do not publish runnable panel artifacts or update the inventory.
GitHub Actions must be enabled and the release/configuration branches must exist remotely.
The final anonymous download and installer checks require public release assets.
