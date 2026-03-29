# Contributing to scAtlasTb-utils

Thank you for contributing! This document explains how to contribute and how our release process works.

## Development setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mumichae/scAtlasTb-utils.git
cd scAtlasTb-utils
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Commit message conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) to automate changelog generation and version bumping via [Release Please](https://github.com/googleapis/release-please).

Please format **all commit messages** (or at minimum **PR titles**) as:

```
<type>(<optional scope>): <short summary>
```

### Common types

| Type       | Description                                    | Changelog section           |
| ---------- | ---------------------------------------------- | --------------------------- |
| `feat`     | A new feature                                  | Features                    |
| `fix`      | A bug fix                                      | Bug Fixes                   |
| `perf`     | A performance improvement                      | Performance Improvements    |
| `docs`     | Documentation only changes                     | Documentation               |
| `revert`   | Reverts a previous commit                      | Reverts                     |
| `refactor` | A code change that is neither fix nor feature  | *(hidden)*                  |
| `test`     | Adding or updating tests                       | *(hidden)*                  |
| `build`    | Changes to the build system or dependencies    | *(hidden)*                  |
| `ci`       | Changes to CI configuration files              | *(hidden)*                  |
| `chore`    | Other changes that don't modify src or tests   | *(hidden)*                  |
| `style`    | Changes that do not affect the meaning of code | *(hidden)*                  |

**Breaking changes** should include `BREAKING CHANGE:` in the commit body or use `!` after the type:

```
feat!: remove deprecated function xyz
```

### Examples

```
feat: add majority voting cluster annotation
fix: handle missing barcode in obs match function
docs: update installation instructions
chore: bump dependency versions
```

## Release process

Releases are fully automated via [Release Please](https://github.com/googleapis/release-please):

1. Commits to `main` following Conventional Commits are analyzed automatically.
2. Release Please opens (or updates) a **Release PR** that:
   - Bumps the version in `pyproject.toml`.
   - Updates `CHANGELOG.md` with all changes since the last release.
3. When the Release PR is merged, Release Please:
   - Creates a **GitHub Release** with the changelog as release notes.
   - Creates a **git tag** (`vX.Y.Z`).
4. The `release.yaml` workflow then publishes the package to PyPI.

> **Note:** You do not need to manually edit `CHANGELOG.md` or `pyproject.toml` version fields. Release Please handles all of that.
