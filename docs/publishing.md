# Publishing

This project is set up for PyPI releases through GitHub Actions trusted
publishing and Read the Docs builds through `.readthedocs.yaml`.

## Before the first public release

Confirm repository and package metadata:

- GitHub repository: `pr1m8/ooai-llm`
- Read the Docs project slug: `ooai-llm`
- package metadata such as author or project URLs if ownership changes

Create the PyPI project and configure a trusted publisher for:

- owner: your GitHub organization or user
- repository: `ooai-llm`
- workflow: `release.yml`
- environment: `pypi`

The release workflow already requests the `id-token: write` permission required
for trusted publishing.

## Release checklist

Run the local checks:

```bash
pdm run pytest
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
pdm build
pdm run twine check dist/*
```

Update `pyproject.toml` and `docs/changelog.md` for the target version, then tag
the release:

```bash
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0
```

Pushing a `v*` tag starts `.github/workflows/release.yml`, builds the source
distribution and wheel, checks them with Twine, and publishes to PyPI from the
`pypi` environment.

## Read the Docs

Import the GitHub repository in Read the Docs and point it at
`.readthedocs.yaml`. The config uses:

- Ubuntu 24.04
- Python 3.13
- Sphinx configuration from `docs/conf.py`
- `pip install .[docs]`
- `fail_on_warning: true`

Local docs builds should use the same warning behavior:

```bash
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
```
