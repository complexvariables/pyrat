# PyPI Deployment Guide

This guide explains how to upload the pyrat package to PyPI.

## Prerequisites

1. Create a PyPI account at https://pypi.org/account/register/
2. Set up API token at https://pypi.org/manage/account/token/
3. Store your API token securely

## Upload to TestPyPI (Recommended First)

Test your package on TestPyPI before uploading to the main PyPI:

```bash
# Upload to TestPyPI
pixi run twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyrat
```

## Upload to PyPI

Once you've verified the package works on TestPyPI:

```bash
# Upload to PyPI
pixi run twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

## Using Environment Variables (Recommended)

To avoid entering credentials each time, set environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your-api-token-here

# Then upload
pixi run twine upload dist/*
```

## Using .pypirc Configuration File

Alternatively, create a `~/.pypirc` file:

```ini
[pypi]
username = __token__
password = your-pypi-api-token

[testpypi]
username = __token__
password = your-testpypi-api-token
```

Then upload with:
```bash
pixi run twine upload dist/*
```

## Verify Upload

After uploading, verify your package at:
- PyPI: https://pypi.org/project/pyrat/
- TestPyPI: https://test.pypi.org/project/pyrat/

## Updating the Package

To release a new version:

1. Update the version in [`pyproject.toml`](pyproject.toml:3)
2. Rebuild the package:
   ```bash
   rm -rf dist/
   pixi run python -m hatchling build
   ```
3. Upload the new version:
   ```bash
   pixi run twine upload dist/*
   ```

## Security Notes

- Never commit API tokens to version control
- Use API tokens instead of passwords
- Consider using GitHub Actions for automated releases
- Add `dist/` to [`.gitignore`](.gitignore:1) (already done)
