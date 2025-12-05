# Publishing to PyPI

This guide covers how to publish GPUQueue to PyPI so users can install it with:

```bash
pip install gpuqueue
```

## Prerequisites

### 1. Create PyPI Accounts

1. **PyPI** (production): https://pypi.org/account/register/
2. **TestPyPI** (testing): https://test.pypi.org/account/register/

Enable 2FA on both accounts (required for trusted publishing).

### 2. Configure Trusted Publishing

Trusted publishing allows GitHub Actions to publish without storing API tokens.

#### On PyPI.org:

1. Go to https://pypi.org/manage/account/publishing/
2. Click **"Add a new pending publisher"**
3. Fill in:
   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `gpuqueue` |
   | Owner | `Flameingmoy` |
   | Repository | `GPGPU_in-mem_msg_q` |
   | Workflow name | `release.yml` |
   | Environment | *(leave blank)* |

4. Click **"Add"**

#### On TestPyPI.org:

Repeat the same steps at https://test.pypi.org/manage/account/publishing/

## Release Process

### Step 1: Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.1.0"  # For stable release
# or
version = "0.1.0a1"  # For alpha
version = "0.1.0b1"  # For beta
version = "0.1.0rc1" # For release candidate
```

### Step 2: Update Changelog

Create/update `CHANGELOG.md` with release notes.

### Step 3: Commit Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release v0.1.0"
git push origin main
```

### Step 4: Create Release Tag

```bash
# For pre-release (goes to TestPyPI)
git tag -a v0.1.0a1 -m "Alpha release 0.1.0a1"
git push origin v0.1.0a1

# For stable release (goes to PyPI)
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

### Step 5: Monitor GitHub Actions

1. Go to https://github.com/Flameingmoy/GPGPU_in-mem_msg_q/actions
2. Watch the "Release" workflow
3. Check for any build errors

### Step 6: Verify Installation

```bash
# For TestPyPI
pip install --index-url https://test.pypi.org/simple/ gpuqueue

# For PyPI
pip install gpuqueue
```

## Manual Publishing (Alternative)

If you need to publish manually without GitHub Actions:

### 1. Build Locally

```bash
# Install build tools
pip install build twine

# Build sdist and wheel
python -m build

# Check the dist/
ls dist/
# gpuqueue-0.1.0.tar.gz
# gpuqueue-0.1.0-cp310-cp310-linux_x86_64.whl
```

### 2. Upload to TestPyPI

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ gpuqueue
```

### 3. Upload to PyPI

```bash
# Upload to PyPI (production)
twine upload dist/*
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

| Version | Meaning | PyPI Behavior |
|---------|---------|---------------|
| `0.1.0a1` | Alpha (unstable) | Marked as pre-release |
| `0.1.0b1` | Beta (feature complete) | Marked as pre-release |
| `0.1.0rc1` | Release candidate | Marked as pre-release |
| `0.1.0` | Stable release | Default install |

Pre-releases are not installed by default:
```bash
pip install gpuqueue          # Gets latest stable
pip install gpuqueue==0.1.0a1 # Gets specific pre-release
pip install --pre gpuqueue    # Gets latest including pre-releases
```

## Troubleshooting

### "Project name already exists"

Someone else has `gpuqueue` on PyPI. Options:
1. Use a different name (e.g., `cuda-gpuqueue`, `gpuqueue-cuda`)
2. Contact PyPI support if you believe it's name-squatting

### Build fails on GitHub Actions

1. Check if CUDA toolkit is installed correctly
2. Verify cibuildwheel configuration
3. Check CMake/pybind11 compatibility

### Wheel not compatible

CUDA wheels are platform-specific. Currently only building for:
- Linux x86_64
- Python 3.10, 3.11, 3.12

For other platforms, users must build from source.

## Checklist Before Release

- [ ] All tests pass locally
- [ ] CI is green on main branch
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] README.md is accurate
- [ ] Documentation is up to date
- [ ] Trusted publishing configured on PyPI/TestPyPI
