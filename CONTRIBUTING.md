# Contributing to Quatham

Thank you for your interest in contributing to Quatham! This document provides guidelines and setup instructions.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jpmorr/quatham.git
cd quatham
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests
uv run pytest

# Check version
uv run python -c "import quatham; print(quatham.__version__)"
```

---

## Development Workflow

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_algebra.py

# Specific test
uv run pytest tests/test_algebra.py::TestQuaternionAlgebra::test_identity_and_conjugate

# With coverage
uv run pytest --cov=src/quatham --cov-report=term-missing
```

### Code Quality

```bash
# Lint with ruff
uv run ruff check src/ tests/

# Format with ruff
uv run ruff format src/ tests/
```

### Building Documentation

```bash
# Build docs
uv run zensical build

# Serve docs locally
uv run zensical serve
```

---

## Version Management

Quatham uses **hatch-vcs** for automatic version management from Git tags.

### Creating a Release

```bash
# 1. Create and push tag
git tag -a 1.0.0 -m "Release version 1.0.0"
git push origin 1.0.0

# 2. GitHub Actions automatically:
#    - Builds the package
#    - Publishes to PyPI
```

### Pre-release Versions

```bash
# Release candidate
git tag -a 1.0.0rc1 -m "Release candidate 1"
git push origin 1.0.0rc1

# Beta
git tag -a 1.0.0b1 -m "Beta 1"
git push origin 1.0.0b1
```

### Check Version

```bash
# From Python
uv run python -c "import quatham; print(quatham.__version__)"

# From hatch
uv run hatch version
```

**See [HATCH_VCS_SETUP.md](HATCH_VCS_SETUP.md) for detailed version management.**

---

## Code Coverage

Quatham tracks code coverage with Codecov.

### Generate Coverage Report

```bash
# Run tests with coverage
uv run pytest --cov=src/quatham --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Targets

- **Overall:** 90%+
- **Core algebra:** 95%+
- **Conversions:** 90%+

### Codecov Integration

Coverage reports are automatically uploaded to Codecov on every push:
- **Coverage:** https://app.codecov.io/gh/jpmorr/quatham
- **Test results:** Available in Codecov Tests tab

---

## Project Structure

```
quatham/
├── src/quatham/          # Package source
│   ├── Quaternion.py     # Main Quaternion class
│   ├── utilities.py      # Numba-jitted functions
│   ├── __init__.py       # Package exports
│   └── __version__.py    # Version (fallback only)
├── tests/                # Test suite
│   ├── test_algebra.py
│   ├── test_conversions.py
│   └── test_utilities.py
├── docs/                 # Documentation
│   ├── getting-started/
│   ├── tutorials/
│   └── reference/
├── .github/workflows/    # CI/CD
│   ├── tests.yml
│   ├── release.yml
│   └── docs.yml
└── pyproject.toml        # Project configuration
```

---

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow existing code style
- Add tests for new functionality
- Update docstrings (NumPy style)
- Update documentation if needed

### 3. Test Locally

```bash
# Run all tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

### 4. Commit and Push

```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to https://github.com/jpmorr/quatham
- Click "New Pull Request"
- Describe your changes
- Link any related issues

---

## Documentation Guidelines

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description.
    
    Extended description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    type
        Description of return value
    
    Examples
    --------
    >>> result = my_function(1, 2)
    >>> print(result)
    3
    """
```

### Documentation Files

- **Getting Started:** `docs/getting-started/`
- **Tutorials:** `docs/tutorials/`
- **API Reference:** `docs/reference/`

Documentation is built with Zensical (successor to MkDocs Material).

---

## Release Process

Releases are automated via GitHub Actions when you push a tag:

1. **Tag the release** → `git tag -a 1.0.0 -m "Release"`
2. **Push tag** → `git push origin 1.0.0`
3. **GitHub Actions** builds and publishes to PyPI
4. **Documentation** is automatically deployed

**See [docs/RELEASE_WORKFLOW.md](docs/RELEASE_WORKFLOW.md) for details.**

---

## Questions?

- **Issues:** https://github.com/jpmorr/quatham/issues
- **Discussions:** https://github.com/jpmorr/quatham/discussions
- **Documentation:** https://jpmorr.github.io/quatham/

---

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.
