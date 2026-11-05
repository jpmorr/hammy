# AGENTS.md

## Project Overview
- **Package**: `quatham` (import as `hammy`) — NumPy/Numba-accelerated quaternion mathematics for 3D rotations
- **Structure**: Single-package layout under `src/quatham/`
- **Python**: 3.12+
- **Package manager**: `uv` (lockfile: `uv.lock`)

## Developer Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=hammy

# Run a single test file
uv run pytest tests/test_algebra.py

# Run a specific test
uv run pytest tests/test_algebra.py::TestQuaternionAlgebra::test_identity_and_conjugate

# Lint with ruff
uv run ruff check src/ tests/

# Format with ruff
uv run ruff format src/ tests/
```

## Architecture Notes

### Source Files
- `src/quatham/Quaternion.py` — Main `Quaternion` class with algebraic operations and conversions
- `src/quatham/utilities.py` — Low-level Numba-jitted math functions (`@njit(cache=True)`)
- `src/quatham/__init__.py` — Exports `Quaternion` class

### Test Files
- `tests/test_algebra.py` — Quaternion algebra, normalization, arithmetic operations
- `tests/test_conversions.py` — Euler/Tait-Bryan angle conversions, rotation matrices, axis-angle

### Key Conventions (critical for correctness)
- **Scalar-first**: Quaternions use `[w, x, y, z]` ordering throughout
- **Intrinsic rotations**: Default for `from_euler()`, `to_euler()`, `to_tait_bryan()`
- **Double-cover SO(3)**: `q` and `-q` represent identical spatial rotations
- **Batch support**: All operations handle single quaternions `(4,)` and arrays `(N, 4)`

### Numba Functions
- Utilities use `@njit(cache=True)` — changes may require clearing `__pycache__` or `.ruff_cache/` to see effects during development


## Lint/Format Rules (ruff)
Maintain PEP 8 compliance. Recommended tools:

- **Linting:** `ruff check dice.py`
- **Formatting:** `ruff format dice.py`
- Line length: 120
- Select: E, F, UP, B, SIM, I, D (pydocstyle numpy convention)
- Ignore: D103 (missing docstring in __init__)

## Code Style Guidelines
### General Principles
- **Simplicity:** Keep the logic straightforward. Avoid over-engineering for a single-file utility.
- **Readability:** Prioritize clear, descriptive names over brevity.

### Naming Conventions
- **Functions:** Use `snake_case` (e.g., `generate_dice_faces_diagram`).
- **Variables:** Use `snake_case` (e.g., `roll_results`).
- **Constants:** Use `SCREAMING_SNAKE_CASE` (e.g., `DICE_ART`, `DIE_HEIGHT`).
- **Internal Helpers:** Prefix with a single underscore (e.g., `_get_dice_faces`).

### Imports
- Place standard library imports at the top of the file.
- Keep them alphabetised within their group.
- Example:
  ```python
  import random
  import sys
  ```

### Documentation Guidelines
- Use triple-quoted docstrings for all functions.
- The first line should be a concise summary.
- Follow with a more detailed explanation if the function's logic or parameters aren't trivial.
- Use NumPy style docstrings. Add details for extended summary, parameters, yields, returns and examples where appropriate.
- Example:
  ```python
  def roll_dice(num_dice):
      """Return a list of random integers between 1 and 6.
      
      Parameters
      ----------
      num_dice : int
          Number of dice to be rolled.
      """
      # Implementation
  ```

### Language
- Use British English spelling throughout the codebase.
- Applies to: comments, docstrings, documentation, error messages, and variable names.
- Examples: `colour` not `color`, `analyse` not `analyze`, `behaviour` not `behavior`.

## Git Workflow
- Do not commit, push, merge, or force-push without approval.
- Do not include unrelated formatting changes.
- Keep migrations and generated files in the same change as the code that requires them.
- Summarise user-visible behaviour and validation in the pull request.

## Python Guidelines
- Follow PEP 8 style conventions
- Add type hints to all new and modified functions
- Prefer specific exception types over bare `except` clauses
- Use context managers for file I/O and database connections
- Write descriptive error messages that include the offending value
- Keep user comments when refactoring or modifying code. Reposition if necessary, but they were put there for a reason.

## CI/CD
- GitLab CI (`.gitlab-ci.yml`): Runs pytest + secret detection
- No GitHub Actions configured
