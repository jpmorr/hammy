# Welcome to Quatham

**Quatham** (import as `hammy`) is a lightweight, high-performance quaternion mathematics package for 3D rotations. Built with NumPy and Numba acceleration, it provides efficient quaternion operations for both single quaternions and batched arrays.

## Quick Navigation

### Getting Started
New to Quatham or quaternions? Start here.
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)
- [Basic Concepts](getting-started/concepts.md)

### Tutorials
Step-by-step guides for common tasks.
- [Creating Quaternions](tutorials/creating-quaternions.md)
- [Quaternion Arithmetic](tutorials/arithmetic.md)
- [Conversions](tutorials/conversions.md)
- [Batch Operations](tutorials/batch-operations.md)

### API Reference
Complete API documentation.
- [Quaternion Class](reference/quaternion.md)
- [Utilities](reference/utilities.md)

## Key Features

- **Numba-accelerated** — JIT-compiled low-level operations for maximum performance
- **Batch processing** — Efficient handling of quaternion arrays
- **Comprehensive conversions** — Euler angles, Tait-Bryan, rotation matrices, axis-angle
- **Intuitive API** — Pythonic interface with operator overloading
- **Scalar-first convention** — Consistent `[w, x, y, z]` ordering

## Example

```python
from hammy import Quaternion
import numpy as np

# Create a quaternion from axis-angle
q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)

# Rotate a vector
v = np.array([1.0, 0.0, 0.0])
v_rotated = q * v  # [0.0, 1.0, 0.0]

# Convert to rotation matrix
R = q.to_rotation_matrix()
```

## Installation

```bash
pip install hammy
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions.
