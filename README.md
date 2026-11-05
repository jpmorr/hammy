# Hammy

[![PyPI version](https://badge.fury.io/py/hammy.svg)](https://pypi.org/project/hammy/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation](https://img.shields.io/badge/docs-zensical-blue.svg)](https://jpmorr.github.io/hammy)

[![Tests](https://github.com/jpmorr/hammy/actions/workflows/tests.yml/badge.svg)](https://github.com/jpmorr/hammy/actions)
[![Coverage](https://codecov.io/gh/jpmorr/hammy/branch/main/graph/badge.svg)](https://codecov.io/gh/jpmorr/hammy)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Hammy** is a lightweight, high-performance quaternion mathematics package for 3D rotations. Built with NumPy and Numba acceleration, it provides efficient quaternion operations for both single quaternions and batched arrays.

## Features

- **Numba-accelerated operations** — Low-level quaternion math functions are JIT-compiled for maximum performance
- **Batch processing** — All operations support both single quaternions and arrays of quaternions
- **Comprehensive conversions** — Convert between quaternions and:
  - Euler angles (proper/classic, 6 sequences)
  - Tait-Bryan angles (Cardan/nautical, 6 sequences including yaw-pitch-roll)
  - Rotation matrices
  - Axis-angle representations
  - Spherical coordinates
- **Intuitive API** — Clean, Pythonic interface with operator overloading
- **Scalar-first convention** — Consistent `[w, x, y, z]` ordering throughout

## Installation

```bash
pip install hammy
```

### Requirements

- Python 3.12+
- NumPy >= 2.0
- Numba >= 0.60

## Quick Start

### Basic Usage

```python
from hammy import Quaternion
import numpy as np

# Create a quaternion from components
q = Quaternion([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

# Create from Euler angles (intrinsic rotations, radians by default)
q = Quaternion.from_euler([np.pi/4, 0, 0], seq='xyz')

# Create from axis-angle representation
axis = np.array([0, 0, 1])  # Z-axis
angle = np.pi / 2  # 90 degrees
q = Quaternion.from_axisangle(axis, angle)

# Create from axis-angle in degrees
q = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)
```

### Quaternion Operations

```python
# Quaternion multiplication (Hamilton product)
q1 = Quaternion([1.0, 0.0, 0.0, 0.0])
q2 = Quaternion.from_euler([np.pi/4, 0, 0], seq='xyz')
q3 = q1 * q2

# Scalar multiplication
q_scaled = q1 * 2.0

# Quaternion division
q_div = q1 / q2

# Conjugate
q_conj = q1.conjugate

# Normalise to unit quaternion
q_norm = q1.normalise()

# Magnitude
magnitude = q1.magnitude
```

### Vector Rotation

```python
# Rotate a 3D vector using quaternion
q = Quaternion.from_axisangle([0, 0, 1], np.pi / 2)
v = np.array([1.0, 0.0, 0.0])
v_rotated = q * v  # Results in [0.0, 1.0, 0.0]
```

### Conversions

```python
# To rotation matrix (3x3)
R = q.to_rotation_matrix()

# To flat rotation matrix (9 elements)
R_flat = q.get_rotation_matrix_flat()

# To axis-angle representation
axis, angle = q.to_axis_angle()
axis, angle_deg = q.to_axis_angle(degrees=True)

# To Euler angles (proper, e.g., 'zxz' sequence)
euler = q.to_euler(seq='zxz', degrees=True)

# To Tait-Bryan angles (yaw-pitch-roll, intrinsic z-y-x)
ypr = q.to_yaw_pitch_roll(degrees=True)

# To Tait-Bryan with custom sequence
tait_bryan = q.to_tait_bryan(seq='xyz', degrees=True, intrinsic=True)
```

### Batch Operations

Hammy efficiently handles arrays of quaternions:

```python
# Create multiple quaternions at once
quats = Quaternion([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

# Batch Euler angle conversion
angles = np.array([
    [np.pi/4, 0, 0],
    [0, np.pi/3, 0],
    [0, 0, np.pi/6],
])
q_batch = Quaternion.from_euler(angles, seq='xyz')

# Batch vector rotation
v = np.array([1.0, 0.0, 0.0])
rotated = q_batch * v  # Shape: (3, 3)

# Access individual quaternions
q_first = Quaternion(q_batch[0])
```

## Supported Rotation Sequences

### Proper Euler Angles (2 distinct axes)
- `xzx`, `xyx`, `yxy`, `yzy`, `zxz`, `zyz`

### Tait-Bryan Angles (3 distinct axes)
- `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx`

Both intrinsic (local/body axes) and extrinsic (fixed/global axes) conventions are supported.

## API Reference

### Quaternion Class

#### Constructors
- `Quaternion(value)` — Create from array-like `[w, x, y, z]` or `[[w, x, y, z], ...]`
- `Quaternion.from_euler(angles, seq, degrees, intrinsic)` — From Euler/Tait-Bryan angles
- `Quaternion.from_axisangle(v, theta, degrees)` — From axis-angle representation
- `Quaternion.from_spherical(theta, phi, degrees)` — From spherical coordinates

#### Properties
- `scalar` / `w` — Real (scalar) component
- `vector` — Imaginary (vector) component `[x, y, z]`
- `x`, `y`, `z` — Individual vector components
- `conjugate` — Conjugate quaternion (negated vector part)
- `magnitude` — Euclidean norm
- `num_quaternions` — Number of quaternions stored

#### Methods
- `normalise()` / `normalize()` — Return unit quaternion
- `to_rotation_matrix()` — Convert to 3×3 rotation matrix
- `get_rotation_matrix_flat()` — Convert to flat 9-element array
- `to_axis_angle(degrees)` — Convert to axis-angle representation
- `get_rotation_vector()` — Get rotation vector (axis × angle)
- `to_euler(seq, degrees, intrinsic)` — Convert to proper Euler angles
- `to_tait_bryan(seq, degrees, intrinsic)` — Convert to Tait-Bryan angles
- `to_yaw_pitch_roll(degrees)` — Convert to yaw-pitch-roll (zyx sequence)
- `vector_norm()` — L2 norm of vector component
- `tolist()` — Convert to Python list

#### Operators
- `+`, `-` — Addition/subtraction with quaternions or arrays
- `*` — Quaternion multiplication, scalar multiplication, or vector rotation
- `/` — Quaternion division or scalar division
- `-q` — Negation
- `q[idx]` — Index/slice access

## Conventions

- **Scalar-first**: Quaternions use `[w, x, y, z]` ordering
- **Intrinsic rotations**: Default for `from_euler()`, `to_euler()`, `to_tait_bryan()`
- **Double-cover SO(3)**: `q` and `-q` represent identical spatial rotations
- **Right-handed coordinate system**: Standard 3D rotation convention

## Performance

Hammy uses Numba JIT compilation for core mathematical operations, providing near-C performance for batch operations. For optimal performance:

1. Use batch operations instead of loops when possible
2. Ensure arrays are contiguous in memory
3. The first call includes JIT compilation overhead; subsequent calls are fast

## Examples

See the [documentation](https://jpmorr.github.io/hammy/) for more detailed examples and tutorials.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests on [GitHub](https://github.com/jpmorr/hammy).

## License

Distributed under the BSD-3-Clause License. See `LICENSE` for more information.

## Citation

If you use Hammy in your research, please cite:

```bibtex
@software{hammy2026,
  author = {Morrissey, JP},
  title = {Hammy: NumPy/Numba-accelerated quaternion mathematics},
  year = {2026},
  url = {https://github.com/jpmorr/hammy}
}
```
