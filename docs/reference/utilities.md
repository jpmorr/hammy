# Utilities Reference

Low-level Numba-accelerated quaternion functions from `hammy.utilities`.

## Overview

The `utilities` module provides JIT-compiled functions for core quaternion mathematics. These are used internally by the `Quaternion` class but can also be called directly for custom applications.

## Normalisation

### `normalise`

```python
def normalise(quat: NDArray) -> NDArray
```

Normalise quaternion(s) to unit length.

**Parameters:**
- `quat` : Array of shape (4,) or (N, 4)

**Returns:** Normalised array of same shape.

**Example:**
```python
from hammy.utilities import normalise
import numpy as np

q = np.array([2.0, 2.0, 2.0, 2.0])
q_unit = normalise(q)  # [0.5, 0.5, 0.5, 0.5]
```

---

### `np_normalise_1d`

```python
@njit(cache=True)
def np_normalise_1d(vect: NDArray) -> NDArray
```

Normalise a 1D vector (internal use).

---

### `np_normalise_2d`

```python
@njit(cache=True)
def np_normalise_2d(row_array: NDArray) -> NDArray
```

Row-wise normalisation of 2D array (internal use).

---

## Axis-Angle Conversions

### `_axis_angle_to_quat`

```python
def _axis_angle_to_quat(
    vec: NDArray | list | tuple, 
    theta: float | NDArray | list | tuple,
    degrees: bool = False
) -> NDArray
```

Convert axis-angle to quaternion.

**Parameters:**
- `vec` : Rotation axis, shape (3,) or (N, 3)
- `theta` : Rotation angle(s)
- `degrees` : If True, theta is in degrees

**Returns:** Quaternion array of shape (4,) or (N, 4).

---

### `_axis_angle_to_quat_nb`

```python
@njit(cache=True)
def _axis_angle_to_quat_nb(vec: NDArray, theta: NDArray) -> NDArray
```

Numba-jitted axis-angle to quaternion (internal use).

---

### `_quat_to_axis_angle`

```python
def _quat_to_axis_angle(
    q: NDArray | list | tuple, 
    degrees: bool = False
) -> tuple[NDArray, NDArray | float]
```

Convert quaternion to axis-angle.

**Parameters:**
- `q` : Quaternion(s), shape (4,) or (N, 4)
- `degrees` : If True, return angle in degrees

**Returns:** Tuple of (axis, angle).

---

### `_quat_to_axis_angle_nb`

```python
@njit(cache=True)
def _quat_to_axis_angle_nb(qarr: NDArray) -> tuple[NDArray, NDArray]
```

Numba-jitted quaternion to axis-angle (internal use).

---

## Rotation Matrices

### `_calc_rotation_mat`

```python
@njit(cache=True)
def _calc_rotation_mat(qarr: NDArray) -> NDArray
```

Compute rotation matrices from quaternions.

**Parameters:**
- `qarr` : Quaternion array of shape (N, 4)

**Returns:** Rotation matrices of shape (N, 3, 3).

---

## Euler and Tait-Bryan Angles

### `_calc_euler_angles`

```python
@njit(cache=True)
def _calc_euler_angles(qarr: NDArray, seq: str) -> NDArray
```

Compute proper Euler angles from quaternions.

**Parameters:**
- `qarr` : Quaternion array of shape (N, 4)
- `seq` : Euler sequence ('xzx', 'xyx', 'yxy', 'yzy', 'zxz', 'zyz')

**Returns:** Euler angles of shape (N, 3).

---

### `_calc_tait_bryan_angles`

```python
@njit(cache=True)
def _calc_tait_bryan_angles(
    qarr: NDArray, 
    seq: str, 
    intrinsic: bool = True
) -> NDArray
```

Compute Tait-Bryan angles from quaternions.

**Parameters:**
- `qarr` : Quaternion array of shape (N, 4)
- `seq` : Tait-Bryan sequence ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')
- `intrinsic` : If True, use intrinsic rotations

**Returns:** Tait-Bryan angles of shape (N, 3).

---

## Quaternion Composition

### `_compose_quaternion_rotation`

```python
def _compose_quaternion_rotation(
    angles: NDArray, 
    seq: str = 'xyz', 
    degrees: bool = False,
    intrinsic: bool = False
) -> NDArray
```

Compose quaternion from Euler/Tait-Bryan angles.

**Parameters:**
- `angles` : Array of shape (N, len(seq)) or (len(seq),)
- `seq` : Rotation sequence
- `degrees` : If True, angles are in degrees
- `intrinsic` : If True, rotations are intrinsic

**Returns:** Quaternion array of shape (N, 4).

---

### `_quat_element_calc`

```python
@njit(cache=True)
def _quat_element_calc(axis: str, angles: NDArray) -> NDArray
```

Compute quaternion for single-axis rotation (internal use).

**Parameters:**
- `axis` : Rotation axis ('x', 'y', or 'z')
- `angles` : Array of rotation angles

**Returns:** Quaternion array of shape (N, 4).

---

## Quaternion Arithmetic

### `_quat_mult`

```python
@njit(cache=True)
def _quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray
```

Hamilton product of quaternions.

**Parameters:**
- `p` : First quaternion(s), shape (N, 4) or (1, 4)
- `q` : Second quaternion(s), shape (N, 4) or (1, 4)

**Returns:** Product quaternion(s) of shape (N, 4).

**Note:** Supports broadcasting when one input is single quaternion.

---

### `_quat_div`

```python
@njit(cache=True)
def _quat_div(p: np.ndarray, q: np.ndarray) -> np.ndarray
```

Quaternion division (p × q⁻¹).

**Parameters:**
- `p` : Numerator quaternion(s)
- `q` : Denominator quaternion(s)

**Returns:** Quotient quaternion(s).

---

### `_quat_single_multiplication`

```python
@njit(cache=True)
def _quat_single_multiplication(p: NDArray, q: NDArray) -> NDArray
```

Multiply two single quaternions (internal use).

---

## Spherical Coordinates

### `_spherical_coords_to_quat`

```python
def _spherical_coords_to_quat(
    theta: float | NDArray, 
    phi: float | NDArray, 
    degrees: bool = False
) -> NDArray
```

Convert spherical coordinates to quaternion.

**Parameters:**
- `theta` : Polar angle
- `phi` : Azimuthal angle
- `degrees` : If True, angles are in degrees

**Returns:** Quaternion of shape (1, 4).

---

### `calc_spherical_coordinates`

```python
def calc_spherical_coordinates(vec) -> tuple
```

Convert 3D vector to spherical coordinates.

**Parameters:**
- `vec` : 3-element vector [x, y, z]

**Returns:** Tuple (r, theta, phi).

---

## Vector Operations

### `calc_cross`

```python
@njit(cache=True)
def calc_cross(a, b) -> tuple
```

Compute cross product of two 3D vectors.

**Parameters:**
- `a`, `b` : 3-element vectors

**Returns:** Tuple (x, y, z) of cross product.

---

### `make_2d`

```python
@njit
def make_2d(arraylist: list[NDArray]) -> NDArray
```

Stack list of 1D arrays into 2D array.

**Parameters:**
- `arraylist` : List of 1D arrays of equal length

**Returns:** 2D array of shape (n, k).

---

## Constants

```python
X_UNIT_AXIS = (1, 0, 0)
Y_UNIT_AXIS = (0, 1, 0)
Z_UNIT_AXIS = (0, 0, 1)
```

Unit vectors for principal axes.

---

## Performance Notes

### JIT Compilation

Functions decorated with `@njit(cache=True)` are compiled on first call:

```python
from hammy.utilities import _quat_mult
import numpy as np
import time

p = np.array([[1.0, 0.0, 0.0, 0.0]])
q = np.array([[1.0, 0.0, 0.0, 0.0]])

# First call includes compilation
start = time.time()
result = _quat_mult(p, q)
print(f"First call: {time.time() - start:.3f}s")

# Subsequent calls are fast
start = time.time()
for _ in range(10000):
    result = _quat_mult(p, q)
print(f"10000 calls: {time.time() - start:.3f}s")
```

### Best Practices

1. **Use batch operations** — Process arrays instead of loops
2. **Ensure contiguous arrays** — Use `np.ascontiguousarray()` if needed
3. **Warm up JIT** — Call once before timing-critical code
4. **Cache persists** — Compiled code is cached in `__pycache__`

---

## See Also

- [Quaternion Class Reference](quaternion.md) — High-level API
- [Batch Operations Tutorial](../tutorials/batch-operations.md)
