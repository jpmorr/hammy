# Quaternion Class Reference

Complete API reference for the `Quaternion` class.

## Class Definition

```python
class Quaternion:
    """Quaternion class for 3D rotation operations."""
```

## Constructors

### `__init__`

```python
def __init__(self, value: NDArray | list | tuple | None = None) -> None
```

Create a quaternion from components.

**Parameters:**
- `value` : Array-like of shape (4,) for single quaternion or (N, 4) for multiple. If None, creates empty quaternion.

**Example:**
```python
q = Quaternion([1.0, 0.0, 0.0, 0.0])
q_batch = Quaternion([[1,0,0,0], [0,1,0,0]])
```

---

### `from_euler`

```python
@classmethod
def from_euler(
    cls, 
    angles: NDArray, 
    seq: str = 'xyz', 
    degrees: bool = False,
    intrinsic: bool = False
) -> Self
```

Create quaternion from Euler or Tait-Bryan angles.

**Parameters:**
- `angles` : Array of shape (3,) or (N, 3) containing rotation angles
- `seq` : Rotation sequence (e.g., 'xyz', 'zxz', 'zyx')
- `degrees` : If True, angles are in degrees
- `intrinsic` : If True, rotations are about local axes

**Supported Sequences:**
- Proper Euler: `xzx`, `xyx`, `yxy`, `yzy`, `zxz`, `zyz`
- Tait-Bryan: `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx`

**Example:**
```python
q = Quaternion.from_euler([30, 45, 60], seq='zyx', degrees=True)
```

---

### `from_axisangle`

```python
@classmethod
def from_axisangle(
    cls, 
    v: NDArray | list | tuple, 
    theta: float | NDArray | list | tuple,
    degrees: bool = False
) -> Self
```

Create quaternion from axis-angle representation.

**Parameters:**
- `v` : Rotation axis, shape (3,) or (N, 3). Will be normalised internally.
- `theta` : Rotation angle(s), broadcastable with v
- `degrees` : If True, theta is in degrees

**Example:**
```python
q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
q_deg = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)
```

---

### `from_spherical`

```python
@classmethod
def from_spherical(
    cls, 
    theta: float, 
    phi: float, 
    degrees: bool = False
) -> Self
```

Create quaternion from spherical coordinates.

**Parameters:**
- `theta` : Polar angle (from z-axis)
- `phi` : Azimuthal angle (in x-y plane from x-axis)
- `degrees` : If True, angles are in degrees

**Example:**
```python
q = Quaternion.from_spherical(np.pi/4, np.pi/3)
```

---

## Properties

### `scalar` / `w`

```python
@property
def scalar(self) -> NDArray
@property
def w(self) -> NDArray
```

The real (scalar) component of the quaternion(s).

**Returns:** Array of shape (N,) containing w values.

---

### `vector`

```python
@property
def vector(self) -> NDArray
```

The imaginary (vector) component [x, y, z].

**Returns:** Array of shape (N, 3) containing vector parts.

---

### `x`, `y`, `z`

```python
@property
def x(self) -> NDArray
@property
def y(self) -> NDArray
@property
def z(self) -> NDArray
```

Individual vector components.

**Returns:** Arrays of shape (N,) containing respective components.

---

### `conjugate`

```python
@property
def conjugate(self) -> Self
```

The conjugate quaternion (negated vector part).

**Returns:** New Quaternion with [w, -x, -y, -z].

**Example:**
```python
q = Quaternion([1, 2, 3, 4])
q_conj = q.conjugate  # [1, -2, -3, -4]
```

---

### `magnitude`

```python
@property
def magnitude(self) -> NDArray
```

Euclidean norm of the quaternion(s).

**Returns:** Array of shape (N,) containing magnitudes.

---

### `num_quaternions`

```python
num_quaternions: int | None
```

Number of quaternions stored, or None if empty.

---

## Methods

### `normalise` / `normalize`

```python
def normalise(self) -> Self
def normalize(self) -> Self  # Alias
```

Return a unit quaternion.

**Returns:** New Quaternion with magnitude 1.

**Example:**
```python
q = Quaternion([2, 2, 2, 2])
q_unit = q.normalise()  # [0.5, 0.5, 0.5, 0.5]
```

---

### `to_rotation_matrix`

```python
def to_rotation_matrix(self) -> NDArray
```

Convert to 3×3 rotation matrix.

**Returns:** Array of shape (3, 3) for single quaternion or (N, 3, 3) for batch.

---

### `get_rotation_matrix_flat`

```python
def get_rotation_matrix_flat(self) -> NDArray
```

Convert to flat 9-element rotation matrix array.

**Returns:** Array of shape (9,) or (N, 9).

---

### `to_axis_angle`

```python
def to_axis_angle(self, degrees: bool = False) -> tuple[NDArray, NDArray | float]
```

Convert to axis-angle representation.

**Parameters:**
- `degrees` : If True, return angle in degrees

**Returns:** Tuple of (axis, angle). Axis shape (3,) or (N, 3), angle is float or shape (N,).

---

### `get_rotation_vector`

```python
def get_rotation_vector(self) -> NDArray
```

Get rotation vector (axis × angle).

**Returns:** Array of shape (3,) or (N, 3).

---

### `to_euler`

```python
def to_euler(
    self, 
    seq: str = 'zxz', 
    degrees: bool = False,
    intrinsic: bool = True
) -> NDArray
```

Convert to proper Euler angles.

**Parameters:**
- `seq` : Euler sequence ('xzx', 'xyx', 'yxy', 'yzy', 'zxz', 'zyz')
- `degrees` : If True, return angles in degrees
- `intrinsic` : If True, use intrinsic rotations

**Returns:** Array of shape (3,) or (N, 3) containing [alpha, beta, gamma].

---

### `to_tait_bryan`

```python
def to_tait_bryan(
    self, 
    seq: str = 'xyz', 
    degrees: bool = False,
    intrinsic: bool = True
) -> NDArray
```

Convert to Tait-Bryan angles.

**Parameters:**
- `seq` : Tait-Bryan sequence ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')
- `degrees` : If True, return angles in degrees
- `intrinsic` : If True, use intrinsic rotations

**Returns:** Array of shape (3,) or (N, 3).

---

### `to_yaw_pitch_roll`

```python
def to_yaw_pitch_roll(self, degrees: bool = False) -> NDArray
```

Get yaw-pitch-roll angles (intrinsic z-y-x sequence).

**Parameters:**
- `degrees` : If True, return angles in degrees

**Returns:** Array of shape (3,) or (N, 3) containing [yaw, pitch, roll].

---

### `vector_norm`

```python
def vector_norm(self) -> NDArray | float
```

L2 norm of the vector component only.

**Returns:** Float for single quaternion or array of shape (N,) for batch.

---

### `tolist`

```python
def tolist(self) -> list
```

Convert to Python list.

**Returns:** Nested list representation.

---

## Operators

### `__add__` / `__radd__`

```python
def __add__(self, other: Self | NDArray | list | tuple) -> Self
```

Element-wise addition.

---

### `__sub__` / `__rsub__`

```python
def __sub__(self, other: Self | NDArray | list | tuple) -> Self
```

Element-wise subtraction.

---

### `__mul__` / `__rmul__`

```python
def __mul__(self, other: Self | NDArray | list | tuple | float | int) -> Self | NDArray
```

Quaternion multiplication, scalar multiplication, or vector rotation.

**Behavior:**
- Quaternion × Quaternion → Hamilton product
- Quaternion × Scalar → Scaled quaternion
- Quaternion × 3D vector → Rotated vector

---

### `__truediv__` / `__rtruediv__`

```python
def __truediv__(self, other: Self | float | int | NDArray) -> Self
```

Quaternion or scalar division.

---

### `__neg__`

```python
def __neg__(self) -> Self
```

Negate all components.

---

### `__getitem__`

```python
def __getitem__(self, item: int | slice) -> NDArray
```

Index or slice the quaternion array.

---

### `__array__`

```python
def __array__(self) -> NDArray
```

Convert to numpy array.

---

### `__str__` / `__repr__`

```python
def __str__(self) -> str
def __repr__(self) -> str
```

String representations.

---

## See Also

- [Utilities Reference](utilities.md) — Low-level Numba functions
- [Creating Quaternions Tutorial](../tutorials/creating-quaternions.md)
- [Arithmetic Tutorial](../tutorials/arithmetic.md)
