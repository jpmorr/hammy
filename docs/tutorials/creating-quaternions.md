# Creating Quaternions

This tutorial covers all the different ways to create quaternions in Quatham.

## From Components

The most direct way to create a quaternion is from its four components:

```python
from hammy import Quaternion
import numpy as np

# Identity quaternion (no rotation)
q_identity = Quaternion([1.0, 0.0, 0.0, 0.0])

# Arbitrary quaternion
q = Quaternion([0.707, 0.0, 0.707, 0.0])

# From numpy array
arr = np.array([1.0, 0.0, 0.0, 0.0])
q = Quaternion(arr)
```

**Remember**: Quatham uses scalar-first ordering `[w, x, y, z]`.

## From Euler Angles

Create a quaternion from a sequence of rotations:

```python
# Single set of angles (radians)
angles = [np.pi/4, 0, 0]  # 45° around X
q = Quaternion.from_euler(angles, seq='xyz')

# In degrees
q_deg = Quaternion.from_euler([45, 0, 0], seq='xyz', degrees=True)

# Intrinsic rotations (default - local axes)
q_intrinsic = Quaternion.from_euler([30, 45, 60], seq='xyz', 
                                     degrees=True, intrinsic=True)

# Extrinsic rotations (fixed axes)
q_extrinsic = Quaternion.from_euler([30, 45, 60], seq='xyz', 
                                     degrees=True, intrinsic=False)
```

### Supported Sequences

**Proper Euler** (2 distinct axes):
- `xzx`, `xyx`, `yxy`, `yzy`, `zxz`, `zyz`

**Tait-Bryan** (3 distinct axes):
- `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx`

## From Axis-Angle

Represent a rotation as an axis vector and an angle:

```python
# 90° rotation around Z-axis
axis = np.array([0, 0, 1])
angle = np.pi / 2
q = Quaternion.from_axisangle(axis, angle)

# In degrees
q_deg = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)

# Non-unit axis (automatically normalised)
q = Quaternion.from_axisangle([0, 0, 5], np.pi/2)  # Same as [0, 0, 1]
```

### Batch Axis-Angle

```python
# Multiple axes, single angle
axes = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
q = Quaternion.from_axisangle(axes, np.pi/2)

# Single axis, multiple angles
angles = [0, np.pi/4, np.pi/2, np.pi]
q = Quaternion.from_axisangle([0, 0, 1], angles)

# Multiple axes, multiple angles (must match)
q = Quaternion.from_axisangle(axes, [np.pi/4, np.pi/3, np.pi/2])
```

## From Spherical Coordinates

Create a quaternion from spherical coordinates:

```python
# theta: polar angle (from z-axis)
# phi: azimuthal angle (in x-y plane from x-axis)
q = Quaternion.from_spherical(theta=np.pi/4, phi=np.pi/3)

# In degrees
q_deg = Quaternion.from_spherical(45, 60, degrees=True)
```

## From Rotation Matrix

While Quatham doesn't have a direct `from_rotation_matrix()` method, you can convert via axis-angle:

```python
# Given a rotation matrix
R = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])

# Extract axis-angle (using scipy or custom code)
# Then create quaternion
from hammy import Quaternion
axis = np.array([0, 0, 1])
angle = np.pi / 2
q = Quaternion.from_axisangle(axis, angle)
```

## Batch Creation

Efficiently create multiple quaternions:

```python
# From array of components
quats = Quaternion([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# From array of Euler angles
angles = np.array([
    [30, 0, 0],
    [0, 45, 0],
    [0, 0, 60],
])
quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)

# Access individual quaternions
first = Quaternion(quats[0])
subset = Quaternion(quats[0:2])
```

## Empty Quaternion

Create an empty quaternion object:

```python
q_empty = Quaternion()
# q_empty.ndarray is None
# q_empty.num_quaternions is None
```

## Best Practices

1. **Use specific constructors** when possible (e.g., `from_euler()` instead of manual calculation)
2. **Normalise** quaternions after arithmetic operations to maintain unit length
3. **Batch operations** are more efficient than loops for multiple quaternions
4. **Be consistent** with angle units (radians vs degrees)

## Next Steps

- [Quaternion Arithmetic](arithmetic.md) — Operations on quaternions
- [Conversions Tutorial](conversions.md) — Converting between representations
- [API Reference: Quaternion](../reference/quaternion.md) — Complete class documentation
