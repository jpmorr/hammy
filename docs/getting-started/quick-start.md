# Quick Start

This guide shows you how to get started with Quatham in just a few minutes.

## Creating Your First Quaternion

```python
from quatham import Quaternion
import numpy as np

# Method 1: From components [w, x, y, z]
q_identity = Quaternion([1.0, 0.0, 0.0, 0.0])

# Method 2: From Euler angles (90° rotation around Z)
q_euler = Quaternion.from_euler([0, 0, np.pi/2], seq='xyz')

# Method 3: From axis-angle (90° around Z-axis)
q_axis = Quaternion.from_axisangle([0, 0, 1], np.pi/2)

# Method 4: From axis-angle in degrees
q_degrees = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)
```

## Basic Operations

```python
# Multiplication (combining rotations)
q1 = Quaternion.from_euler([np.pi/4, 0, 0], seq='xyz')
q2 = Quaternion.from_euler([0, np.pi/4, 0], seq='xyz')
q_combined = q1 * q2

# Scalar multiplication
q_scaled = q1 * 2.0

# Conjugate (inverse for unit quaternions)
q_conj = q1.conjugate

# Normalise to unit quaternion
q_unit = q1.normalise()
```

## Rotating Vectors

```python
# Create a rotation quaternion
q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)  # 90° around Z

# Rotate a vector
v = np.array([1.0, 0.0, 0.0])
v_rotated = q * v

print(f"Original: {v}")
print(f"Rotated: {v_rotated}")
# Output: [0. 1. 0.]
```

## Conversions

```python
# To rotation matrix
R = q.to_rotation_matrix()

# To axis-angle
axis, angle = q.to_axis_angle()

# To Euler angles (proper sequence)
euler = q.to_euler(seq='zxz', degrees=True)

# To yaw-pitch-roll (Tait-Bryan z-y-x)
yaw, pitch, roll = q.to_yaw_pitch_roll(degrees=True)
```

## Working with Arrays

Quatham efficiently handles batches of quaternions:

```python
# Create multiple quaternions
quats = Quaternion([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

# Rotate multiple vectors at once
v = np.array([1.0, 0.0, 0.0])
rotated = quats * v  # Shape: (3, 3)

# Access individual quaternions
first = Quaternion(quats[0])
```

## Next Steps

- Learn about [creating quaternions](../tutorials/creating-quaternions.md) in detail
- Explore [quaternion arithmetic](../tutorials/arithmetic.md)
- See all available [conversions](../tutorials/conversions.md)
- Check the [API reference](../reference/quaternion.md)
