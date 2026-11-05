# Quaternion Conversions

This tutorial covers converting between quaternions and other rotation representations.

## To and From Axis-Angle

### Creating from Axis-Angle

```python
from quatham import Quaternion
import numpy as np

# Rotation around Z-axis by 90 degrees
axis = np.array([0, 0, 1])
angle = np.pi / 2

q = Quaternion.from_axisangle(axis, angle)

# In degrees
q_deg = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)
```

### Converting to Axis-Angle

```python
q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)

# Get axis and angle (radians)
axis, angle = q.to_axis_angle()

# Get axis and angle (degrees)
axis, angle_deg = q.to_axis_angle(degrees=True)

# For single quaternion, returns 1D arrays
print(f"Axis: {axis}, Angle: {angle_deg:.1f}°")
```

### Rotation Vector

The rotation vector combines axis and angle into a single vector:

```python
q = Quaternion.from_axisangle([0, 1, 0], 1.2)
rot_vec = q.get_rotation_vector()  # axis * angle

# Direction = rotation axis
# Magnitude = rotation angle
axis = rot_vec / np.linalg.norm(rot_vec)
angle = np.linalg.norm(rot_vec)
```

## To and From Euler Angles

### Proper Euler Angles

Proper Euler angles use rotations about 2 distinct axes:

```python
# Supported sequences: xzx, xyx, yxy, yzy, zxz, zyz
q = Quaternion.from_euler([30, 45, 60], seq='zxz', degrees=True)

# Convert back to Euler angles
euler = q.to_euler(seq='zxz', degrees=True)

# Intrinsic (local axes) - default
euler_intrinsic = q.to_euler(seq='zxz', intrinsic=True)

# Extrinsic (fixed axes)
euler_extrinsic = q.to_euler(seq='zxz', intrinsic=False)
```

### Tait-Bryan Angles

Tait-Bryan angles use rotations about 3 distinct axes:

```python
# Supported sequences: xyz, xzy, yxz, yzx, zxy, zyx
q = Quaternion.from_euler([30, 45, 60], seq='xyz', degrees=True)

# Convert to Tait-Bryan angles
tait_bryan = q.to_tait_bryan(seq='xyz', degrees=True)

# Intrinsic vs extrinsic
tait_intrinsic = q.to_tait_bryan(seq='xyz', intrinsic=True)
tait_extrinsic = q.to_tait_bryan(seq='xyz', intrinsic=False)
```

### Yaw-Pitch-Roll

The most common Tait-Bryan sequence is z-y-x (yaw-pitch-roll):

```python
q = Quaternion.from_euler([30, 45, 60], seq='zyx', degrees=True)

# Get yaw, pitch, roll directly
ypr = q.to_yaw_pitch_roll(degrees=True)
yaw, pitch, roll = ypr

print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")
```

## To and From Rotation Matrices

### Creating Rotation Matrices

```python
q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)

# Get 3x3 rotation matrix
R = q.to_rotation_matrix()
print(R.shape)  # (3, 3)

# Get flat 9-element array
R_flat = q.get_rotation_matrix_flat()
print(R_flat.shape)  # (9,)
```

### Rotation Matrix Properties

```python
q = Quaternion.from_euler([30, 45, 60], seq='xyz', degrees=True)
R = q.to_rotation_matrix()

# Verify orthogonality: R * R^T = I
assert np.allclose(R @ R.T, np.eye(3))

# Verify determinant: det(R) = 1
assert np.isclose(np.linalg.det(R), 1.0)
```

## To and From Spherical Coordinates

```python
# Create from spherical coordinates
# theta: polar angle (from z-axis)
# phi: azimuthal angle (in x-y plane)
q = Quaternion.from_spherical(theta=np.pi/4, phi=np.pi/3)

# In degrees
q_deg = Quaternion.from_spherical(45, 60, degrees=True)
```

## Complete Conversion Example

```python
# Start with axis-angle
axis = np.array([0.577, 0.577, 0.577])  # Normalised [1,1,1]
angle = np.pi / 3  # 60 degrees

# Create quaternion
q = Quaternion.from_axisangle(axis, angle)

# Convert to various representations
R = q.to_rotation_matrix()
euler = q.to_euler(seq='zxz', degrees=True)
ypr = q.to_yaw_pitch_roll(degrees=True)
axis_rec, angle_rec = q.to_axis_angle()
rot_vec = q.get_rotation_vector()

print(f"Original angle: {np.degrees(angle):.1f}°")
print(f"Recovered angle: {np.degrees(angle_rec):.1f}°")
print(f"Yaw-Pitch-Roll: {ypr}")
```

## Batch Conversions

All conversions work efficiently on arrays:

```python
# Create batch of quaternions
angles = np.array([
    [30, 0, 0],
    [0, 45, 0],
    [0, 0, 60],
])
quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)

# Batch conversion to rotation matrices
R_batch = quats.to_rotation_matrix()
print(R_batch.shape)  # (3, 3, 3)

# Batch conversion to yaw-pitch-roll
ypr_batch = quats.to_yaw_pitch_roll(degrees=True)
print(ypr_batch.shape)  # (3, 3)

# Access individual results
first_ypr = ypr_batch[0]
```

## Round-Trip Verification

Verify conversions preserve the rotation:

```python
# Start with Euler angles
original_angles = np.array([30.0, 45.0, 60.0])

# Convert to quaternion and back
q = Quaternion.from_euler(original_angles, seq='xyz', degrees=True)
recovered = q.to_euler(seq='xyz', degrees=True)

# Note: Euler angles may differ by 360° or use alternative representations
# but should produce the same rotation
v = np.array([1.0, 2.0, 3.0])
q_orig = Quaternion.from_euler(original_angles, seq='xyz', degrees=True)
q_rec = Quaternion.from_euler(recovered, seq='xyz', degrees=True)

assert np.allclose(q_orig * v, q_rec * v)
```

## Gimbal Lock Awareness

Euler angles suffer from gimbal lock at certain configurations:

```python
# Pitch = 90° causes gimbal lock for xyz sequence
q = Quaternion.from_euler([0, 90, 0], seq='xyz', degrees=True)
euler = q.to_euler(seq='xyz', degrees=True)

# The first and third angles become coupled
# Use quaternions or rotation matrices for robust computations
```

## Best Practices

1. **Use quaternions for storage and interpolation** — avoid gimbal lock
2. **Convert to Euler angles only for display** — human-readable format
3. **Use rotation matrices for transforming many vectors** — efficient batch operations
4. **Be consistent with conventions** — intrinsic vs extrinsic, radians vs degrees
5. **Verify round-trips with vector rotation** — not direct value comparison

## Next Steps

- [Batch Operations Tutorial](batch-operations.md) — Efficient array processing
- [API Reference: Quaternion](../reference/quaternion.md) — Complete class documentation
- [API Reference: Utilities](../reference/utilities.md) — Low-level functions
