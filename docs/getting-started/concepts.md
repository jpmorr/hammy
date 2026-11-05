# Basic Concepts

This page introduces the fundamental concepts you need to understand when working with Hammy.

## What is a Quaternion?

A quaternion is a mathematical object that extends complex numbers to four dimensions. In 3D graphics and robotics, quaternions are used to represent rotations because they:

- Avoid **gimbal lock** (a problem with Euler angles)
- Enable **smooth interpolation** (SLERP)
- Are **computationally efficient** for combining rotations
- Provide **numerical stability**

## Quaternion Representation

Hammy uses the **scalar-first** convention:

```
q = [w, x, y, z]
```

Where:
- `w` is the scalar (real) part
- `[x, y, z]` is the vector (imaginary) part

This can also be written as:

```
q = w + xi + yj + zk
```

## Unit Quaternions

For rotation operations, we use **unit quaternions** where:

```
|q| = √(w² + x² + y² + z²) = 1
```

Hammy provides the `normalise()` method to convert any quaternion to a unit quaternion:

```python
q = Quaternion([2.0, 2.0, 2.0, 2.0])
q_unit = q.normalise()  # [0.5, 0.5, 0.5, 0.5]
```

## Double Cover Property

Quaternions have a **double cover** property: `q` and `-q` represent the same rotation. This is important when comparing quaternions or interpolating between them.

```python
q1 = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
q2 = -q1

# Both represent the same rotation
assert np.allclose(q1.to_rotation_matrix(), q2.to_rotation_matrix())
```

## Rotation Conventions

### Intrinsic vs Extrinsic

- **Intrinsic rotations**: Each rotation is about the local (moving) axes
- **Extrinsic rotations**: Each rotation is about the fixed (global) axes

Hammy uses **intrinsic rotations** by default:

```python
# Intrinsic (default)
q_intrinsic = Quaternion.from_euler([a, b, c], seq='xyz', intrinsic=True)

# Extrinsic
q_extrinsic = Quaternion.from_euler([a, b, c], seq='xyz', intrinsic=False)
```

### Rotation Sequences

Hammy supports two types of rotation sequences:

**Proper Euler Angles** (2 distinct axes):
- `xzx`, `xyx`, `yxy`, `yzy`, `zxz`, `zyz`

**Tait-Bryan Angles** (3 distinct axes):
- `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx`

The Tait-Bryan `zyx` sequence corresponds to **yaw-pitch-roll** in aerospace applications.

## Batch Operations

All Hammy operations support both single quaternions and arrays:

```python
# Single quaternion
q_single = Quaternion([1.0, 0.0, 0.0, 0.0])

# Array of quaternions
q_array = Quaternion([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

# Same operations work on both
v = np.array([1.0, 0.0, 0.0])
rotated_single = q_single * v      # Shape: (3,)
rotated_array = q_array * v        # Shape: (3, 3)
```

## Performance Considerations

Hammy uses Numba JIT compilation for optimal performance:

1. **First call overhead**: The first operation includes JIT compilation time
2. **Subsequent calls**: Much faster due to caching
3. **Batch operations**: Prefer array operations over loops for best performance

```python
# Slower: loop over individual quaternions
results = []
for q in quaternion_list:
    results.append(q * v)

# Faster: batch operation
q_array = Quaternion(quaternion_list)
results = q_array * v
```

## Next Steps

- [Installation Guide](installation.md) — Set up Hammy
- [Quick Start](quick-start.md) — Hands-on examples
- [Creating Quaternions Tutorial](../tutorials/creating-quaternions.md) — Detailed construction methods
