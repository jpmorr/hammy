# Batch Operations

This tutorial covers efficient batch processing of quaternion arrays in Hammy.

## Why Batch Operations?

Hammy is designed for high-performance batch processing using Numba JIT compilation. Processing arrays of quaternions is significantly faster than looping:

```python
import numpy as np
from hammy import Quaternion

# Create 1000 random rotations
angles = np.random.uniform(-np.pi, np.pi, size=(1000, 3))

# Batch creation (fast)
quats = Quaternion.from_euler(angles, seq='xyz')

# Batch rotation (fast)
v = np.array([1.0, 0.0, 0.0])
rotated = quats * v  # Shape: (1000, 3)
```

## Creating Batch Quaternions

### From Arrays

```python
# From 2D array of components
quats = Quaternion([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# From numpy array
arr = np.random.randn(100, 4)
quats = Quaternion(arr)
```

### From Euler Angles

```python
# Batch of Euler angles
angles = np.array([
    [30, 0, 0],
    [0, 45, 0],
    [0, 0, 60],
])
quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)

# Large batch
n = 10000
angles = np.random.uniform(0, 360, size=(n, 3))
quats = Quaternion.from_euler(angles, seq='zyx', degrees=True)
```

### From Axis-Angle

```python
# Multiple axes, single angle
axes = np.random.randn(100, 3)
quats = Quaternion.from_axisangle(axes, np.pi/4)

# Single axis, multiple angles
angles = np.linspace(0, 2*np.pi, 100)
quats = Quaternion.from_axisangle([0, 0, 1], angles)

# Multiple axes, multiple angles (must match)
axes = np.random.randn(50, 3)
angles = np.random.uniform(0, np.pi, 50)
quats = Quaternion.from_axisangle(axes, angles)
```

## Accessing Batch Elements

```python
quats = Quaternion(np.random.randn(100, 4))

# Single quaternion (returns array)
q0 = quats[0]  # Shape: (4,)

# Slice
subset = quats[0:10]  # Shape: (10, 4)

# Boolean indexing
mask = quats.w > 0
filtered = Quaternion(quats.ndarray[mask])

# Convert single element back to Quaternion
q_single = Quaternion(quats[0])
```

## Component Access

All properties work on batches:

```python
quats = Quaternion([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
])

# Scalar component (w)
scalars = quats.scalar  # Shape: (2,)
w = quats.w  # Same as scalar

# Vector components (x, y, z)
vectors = quats.vector  # Shape: (2, 3)
x = quats.x  # Shape: (2,)
y = quats.y  # Shape: (2,)
z = quats.z  # Shape: (2,)

# Magnitude
mags = quats.magnitude  # Shape: (2,)
```

## Batch Arithmetic

```python
qa = Quaternion(np.random.randn(100, 4))
qb = Quaternion(np.random.randn(100, 4))

# Element-wise operations
q_sum = qa + qb
q_prod = qa * qb
q_diff = qa - qb
q_div = qa / qb

# Broadcasting with single quaternion
q_single = Quaternion([1.0, 0.0, 0.0, 0.0])
result = qa * q_single

# Scalar operations
q_scaled = qa * 2.0
q_norm = qa / qa.magnitude[:, None]
```

## Batch Conversions

All conversion methods support batches:

```python
# Create batch
angles = np.random.uniform(0, 360, size=(100, 3))
quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)

# Batch rotation matrices
R_batch = quats.to_rotation_matrix()  # Shape: (100, 3, 3)
R_flat = quats.get_rotation_matrix_flat()  # Shape: (100, 9)

# Batch axis-angle
axes, thetas = quats.to_axis_angle()  # Shapes: (100, 3), (100,)

# Batch Euler angles
euler = quats.to_euler(seq='zxz', degrees=True)  # Shape: (100, 3)

# Batch yaw-pitch-roll
ypr = quats.to_yaw_pitch_roll(degrees=True)  # Shape: (100, 3)
```

## Batch Vector Rotation

```python
# Create batch of rotation quaternions
n = 1000
quats = Quaternion.from_euler(
    np.random.uniform(0, 360, size=(n, 3)), 
    seq='zyx', 
    degrees=True
)

# Rotate single vector by all quaternions
v = np.array([1.0, 0.0, 0.0])
rotated = quats * v  # Shape: (1000, 3)

# Rotate batch of vectors (one per quaternion)
vectors = np.random.randn(n, 3)
# Note: Currently rotates each vector by corresponding quaternion
# For advanced use, iterate or use broadcasting
```

## Performance Tips

### 1. Prefer Batch Operations

```python
# Slower: Python loop
results = []
for i in range(len(quats)):
    results.append(quats[i] * v)
results = np.array(results)

# Faster: Batch operation
results = quats * v
```

### 2. Pre-allocate When Possible

```python
# If you know the size, create the full array at once
n = 10000
angles = np.zeros((n, 3))
angles[:, 0] = np.linspace(0, 2*np.pi, n)
quats = Quaternion.from_euler(angles, seq='xyz')
```

### 3. Minimise Conversions

```python
# Keep data in quaternion form during computations
q1 = Quaternion.from_euler(angles1, seq='xyz')
q2 = Quaternion.from_euler(angles2, seq='xyz')
q_combined = q1 * q2

# Convert only at the end if needed
result_matrix = q_combined.to_rotation_matrix()
```

### 4. Use Numpy Operations on Components

```python
# Access components directly for custom operations
w = quats.w
x = quats.x
y = quats.y
z = quats.z

# Vectorised operations
custom_metric = w**2 + x**2
```

## Practical Example: Animation Interpolation

```python
# Create keyframe rotations
keyframes = Quaternion.from_euler([
    [0, 0, 0],
    [0, 90, 0],
    [0, 180, 0],
    [0, 270, 0],
], seq='zyx', degrees=True)

# Generate interpolated frames
n_frames = 100
t = np.linspace(0, 1, n_frames)

# Simple linear interpolation of components (approximate)
interpolated = []
for i in range(len(keyframes) - 1):
    q_start = keyframes[i]
    q_end = keyframes[i + 1]
    # Linear interpolation (for SLERP, use external library)
    t_local = t[i * 25:(i + 1) * 25]
    for ti in t_local:
        q_interp = Quaternion(
            (1 - ti) * q_start.ndarray[0] + ti * q_end.ndarray[0]
        )
        interpolated.append(q_interp.normalise())

# Convert to rotation matrices for rendering
matrices = [q.to_rotation_matrix() for q in interpolated]
```

## Memory Considerations

```python
# Check memory usage
import sys

n = 1000000
quats = Quaternion(np.random.randn(n, 4))

# ndarray is the main data
print(f"Array shape: {quats.ndarray.shape}")
print(f"Array dtype: {quats.ndarray.dtype}")
print(f"Memory: {quats.ndarray.nbytes / 1e6:.1f} MB")
```

## Best Practices

1. **Use batches for 10+ quaternions** — amortise JIT overhead
2. **Keep data in array form** — avoid converting to lists
3. **Normalise periodically** — prevent numerical drift in long chains
4. **Profile your code** — identify bottlenecks with `timeit` or `cProfile`

## Next Steps

- [API Reference: Quaternion](../reference/quaternion.md) — Complete class documentation
- [Utilities Reference](../reference/utilities.md) — Low-level Numba functions
- [Back to Quick Start](../getting-started/quick-start.md)
