# Quaternion Arithmetic

This tutorial covers arithmetic operations with quaternions in Hammy.

## Addition and Subtraction

```python
from hammy import Quaternion
import numpy as np

q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
q2 = Quaternion([5.0, 6.0, 7.0, 8.0])

# Addition
q_sum = q1 + q2  # [6.0, 8.0, 10.0, 12.0]

# Subtraction
q_diff = q1 - q2  # [-4.0, -4.0, -4.0, -4.0]

# With arrays
q_arr = q1 + np.array([1.0, 1.0, 1.0, 1.0])

# With lists
q_list = q1 + [1.0, 1.0, 1.0, 1.0]

# Right-hand operations
q_right = [1.0, 1.0, 1.0, 1.0] + q1
```

## Scalar Multiplication

```python
q = Quaternion([1.0, 2.0, 3.0, 4.0])

# Multiply by scalar
q_scaled = q * 2.0  # [2.0, 4.0, 6.0, 8.0]

# Right multiplication
q_scaled = 2.0 * q

# Division by scalar
q_div = q / 2.0  # [0.5, 1.0, 1.5, 2.0]

# Scalar divided by quaternion
q_inv_scaled = 5.0 / q  # 5 * q^(-1)
```

## Quaternion Multiplication

Quaternion multiplication (Hamilton product) combines rotations:

```python
q1 = Quaternion.from_euler([30, 0, 0], seq='xyz', degrees=True)
q2 = Quaternion.from_euler([0, 45, 0], seq='xyz', degrees=True)

# Combine rotations: first q2, then q1
q_combined = q1 * q2

# Note: quaternion multiplication is NOT commutative
assert not np.allclose((q1 * q2).ndarray, (q2 * q1).ndarray)

# But it IS associative
q3 = Quaternion.from_euler([0, 0, 60], seq='xyz', degrees=True)
assert np.allclose(((q1 * q2) * q3).ndarray, (q1 * (q2 * q3)).ndarray)
```

### Hamilton Product Rules

The fundamental quaternion units follow:
- `i² = j² = k² = ijk = -1`
- `ij = k`, `jk = i`, `ki = j`
- `ji = -k`, `kj = -i`, `ik = -j`

```python
# Verify with Hammy
qi = Quaternion([0.0, 1.0, 0.0, 0.0])
qj = Quaternion([0.0, 0.0, 1.0, 0.0])
qk = Quaternion([0.0, 0.0, 0.0, 1.0])

assert np.allclose((qi * qi).ndarray, [-1, 0, 0, 0])
assert np.allclose((qi * qj).ndarray, qk.ndarray)
assert np.allclose((qj * qi).ndarray, (-qk).ndarray)
```

## Quaternion Division

```python
q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
q2 = Quaternion([2.0, -1.0, 0.5, 1.5])

# Divide q1 by q2: q1 * q2^(-1)
q_div = q1 / q2

# Verify: (q1 / q2) * q2 = q1
assert np.allclose((q_div * q2).ndarray, q1.ndarray)

# Self-division gives identity
q_identity = q1 / q1
assert np.allclose(q_identity.ndarray, [1.0, 0.0, 0.0, 0.0])
```

## Conjugate and Inverse

The conjugate negates the vector part:

```python
q = Quaternion([1.0, 2.0, 3.0, 4.0])
q_conj = q.conjugate  # [1.0, -2.0, -3.0, -4.0]

# For unit quaternions: q^(-1) = q*
q_unit = Quaternion([0.5, 0.5, 0.5, 0.5])
q_inv = q_unit.conjugate

# q * q* = |q|^2 (as a scalar quaternion)
product = q * q.conjugate
assert np.allclose(product.ndarray, [[q.magnitude[0]**2, 0, 0, 0]])
```

## Normalisation

Normalise quaternions to unit length:

```python
q = Quaternion([2.0, 2.0, 2.0, 2.0])

# Normalise (British spelling)
q_unit = q.normalise()
assert np.isclose(q_unit.magnitude[0], 1.0)

# Normalize (American spelling alias)
q_unit = q.normalize()

# Original is unchanged (non-destructive)
assert np.allclose(q.ndarray, [2.0, 2.0, 2.0, 2.0])
```

## Magnitude and Norm

```python
q = Quaternion([1.0, 2.0, 3.0, 6.0])

# Full magnitude
mag = q.magnitude  # sqrt(1 + 4 + 9 + 36) = sqrt(50)

# Vector part norm only
vec_norm = q.vector_norm()  # sqrt(4 + 9 + 36) = 7.0
```

## Negation

```python
q = Quaternion([1.0, 2.0, 3.0, 4.0])
q_neg = -q  # [-1.0, -2.0, -3.0, -4.0]

# Note: q and -q represent the same rotation
assert np.allclose(q.to_rotation_matrix(), (-q).to_rotation_matrix())
```

## Batch Arithmetic

All operations work on arrays of quaternions:

```python
qa = Quaternion([
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 20.0, 30.0, 40.0],
])
qb = Quaternion([
    [5.0, 6.0, 7.0, 8.0],
    [50.0, 60.0, 70.0, 80.0],
])

# Element-wise operations
q_sum = qa + qb
q_prod = qa * qb

# Broadcasting (single with batch)
q_single = Quaternion([1.0, 1.0, 1.0, 1.0])
q_result = qa + q_single

# Scalar operations on batch
q_scaled = qa * 2.0
```

## Practical Example: Combining Rotations

```python
# Start with identity orientation
orientation = Quaternion([1.0, 0.0, 0.0, 0.0])

# Apply successive rotations
pitch = Quaternion.from_euler([30, 0, 0], seq='xyz', degrees=True)
yaw = Quaternion.from_euler([0, 45, 0], seq='xyz', degrees=True)
roll = Quaternion.from_euler([0, 0, 15], seq='xyz', degrees=True)

# Combine (order matters!)
orientation = roll * yaw * pitch

# Get final rotation matrix
R = orientation.to_rotation_matrix()
```

## Best Practices

1. **Normalise after operations** to prevent numerical drift
2. **Remember non-commutativity**: `q1 * q2 ≠ q2 * q1`
3. **Use conjugate for inverse** when working with unit quaternions
4. **Batch operations** are more efficient than loops

## Next Steps

- [Conversions Tutorial](conversions.md) — Convert between representations
- [Batch Operations Tutorial](batch-operations.md) — Efficient array processing
- [API Reference](../reference/quaternion.md) — Complete documentation
