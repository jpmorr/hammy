#!/usr/bin/env python
"""
Quaternion Mathematics Module
=============================

Provides quaternion-based calculations and conversions for 3D rotations.
Supports operations on single quaternions and arrays of quaternions.

Features
--------
- Quaternion class with algebraic operations (multiplication, division, conjugate)
- Conversions between quaternions and:
  - Euler angles (proper/classic, 6 sequences)
  - Tait-Bryan angles (Cardan/nutical, 6 sequences)
  - Rotation matrices
  - Axis-angle representations
  - Spherical coordinates
- Numba-accelerated low-level operations in `utilities` submodule

Submodules
----------
utilities
    Low-level Numba-jitted quaternion math functions.

Examples
--------
Create a quaternion from Euler angles::

    from granulysed.quaternion import Quaternion
    import numpy as np

    q = Quaternion.from_euler([np.pi/4, 0, 0], seq='xyz', degrees=False)
    print(q.get_rotation_matrix())

Convert to Tait-Bryan (yaw-pitch-roll)::

    ypr = q.get_tait_bryan(seq='zyx', degrees=True)
    print(f"Yaw: {ypr[0]:.1f}°, Pitch: {ypr[1]:.1f}°, Roll: {ypr[2]:.1f}°")

Notes
-----
- All angles in radians unless `degrees=True` specified
- Intrinsic rotations (local axes) are default; use `intrinsic=False` for extrinsic
- Quaternion convention: scalar-first [w, x, y, z]
"""

from .Quaternion import Quaternion

__all__ = ["Quaternion"]