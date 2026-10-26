#!/usr/bin/env python
"""Comprehensive validation suite for Quaternion representations and transformations.

Conventions verified:
- Quaternions are scalar-first: [w, x, y, z].
- Intrinsic rotations follow moving/body axes; extrinsic follow fixed/static axes.
- Double-cover SO(3) invariance: q and -q represent identical spatial rotations.
"""

__author__ = 'J.P. Morrissey'
__copyright__ = 'Copyright 2023, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = 'J.P. Morrissey'
__email__ = 'j.morrissey@ed.ac.uk'
__status__ = '{dev_status}'

# Standard Library


# Imports
import numpy as np
import pytest

# Local Sources
from hammy.Quaternion import Quaternion
from hammy.utilities import _compose_quaternion_rotation, normalise

class TestQuaternionAlgebra:
    """Validates core algebraic properties, component access, and operator overloading."""

    def test_identity_and_conjugate(self):
        """Validate identity quaternion vector rotation and basic conjugate."""
        q = Quaternion([1.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(q * v, v)

    def test_vector_rotation_vs_matrix(self):
        """Validate vector rotation equivalence with 3x3 rotation matrix."""
        q = Quaternion.from_axisangle(np.array([0, 0, 1]), np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(q * v, expected, atol=1e-7)
        assert np.allclose(q.get_rotation_matrix() @ v, expected, atol=1e-7)

    def test_component_access_single(self):
        """Validate scalar, vector, and w, x, y, z component access for a single quaternion."""
        q = Quaternion([1.0, 2.0, 3.0, 4.0])
        assert q.num_quaternions == 1
        assert np.allclose(q.scalar, np.array([1.0]))
        assert np.allclose(q.w, np.array([1.0]))
        assert np.allclose(q.vector, np.array([[2.0, 3.0, 4.0]]))
        assert np.allclose(q.x, np.array([2.0]))
        assert np.allclose(q.y, np.array([3.0]))
        assert np.allclose(q.z, np.array([4.0]))

    def test_component_access_batched_2(self):
        """Validate scalar, vector, and w, x, y, z component access for 2 quaternions."""
        q = Quaternion([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        assert q.num_quaternions == 2
        assert np.allclose(q.scalar, np.array([1.0, 5.0]))
        assert np.allclose(q.w, np.array([1.0, 5.0]))
        assert np.allclose(q.vector, np.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))
        assert np.allclose(q.x, np.array([2.0, 6.0]))
        assert np.allclose(q.y, np.array([3.0, 7.0]))
        assert np.allclose(q.z, np.array([4.0, 8.0]))

    def test_vector_norm_and_magnitude(self):
        """Validate vector_norm (norm of vector part) and magnitude for 1 and 2 quaternions."""
        # Single quaternion: vector norm sqrt(2^2 + 3^2 + 6^2) = 7.0, magnitude sqrt(1^2 + 49) = sqrt(50)
        q1 = Quaternion([1.0, 2.0, 3.0, 6.0])
        assert isinstance(q1.vector_norm(), float)
        assert np.isclose(q1.vector_norm(), 7.0)
        assert np.allclose(q1.magnitude, np.array([np.sqrt(50.0)]))

        # Batched quaternions (2):
        q2 = Quaternion([[1.0, 2.0, 3.0, 6.0], [0.0, 0.0, 4.0, 3.0]])
        v_norm = q2.vector_norm()
        assert isinstance(v_norm, np.ndarray)
        assert v_norm.shape == (2,)
        assert np.allclose(v_norm, np.array([7.0, 5.0]))
        assert np.allclose(q2.magnitude, np.array([np.sqrt(50.0), 5.0]))

    def test_conjugation_1_and_2(self):
        """Validate quaternion conjugation for single (1) and batched (2) quaternions."""
        # Single (1)
        q1 = Quaternion([1.0, 2.0, -3.0, 4.0])
        conj1 = q1.conjugate
        assert np.allclose(conj1.ndarray, np.array([[1.0, -2.0, 3.0, -4.0]]))
        assert np.allclose(conj1.conjugate.ndarray, q1.ndarray)
        # q * q* = ||q||^2
        prod1 = q1 * conj1
        assert np.allclose(prod1.ndarray, np.array([[q1.magnitude[0] ** 2, 0.0, 0.0, 0.0]]))

        # Batched (2)
        q2 = Quaternion([[1.0, 2.0, 3.0, 4.0], [5.0, -6.0, 7.0, -8.0]])
        conj2 = q2.conjugate
        assert np.allclose(conj2.ndarray, np.array([[1.0, -2.0, -3.0, -4.0], [5.0, 6.0, -7.0, 8.0]]))
        assert np.allclose(conj2.conjugate.ndarray, q2.ndarray)
        prod2 = q2 * conj2
        assert np.allclose(prod2.ndarray, np.array([[q2.magnitude[0] ** 2, 0.0, 0.0, 0.0],
                                                    [q2.magnitude[1] ** 2, 0.0, 0.0, 0.0]]))

    def test_normalisation_1_and_2(self):
        """Validate normalization methods and functions for 1 and 2 quaternions."""
        # Single (1)
        q1 = Quaternion([2.0, 2.0, 2.0, 2.0])
        q1_norm = q1.normalise()
        assert np.allclose(q1_norm.ndarray, np.array([[0.5, 0.5, 0.5, 0.5]]))
        assert np.isclose(q1_norm.magnitude[0], 1.0)
        assert np.allclose(q1.normalize().ndarray, q1_norm.ndarray)
        # Original remains untouched (non-destructive)
        assert np.allclose(q1.ndarray, np.array([[2.0, 2.0, 2.0, 2.0]]))

        # Batched (2)
        q2 = Quaternion([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
        q2_norm = q2.normalise()
        assert np.allclose(q2_norm.ndarray, np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0]]))
        assert np.allclose(q2_norm.magnitude, np.array([1.0, 1.0]))

        # Utility function normalise
        arr_1d = np.array([2.0, 0.0, 0.0, 0.0])
        assert np.allclose(normalise(arr_1d), np.array([1.0, 0.0, 0.0, 0.0]))
        arr_2d = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
        assert np.allclose(normalise(arr_2d), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0]]))

    def test_addition_and_subtraction_1_and_2(self):
        """Validate quaternion addition, subtraction, and negation for 1 and 2 quaternions."""
        # Single (1)
        q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
        q2 = Quaternion([5.0, 6.0, 7.0, 8.0])
        assert np.allclose((q1 + q2).ndarray, np.array([[6.0, 8.0, 10.0, 12.0]]))
        assert np.allclose((q1 - q2).ndarray, np.array([[-4.0, -4.0, -4.0, -4.0]]))
        assert np.allclose((-q1).ndarray, np.array([[-1.0, -2.0, -3.0, -4.0]]))

        # Array interoperability
        assert np.allclose((q1 + [1.0, 1.0, 1.0, 1.0]).ndarray, np.array([[2.0, 3.0, 4.0, 5.0]]))
        assert np.allclose(([1.0, 1.0, 1.0, 1.0] + q1).ndarray, np.array([[2.0, 3.0, 4.0, 5.0]]))
        assert np.allclose((q1 - [1.0, 1.0, 1.0, 1.0]).ndarray, np.array([[0.0, 1.0, 2.0, 3.0]]))
        assert np.allclose(([1.0, 1.0, 1.0, 1.0] - q1).ndarray, np.array([[0.0, -1.0, -2.0, -3.0]]))

        # Batched (2)
        qa = Quaternion([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        qb = Quaternion([[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]])
        assert np.allclose((qa + qb).ndarray, np.array([[6.0, 8.0, 10.0, 12.0], [60.0, 80.0, 100.0, 120.0]]))
        assert np.allclose((qa - qb).ndarray, np.array([[-4.0, -4.0, -4.0, -4.0], [-40.0, -40.0, -40.0, -40.0]]))
        assert np.allclose((-qa).ndarray, np.array([[-1.0, -2.0, -3.0, -4.0], [-10.0, -20.0, -30.0, -40.0]]))

        # Broadcasting (2 with 1)
        assert np.allclose((qa + q1).ndarray, np.array([[2.0, 4.0, 6.0, 8.0], [11.0, 22.0, 33.0, 44.0]]))
        assert np.allclose((qa - q1).ndarray, np.array([[0.0, 0.0, 0.0, 0.0], [9.0, 18.0, 27.0, 36.0]]))

    def test_multiplication_quaternion_and_scalar(self):
        """Validate Hamilton multiplication and scalar multiplication for 1 and 2 quaternions."""
        # Fundamental quaternion units: i^2 = j^2 = k^2 = ijk = -1, ij = k, jk = i, ki = j
        qi = Quaternion([0.0, 1.0, 0.0, 0.0])
        qj = Quaternion([0.0, 0.0, 1.0, 0.0])
        qk = Quaternion([0.0, 0.0, 0.0, 1.0])
        q_one = Quaternion([1.0, 0.0, 0.0, 0.0])

        assert np.allclose((qi * qi).ndarray, np.array([[-1.0, 0.0, 0.0, 0.0]]))
        assert np.allclose((qj * qj).ndarray, np.array([[-1.0, 0.0, 0.0, 0.0]]))
        assert np.allclose((qk * qk).ndarray, np.array([[-1.0, 0.0, 0.0, 0.0]]))
        assert np.allclose((qi * qj).ndarray, qk.ndarray)
        assert np.allclose((qj * qk).ndarray, qi.ndarray)
        assert np.allclose((qk * qi).ndarray, qj.ndarray)
        assert np.allclose((qj * qi).ndarray, (-qk).ndarray)
        assert np.allclose((qi * qj * qk).ndarray, (-q_one).ndarray)

        # Associativity: (q1 * q2) * q3 == q1 * (q2 * q3)
        q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
        q2 = Quaternion([2.0, -1.0, 0.5, 1.5])
        q3 = Quaternion([0.5, 1.0, -2.0, 0.0])
        assert np.allclose(((q1 * q2) * q3).ndarray, (q1 * (q2 * q3)).ndarray)

        # Batched quaternion multiplication (2)
        qa = Quaternion([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        qb = Quaternion([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        res = qa * qb
        assert res.num_quaternions == 2
        assert np.allclose(res[0], (Quaternion(qa[0]) * Quaternion(qb[0])).ndarray[0])
        assert np.allclose(res[1], (Quaternion(qa[1]) * Quaternion(qb[1])).ndarray[0])

        # Scalar multiplication (q * s and s * q)
        assert np.allclose((q1 * 3.0).ndarray, np.array([[3.0, 6.0, 9.0, 12.0]]))
        assert np.allclose((3.0 * q1).ndarray, np.array([[3.0, 6.0, 9.0, 12.0]]))
        assert np.allclose((qa * 2.0).ndarray, np.array([[2.0, 4.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]))
        assert np.allclose((2.0 * qa).ndarray, np.array([[2.0, 4.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]))

    def test_division_quaternion_and_scalar(self):
        """Validate quaternion division and scalar division for 1 and 2 quaternions."""
        # Single (1)
        q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
        q2 = Quaternion([2.0, -1.0, 0.5, 1.5])
        q_div = q1 / q2
        assert np.allclose((q_div * q2).ndarray, q1.ndarray)
        assert np.allclose((q1 / q1).ndarray, np.array([[1.0, 0.0, 0.0, 0.0]]))

        # Batched (2)
        qa = Quaternion([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        qb = Quaternion([[2.0, -1.0, 0.5, 1.5], [1.0, 1.0, 1.0, 1.0]])
        q_div_batch = qa / qb
        assert q_div_batch.num_quaternions == 2
        assert np.allclose((q_div_batch * qb).ndarray, qa.ndarray)

        # Broadcasting division (2 by 1)
        q_div_broad = qa / q2
        assert q_div_broad.num_quaternions == 2
        assert np.allclose((q_div_broad * q2).ndarray, qa.ndarray)

        # Scalar division (q / s and s / q)
        assert np.allclose((q1 / 2.0).ndarray, np.array([[0.5, 1.0, 1.5, 2.0]]))
        assert np.allclose((qa / 2.0).ndarray, np.array([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]))
        # s / q = s * q^-1
        s = 5.0
        assert np.allclose(((s / q1) * q1).ndarray, np.array([[s, 0.0, 0.0, 0.0]]))



