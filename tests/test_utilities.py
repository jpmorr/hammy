#!/usr/bin/env python
"""Tests for utility functions in quatham.utilities module."""

import numpy as np
import pytest

from quatham.utilities import calc_cross, calc_spherical_coordinates, make_2d


class TestCalcCross:
    """Tests for calc_cross vector cross-product function."""

    def test_cross_product_unit_vectors_xy(self):
        """Test cross product of x and y unit vectors gives z."""
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        
        result = calc_cross(x_axis, y_axis)
        
        assert result == pytest.approx((0.0, 0.0, 1.0))

    def test_cross_product_unit_vectors_yz(self):
        """Test cross product of y and z unit vectors gives x."""
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        
        result = calc_cross(y_axis, z_axis)
        
        assert result == pytest.approx((1.0, 0.0, 0.0))

    def test_cross_product_anticommutative(self):
        """Test that cross product is anticommutative: a × b = -(b × a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        
        result_ab = calc_cross(a, b)
        result_ba = calc_cross(b, a)
        
        # a × b = -(b × a)
        assert result_ab[0] == pytest.approx(-result_ba[0])
        assert result_ab[1] == pytest.approx(-result_ba[1])
        assert result_ab[2] == pytest.approx(-result_ba[2])

    def test_cross_product_parallel_vectors(self):
        """Test cross product of parallel vectors is zero."""
        v = np.array([1.0, 2.0, 3.0])
        
        result = calc_cross(v, v)
        
        assert result == pytest.approx((0.0, 0.0, 0.0))


class TestCalcSphericalCoordinates:
    """Tests for calc_spherical_coordinates function."""

    def test_spherical_coordinates_z_axis(self):
        """Test spherical coordinates for vector along z-axis."""
        vec = np.array([0.0, 0.0, 5.0])
        
        r, theta, phi = calc_spherical_coordinates(vec)
        
        assert r == pytest.approx(5.0)
        assert phi == pytest.approx(0.0)  # Along z-axis

    def test_spherical_coordinates_xy_plane(self):
        """Test spherical coordinates for vector in x-y plane."""
        vec = np.array([3.0, 4.0, 0.0])
        
        r, theta, phi = calc_spherical_coordinates(vec)
        
        assert r == pytest.approx(5.0)  # sqrt(3² + 4²)
        assert phi == pytest.approx(np.pi / 2)  # In x-y plane


class TestMake2d:
    """Tests for make_2d function."""

    def test_make_2d_simple(self):
        """Test stacking 1D arrays into 2D array."""
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        
        result = make_2d(arrays)
        
        assert result.shape == (3, 3)
        assert np.allclose(result[0], [1.0, 2.0, 3.0])
        assert np.allclose(result[1], [4.0, 5.0, 6.0])
        assert np.allclose(result[2], [7.0, 8.0, 9.0])
