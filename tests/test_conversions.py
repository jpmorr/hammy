#!/usr/bin/env python
"""Comprehensive validation suite for Quaternion representations and transformations.

Conventions verified:
- Quaternions are scalar-first: [w, x, y, z].
- Intrinsic rotations follow moving/body axes; extrinsic follow fixed/static axes.
- Double-cover SO(3) invariance: q and -q represent identical spatial rotations.
"""

__author__ = 'J.P. Morrissey'
__copyright__ = 'Copyright 2023, {project_name}'

# Standard Library


# Imports
import numpy as np
import pytest

# Local Sources
from hammy.Quaternion import Quaternion
from hammy.utilities import _compose_quaternion_rotation, normalise

rots2 = (0.4636476090008061, 0.3217505543966422, 0.16514867741462683)
angles = np.asarray(rots2, dtype=float)

@pytest.mark.parametrize("angle_input, seq_input, expected_output",
                         [(angles, 'xzy', np.array([[0.95437304, 0.23887198, 0.11591449, 0.13666057]])),
                          (angles, 'xyz', np.array([[0.96044399, 0.21315501, 0.17407146, 0.04256061]])),
                          (angles, 'yzx', np.array([[0.96044399, 0.04256061, 0.21315501, 0.17407146]])),
                          (angles, 'yxz', np.array([[0.95437304, 0.13666057, 0.23887198, 0.11591449]])),
                          (angles, 'zxy', np.array([[0.96044399, 0.17407146, 0.04256061, 0.21315501]])),
                          (angles, 'zyx', np.array([[0.95437304, 0.11591449, 0.13666057, 0.23887198]]))])
def test_compose_quaternion(angle_input, seq_input, expected_output):
    ans = _compose_quaternion_rotation(angle_input, seq=seq_input)
    assert np.allclose(ans, expected_output)


@pytest.mark.parametrize("angle_input, seq_input, expected_output",
                         [(angles, 'xzy', np.array([29.884901610345047, 11.85718184843569, 17.024904492212205])),
                          (angles, 'xyz', np.array([26.284766549922825, 16.650929234610523, 12.384188650990287])),
                          (angles, 'yzx', np.array([9.462322208025615, 18.43494882292201, 26.565051177077994])),
                          (angles, 'yxz', np.array([18.4471991397113, 25.104090250221386, 20.439317573258254])),
                          (angles, 'zxy', np.array([13.612266859928262, 29.184784532900252, 10.29059603838653])),
                          (angles, 'zyx', np.array([18.671718064007877, 8.972626614896392, 29.582012486893984]))])
def test_get_tait_bryan(angle_input, seq_input, expected_output):
    r4 = Quaternion.from_euler(angles, seq='xzy')
    ans = r4.to_tait_bryan(seq=seq_input, degrees=True)
    assert np.allclose(ans, expected_output)

@pytest.mark.parametrize("quat_input, seq_input, intrinsic, expected_output",
                         [(angles, 'xyz', True, np.array([-121.276486, 19.5007892, -157.5187063])),
                          (angles, 'xzy', True, np.array([66.58885942, -21.12766087, 159.0301626])),
                          (angles, 'yxz', True, np.array([145.7014007, -53.67262417, 51.27207389])),
                          (angles, 'yzx', True, np.array([-169.1669535, 27.52560531, -65.29633378])),
                          (angles, 'zxy', True, np.array([-44.20294005, 58.86936406, -161.1924521])),
                          (angles, 'zyx', True, np.array([152.0499881, -9.59443435, 119.7574795])),
                          (angles, 'xyz', False, np.array([119.7574795, -9.59443435, 152.0499881])),
                          (angles, 'xzy', False, np.array([-65.29633378, 27.52560531, -169.1669535])),
                          (angles, 'yxz', False, np.array([-161.1924521, 58.86936406, -44.20294005])),
                          (angles, 'yzx', False, np.array([159.0301626, -21.12766087, 66.58885942])),
                          (angles, 'zxy', False, np.array([51.27207389, -53.67262417, 145.7014007])),
                          (angles, 'zyx', False, np.array([-157.5187063, 19.5007892, -121.276486]))])
def test_get_tait_bryan_from_quat(quat_input, seq_input, intrinsic, expected_output):
    r5 = Quaternion((-0.05057163, -0.24888186, -0.82628776, -0.50274135))
    ans = r5.to_tait_bryan(seq=seq_input, degrees=True, intrinsic=intrinsic)
    assert np.allclose(ans, expected_output)



@pytest.mark.parametrize("angle_input, seq_input, intrinsic, expected_output",
                         [(angles, 'xzx', True, np.array([-26.25237512, 20.64621396, 54.35635682])),
                          (angles, 'xyx', True, np.array([63.74762488, 20.64621396, -35.64364318])),
                          (angles, 'yxy', True, np.array([-22.84921103, 31.94805943, 36.6992252])),
                          (angles, 'yzy', True, np.array([67.15078897, 31.94805943, -53.3007748])),
                          (angles, 'zxz', True, np.array([34.03436266, 30.79461491, -17.73632227])),
                          (angles, 'zyz', True, np.array([-55.96563734, 30.79461491, 72.26367773]))])
def test_get_euler(angle_input, seq_input, intrinsic, expected_output):
    r4 = Quaternion.from_euler(angles, seq='xzy')
    ans = r4.to_euler(seq=seq_input, degrees=True, intrinsic=intrinsic)
    assert np.allclose(ans, expected_output)

@pytest.mark.parametrize("quat_input, seq_input, intrinsic, expected_output",
                         [(angles, 'xzx', True, np.array([19.83188107, 150.5751316, 137.1963972])),
                          (angles, 'xyx', True, np.array([109.8318811, 150.5751316, 47.19639719])),
                          (angles, 'yxy', True, np.array([22.83536002, 68.2462697, 150.159983])),
                          (angles, 'yzy', True, np.array([112.83536, 68.2462697, 60.15998304])),
                          (angles, 'zxz', True, np.array([157.4932852, 119.3003178, 11.01839475])),
                          (angles, 'zyz', True, np.array([67.49328524, 119.3003178, 101.0183948])),
                          (angles, 'xzx', False, np.array([137.1963972, 150.5751316, 19.83188107])),
                          (angles, 'xyx', False, np.array([47.19639719, 150.5751316, 109.8318811])),
                          (angles, 'yxy', False, np.array([150.159983, 68.2462697, 22.83536002])),
                          (angles, 'yzy', False, np.array([60.15998304, 68.2462697, 112.83536])),
                          (angles, 'zxz', False, np.array([11.01839475, 119.3003178, 157.4932852])),
                          (angles, 'zyz', False, np.array([101.0183948, 119.3003178, 67.49328524]))])
def test_get_euler_from_quat(quat_input, seq_input, intrinsic, expected_output):
    r5 = Quaternion((-0.05057163, -0.24888186, -0.82628776, -0.50274135))
    ans = r5.to_euler(seq=seq_input, degrees=True, intrinsic=intrinsic)
    assert np.allclose(ans, expected_output)


def test_get_rotation_matrix():
    r4 = Quaternion.from_euler(angles, seq='xzy')
    ans = r4.to_rotation_matrix()
    expected_output = np.array([[0.93577544, -0.20547287, 0.28654009],
                                [0.31622777, 0.84852814, -0.42426407],
                                [-0.15596257, 0.48762783, 0.85900802]])
    assert np.allclose(ans, expected_output)


def test_axis_angle_to_quat():
    vec = np.array([0.7999276420664455, 0.38817112273710197, 0.4576450009898877])
    theta = 0.6064863777699238
    quat = Quaternion.from_axisangle(vec, theta)
    expected_output = np.array([0.9543730387151051, 0.23887197967840104,  0.11591448984394159, 0.13666056981601069])

    assert np.allclose(quat.ndarray, expected_output)


class TestEulerAndTaitBryanConversions:
    """Validates all 12 rotation sequences across intrinsic and extrinsic modes."""

    @pytest.mark.parametrize("seq", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    @pytest.mark.parametrize("intrinsic", [True, False])
    def test_tait_bryan_invariants(self, seq, intrinsic):
        """Validate Tait-Bryan angle round-trip invariance across intrinsic and extrinsic modes."""
        test_angles = np.array([15.0, 25.0, 35.0])
        q = Quaternion.from_euler(test_angles, seq=seq, degrees=True, intrinsic=intrinsic)
        recovered = q.to_tait_bryan(seq=seq, degrees=True, intrinsic=intrinsic)
        q_rec = Quaternion.from_euler(recovered, seq=seq, degrees=True, intrinsic=intrinsic)

        v = np.array([1.0, -2.0, 3.5])
        assert np.allclose(q * v, q_rec * v, atol=1e-6)

    @pytest.mark.parametrize("seq", ["xzx", "xyx", "yxy", "yzy", "zxz", "zyz"])
    @pytest.mark.parametrize("intrinsic", [True, False])
    def test_proper_euler_invariants(self, seq, intrinsic):
        """Validate proper Euler angle round-trip invariance across intrinsic and extrinsic modes."""
        test_angles = np.array([20.0, 45.0, 70.0])
        q = Quaternion.from_euler(test_angles, seq=seq, degrees=True, intrinsic=intrinsic)
        recovered = q.to_euler(seq=seq, degrees=True, intrinsic=intrinsic)
        q_rec = Quaternion.from_euler(recovered, seq=seq, degrees=True, intrinsic=intrinsic)

        v = np.array([2.0, 1.0, -1.0])
        assert np.allclose(q * v, q_rec * v, atol=1e-6)


class TestVectorizedBatchOperations:
    """Ensures high-throughput array computations match single-quaternion evaluations."""

    def test_batch_vector_rotation(self):
        """Validate batch vector rotation matches single quaternion evaluations."""
        n = 100
        angles = np.random.uniform(-np.pi, np.pi, size=(n, 3))
        q_batch = Quaternion.from_euler(angles, seq="zyx")
        v = np.array([1.0, 0.0, 0.0])

        rotated_batch = q_batch * v
        assert rotated_batch.shape == (n, 3)

        for i in range(min(5, n)):
            q_single = Quaternion(q_batch[i])
            assert np.allclose(rotated_batch[i], q_single * v)


class TestAxisAngleConversions:
    """Validates axis-angle conversions for both single and batched quaternions."""

    def test_single_axis_angle_roundtrip(self):
        """Validate single axis-angle conversion and recovery."""
        axis = np.array([0.0, 1.0, 0.0])
        theta = np.pi / 3
        q = Quaternion.from_axisangle(axis, theta)
        rec_axis, rec_theta = q.to_axis_angle()
        assert np.allclose(axis, rec_axis)
        assert np.isclose(theta, rec_theta)

    def test_single_axis_angle_degrees(self):
        """Validate degrees parameter in single axis-angle conversion."""
        axis = np.array([1.0, 0.0, 0.0])
        theta_deg = 45.0
        q = Quaternion.from_axisangle(axis, theta_deg, degrees=True)
        rec_axis, rec_theta = q.to_axis_angle(degrees=True)
        assert np.allclose(axis, rec_axis)
        assert np.isclose(theta_deg, rec_theta)

    def test_batch_axis_angle_roundtrip(self):
        """Validate batched axis-angle conversion and SO(3) vector rotation invariance."""
        np.random.seed(42)
        n = 50
        axes = np.random.randn(n, 3)
        thetas = np.random.uniform(0.1, np.pi - 0.1, size=n)
        q = Quaternion.from_axisangle(axes, thetas)
        assert q.num_quaternions == n
        rec_axes, rec_thetas = q.to_axis_angle()
        assert rec_axes.shape == (n, 3)
        assert rec_thetas.shape == (n,)

        # Invariant: q * v == q_rec * v
        q_rec = Quaternion.from_axisangle(rec_axes, rec_thetas)
        v = np.array([1.0, 2.0, -1.0])
        assert np.allclose(q * v, q_rec * v)

    def test_broadcasting_axis_and_angles(self):
        """Validate broadcasting between axes and angles arrays."""
        # Single axis, multiple angles
        axis = np.array([0.0, 0.0, 1.0])
        thetas = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        q = Quaternion.from_axisangle(axis, thetas)
        assert q.num_quaternions == 4
        rec_axes, rec_thetas = q.to_axis_angle()
        assert np.allclose(rec_thetas, thetas)

        # Multiple axes, single angle
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        theta = np.pi / 2
        q2 = Quaternion.from_axisangle(axes, theta)
        assert q2.num_quaternions == 3
        rec_axes2, rec_thetas2 = q2.to_axis_angle()
        assert np.allclose(rec_thetas2, [theta, theta, theta])

    def test_rotation_vector(self):
        """Validate rotation vector generation for single and batched quaternions."""
        # Single
        axis = np.array([0.0, 1.0, 0.0])
        theta = 1.2
        q = Quaternion.from_axisangle(axis, theta)
        rotvec = q.get_rotation_vector()
        assert rotvec.shape == (3,)
        assert np.allclose(rotvec, axis * theta)

        # Batch
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        thetas = np.array([0.5, 1.5])
        q_batch = Quaternion.from_axisangle(axes, thetas)
        rotvec_batch = q_batch.get_rotation_vector()
        assert rotvec_batch.shape == (2, 3)
        assert np.allclose(rotvec_batch[0], axes[0] * thetas[0])
        assert np.allclose(rotvec_batch[1], axes[1] * thetas[1])

    def test_unnormalized_axis_and_quat(self):
        """Validate automatic normalisation and scale invariance for unnormalized inputs."""
        # Unnormalized axis
        axis_raw = np.array([0.0, 5.0, 0.0])
        theta = np.pi / 2
        q = Quaternion.from_axisangle(axis_raw, theta)
        rec_axis, rec_theta = q.to_axis_angle()
        assert np.allclose(rec_axis, [0.0, 1.0, 0.0])
        assert np.isclose(rec_theta, theta)

        # Unnormalized quaternion
        q_scaled = Quaternion(q.ndarray * 3.5)
        rec_axis2, rec_theta2 = q_scaled.to_axis_angle()
        assert np.allclose(rec_axis2, [0.0, 1.0, 0.0])
        assert np.isclose(rec_theta2, theta)