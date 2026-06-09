"""Unit tests for motionforge.geometry (CPU, no GPU)."""

import numpy as np
import pytest

from motionforge.geometry import (
    Pose,
    axis_to_vector,
    grasp_transform,
    offset_along_axis,
    pre_grasp_pose,
    quat_multiply,
    quat_rotate_vector,
)

# 90-degree rotation about +Z, wxyz.
QZ90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def test_axis_to_vector():
    assert np.allclose(axis_to_vector("x"), [1, 0, 0])
    assert np.allclose(axis_to_vector("y"), [0, 1, 0])
    assert np.allclose(axis_to_vector("Z"), [0, 0, 1])
    with pytest.raises(ValueError):
        axis_to_vector("w")


def test_quat_rotate_vector_z90():
    # Rotating +X by +90 deg about Z yields +Y.
    v = quat_rotate_vector(QZ90, [1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_multiply_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_multiply(q, QZ90), QZ90)


def test_pose_identity_and_normalization():
    # A non-unit quaternion is normalized; w-sign canonicalized.
    p = Pose(position=[1, 2, 3], quaternion=[0.0, 0.0, 0.0, -2.0])
    assert np.isclose(np.linalg.norm(p.quaternion), 1.0)
    assert p.quaternion[0] >= 0.0  # canonical sign (here w=0, z flipped positive)


def test_pose_inverse_roundtrip():
    p = Pose(position=[0.5, -0.2, 0.3], quaternion=QZ90)
    composed = p.multiply(p.inverse())
    assert composed.approx_equal(Pose.identity(), atol=1e-9)
    # Other order too.
    assert p.inverse().multiply(p).approx_equal(Pose.identity(), atol=1e-9)


def test_pose_multiply_local_frame_offset():
    # A pose rotated +90 about Z at the origin; offset (1,0,0) in its LOCAL frame
    # should land at world (0,1,0).
    p = Pose(position=[0, 0, 0], quaternion=QZ90)
    offset = Pose(position=[1, 0, 0], quaternion=[1, 0, 0, 0])
    out = p.multiply(offset)
    assert np.allclose(out.position, [0, 1, 0], atol=1e-9)


def test_curobo_list_roundtrip():
    p = Pose(position=[0.1, 0.2, 0.3], quaternion=QZ90)
    lst = p.to_curobo_list()
    assert len(lst) == 7
    assert lst[:3] == [0.1, 0.2, 0.3]
    assert Pose.from_list(lst).approx_equal(p)


def test_pre_grasp_pose_offsets_against_approach():
    grasp = Pose(position=[0.5, 0.0, 0.3], quaternion=[1, 0, 0, 0])
    pre = pre_grasp_pose(grasp, approach_axis_base=[0, 0, 1], standoff_m=0.05)
    # Pre-grasp is 5 cm back along -Z (below the grasp), orientation unchanged.
    assert np.allclose(pre.position, [0.5, 0.0, 0.25], atol=1e-12)
    assert np.allclose(pre.quaternion, grasp.quaternion)


def test_pre_grasp_uses_negative_direction_regardless_of_sign():
    grasp = Pose(position=[0.0, 0.0, 0.0], quaternion=[1, 0, 0, 0])
    # Passing a negative standoff still moves opposite the approach axis.
    pre = pre_grasp_pose(grasp, approach_axis_base=[1, 0, 0], standoff_m=-0.1)
    assert np.allclose(pre.position, [-0.1, 0.0, 0.0])


def test_offset_along_axis_normalizes_axis():
    p = Pose(position=[0, 0, 0], quaternion=[1, 0, 0, 0])
    out = offset_along_axis(p, axis_unit_base=[0, 0, 5.0], distance_m=0.2)
    assert np.allclose(out.position, [0, 0, 0.2])


def test_grasp_transform_recovers_object_in_tcp_frame():
    grasp = Pose(position=[0.5, 0.0, 0.3], quaternion=QZ90)
    obj = Pose(position=[0.5, 0.1, 0.3], quaternion=QZ90)
    t_tcp_obj = grasp_transform(obj, grasp)
    # Re-composing the grasp with the relative transform recovers the object pose.
    assert grasp.multiply(t_tcp_obj).approx_equal(obj, atol=1e-9)
