"""Unit tests for constraint-weight logic (pure, no GPU) — SPEC §5.3 ordering caveat."""

import numpy as np
import pytest

from motionforge.planner.constraints import (
    approach_axis_in_goal_frame,
    hold_orientation_weights,
    reorder_hold_weight,
    vector_to_principal_axis,
)
from motionforge.types import SegmentConstraints

# wxyz quaternions for 90° rotations about base axes.
QX90 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
QY90 = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]


def test_reorder_hold_weight_orientation_first_to_position_first():
    # SPEC order [rx,ry,rz, x,y,z] -> ToolPoseCriteria order [x,y,z, roll,pitch,yaw].
    assert reorder_hold_weight([1, 1, 1, 0, 0, 0]) == [0, 0, 0, 1, 1, 1]
    assert reorder_hold_weight([0, 0, 0, 1, 1, 1]) == [1, 1, 1, 0, 0, 0]
    assert reorder_hold_weight([1, 2, 3, 4, 5, 6]) == [4, 5, 6, 1, 2, 3]


def test_reorder_hold_weight_validates_length():
    with pytest.raises(ValueError):
        reorder_hold_weight([1, 2, 3])


def test_vector_to_principal_axis():
    assert vector_to_principal_axis([0, 0, 1]) == "z"
    assert vector_to_principal_axis([0.9, 0.1, 0.0]) == "x"
    assert vector_to_principal_axis([0.0, -1.0, 0.2]) == "y"


def test_hold_orientation_weights_locks_orientation_along_path():
    c = SegmentConstraints(hold_orientation=True, hold_vec_weight=(1, 1, 1, 0, 0, 0))
    terminal, non_terminal = hold_orientation_weights(c)
    assert terminal == [1, 1, 1, 1, 1, 1]  # reach full pose at the end
    # Along the path, only orientation (roll,pitch,yaw) is penalized; position is free.
    assert non_terminal == [0, 0, 0, 1, 1, 1]


# -- goal-frame approach-axis mapping (the linear-approach bug fix) --


def test_approach_axis_identity_goal_unchanged():
    # Identity goal: goal frame == base frame, axis is preserved.
    out = approach_axis_in_goal_frame([0, 0, -1], [1, 0, 0, 0])
    assert np.allclose(out, [0, 0, -1])
    assert vector_to_principal_axis(out) == "z"


def test_approach_axis_rotated_about_x_maps_z_to_y():
    # Goal rotated 90° about base-x: base -Z maps to goal +Y (or -Y); principal axis becomes 'y'.
    out = approach_axis_in_goal_frame([0, 0, -1], QX90)
    assert vector_to_principal_axis(out) == "y"
    assert np.isclose(np.linalg.norm(out), 1.0)  # rotation preserves length


def test_approach_axis_rotated_about_y_maps_z_to_x():
    # Goal rotated 90° about base-y: base -Z maps to goal ±X; principal axis becomes 'x'.
    out = approach_axis_in_goal_frame([0, 0, -1], QY90)
    assert vector_to_principal_axis(out) == "x"
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_approach_axis_in_goal_frame_is_pure_rotation():
    # An axis aligned with the rotation axis is unchanged (rotation about x keeps x fixed).
    out = approach_axis_in_goal_frame([1, 0, 0], QX90)
    assert np.allclose(out, [1, 0, 0])
