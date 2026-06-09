"""Unit tests for constraint-weight logic (pure, no GPU) — SPEC §5.3 ordering caveat."""

import pytest

from motionforge.planner.constraints import (
    hold_orientation_weights,
    reorder_hold_weight,
    vector_to_principal_axis,
)
from motionforge.types import SegmentConstraints


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
