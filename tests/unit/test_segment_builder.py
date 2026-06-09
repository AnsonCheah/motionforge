"""Unit tests for the Segment / Constraint Builder (SPEC §5.3) — pure, no GPU."""

import numpy as np

from motionforge.config import DEFAULTS
from motionforge.geometry import Pose
from motionforge.planner.segment_builder import (
    build_cycle,
    build_pick_segments,
    build_place_segments,
)
from motionforge.types import GraspCandidate, GripConfig, PlaceCandidate


def _grasp():
    return GraspCandidate(
        tcp_pose=Pose([0.5, 0.0, 0.3], [1, 0, 0, 0]),
        approach_axis=[0, 0, 1],
        standoff_m=0.05,
        tool_id="jaw",
        grip=GripConfig(width_m=0.02, force=30.0, mode="inward"),
    )


def _place():
    return PlaceCandidate(
        tcp_pose=Pose([0.4, 0.3, 0.2], [1, 0, 0, 0]),
        approach_axis=[0, 0, 1],
        standoff_m=0.05,
        tool_id="jaw",
        grip=GripConfig(width_m=0.08, mode="outward"),
    )


def test_pick_segment_order_and_actions():
    segs = build_pick_segments(_grasp())
    assert [s.name for s in segs] == ["approach", "grasp", "retract"]

    approach, grasp, retract = segs
    # Concurrent gripper open to standby width, non-blocking.
    assert approach.pre_action is not None
    assert approach.pre_action.blocking is False
    assert approach.pre_action.grip.width_m == DEFAULTS.gripper_standby_width_m
    # Grip closes to the grasp width after the grasp move, as a blocking barrier.
    assert grasp.post_action is not None
    assert grasp.post_action.blocking is True
    assert grasp.post_action.grip.width_m == 0.02
    # Grasp + retract are straight-line moves.
    assert grasp.constraints.linear_approach is True
    assert retract.constraints.linear_approach is True


def test_pick_approach_goal_is_pregrasp_offset():
    g = _grasp()
    approach = build_pick_segments(g)[0]
    # Pre-grasp is standoff back along -approach_axis (below the grasp by 5 cm).
    assert np.allclose(approach.goal.position, [0.5, 0.0, 0.25], atol=1e-12)
    assert np.allclose(approach.goal.quaternion, g.tcp_pose.quaternion)
    # The grasp segment lands exactly on the candidate pose (no relaxation).
    grasp = build_pick_segments(g)[1]
    assert np.allclose(grasp.goal.position, [0.5, 0.0, 0.3])


def test_pick_retract_lifts_opposite_approach():
    retract = build_pick_segments(_grasp())[2]
    # Retract = grasp lifted by retract_m along -approach (up).
    assert np.allclose(retract.goal.position, [0.5, 0.0, 0.3 - DEFAULTS.retract_m], atol=1e-12)


def test_place_segment_order_and_constraints():
    segs = build_place_segments(_place())
    assert [s.name for s in segs] == ["transport", "place", "release", "retract"]

    transport, place, release, retract = segs
    # Transport carries with orientation held (vacuum-carry weights, orientation-first SPEC order).
    assert transport.constraints.hold_orientation is True
    assert transport.constraints.hold_vec_weight == DEFAULTS.hold_vec_weight_vacuum_carry
    # Place is a linear approach; release is a blocking barrier opening to the release config.
    assert place.constraints.linear_approach is True
    assert release.post_action is not None
    assert release.post_action.blocking is True
    assert release.post_action.grip.width_m == 0.08


def test_place_transport_goal_is_preplace_offset():
    p = _place()
    transport = build_place_segments(p)[0]
    assert np.allclose(transport.goal.position, [0.4, 0.3, 0.15], atol=1e-12)


def test_build_cycle_returns_both_phases():
    cycle = build_cycle(_grasp(), _place())
    assert [s.name for s in cycle["pick"]] == ["approach", "grasp", "retract"]
    assert [s.name for s in cycle["place"]] == ["transport", "place", "release", "retract"]
