"""Segment / Constraint Builder (SPEC §5.3) — expand a phase template into MotionSegments.

Pure geometry/data (numpy, CPU), unit-testable without a GPU. The pick template
(approach → grasp → retract) maps onto cuRobo's native ``plan_grasp`` (the coordinator calls
it once and associates the approach/grasp/lift trajectories with these segments); the place
template (transport → place → release → retract) maps onto ``plan_pose`` with per-segment
:class:`SegmentConstraints` (hold-orientation carry, linear approach).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from motionforge.config import DEFAULTS, Config
from motionforge.geometry import Pose, offset_along_axis, pre_grasp_pose
from motionforge.types import (
    GraspCandidate,
    GripConfig,
    MotionSegment,
    PlaceCandidate,
    SegmentConstraints,
    ToolAction,
)


def _retract_pose(tcp: Pose, approach_axis, distance_m: float) -> Pose:
    """Straight-line back-off: move opposite the approach direction (lift out)."""
    return offset_along_axis(tcp, approach_axis, -abs(distance_m))


def build_pick_segments(
    grasp: GraspCandidate,
    config: Config = DEFAULTS,
    standby_width_m: Optional[float] = None,
) -> List[MotionSegment]:
    """approach → grasp → retract, with concurrent gripper-open and post-grasp grip barriers."""
    standby = standby_width_m if standby_width_m is not None else config.gripper_standby_width_m
    axis = grasp.approach_axis
    pre = pre_grasp_pose(grasp.tcp_pose, axis, grasp.standoff_m)
    retract = _retract_pose(grasp.tcp_pose, axis, config.retract_m)

    free = SegmentConstraints(min_clearance_m=config.min_clearance_m)
    linear = SegmentConstraints(
        linear_approach=True, approach_axis=axis, offset_m=grasp.standoff_m,
        min_clearance_m=config.min_clearance_m,
    )
    # Open to standby concurrently with the approach (non-blocking); grip after the grasp move.
    open_to_standby = ToolAction(
        tool_id=grasp.tool_id, grip=GripConfig(width_m=standby, mode=grasp.grip.mode), blocking=False
    )
    close_and_grip = ToolAction(tool_id=grasp.tool_id, grip=grasp.grip, blocking=True)

    return [
        MotionSegment("approach", pre, free, pre_action=open_to_standby),
        MotionSegment("grasp", grasp.tcp_pose, linear, post_action=close_and_grip),
        MotionSegment("retract", retract, linear),
    ]


def build_place_segments(
    place: PlaceCandidate,
    config: Config = DEFAULTS,
) -> List[MotionSegment]:
    """transport(orientation-held) → place(linear) → release → retract."""
    axis = place.approach_axis
    pre = pre_grasp_pose(place.tcp_pose, axis, place.standoff_m)
    retract = _retract_pose(place.tcp_pose, axis, config.retract_m)

    free = SegmentConstraints(min_clearance_m=config.min_clearance_m)
    hold = SegmentConstraints(
        hold_orientation=True, approach_axis=axis,
        hold_vec_weight=config.hold_vec_weight_vacuum_carry,
        min_clearance_m=config.min_clearance_m,
    )
    linear = SegmentConstraints(
        linear_approach=True, approach_axis=axis, offset_m=place.standoff_m,
        min_clearance_m=config.min_clearance_m,
    )
    release = ToolAction(tool_id=place.tool_id, grip=place.grip, blocking=True)

    return [
        MotionSegment("transport", pre, hold),
        MotionSegment("place", place.tcp_pose, linear),
        # Actuation-only: no motion (goal == place); the coordinator just runs the release barrier.
        MotionSegment("release", place.tcp_pose, free, post_action=release),
        MotionSegment("retract", retract, linear),
    ]


def build_cycle(
    grasp: GraspCandidate,
    place: PlaceCandidate,
    config: Config = DEFAULTS,
    standby_width_m: Optional[float] = None,
) -> Dict[str, List[MotionSegment]]:
    """Full pick-and-place template: ``{"pick": [...], "place": [...]}``."""
    return {
        "pick": build_pick_segments(grasp, config, standby_width_m),
        "place": build_place_segments(place, config),
    }
