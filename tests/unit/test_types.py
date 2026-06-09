"""Unit tests for motionforge.types data contracts (SPEC §4)."""

from motionforge.geometry import Pose
from motionforge.types import (
    CollisionBody,
    GraspCandidate,
    GripConfig,
    JointTrajectory,
    MotionSegment,
    PlaceCandidate,
    PlanResult,
    SegmentConstraints,
    ToolAction,
    ToolDescriptor,
)


def test_grip_config_defaults():
    g = GripConfig(width_m=0.04)
    assert g.width_m == 0.04
    assert g.force == 0.0
    assert g.mode == "inward"


def test_grasp_and_place_candidates_share_shape():
    grip = GripConfig(width_m=0.02, force=20.0, mode="inward")
    grasp = GraspCandidate(
        tcp_pose=Pose([0.5, 0.0, 0.3], [1, 0, 0, 0]),
        approach_axis=[0, 0, 1],
        standoff_m=0.05,
        tool_id="parallel_jaw",
        grip=grip,
    )
    place = PlaceCandidate(
        tcp_pose=Pose([0.5, 0.3, 0.2], [1, 0, 0, 0]),
        approach_axis=[0, 0, 1],
        standoff_m=0.05,
        tool_id="parallel_jaw",
        grip=GripConfig(width_m=0.08, mode="outward"),
    )
    assert grasp.tool_id == place.tool_id
    assert grasp.grip.width_m == 0.02


def test_segment_constraints_defaults():
    c = SegmentConstraints()
    assert c.linear_approach is False
    assert c.approach_axis is None
    assert c.hold_orientation is False
    assert c.hold_vec_weight == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert c.min_clearance_m == 0.01


def test_motion_segment_with_pre_post_actions():
    open_action = ToolAction(tool_id="jaw", grip=GripConfig(0.08), blocking=False)
    grip_action = ToolAction(tool_id="jaw", grip=GripConfig(0.02, force=30.0), blocking=True)
    seg = MotionSegment(
        name="grasp",
        goal=Pose([0.5, 0.0, 0.3], [1, 0, 0, 0]),
        pre_action=open_action,
        post_action=grip_action,
    )
    assert seg.name == "grasp"
    assert seg.pre_action.blocking is False
    assert seg.post_action.blocking is True
    # Default constraints are attached.
    assert isinstance(seg.constraints, SegmentConstraints)


def test_collision_body_and_tool_descriptor():
    def geom_fn(grip: GripConfig) -> CollisionBody:
        return CollisionBody(kind="primitive", data={"width": grip.width_m}, frame="tcp")

    tool = ToolDescriptor(
        tool_id="jaw",
        tcp_pose=Pose([0, 0, 0.1], [1, 0, 0, 0]),
        collision_geom_fn=geom_fn,
        actuation_iface="socket://gripper",
        payload_kg=0.5,
    )
    body = tool.collision_geom_fn(GripConfig(0.05))
    assert body.kind == "primitive"
    assert body.frame == "tcp"
    assert body.data["width"] == 0.05


def test_joint_trajectory_duration_and_len():
    traj = JointTrajectory(
        joint_names=["j1", "j2"],
        points=[
            ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0),
            ([0.1, 0.2], [0.0, 0.0], [0.0, 0.0], 0.5),
        ],
    )
    assert len(traj) == 2
    assert traj.duration_s == 0.5


def test_plan_result_defaults():
    r = PlanResult(success=True)
    assert r.success is True
    assert r.trajectory is None
    assert r.candidate_index == -1
    assert r.metrics == {}
