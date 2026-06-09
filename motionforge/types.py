"""Core data contracts (SPEC §4).

These are **our** module contracts, not cuRobo's API. The Motion Planner adapter
(``motionforge.planner.motion_planner``) maps them onto cuRobo's ``GoalToolPose`` /
``plan_pose`` / ``plan_grasp``. All poses are SE3 in the robot **base frame** unless
stated, using :class:`motionforge.geometry.Pose` (position metres, quaternion wxyz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from motionforge.geometry import Pose

Vec = Sequence[float]
Vec3 = Sequence[float]
#: cuRobo pose-cost weight order per SPEC: ``[rx, ry, rz, x, y, z]`` (orientation first).
#: NOTE: cuRobo's ``ToolPoseCriteria`` uses the OPPOSITE order ``[x, y, z, roll, pitch,
#: yaw]`` (position first); ``planner.constraints`` performs the reordering.
Tuple6 = Tuple[float, float, float, float, float, float]


@dataclass
class GripConfig:
    """Gripper command. ``width_m`` drives collision geometry; ``force``/``mode`` are
    execution-only and ignored by the planner."""

    width_m: float
    force: float = 0.0
    mode: str = "inward"  # e.g. "inward" | "outward" | "vacuum_on"


@dataclass
class GraspCandidate:
    """One ranked grasp target. Priority is implicit in list order (no score field).

    The ranked list becomes the ``num_goalset`` dimension of a cuRobo ``GoalToolPose``;
    the planner selects the most reachable entry.
    """

    tcp_pose: Pose  # EXACT target pose (no relaxation), base frame
    approach_axis: Vec3  # unit vector, base frame; usually tool +Z, configurable (XY allowed)
    standoff_m: float  # pre-grasp offset along -approach_axis -> grasp_approach_offset
    tool_id: str
    grip: GripConfig  # grip state to achieve at grasp


@dataclass
class PlaceCandidate:
    """Symmetric to :class:`GraspCandidate`; ``grip`` is the release config."""

    tcp_pose: Pose  # EXACT, base frame
    approach_axis: Vec3
    standoff_m: float
    tool_id: str
    grip: GripConfig  # release config


@dataclass
class SegmentConstraints:
    """Per-segment constraint spec mapped onto cuRobo ``ToolPoseCriteria`` / ``plan_grasp``."""

    linear_approach: bool = False  # straight-line along approach_axis
    approach_axis: Optional[Vec3] = None  # frame to align the linear/orientation cost to
    offset_m: float = 0.0  # standoff for the grasp-approach metric
    hold_orientation: bool = False  # lock all 3 rotation DOF (e.g. vacuum stays level)
    #: cuRobo pose-cost order [rx,ry,rz, x,y,z] (orientation first). See :data:`Tuple6`.
    hold_vec_weight: Tuple6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_clearance_m: float = 0.01  # hard filter threshold (reject below)


@dataclass
class ToolAction:
    """A gripper/tool actuation tied to a segment. If ``blocking``, the coordinator waits
    for completion (sync barrier)."""

    tool_id: str
    grip: GripConfig
    blocking: bool = True


@dataclass
class MotionSegment:
    """One step of the pick-place template with its own goal, constraints, and optional
    pre/post tool actions."""

    name: str  # "approach"|"grasp"|"retract"|"transport"|"place"|"release"|"free"
    goal: Pose  # base-frame TCP goal (exact)
    constraints: SegmentConstraints = field(default_factory=SegmentConstraints)
    pre_action: Optional[ToolAction] = None  # actuation to complete BEFORE this segment runs
    post_action: Optional[ToolAction] = None  # actuation to complete AFTER this segment runs


@dataclass
class CollisionBody:
    """A collision body for the cuRobo scene or an attached object."""

    kind: str  # "mesh" | "voxel" | "primitive"
    data: Any  # trimesh / VoxelGrid / primitive params
    frame: str  # "tcp" for attached object, "base" for world


@dataclass
class ToolDescriptor:
    """A tool in the tool library. ``collision_geom_fn`` returns geometry as a function of
    commanded grip width (SPEC §5.4)."""

    tool_id: str
    tcp_pose: Pose  # TCP relative to flange; becomes the active TCP
    collision_geom_fn: Callable[[GripConfig], CollisionBody]
    actuation_iface: str  # topic/service/socket descriptor
    payload_kg: float = 0.0


@dataclass
class JointTrajectory:
    """Timed, dense joint trajectory (internal representation).

    ``points`` are ``(positions, velocities, accelerations, time_from_start_s)`` tuples.
    """

    joint_names: List[str]
    points: List[Tuple[Vec, Vec, Vec, float]]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def duration_s(self) -> float:
        return self.points[-1][3] if self.points else 0.0


@dataclass
class PlanResult:
    """Result of a planning call mapped from cuRobo's solver result."""

    success: bool
    trajectory: Optional[JointTrajectory] = None  # timed, dense
    candidate_index: int = -1  # which goalset entry was selected
    metrics: Dict[str, float] = field(default_factory=dict)  # cycle_time, peak_jerk, ...
