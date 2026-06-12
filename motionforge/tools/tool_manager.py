"""Tool & Gripper Manager (SPEC §5.4).

Tool library keyed by ``tool_id``; tracks the active TCP, builds gripper collision geometry
as a function of the **commanded** grip width (never a fixed worst-case envelope), derives the
grasp transform (object pose in the TCP frame), and emits actuation commands. Pure data/numpy
— no GPU. ``collision_geom_fn`` returns a :class:`CollisionBody` with primitive params; the
collision world manager converts those to cuRobo geometry when adding the attached body.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from motionforge.geometry import Pose
from motionforge.geometry import grasp_transform as _grasp_transform
from motionforge.types import CollisionBody, GripConfig, ToolAction, ToolDescriptor

#: Identity pose list in cuRobo order [x,y,z, qw,qx,qy,qz].
_IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def collision_body_to_cuboid_specs(body: CollisionBody) -> List[dict]:
    """Flatten a gripper ``CollisionBody`` primitive into TCP-frame cuboid specs.

    Returns ``[{"name", "dims", "pose"}]`` where ``pose`` is ``[x,y,z, qw,qx,qy,qz]`` in the
    TCP frame (the convention :meth:`MotionPlannerAdapter.attach_tool` expects). Handles the
    parallel-jaw schema (``base`` + two ``jaws``) and the vacuum schema (a ``cylinder``, taken
    as its bounding cuboid for the MVP). The held-width dependence comes through the jaw
    offsets, so the geometry always reflects the COMMANDED width (SPEC §5.4).
    """
    if body.kind != "primitive":
        raise ValueError(f"collision_body_to_cuboid_specs expects kind='primitive', got {body.kind!r}")
    data = body.data
    specs: List[dict] = []
    if "jaws" in data:  # parallel-jaw
        if "base" in data:
            b = data["base"]
            specs.append({"name": "tool_base", "dims": list(b["dims"]), "pose": [*b["offset"], 1.0, 0.0, 0.0, 0.0]})
        for i, jaw in enumerate(data["jaws"]):
            specs.append({"name": f"tool_jaw{i}", "dims": list(jaw["dims"]), "pose": [*jaw["offset"], 1.0, 0.0, 0.0, 0.0]})
    elif "cylinder" in data:  # vacuum: bounding cuboid (2r × 2r × length) along +Z
        c = data["cylinder"]
        r, length = float(c["radius"]), float(c["length"])
        specs.append({"name": "tool_vacuum", "dims": [2 * r, 2 * r, length], "pose": [0.0, 0.0, length / 2.0, 1.0, 0.0, 0.0, 0.0]})
    else:
        raise ValueError(f"unrecognized gripper primitive schema: keys={sorted(data)}")
    return specs


class ToolManager:
    """Holds the tool library and the active tool/TCP."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}
        self._active: Optional[str] = None

    def register(self, tool: ToolDescriptor) -> None:
        self._tools[tool.tool_id] = tool
        if self._active is None:
            self._active = tool.tool_id

    def set_active(self, tool_id: str) -> None:
        if tool_id not in self._tools:
            raise KeyError(f"unknown tool_id: {tool_id!r}")
        self._active = tool_id

    def get(self, tool_id: str) -> ToolDescriptor:
        return self._tools[tool_id]

    @property
    def active_id(self) -> Optional[str]:
        return self._active

    @property
    def active(self) -> ToolDescriptor:
        if self._active is None:
            raise RuntimeError("no active tool registered")
        return self._tools[self._active]

    def active_tcp(self) -> Pose:
        """Active TCP pose relative to the flange (drives ``GoalToolPose.tool_frames``)."""
        return self.active.tcp_pose

    def collision_geom(self, grip: GripConfig, tool_id: Optional[str] = None) -> CollisionBody:
        """Gripper collision body for the **commanded** width (SPEC §5.4)."""
        tool = self._tools[tool_id or self._active]
        return tool.collision_geom_fn(grip)

    def grasp_transform(self, object_pose: Pose, grasp_pose: Pose) -> Pose:
        """Object pose in the TCP/grasp frame (``object_pose ⊖ grasp_pose``)."""
        return _grasp_transform(object_pose, grasp_pose)

    def actuation(
        self, grip: GripConfig, blocking: bool = True, tool_id: Optional[str] = None
    ) -> ToolAction:
        """Build a :class:`ToolAction` for the (active) tool."""
        return ToolAction(tool_id=tool_id or self._active, grip=grip, blocking=blocking)

    def payload_kg(self, tool_id: Optional[str] = None) -> float:
        return self._tools[tool_id or self._active].payload_kg


def parallel_jaw_geom_fn(
    jaw_length: float = 0.04,
    jaw_thickness: float = 0.02,
    jaw_height: float = 0.05,
    base_dims: Tuple[float, float, float] = (0.06, 0.08, 0.04),
):
    """Build a ``collision_geom_fn`` for a parallel-jaw gripper.

    Returns a function ``GripConfig -> CollisionBody`` whose two jaws are separated by the
    commanded ``width_m`` (so an open/standby width yields a wider envelope that must clear
    neighbours). Geometry is in the TCP frame; +Z points along the approach.
    """

    def fn(grip: GripConfig) -> CollisionBody:
        w = float(grip.width_m)
        half_sep = (w + jaw_thickness) / 2.0
        data = {
            "width": w,
            "base": {"dims": list(base_dims), "offset": [0.0, 0.0, -jaw_height / 2.0 - base_dims[2] / 2.0]},
            "jaws": [
                {"dims": [jaw_thickness, jaw_length, jaw_height], "offset": [half_sep, 0.0, 0.0]},
                {"dims": [jaw_thickness, jaw_length, jaw_height], "offset": [-half_sep, 0.0, 0.0]},
            ],
        }
        return CollisionBody(kind="primitive", data=data, frame="tcp")

    return fn


def vacuum_geom_fn(radius: float = 0.02, length: float = 0.04):
    """Build a ``collision_geom_fn`` for a vacuum/suction tool (cylinder along approach +Z)."""

    def fn(grip: GripConfig) -> CollisionBody:
        data = {"cylinder": {"radius": radius, "length": length}, "width": float(grip.width_m)}
        return CollisionBody(kind="primitive", data=data, frame="tcp")

    return fn
