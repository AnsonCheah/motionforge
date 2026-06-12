"""Map :class:`SegmentConstraints` onto cuRobo ``ToolPoseCriteria`` (SPEC §5.3).

cuRobo applies per-frame pose-cost criteria via ``planner.update_tool_pose_criteria``. Two
constraint kinds matter here:

- **linear approach** — straight-line along a principal axis (``ToolPoseCriteria.linear_motion``).
- **hold orientation** (vacuum/suction carry) — keep the tool face level along the whole path
  while still reaching the goal pose.

**Weight-vector ordering caveat (verified in source):** ``ToolPoseCriteria`` uses
``[x, y, z, roll, pitch, yaw]`` (position first). Our :data:`SegmentConstraints.hold_vec_weight`
follows the SPEC's ``[rx, ry, rz, x, y, z]`` (orientation first). :func:`reorder_hold_weight`
performs the conversion. The pure-logic helpers here are GPU-free and unit-tested; only the
final ``ToolPoseCriteria`` construction touches CUDA.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from motionforge.geometry import Pose, quat_conjugate, quat_rotate_vector
from motionforge.types import SegmentConstraints


def reorder_hold_weight(spec_weight) -> List[float]:
    """Convert SPEC order ``[rx,ry,rz, x,y,z]`` → ToolPoseCriteria ``[x,y,z, roll,pitch,yaw]``."""
    s = list(spec_weight)
    if len(s) != 6:
        raise ValueError(f"hold_vec_weight must have 6 entries, got {len(s)}")
    return [s[3], s[4], s[5], s[0], s[1], s[2]]


def vector_to_principal_axis(vec3) -> str:
    """Dominant principal axis ('x'|'y'|'z') of a vector."""
    a = np.abs(np.asarray(vec3, dtype=float).reshape(-1))
    return ["x", "y", "z"][int(np.argmax(a))]


def approach_axis_in_goal_frame(axis_base, goal_quat) -> np.ndarray:
    """Express a base-frame axis in the GOAL frame: ``R(goal_quat)^T @ axis_base``.

    cuRobo's ``ToolPoseCriteria.linear_motion(axis=...)`` with ``project_distance_to_goal=True``
    applies the axis weighting in the goal/tool frame (verified in
    ``curobo/_src/cost/wp_torch_pose_dist.py``), so a base-frame approach vector must be rotated
    into the goal frame before choosing the principal axis. Otherwise a rotated grasp gets the
    wrong free/constrained axes and the "straight-line" approach bends.
    """
    return quat_rotate_vector(quat_conjugate(goal_quat), np.asarray(axis_base, dtype=float))


def hold_orientation_weights(constraints: SegmentConstraints) -> tuple[List[float], List[float]]:
    """Return ``(terminal, non_terminal)`` weight vectors (ToolPoseCriteria order) for a
    hold-orientation segment: reach the full pose at the end, penalize orientation drift
    along the path. Pure logic (unit-tested)."""
    terminal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    non_terminal = reorder_hold_weight(constraints.hold_vec_weight)
    return terminal, non_terminal


def build_tool_pose_criteria(
    constraints: SegmentConstraints,
    device_cfg=None,
    axis: Optional[str] = None,
    goal: Optional[Pose] = None,
):
    """Build the cuRobo ``ToolPoseCriteria`` for a segment (CUDA — lazy import).

    ``goal`` (the segment's target pose) is used to map the base-frame ``approach_axis`` into
    the goal frame before picking the linear-motion principal axis. Without it, the base-frame
    axis is used directly (correct only when the goal orientation is axis-aligned with base).
    """
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo._src.types.device_cfg import DeviceCfg

    dc = device_cfg or DeviceCfg()

    if constraints.linear_approach:
        if axis is not None:
            principal = axis
        elif constraints.approach_axis is not None:
            axis_vec = constraints.approach_axis
            if goal is not None:
                axis_vec = approach_axis_in_goal_frame(axis_vec, goal.quaternion)
            principal = vector_to_principal_axis(axis_vec)
        else:
            principal = "z"
        return ToolPoseCriteria.linear_motion(
            axis=principal, non_terminal_scale=1.0, project_distance_to_goal=True
        )

    if constraints.hold_orientation:
        terminal, non_terminal = hold_orientation_weights(constraints)
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=terminal,
            non_terminal_pose_axes_weight_factor=non_terminal,
            device_cfg=dc,
        )

    return ToolPoseCriteria(device_cfg=dc)
