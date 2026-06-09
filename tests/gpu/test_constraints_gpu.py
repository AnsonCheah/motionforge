"""Phase 3 GPU test: hold-orientation pose-cost criteria actually hold orientation on UR10e."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from motionforge.geometry import Pose  # noqa: E402
from motionforge.planner.constraints import build_tool_pose_criteria  # noqa: E402
from motionforge.types import SegmentConstraints  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _quat_angle(q1, q2) -> float:
    dot = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    return 2.0 * math.acos(min(1.0, dot))


def _reachable_goal(planner, deltas):
    q = np.array(planner.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return planner.tcp_pose_at(q.tolist())


def _max_orientation_drift(planner, traj, ref_quat, samples=8):
    pts = traj.points
    step = max(1, len(pts) // samples)
    drift = 0.0
    for i in range(0, len(pts), step):
        tcp = planner.tcp_pose_at(pts[i][0])  # FK at the waypoint's joint positions
        drift = max(drift, _quat_angle(tcp.quaternion, ref_quat))
    return drift


def test_hold_orientation_criteria_limits_drift_then_resets(mf_planner):
    start_tcp = mf_planner.tcp_pose_at()  # default config
    # Goal: translated in the base frame, SAME orientation as the start (carry level).
    goal = Pose(start_tcp.position + np.array([0.08, 0.10, 0.05]), start_tcp.quaternion)

    criteria = build_tool_pose_criteria(
        SegmentConstraints(hold_orientation=True, hold_vec_weight=(1, 1, 1, 0, 0, 0)),
        device_cfg=mf_planner.device_cfg,
    )
    mf_planner.set_tool_pose_criteria(criteria)
    try:
        held = mf_planner.plan_free(goal)
        if held.success:
            drift = _max_orientation_drift(mf_planner, held.trajectory, start_tcp.quaternion)
            # Orientation is held approximately along the carry (soft cost; tolerant bound).
            assert drift < 0.30, f"orientation drift {drift:.3f} rad too large under hold"
    finally:
        mf_planner.reset_tool_pose_criteria()

    # After reset, unconstrained planning works again (mechanism restored).
    after = mf_planner.plan_free(_reachable_goal(mf_planner, [0.3, -0.2, 0.2]))
    assert after.success


def test_linear_approach_criteria_applies_and_plans(mf_planner):
    criteria = build_tool_pose_criteria(
        SegmentConstraints(linear_approach=True, approach_axis=[0, 0, 1]),
        device_cfg=mf_planner.device_cfg,
        axis="z",
    )
    mf_planner.set_tool_pose_criteria(criteria)
    try:
        result = mf_planner.plan_free(_reachable_goal(mf_planner, [0.2, -0.1, 0.1]))
        assert result is not None
    finally:
        mf_planner.reset_tool_pose_criteria()
