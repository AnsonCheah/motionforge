"""Phase 1 GPU tests: MotionPlannerAdapter on UR10e (mirrors cuRobo's ESDF test patterns)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from motionforge.geometry import Pose  # noqa: E402
from motionforge.planner import GraspPlan  # noqa: E402
from motionforge.types import GraspCandidate, GripConfig  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _reachable_goal(planner, deltas) -> Pose:
    """A guaranteed-reachable TCP pose: FK of the default config perturbed by ``deltas``."""
    q = np.array(planner.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return planner.tcp_pose_at(q.tolist())


# ── Introspection ──


def test_robot_is_ur10e_6dof(mf_planner):
    assert len(mf_planner.joint_names) == 6
    assert mf_planner.tool_frames == ["tool0"]
    assert len(mf_planner.default_q0) == 6


def test_warmup_succeeded(mf_planner):
    assert mf_planner.warmup() is True


def test_tcp_pose_at_returns_pose(mf_planner):
    p = mf_planner.tcp_pose_at()
    assert isinstance(p, Pose)
    assert p.position.shape == (3,)


# ── plan_free ──


def test_plan_free_reachable_pose(mf_planner):
    goal = _reachable_goal(mf_planner, [0.3, -0.2, 0.2])
    result = mf_planner.plan_free(goal)
    assert result.success
    assert result.trajectory is not None
    assert len(result.trajectory) > 1
    assert result.trajectory.joint_names == mf_planner.joint_names


def test_plan_free_within_time_budget(mf_planner):
    goal = _reachable_goal(mf_planner, [0.25, 0.2, -0.15])
    result = mf_planner.plan_free(goal)
    assert result.success
    # SPEC §8: < 1 s per planning call (steady-state, post-warmup).
    assert result.metrics["total_time"] < 1.0
    assert "cycle_time" in result.metrics


def test_plan_free_goalset_selects_reachable(mf_planner):
    goals = [
        _reachable_goal(mf_planner, [0.3, 0.0, 0.0]),
        _reachable_goal(mf_planner, [0.2, -0.2, 0.2]),
        _reachable_goal(mf_planner, [-0.2, 0.2, 0.1]),
    ]
    result = mf_planner.plan_free(goals)
    assert result.success
    assert 0 <= result.candidate_index < len(goals)


def test_plan_free_unreachable_fails(mf_planner):
    far = Pose([5.0, 5.0, 5.0], [1.0, 0.0, 0.0, 0.0])
    result = mf_planner.plan_free(far, max_attempts=1)
    assert not result.success
    assert result.trajectory is None


# ── plan_grasp ──


def _grasp_candidate(planner, deltas, standoff=0.1) -> GraspCandidate:
    return GraspCandidate(
        tcp_pose=_reachable_goal(planner, deltas),
        approach_axis=[0.0, 0.0, 1.0],
        standoff_m=standoff,
        tool_id="jaw",
        grip=GripConfig(width_m=0.04, force=20.0),
    )


def test_plan_grasp_reach_only_succeeds(mf_planner):
    cand = _grasp_candidate(mf_planner, [0.3, -0.1, 0.1])
    plan = mf_planner.plan_grasp(
        [cand], plan_approach_to_grasp=False, plan_grasp_to_lift=False
    )
    assert isinstance(plan, GraspPlan)
    assert plan.success
    assert plan.candidate_index >= 0
    assert plan.status


def test_plan_grasp_full_approach_lift_returns_phases(mf_planner):
    cands = [
        _grasp_candidate(mf_planner, [0.3, -0.1, 0.1]),
        _grasp_candidate(mf_planner, [0.28, -0.08, 0.12]),
    ]
    plan = mf_planner.plan_grasp(
        cands,
        grasp_approach_axis="z",
        grasp_approach_offset=-0.1,
        grasp_lift_axis="z",
        grasp_lift_offset=0.1,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
    )
    assert isinstance(plan, GraspPlan)
    assert plan.status  # human-readable status always set
    if plan.success:
        # On success the approach phase trajectory must be present.
        assert plan.approach is not None and len(plan.approach) > 1


def test_plan_grasp_disable_collision_links_path_runs(mf_planner):
    cand = _grasp_candidate(mf_planner, [0.3, -0.1, 0.1])
    plan = mf_planner.plan_grasp(
        [cand],
        disable_collision_links=[],
        plan_approach_to_grasp=False,
        plan_grasp_to_lift=False,
    )
    assert isinstance(plan, GraspPlan)


def test_plan_grasp_no_candidates(mf_planner):
    plan = mf_planner.plan_grasp([])
    assert isinstance(plan, GraspPlan)
    assert not plan.success
