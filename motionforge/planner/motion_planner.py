"""cuRobo (curobov2) MotionPlanner adapter — the core planning engine.

Wraps :class:`curobo.motion_planner.MotionPlanner` and maps our SPEC §4 contracts onto
cuRobo's ``GoalToolPose`` / ``plan_pose`` / ``plan_grasp``. Verified API facts (see the
build plan) drive this module:

- ``plan_pose`` returns a ``TrajOptSolverResult`` (``.success``, ``.total_time``,
  ``.solve_time``, ``.goalset_index``, ``.get_interpolated_plan()``), or ``None`` on total
  IK failure. Goalset is the ``num_goalset`` axis of ``GoalToolPose``.
- ``plan_grasp`` returns a ``GraspPlanResult`` with approach/grasp/lift interpolated
  trajectories; it natively chains pre-grasp offset → linear approach → grasp → linear lift.

torch/curobo are imported lazily so this module stays importable on a machine without a GPU
(unit-test collection, CI). Instantiating :class:`MotionPlannerAdapter` requires CUDA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from motionforge.config import DEFAULTS, Config
from motionforge.geometry import Pose
from motionforge.types import GraspCandidate, JointTrajectory, PlanResult

# Default collision-cache so the world checker is pre-allocated even with no initial scene,
# letting CollisionWorldManager.update_world(...) populate cuboids/meshes later (Phase 2).
# cuboid/mesh take int counts; voxel needs {"layers","dims","voxel_size"} — omitted here
# (no ESDF in Phase 1). Phase 2 passes a voxel-sized cache when it needs ESDF layers.
DEFAULT_COLLISION_CACHE: Dict[str, int] = {"cuboid": 30, "mesh": 30}

#: Default link name cuRobo uses for an attached object (see AttachmentManager).
ATTACHED_OBJECT_LINK = "attached_object"


def _robot_config(robot_yaml: str, attached_object_spheres: int, parent_link: str = "tool0"):
    """Return the robot spec for ``MotionPlannerCfg.create``.

    If ``attached_object_spheres > 0``, load the bundled yaml as a dict and declare an
    ``attached_object`` collision link (mirroring franka.yml's structure) parented to
    ``parent_link`` (the TCP frame), so the held workpiece can be attached at runtime
    (SPEC §5.2). UR10e ships no such link, so we inject the full extra-link + sphere-slot +
    collision-link-name set. Otherwise pass the yaml filename through.
    """
    if attached_object_spheres <= 0:
        return robot_yaml
    from curobo.content import get_robot_configs_path
    from curobo._src.util.config_io import join_path
    from curobo._src.util_file import load_yaml

    data = load_yaml(join_path(get_robot_configs_path(), robot_yaml))
    kin = data["robot_cfg"]["kinematics"]
    kin["extra_collision_spheres"] = {ATTACHED_OBJECT_LINK: int(attached_object_spheres)}
    extra_links = kin.setdefault("extra_links", {})
    extra_links[ATTACHED_OBJECT_LINK] = {
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],  # identity; spheres are placed at attach time
        "joint_name": "attach_joint",
        "joint_type": "FIXED",
        "link_name": ATTACHED_OBJECT_LINK,
        "parent_link_name": parent_link,
    }
    cln = kin.setdefault("collision_link_names", [])
    if ATTACHED_OBJECT_LINK not in cln:
        cln.append(ATTACHED_OBJECT_LINK)
    return data


@dataclass
class GraspPlan:
    """Result of a ``plan_grasp`` call, split into executable phases (SPEC §5.3)."""

    success: bool
    candidate_index: int = -1
    status: str = ""
    approach: Optional[JointTrajectory] = None
    grasp: Optional[JointTrajectory] = None
    lift: Optional[JointTrajectory] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class MotionPlannerAdapter:
    """Embeds a cuRobo ``MotionPlanner`` for one robot (default: UR10e, 6-DOF).

    Args:
        config: planner/robot defaults (see :class:`motionforge.config.Config`).
        scene: optional cuRobo ``SceneCfg`` for the initial collision world. If ``None``,
            the planner starts in free space but with a pre-allocated collision cache so a
            scene can be pushed later via :meth:`update_world`.
        device: CUDA device string.
        collision_cache: override the pre-allocated cache sizes.
    """

    def __init__(
        self,
        config: Config = DEFAULTS,
        scene=None,
        device: str = "cuda:0",
        collision_cache: Optional[Dict[str, int]] = None,
        attached_object_spheres: int = 0,
    ) -> None:
        import torch  # lazy
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.types.device_cfg import DeviceCfg

        self._cfg = config
        self._torch = torch
        self._device = torch.device(device)
        self._device_cfg = DeviceCfg(device=self._device, dtype=torch.float32)

        # When a scene is supplied it sizes the collision buffers; otherwise pre-allocate a
        # default cuboid/mesh cache so update_world() can add obstacles later.
        if collision_cache is None:
            collision_cache = None if scene is not None else DEFAULT_COLLISION_CACHE

        mpc = MotionPlannerCfg.create(
            robot=_robot_config(config.robot_yaml, attached_object_spheres, config.tcp_frame),
            scene_model=scene,
            collision_cache=collision_cache,
            device_cfg=self._device_cfg,
            num_ik_seeds=config.num_ik_seeds,
            num_trajopt_seeds=config.num_trajopt_seeds,
            use_cuda_graph=config.use_cuda_graph,
            max_goalset=config.max_goalset,
        )
        self._planner = MotionPlanner(mpc)
        self._warm = False

    # -- introspection --

    @property
    def planner(self):
        """The underlying cuRobo ``MotionPlanner`` (for Phase 2 world updates etc.)."""
        return self._planner

    @property
    def joint_names(self) -> List[str]:
        return list(self._planner.joint_names)

    @property
    def tool_frames(self) -> List[str]:
        return list(self._planner.tool_frames)

    @property
    def default_q0(self) -> List[float]:
        return self._planner.default_joint_state.position.view(-1).tolist()

    def tcp_pose_at(self, q: Optional[Sequence[float]] = None) -> Pose:
        """Forward kinematics: base-frame TCP (``tool_frames[0]``) pose at joint config ``q``.

        Used by tests to obtain guaranteed-reachable goals, and by the EIH perception path
        (camera→base = extrinsic × FK at capture).
        """
        js = self._make_joint_state(q)
        kin = self._planner.compute_kinematics(js)
        cp = kin.tool_poses.to_dict()[self.tool_frames[0]]
        return Pose(cp.position.view(-1).tolist(), cp.quaternion.view(-1).tolist())

    # -- lifecycle --

    def warmup(self) -> bool:
        """JIT/compile + (optionally) capture CUDA graphs. ~14 s with graphs on first call."""
        ok = self._planner.warmup(
            enable_graph=self._cfg.use_cuda_graph,
            num_warmup_iterations=self._cfg.warmup_iterations,
        )
        self._warm = bool(ok)
        return self._warm

    def update_world(self, scene) -> None:
        """Replace the collision world with a cuRobo ``SceneCfg`` (Phase 2 hand-off)."""
        self._planner.update_world(scene)

    def to_joint_state(self, q: Optional[Sequence[float]] = None):
        """Public ``JointState`` builder (1×dof) for the configured robot."""
        return self._make_joint_state(q)

    @property
    def device_cfg(self):
        """cuRobo ``DeviceCfg`` for this planner (for building ToolPoseCriteria etc.)."""
        return self._device_cfg

    # -- per-frame pose-cost criteria (linear approach / hold-orientation) --

    def set_tool_pose_criteria(self, criteria) -> None:
        """Apply a ``ToolPoseCriteria`` to every tool frame (IK + trajopt)."""
        self._planner.update_tool_pose_criteria({f: criteria for f in self.tool_frames})

    def reset_tool_pose_criteria(self) -> None:
        """Restore the default (full-pose) criteria on all tool frames."""
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        std = ToolPoseCriteria(device_cfg=self._device_cfg)
        self._planner.update_tool_pose_criteria({f: std for f in self.tool_frames})

    # -- attached object (held workpiece) --
    #
    # NOTE: cuRobo's MotionPlanner.attachment_manager property is broken in this build
    # (TrajOptSolver doesn't inherit SolverCore, so it has no attachment_manager). The IK and
    # TrajOpt solvers also hold SEPARATE kinematics, so we build an AttachmentManager per
    # kinematics and attach to both — the held object then affects IK seeding AND the swept
    # trajopt collision check. Only the trajopt manager owns scene_collision (obstacle disable).

    def _attach_managers(self):
        if getattr(self, "_attach_mgrs", None) is None:
            from curobo._src.collision.attachment_manager import AttachmentManager

            self._attach_mgrs = [
                AttachmentManager(
                    kinematics=self._planner.trajopt_solver.kinematics,
                    scene_collision=self._planner.scene_collision_checker,
                    device_cfg=self._device_cfg,
                ),
                AttachmentManager(
                    kinematics=self._planner.ik_solver.kinematics,
                    scene_collision=None,
                    device_cfg=self._device_cfg,
                ),
            ]
        return self._attach_mgrs

    def attach_object(
        self,
        obstacles,
        q_grasp: Optional[Sequence[float]] = None,
        link_name: str = ATTACHED_OBJECT_LINK,
        world_objects_pose_offset=None,
        disable_obstacle_names: Optional[List[str]] = None,
        num_spheres: Optional[int] = None,
    ) -> None:
        """Attach obstacle(s) to ``link_name`` at the grasp config (fits collision spheres).

        Requires the adapter to have been built with ``attached_object_spheres > 0``.
        """
        js = self._make_joint_state(q_grasp)
        obstacles = list(obstacles)
        for i, mgr in enumerate(self._attach_managers()):
            mgr.attach(
                joint_states=js,
                obstacles=obstacles,
                link_name=link_name,
                num_spheres=num_spheres,
                world_objects_pose_offset=world_objects_pose_offset,
                # Only the scene-bound (trajopt) manager toggles world obstacles.
                disable_obstacle_names=disable_obstacle_names if i == 0 else None,
            )

    def detach_object(self, link_name: Optional[str] = None) -> None:
        """Detach the held object and re-enable any obstacles disabled at attach time."""
        for mgr in self._attach_managers():
            mgr.detach(link_name=link_name)

    @property
    def attached_link_name(self) -> Optional[str]:
        return self._attach_managers()[0]._attached_link_name

    # -- planning --

    def plan_free(
        self,
        goals: Union[Pose, Sequence[Pose]],
        q0: Optional[Sequence[float]] = None,
        max_attempts: int = 5,
    ) -> PlanResult:
        """Free-motion plan to one pose or a ranked goalset (planner picks the best)."""
        if isinstance(goals, Pose):
            goals = [goals]
        goalset = self._build_goalset(list(goals))
        current = self._make_joint_state(q0)
        t0 = time.perf_counter()
        result = self._planner.plan_pose(goalset, current, max_attempts=max_attempts)
        wall = time.perf_counter() - t0
        return self._to_plan_result(result, wall)

    def plan_segment(self, goal: Pose, constraints, q0: Optional[Sequence[float]] = None, axis: Optional[str] = None) -> PlanResult:
        """Plan one constrained free-motion segment: build the ToolPoseCriteria from
        :class:`SegmentConstraints`, plan, then restore the default criteria. Keeps cuRobo
        constraint construction out of the coordinator (which stays GPU-free)."""
        from motionforge.planner.constraints import build_tool_pose_criteria

        criteria = build_tool_pose_criteria(constraints, device_cfg=self._device_cfg, axis=axis)
        self.set_tool_pose_criteria(criteria)
        try:
            return self.plan_free(goal, q0=q0)
        finally:
            self.reset_tool_pose_criteria()

    def plan_grasp(
        self,
        grasp_candidates: Sequence[GraspCandidate],
        q0: Optional[Sequence[float]] = None,
        grasp_approach_axis: Optional[str] = None,
        grasp_approach_offset: Optional[float] = None,
        grasp_lift_axis: str = "z",
        grasp_lift_offset: float = 0.1,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_lift: bool = True,
        disable_collision_links: Optional[List[str]] = None,
    ) -> GraspPlan:
        """Native approach→grasp→lift over a ranked grasp goalset (cuRobo ``plan_grasp``).

        ``grasp_approach_axis`` is a tool-frame principal axis ("x"|"y"|"z"); the standoff
        is taken from the first candidate (or :attr:`Config.standoff_m`). Aligning an
        arbitrary base-frame ``approach_axis`` to a principal axis is the segment builder's
        job (Phase 3).
        """
        if not grasp_candidates:
            return GraspPlan(success=False, status="no grasp candidates")
        axis = grasp_approach_axis or self._cfg.grasp_approach_axis
        standoff = grasp_candidates[0].standoff_m or self._cfg.standoff_m
        offset = grasp_approach_offset if grasp_approach_offset is not None else -abs(standoff)

        goalset = self._build_goalset([c.tcp_pose for c in grasp_candidates])
        current = self._make_joint_state(q0)

        t0 = time.perf_counter()
        res = self._planner.plan_grasp(
            grasp_poses=goalset,
            current_state=current,
            grasp_approach_axis=axis,
            grasp_approach_offset=offset,
            grasp_lift_axis=grasp_lift_axis,
            grasp_lift_offset=grasp_lift_offset,
            plan_approach_to_grasp=plan_approach_to_grasp,
            plan_grasp_to_lift=plan_grasp_to_lift,
            disable_collision_links=disable_collision_links,
        )
        wall = time.perf_counter() - t0
        return self._grasp_to_plan(res, wall)

    # -- helpers --

    def _make_joint_state(self, q0: Optional[Sequence[float]]):
        from curobo.types import JointState

        torch = self._torch
        if q0 is None:
            pos = self._planner.default_joint_state.position.view(1, -1)
        else:
            pos = torch.as_tensor(q0, device=self._device, dtype=torch.float32).view(1, -1)
        return JointState.from_position(pos, joint_names=self._planner.joint_names)

    def _build_goalset(self, poses: List[Pose]):
        """Build a ``GoalToolPose`` whose ``num_goalset`` dim is the ranked-candidate axis."""
        from curobo.types import GoalToolPose

        torch = self._torch
        pos = torch.tensor(
            [[p.position.tolist() for p in poses]], device=self._device, dtype=torch.float32
        )  # (1, G, 3)
        quat = torch.tensor(
            [[p.quaternion.tolist() for p in poses]], device=self._device, dtype=torch.float32
        )  # (1, G, 4)
        # Insert horizon + link dims -> [B, H, L, G, *] with H=L=1 (single tool frame).
        return GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos[:, None, None],
            quaternion=quat[:, None, None],
        )

    def _to_plan_result(self, result, wall_s: float) -> PlanResult:
        if result is None:
            return PlanResult(success=False, metrics={"wall_s": wall_s})
        success = bool(result.success.any().item())
        metrics = {
            "wall_s": wall_s,
            "solve_time": float(getattr(result, "solve_time", 0.0) or 0.0),
            "total_time": float(getattr(result, "total_time", 0.0) or 0.0),
        }
        if not success:
            return PlanResult(success=False, metrics=metrics)

        cand_idx = self._goalset_index(result)
        traj = self._extract_trajectory(result)
        if traj is not None:
            metrics["cycle_time"] = traj.duration_s
            pj = self._peak_jerk(result)
            if pj is not None:
                metrics["peak_jerk"] = pj
        # NOTE: min_clearance_m hard filter is deferred — cuRobo `success` already enforces a
        # collision-free trajectory within the optimizer's activation margin. A dedicated ESDF
        # clearance query (scene_collision_checker.get_sphere_distance) is a later refinement.
        return PlanResult(success=True, trajectory=traj, candidate_index=cand_idx, metrics=metrics)

    def _grasp_to_plan(self, res, wall_s: float) -> GraspPlan:
        metrics = {"wall_s": wall_s, "planning_time": float(getattr(res, "planning_time", 0.0) or 0.0)}
        success = bool(res.success.any().item()) if res.success is not None else False
        cand_idx = -1
        if res.goalset_index is not None:
            cand_idx = int(res.goalset_index.view(-1)[0].item())
        return GraspPlan(
            success=success,
            candidate_index=cand_idx,
            status=res.status or "",
            approach=self._js_to_trajectory(res.approach_interpolated_trajectory, res.approach_trajectory_dt),
            grasp=self._js_to_trajectory(res.grasp_interpolated_trajectory, res.grasp_trajectory_dt),
            lift=self._js_to_trajectory(res.lift_interpolated_trajectory, res.lift_trajectory_dt),
            metrics=metrics,
        )

    def _goalset_index(self, result) -> int:
        gi = getattr(result, "goalset_index", None)
        if gi is None:
            return 0
        return int(gi.view(-1)[0].item())

    def _extract_trajectory(self, result) -> Optional[JointTrajectory]:
        js = result.get_interpolated_plan()
        return self._js_to_trajectory(js, getattr(js, "dt", None))

    def _js_to_trajectory(self, js, dt_tensor) -> Optional[JointTrajectory]:
        if js is None or js.position is None:
            return None
        pos = self._as_2d(js.position)
        n = pos.shape[0]
        vel = self._as_2d(js.velocity, like=pos)
        acc = self._as_2d(js.acceleration, like=pos)
        dt = self._scalar(dt_tensor if dt_tensor is not None else getattr(js, "dt", None))
        names = list(js.joint_names) if js.joint_names is not None else list(self._planner.joint_names)
        points = [
            (pos[i].tolist(), vel[i].tolist(), acc[i].tolist(), float(i * dt))
            for i in range(n)
        ]
        return JointTrajectory(joint_names=names, points=points)

    def _as_2d(self, tensor, like=None) -> np.ndarray:
        if tensor is None:
            return np.zeros_like(like) if like is not None else np.zeros((0, 0))
        arr = tensor.detach().cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        return arr

    def _scalar(self, tensor) -> float:
        if tensor is None:
            return 0.0
        try:
            return float(tensor.view(-1)[0].item())
        except AttributeError:
            return float(tensor)

    def _peak_jerk(self, result) -> Optional[float]:
        js = result.get_interpolated_plan()
        if getattr(js, "jerk", None) is None:
            return None
        return float(js.jerk.abs().max().item())
