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
#: Injected fixed link carrying the gripper's own collision geometry (width-dependent).
TOOL_COLLISION_LINK = "tool_collision"
#: Injected fixed link used as the planned tool frame when a tool TCP offset is supplied.
TCP_LINK = "tcp"
#: UR10e wrist/flange links the held object and gripper geometry must not self-collide with.
_WRIST_LINKS = ["tool0", "wrist_3_link", "wrist_2_link", "camera_mount"]

_IDENTITY_TRANSFORM = [0, 0, 0, 1, 0, 0, 0]  # [x,y,z, qw,qx,qy,qz]


def _add_extra_link(kin: dict, name: str, parent: str, joint: str, transform, spheres: int = 0):
    """Inject a FIXED extra link (+ optional collision sphere slots) into a kinematics dict."""
    extra_links = kin.setdefault("extra_links", {})
    extra_links[name] = {
        "fixed_transform": list(transform),
        "joint_name": joint,
        "joint_type": "FIXED",
        "link_name": name,
        "parent_link_name": parent,
    }
    if spheres > 0:
        kin.setdefault("extra_collision_spheres", {})[name] = int(spheres)
        cln = kin.setdefault("collision_link_names", [])
        if name not in cln:
            cln.append(name)


def _ignore_pairs(kin: dict, link_name: str, ignore: List[str]) -> None:
    """Add self-collision ignores for ``link_name`` (cuRobo symmetrizes via min, so one side
    is enough — see self_collision_params.compute_sphere_pair_distance...)."""
    sci = kin.setdefault("self_collision_ignore", {})
    cur = sci.setdefault(link_name, [])
    for other in ignore:
        if other not in cur:
            cur.append(other)


def _robot_config(
    robot_yaml: str,
    attached_object_spheres: int,
    parent_link: str = "tool0",
    tcp_offset=None,
    tool_spheres: int = 0,
):
    """Return the robot spec for ``MotionPlannerCfg.create``.

    UR10e ships a bare arm, so optional runtime collision links are injected into the
    kinematics dict (mirroring franka.yml's ``attached_object`` wiring):

    - ``attached_object`` (``attached_object_spheres > 0``): the held workpiece, parented to
      ``parent_link`` with ``fixed_transform`` = the TCP offset so its frame coincides with
      the planned tool frame ``tool_frames[0]`` — required for the attach offset math
      (``AttachmentManager.update`` resolves the offset against ``tool_frames[0]``).
    - ``tcp`` (``tcp_offset`` given): a fixed tool frame at the tool's TCP, set as the SOLE
      ``tool_frames`` entry so IK/goals target the real TCP instead of the flange. The
      transform is immutable, so a tool swap that changes the TCP needs a planner rebuild.
    - ``tool_collision`` (``tool_spheres > 0``): the gripper's own width-dependent geometry,
      attached at runtime via a second AttachmentManager.

    Self-collision ignores are added so these links don't false-collide with the wrist/flange
    or each other. With none of the options requested, the yaml filename is passed through.
    """
    if attached_object_spheres <= 0 and tcp_offset is None and tool_spheres <= 0:
        return robot_yaml
    from curobo.content import get_robot_configs_path
    from curobo._src.util.config_io import join_path
    from curobo._src.util_file import load_yaml

    data = load_yaml(join_path(get_robot_configs_path(), robot_yaml))
    kin = data["robot_cfg"]["kinematics"]
    tcp_transform = list(tcp_offset.to_curobo_list()) if tcp_offset is not None else _IDENTITY_TRANSFORM

    if tcp_offset is not None:
        _add_extra_link(kin, TCP_LINK, parent_link, "tcp_joint", tcp_transform)
        kin["tool_frames"] = [TCP_LINK]  # plan/IK to the real TCP, not the flange
        _ignore_pairs(kin, TCP_LINK, _WRIST_LINKS)

    if attached_object_spheres > 0:
        # Frame must coincide with tool_frames[0] for the attach offset math; when a TCP offset
        # is present, the tcp link IS tool_frames[0] and sits at the same transform.
        _add_extra_link(
            kin, ATTACHED_OBJECT_LINK, parent_link, "attach_joint", tcp_transform,
            spheres=attached_object_spheres,
        )
        _ignore_pairs(kin, ATTACHED_OBJECT_LINK, _WRIST_LINKS + [TOOL_COLLISION_LINK, TCP_LINK])

    if tool_spheres > 0:
        # Gripper geometry is authored in the TCP frame, so place the link at the TCP offset.
        _add_extra_link(
            kin, TOOL_COLLISION_LINK, parent_link, "tool_collision_joint", tcp_transform,
            spheres=tool_spheres,
        )
        _ignore_pairs(kin, TOOL_COLLISION_LINK, _WRIST_LINKS + [ATTACHED_OBJECT_LINK, TCP_LINK])

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
        attached_object_spheres: sphere-slot budget for the held workpiece (0 disables attach).
        tcp_offset: tool TCP relative to the flange (``motionforge.geometry.Pose``). When set,
            a fixed ``tcp`` link is injected and becomes the planned tool frame so goals/IK
            target the real TCP. Immutable — a TCP change needs a planner rebuild.
        tool_spheres: sphere-slot budget for the gripper's own width-dependent geometry
            (0 disables tool collision).
    """

    def __init__(
        self,
        config: Config = DEFAULTS,
        scene=None,
        device: str = "cuda:0",
        collision_cache: Optional[Dict[str, int]] = None,
        attached_object_spheres: int = 0,
        tcp_offset=None,
        tool_spheres: int = 0,
    ) -> None:
        import torch  # lazy
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.types.device_cfg import DeviceCfg

        self._cfg = config
        self._torch = torch
        self._device = torch.device(device)
        self._device_cfg = DeviceCfg(device=self._device, dtype=torch.float32)
        self._tcp_offset = tcp_offset

        # When a scene is supplied it sizes the collision buffers; otherwise pre-allocate a
        # default cuboid/mesh cache so update_world() can add obstacles later.
        if collision_cache is None:
            collision_cache = None if scene is not None else DEFAULT_COLLISION_CACHE

        mpc = MotionPlannerCfg.create(
            robot=_robot_config(
                config.robot_yaml, attached_object_spheres, config.tcp_frame,
                tcp_offset=tcp_offset, tool_spheres=tool_spheres,
            ),
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
        self._attach_mgrs_by_link: Dict[str, list] = {}

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

    # -- CUDA graph lifecycle --
    #
    # Runtime mutations (world/voxel updates, attach/detach, tool-pose-criteria swaps) write
    # in-place into pre-allocated tensors, which captured CUDA graphs replay against — so the
    # mutation is normally visible WITHOUT re-capture. GRAPH_RESET_KINDS lists the kinds that
    # empirically need a reset anyway (validated by tests/gpu/test_cuda_graph_e2e.py); it is
    # empty until a freshness test proves a kind goes stale, because re-capturing on every
    # per-segment mutation would blow the <1 s plan budget.
    GRAPH_RESET_KINDS: frozenset = frozenset()

    def reset_cuda_graphs(self) -> None:
        """Drop captured IK/TrajOpt/graph-planner CUDA graphs so the next call re-captures."""
        if not self._cfg.use_cuda_graph:
            return
        self._planner.ik_solver.reset_cuda_graph()
        self._planner.trajopt_solver.reset_cuda_graph()
        gp = getattr(self._planner, "graph_planner", None)
        if gp is not None and hasattr(gp, "reset_cuda_graph"):
            gp.reset_cuda_graph()

    def _after_mutation(self, kind: str) -> None:
        """Hook invoked after every collision/criteria mutation; resets graphs only for kinds
        known to go stale under capture (see :attr:`GRAPH_RESET_KINDS`)."""
        if self._cfg.use_cuda_graph and kind in self.GRAPH_RESET_KINDS:
            self.reset_cuda_graphs()

    def update_world(self, scene) -> None:
        """Replace the collision world with a cuRobo ``SceneCfg`` (Phase 2 hand-off)."""
        self._planner.update_world(scene)
        self._after_mutation("world")

    def load_voxel_layers(self, grids: Sequence) -> None:
        """Batch-load all ESDF ``VoxelGrid`` layers into the shared collision world.

        Pushes every layer in ONE ``VoxelData.load_batch`` call so they compose at their own
        poses. The cuRobo ``Scene`` voxel path replaces instead of composing (last-one-wins),
        so :class:`CollisionWorldManager` routes layers here rather than through ``update_world``.
        Call AFTER ``update_world`` (which clears the env's voxels). Empty ``grids`` clears them.
        """
        checker = self._planner.scene_collision_checker
        voxels = getattr(getattr(checker, "data", None), "voxels", None)
        if voxels is None:
            if grids:
                raise RuntimeError(
                    "planner has no voxel collision cache; build it with a voxel `scene` or a "
                    "`collision_cache` containing a 'voxel' entry to push ESDF layers"
                )
            return
        voxels.load_batch(list(grids), 0)
        self._after_mutation("voxel")

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
        self._after_mutation("criteria")

    def reset_tool_pose_criteria(self) -> None:
        """Restore the default (full-pose) criteria on all tool frames."""
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        std = ToolPoseCriteria(device_cfg=self._device_cfg)
        self._planner.update_tool_pose_criteria({f: std for f in self.tool_frames})
        self._after_mutation("criteria")

    # -- attached bodies (held workpiece + gripper geometry) --
    #
    # NOTE: cuRobo's MotionPlanner.attachment_manager property is broken in this build
    # (TrajOptSolver doesn't inherit SolverCore, so it has no attachment_manager). The IK and
    # TrajOpt solvers also hold SEPARATE kinematics, so we build an AttachmentManager pair per
    # kinematics and attach to both — the body then affects IK seeding AND the swept trajopt
    # collision check. Only the trajopt manager owns scene_collision (obstacle disable).
    #
    # AttachmentManager tracks a single attached link per INSTANCE, so each injected link
    # (attached_object, tool_collision) gets its own (trajopt, ik) pair, keyed by link name.

    def _attach_managers(self, link_name: str = ATTACHED_OBJECT_LINK):
        mgrs = self._attach_mgrs_by_link.get(link_name)
        if mgrs is None:
            from curobo._src.collision.attachment_manager import AttachmentManager

            mgrs = [
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
            self._attach_mgrs_by_link[link_name] = mgrs
        return mgrs

    def _identity_offset(self):
        """A [1]-batch identity ``Pose`` for ``world_objects_pose_offset``.

        Passing identity (rather than ``None``) makes ``AttachmentManager.update`` resolve the
        obstacle's world-frame spheres into ``tool_frames[0]``-local coordinates: with the
        attached link's frame coinciding with ``tool_frames[0]``, the body lands EXACTLY where
        it was grasped and rides rigidly with the tool. ``None`` would write world coordinates
        as link-local spheres — a phantom body offset from the gripper (the prior bug).
        """
        from curobo._src.types.pose import Pose as CuPose

        return CuPose.from_list(list(_IDENTITY_TRANSFORM), self._device_cfg)

    def _require_link(self, link_name: str) -> None:
        kc = self._planner.trajopt_solver.kinematics.config.kinematics_config
        if link_name not in kc.link_name_to_idx_map:
            raise RuntimeError(
                f"planner has no '{link_name}' collision link — rebuild the adapter with "
                f"{'attached_object_spheres' if link_name == ATTACHED_OBJECT_LINK else 'tool_spheres'}"
                " > 0 to enable it"
            )

    def _attach_to_link(
        self, link_name, obstacles, js, world_offset, disable_obstacle_names, num_spheres
    ) -> None:
        """Fit spheres once and write them to both (trajopt, ik) managers for ``link_name``.

        Fitting is geometry-only (kinematics-independent), so we fit on one manager and reuse
        the tensor. A lone fitted sphere is duplicated: cuRobo's ``AttachmentManager.update``
        squeezes a ``[1,3]`` centers tensor to ``[3]`` and then fails to concat the radius — a
        harmless duplicate (same center/radius) sidesteps that single-sphere edge case.
        """
        self._require_link(link_name)
        mgrs = self._attach_managers(link_name)
        sphere_tensor = mgrs[0].fit_spheres(obstacles, num_spheres=num_spheres)
        if sphere_tensor.shape[0] == 1:
            sphere_tensor = sphere_tensor.repeat(2, 1)
        for i, mgr in enumerate(mgrs):
            mgr.update(sphere_tensor, js, link_name, world_offset)
            # Only the scene-bound (trajopt) manager toggles world obstacles.
            if i == 0 and disable_obstacle_names and mgr._scene_collision is not None:
                n_envs = mgr._get_num_envs(js)
                for name in disable_obstacle_names:
                    for env_idx in range(n_envs):
                        mgr._scene_collision.enable_obstacle(name, enable=False, env_idx=env_idx)
                mgr._disabled_obstacle_names = list(disable_obstacle_names)
                mgr._disabled_num_envs = n_envs
        self._after_mutation("attach")

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

        Obstacles carry their base-frame world pose; the default identity offset localizes
        them into the tool frame (see :meth:`_identity_offset`). Requires the adapter to have
        been built with ``attached_object_spheres > 0`` (or ``tool_spheres`` for the tool link).
        """
        js = self._make_joint_state(q_grasp)
        if world_objects_pose_offset is None:
            world_objects_pose_offset = self._identity_offset()
        self._attach_to_link(
            link_name, list(obstacles), js, world_objects_pose_offset,
            disable_obstacle_names, num_spheres,
        )

    def detach_object(self, link_name: str = ATTACHED_OBJECT_LINK) -> None:
        """Detach the held object and re-enable any obstacles disabled at attach time."""
        for mgr in self._attach_managers(link_name):
            mgr.detach(link_name=link_name)
        self._after_mutation("attach")

    def attach_tool(self, obstacles, num_spheres: Optional[int] = None) -> None:
        """Attach the gripper's own collision geometry to the ``tool_collision`` link.

        Unlike the held workpiece, tool geometry is authored directly in the TCP/link-local
        frame (the cuboid poses ARE the TCP-frame offsets), so it is attached with NO world
        offset (``None``) — the fitted spheres are already link-local. Re-call when the
        commanded width changes; the previous tool geometry is replaced. Requires
        ``tool_spheres > 0``.
        """
        js = self._make_joint_state(None)  # tool geometry is link-local; q is irrelevant
        self._attach_to_link(
            TOOL_COLLISION_LINK, list(obstacles), js, None, None, num_spheres,
        )

    def detach_tool(self) -> None:
        """Remove the gripper collision geometry from the ``tool_collision`` link."""
        for mgr in self._attach_managers(TOOL_COLLISION_LINK):
            mgr.detach(link_name=TOOL_COLLISION_LINK)
        self._after_mutation("attach")

    @property
    def attached_link_name(self) -> Optional[str]:
        return self._attach_managers(ATTACHED_OBJECT_LINK)[0]._attached_link_name

    @property
    def tool_attached(self) -> bool:
        return self._attach_managers(TOOL_COLLISION_LINK)[0]._attached_link_name is not None

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

        criteria = build_tool_pose_criteria(
            constraints, device_cfg=self._device_cfg, axis=axis, goal=goal
        )
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
        acknowledge_broken: bool = False,
    ) -> GraspPlan:
        """Native approach→grasp→lift over a ranked grasp goalset (cuRobo ``plan_grasp``).

        ``grasp_approach_axis`` is a tool-frame principal axis ("x"|"y"|"z"); the standoff
        is taken from the first candidate (or :attr:`Config.standoff_m`). Aligning an
        arbitrary base-frame ``approach_axis`` to a principal axis is the segment builder's
        job (Phase 3).

        .. warning::
           **Broken in this curobov2 build — do not use in the pipeline.** ``plan_grasp``'s
           internal linear approach→grasp step fails to converge here AND a failed call
           **corrupts planner state**, so subsequent goalset plans return ``None``. The
           coordinator deliberately plans the pick as a per-segment ``plan_segment`` chain
           (approach/grasp/retract) instead. This method is kept only for the Phase-1 API
           tests and as the drop-in for when the upstream issue is fixed; calling it emits a
           ``RuntimeWarning`` unless ``acknowledge_broken=True``.
        """
        if not acknowledge_broken:
            import warnings

            warnings.warn(
                "MotionPlannerAdapter.plan_grasp is broken in this curobov2 build (fails to "
                "converge and corrupts planner state); use the per-segment plan_segment chain. "
                "Pass acknowledge_broken=True to silence this once the upstream fix lands.",
                RuntimeWarning,
                stacklevel=2,
            )
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
