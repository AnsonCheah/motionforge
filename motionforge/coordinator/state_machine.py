"""Task Coordinator (SPEC §5.5, §6) — the pick-and-place state machine.

Drives: PERCEIVE_PICK → PLAN_PICK → EXEC_PICK → PERCEIVE_PLACE → PLAN_PLACE → EXEC_PLACE →
DONE, with RECAPTURE and FAULT. Owns the sync barriers (open-to-standby before the final
approach; close-and-grip before attach), the fallback/recapture ladder (SPEC §7), and
attach/detach of the held workpiece.

GPU-free: all cuRobo work lives behind the injected planner/world (the coordinator only calls
``plan_grasp`` / ``plan_segment`` / ``attach_object`` etc.), so it is fully unit-testable with
fakes and reused unchanged for the GPU acceptance run and the Isaac twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from motionforge.collision.world_manager import PICK_BIN_LAYER, PLACE_TRAY_LAYER
from motionforge.config import DEFAULTS, Config
from motionforge.planner.segment_builder import build_pick_segments, build_place_segments
from motionforge.types import JointTrajectory, PlaceCandidate


class CoordinatorState(Enum):
    PERCEIVE_PICK = "perceive_pick"
    PLAN_PICK = "plan_pick"
    EXEC_PICK = "exec_pick"
    PERCEIVE_PLACE = "perceive_place"
    PLAN_PLACE = "plan_place"
    EXEC_PLACE = "exec_place"
    RECAPTURE = "recapture"
    DONE = "done"
    FAULT = "fault"


@dataclass
class _PickPlan:
    candidate: Any
    candidate_index: int
    approach: Optional[JointTrajectory]
    grasp: Optional[JointTrajectory]
    retract: Optional[JointTrajectory]
    q_grasp: Optional[List[float]]


@dataclass
class _PlacePlan:
    candidate: PlaceCandidate
    candidate_index: int
    transport: Optional[JointTrajectory]
    place: Optional[JointTrajectory]
    retract: Optional[JointTrajectory]


@dataclass
class CycleResult:
    success: bool
    state: CoordinatorState
    recaptures: int = 0
    fault_reason: str = ""
    pick_candidate_index: int = -1
    place_candidate_index: int = -1
    plan_times_s: List[float] = field(default_factory=list)
    pick_plan: Optional[_PickPlan] = None
    place_plan: Optional[_PlacePlan] = None

    @property
    def max_plan_time_s(self) -> float:
        return max(self.plan_times_s) if self.plan_times_s else 0.0


class TaskCoordinator:
    def __init__(
        self,
        planner,
        world,
        tools,
        perception,
        gripper,
        execution,
        joint_state_source,
        config: Config = DEFAULTS,
    ) -> None:
        self.planner = planner
        self.world = world
        self.tools = tools
        self.perception = perception
        self.gripper = gripper
        self.execution = execution
        self.joints = joint_state_source
        self.config = config

        self.state = CoordinatorState.PERCEIVE_PICK
        self.recaptures = 0
        self._plan_times: List[float] = []

    # -- public --

    def run_cycle(self) -> CycleResult:
        """Run one full pick-and-place cycle; returns the terminal result."""
        self.recaptures = 0
        self._plan_times = []

        pick, pick_plan = self._perceive_and_plan_pick()
        if pick_plan is None:
            return self._fault("pick planning failed after recapture cap")
        self._execute_pick(pick, pick_plan)

        # Place continues from where the lift/retract ENDED (arm up), not the grasp config —
        # q_grasp is only for the attach. Starting place from q_grasp would jump the arm back
        # down to the grasp pose before transporting.
        q_after_lift = self._last_q(pick_plan.retract, pick_plan.grasp, pick_plan.approach)
        place_plan = self._perceive_and_plan_place(q_after_lift)
        if place_plan is None:
            return self._fault("place planning failed after recapture cap")
        self._execute_place(place_plan)

        self.state = CoordinatorState.DONE
        return CycleResult(
            success=True,
            state=self.state,
            recaptures=self.recaptures,
            fault_reason="",
            pick_candidate_index=pick_plan.candidate_index,
            place_candidate_index=place_plan.candidate_index,
            plan_times_s=list(self._plan_times),
            pick_plan=pick_plan,
            place_plan=place_plan,
        )

    # -- pick --
    #
    # The pick uses the same per-segment planning path as place (plan_segment per segment with
    # the segment's constraints), not cuRobo's native plan_grasp: in this curobov2 build
    # plan_grasp's internal linear approach→grasp step fails to converge and corrupts planner
    # state across calls. plan_grasp remains available on the adapter (Phase 1) if/when fixed.

    def _perceive_and_plan_pick(self):
        pick = None
        for _ in range(self.config.recapture_cap + 1):
            self.state = CoordinatorState.PERCEIVE_PICK
            q0 = self.joints.read_joint_state()
            pick = self.perception.perceive_pick()
            if pick.bin_voxels is not None:
                self.world.set_voxel_layer(PICK_BIN_LAYER, pick.bin_voxels)
                self.world.commit()

            self.state = CoordinatorState.PLAN_PICK
            plan = self._plan_pick_candidates(pick.grasps, q0)
            if plan is not None:
                return pick, plan

            self.state = CoordinatorState.RECAPTURE
            self.recaptures += 1
        return pick, None

    def _plan_pick_candidates(self, grasps, q_start) -> Optional[_PickPlan]:
        """Fallback ladder: try each ranked grasp; return the first that plans fully."""
        for idx, grasp in enumerate(grasps):
            chain = self._plan_pick_chain(grasp, idx, q_start)
            if chain is not None:
                return chain
        return None

    def _plan_pick_chain(self, grasp, idx: int, q_start) -> Optional[_PickPlan]:
        approach_seg, grasp_seg, retract_seg = build_pick_segments(grasp, self.config)

        r_a = self.planner.plan_segment(approach_seg.goal, approach_seg.constraints, q0=q_start or None)
        self._record_plan_time(r_a)
        if not r_a.success:
            return None
        q1 = self._last_q(r_a.trajectory)

        r_g = self.planner.plan_segment(grasp_seg.goal, grasp_seg.constraints, q0=q1 or None)
        self._record_plan_time(r_g)
        if not r_g.success:
            return None
        q2 = self._last_q(r_g.trajectory)

        r_r = self.planner.plan_segment(retract_seg.goal, retract_seg.constraints, q0=q2 or None)
        self._record_plan_time(r_r)
        if not r_r.success:
            return None

        return _PickPlan(grasp, idx, r_a.trajectory, r_g.trajectory, r_r.trajectory, q_grasp=q2)

    def _execute_pick(self, pick, plan: _PickPlan) -> None:
        self.state = CoordinatorState.EXEC_PICK
        approach_seg, grasp_seg, _ = build_pick_segments(plan.candidate, self.config)

        # Open to standby concurrently with the approach move (non-blocking), then barrier.
        self.gripper.command(approach_seg.pre_action)
        self._exec(plan.approach)
        self.gripper.wait()  # barrier: standby width reached before the final approach

        self._exec(plan.grasp)
        self.gripper.command(grasp_seg.post_action)
        self.gripper.wait()  # barrier: grip confirmed

        # Attach the held workpiece on grip confirmation, at the grasp config.
        if pick.workpiece is not None:
            self.world.attach_object([pick.workpiece], q_grasp=plan.q_grasp or None)

        self._exec(plan.retract)

    # -- place --

    def _perceive_and_plan_place(self, q_start) -> Optional[_PlacePlan]:
        for _ in range(self.config.recapture_cap + 1):
            self.state = CoordinatorState.PERCEIVE_PLACE
            place = self.perception.perceive_place()
            if place.tray_voxels is not None:
                self.world.set_voxel_layer(PLACE_TRAY_LAYER, place.tray_voxels)
                self.world.commit()

            self.state = CoordinatorState.PLAN_PLACE
            plan = self._plan_place_candidates(place.places, q_start)
            if plan is not None:
                return plan

            self.state = CoordinatorState.RECAPTURE
            self.recaptures += 1
        return None

    def _plan_place_candidates(self, places, q_start) -> Optional[_PlacePlan]:
        """Fallback ladder: try each ranked candidate; return the first that plans fully."""
        for idx, place in enumerate(places):
            chain = self._plan_place_chain(place, idx, q_start)
            if chain is not None:
                return chain
        return None

    def _plan_place_chain(self, place: PlaceCandidate, idx: int, q_start) -> Optional[_PlacePlan]:
        transport_seg, place_seg, _release_seg, retract_seg = build_place_segments(place, self.config)

        r_t = self.planner.plan_segment(transport_seg.goal, transport_seg.constraints, q0=q_start or None)
        self._record_plan_time(r_t)
        if not r_t.success:
            return None
        q1 = self._last_q(r_t.trajectory)

        r_p = self.planner.plan_segment(place_seg.goal, place_seg.constraints, q0=q1 or None)
        self._record_plan_time(r_p)
        if not r_p.success:
            return None
        q2 = self._last_q(r_p.trajectory)

        r_r = self.planner.plan_segment(retract_seg.goal, retract_seg.constraints, q0=q2 or None)
        self._record_plan_time(r_r)
        if not r_r.success:
            return None

        return _PlacePlan(place, idx, r_t.trajectory, r_p.trajectory, r_r.trajectory)

    def _execute_place(self, plan: _PlacePlan) -> None:
        self.state = CoordinatorState.EXEC_PLACE
        _t, _p, release_seg, _r = build_place_segments(plan.candidate, self.config)

        self._exec(plan.transport)
        self._exec(plan.place)

        # Release (blocking), then detach the held body.
        self.gripper.command(release_seg.post_action)
        self.gripper.wait()
        self.world.detach_object()

        self._exec(plan.retract)

    # -- helpers --

    def _exec(self, traj: Optional[JointTrajectory]) -> None:
        if traj is not None and len(traj) > 0:
            self.execution.send_trajectory(traj)

    def _fault(self, reason: str) -> CycleResult:
        self.state = CoordinatorState.FAULT
        return CycleResult(
            success=False, state=self.state, recaptures=self.recaptures,
            fault_reason=reason, plan_times_s=list(self._plan_times),
        )

    def _safe_index(self, idx) -> int:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return 0
        return i if i >= 0 else 0

    @staticmethod
    def _last_q(*trajs: Optional[JointTrajectory]) -> Optional[List[float]]:
        for traj in trajs:
            if traj is not None and len(traj) > 0:
                return list(traj.points[-1][0])
        return None

    def _record_plan_time(self, result) -> None:
        if result is not None and getattr(result, "metrics", None):
            t = result.metrics.get("total_time") or result.metrics.get("wall_s")
            if t:
                self._plan_times.append(float(t))
