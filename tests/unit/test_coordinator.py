"""Unit tests for the Task Coordinator state machine (SPEC §5.5–§7) — fakes, no GPU."""

import numpy as np

from motionforge.config import DEFAULTS
from motionforge.coordinator import CoordinatorState, TaskCoordinator
from motionforge.coordinator.fakes import FakeGripper, RecordingExecution, ScriptedPerception
from motionforge.coordinator.interfaces import PickPerception, PlacePerception
from motionforge.geometry import Pose
from motionforge.joint_state import FakeJointStateSource
from motionforge.planner import GraspPlan
from motionforge.types import GraspCandidate, GripConfig, JointTrajectory, PlaceCandidate, PlanResult

DOF = 6


def _traj(q=None):
    q = list(q) if q is not None else [0.0] * DOF
    pts = [(q, [0.0] * DOF, [0.0] * DOF, 0.0), (q, [0.0] * DOF, [0.0] * DOF, 0.1)]
    return JointTrajectory(joint_names=[f"j{i}" for i in range(DOF)], points=pts)


def _reachable(pose: Pose) -> bool:
    return float(np.max(np.abs(pose.position))) <= 10.0


class FakePlanner:
    """Plans succeed only for 'reachable' goals (|position| <= 10); marks the candidate index."""

    def plan_grasp(self, grasps, q0=None):
        for i, g in enumerate(grasps):
            if _reachable(g.tcp_pose):
                t = _traj(q0)
                return GraspPlan(True, candidate_index=i, status="ok", approach=t, grasp=t,
                                 lift=t, metrics={"planning_time": 0.03})
        return GraspPlan(False, status="unreachable")

    def plan_segment(self, goal, constraints, q0=None, axis=None):
        if _reachable(goal):
            return PlanResult(success=True, trajectory=_traj(q0), metrics={"total_time": 0.02})
        return PlanResult(success=False, metrics={"total_time": 0.02})


class FakeWorld:
    def __init__(self, log):
        self.log = log
        self.attached = False
        self.layers = {}

    def set_voxel_layer(self, name, grid):
        self.layers[name] = grid

    def commit(self):
        self.log.append(("world.commit",))

    def attach_object(self, obstacles, q_grasp=None):
        self.attached = True
        self.log.append(("world.attach",))

    def detach_object(self):
        self.attached = False
        self.log.append(("world.detach",))


def _grasp(reachable=True):
    x = 0.5 if reachable else 999.0
    return GraspCandidate(Pose([x, 0.0, 0.3], [1, 0, 0, 0]), [0, 0, 1], 0.05, "jaw",
                          GripConfig(width_m=0.02, force=30.0))


def _place(reachable=True):
    x = 0.4 if reachable else 999.0
    return PlaceCandidate(Pose([x, 0.3, 0.2], [1, 0, 0, 0]), [0, 0, 1], 0.05, "jaw",
                          GripConfig(width_m=0.08, mode="outward"))


def _make_coordinator(picks, places, planner=None, log=None):
    log = log if log is not None else []
    gripper = FakeGripper(log)
    execution = RecordingExecution(log)
    world = FakeWorld(log)
    perception = ScriptedPerception(picks, places)
    return (
        TaskCoordinator(
            planner=planner or FakePlanner(),
            world=world,
            tools=None,
            perception=perception,
            gripper=gripper,
            execution=execution,
            joint_state_source=FakeJointStateSource([0.0] * DOF),
            config=DEFAULTS,
        ),
        log,
        gripper,
        execution,
        world,
        perception,
    )


def test_happy_path_reaches_done():
    picks = [PickPerception([_grasp()], workpiece=object())]
    places = [PlacePerception([_place()])]
    coord, log, gripper, execution, world, _ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert result.success
    assert result.state == CoordinatorState.DONE
    assert result.recaptures == 0
    # approach, grasp, lift + transport, place, retract = 6 streamed trajectories.
    assert len(execution.sent) == 6
    assert result.max_plan_time_s < 1.0


def test_barrier_and_attach_detach_ordering():
    picks = [PickPerception([_grasp()], workpiece=object())]
    places = [PlacePerception([_place()])]
    coord, log, *_ = _make_coordinator(picks, places)
    coord.run_cycle()

    tags = [e[0] for e in log]
    first_send = tags.index("exec.send")
    # Open-to-standby is commanded (non-blocking) before the first motion.
    open_cmd = next(i for i, e in enumerate(log) if e[0] == "gripper.command" and e[2] is False)
    assert open_cmd < first_send
    # Attach happens after a blocking grip command + barrier, and before detach.
    grip_cmd = next(i for i, e in enumerate(log) if e[0] == "gripper.command" and e[2] is True)
    attach = tags.index("world.attach")
    detach = tags.index("world.detach")
    assert grip_cmd < attach < detach


def test_pick_candidate_fallback_advances():
    # First grasp unreachable, second reachable -> fallback within the goalset (no recapture).
    picks = [PickPerception([_grasp(reachable=False), _grasp(reachable=True)])]
    places = [PlacePerception([_place()])]
    coord, *_ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert result.success
    assert result.pick_candidate_index == 1
    assert result.recaptures == 0


def test_place_candidate_fallback_advances():
    # First place candidate unreachable, second reachable -> fallback (no recapture).
    picks = [PickPerception([_grasp()])]
    places = [PlacePerception([_place(reachable=False), _place(reachable=True)])]
    coord, *_ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert result.success
    assert result.place_candidate_index == 1
    assert result.recaptures == 0


def test_pick_recapture_then_success():
    # No grasps for 3 captures, then a reachable grasp on the 4th (within recapture_cap=3).
    empty = PickPerception([])
    good = PickPerception([_grasp()])
    picks = [empty, empty, empty, good]
    places = [PlacePerception([_place()])]
    coord, *_ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert result.success
    assert result.recaptures == 3


def test_pick_fault_after_recapture_cap():
    picks = [PickPerception([])]  # always empty
    places = [PlacePerception([_place()])]
    coord, *_ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert not result.success
    assert result.state == CoordinatorState.FAULT
    assert "pick" in result.fault_reason


def test_place_recapture_then_success():
    picks = [PickPerception([_grasp()])]
    bad = PlacePerception([_place(reachable=False)])
    good = PlacePerception([_place(reachable=True)])
    coord, *_ = _make_coordinator(picks, [bad, good])
    result = coord.run_cycle()
    assert result.success
    assert result.recaptures == 1


def test_place_fault_after_recapture_cap():
    picks = [PickPerception([_grasp()])]
    places = [PlacePerception([_place(reachable=False)])]  # never reachable
    coord, *_ = _make_coordinator(picks, places)
    result = coord.run_cycle()
    assert not result.success
    assert result.state == CoordinatorState.FAULT
    assert "place" in result.fault_reason
