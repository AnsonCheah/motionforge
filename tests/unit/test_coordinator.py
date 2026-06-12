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
    def __init__(self, log, tool_geometry=False):
        self.log = log
        self.attached = False
        self.layers = {}
        self.tool_widths = []  # commanded widths re-attached, in order
        self._tool_geometry = tool_geometry

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

    def attach_tool_geometry(self, body):
        self.tool_widths.append(body.data["width"])
        self.log.append(("world.attach_tool", body.data["width"]))


class FakeTools:
    """Minimal ToolManager stand-in: builds a parallel-jaw CollisionBody per commanded width."""

    def collision_geom(self, grip, tool_id=None):
        from motionforge.tools import parallel_jaw_geom_fn

        return parallel_jaw_geom_fn()(grip)


def _grasp(reachable=True):
    x = 0.5 if reachable else 999.0
    return GraspCandidate(Pose([x, 0.0, 0.3], [1, 0, 0, 0]), [0, 0, 1], 0.05, "jaw",
                          GripConfig(width_m=0.02, force=30.0))


def _place(reachable=True):
    x = 0.4 if reachable else 999.0
    return PlaceCandidate(Pose([x, 0.3, 0.2], [1, 0, 0, 0]), [0, 0, 1], 0.05, "jaw",
                          GripConfig(width_m=0.08, mode="outward"))


def _make_coordinator(picks, places, planner=None, log=None, joints=None, execution=None):
    log = log if log is not None else []
    gripper = FakeGripper(log)
    joints = joints if joints is not None else FakeJointStateSource([0.0] * DOF)
    # The execution drives the joint-state sink so post-segment verification sees the
    # achieved config (mirrors the real adapter, which is both executor and joint source).
    execution = execution if execution is not None else RecordingExecution(log, joint_state=joints)
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
            joint_state_source=joints,
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


def test_tool_geometry_reattached_at_three_widths():
    # With a tool library wired, the gripper geometry is re-attached at standby (pick plan),
    # grasp width (pre-place), and release width (pre-final-retract), in that order (SPEC §5.4).
    log = []
    world = FakeWorld(log)
    gripper = FakeGripper(log)
    execution = RecordingExecution(log)
    picks = [PickPerception([_grasp()], workpiece=object())]  # grasp width 0.02
    places = [PlacePerception([_place()])]                    # release width 0.08
    coord = TaskCoordinator(
        planner=FakePlanner(), world=world, tools=FakeTools(),
        perception=ScriptedPerception(picks, places), gripper=gripper, execution=execution,
        joint_state_source=FakeJointStateSource([0.0] * DOF), config=DEFAULTS,
    )
    result = coord.run_cycle()
    assert result.success
    # standby (config default 0.08), grasp width (0.02), release width (0.08).
    assert world.tool_widths == [DEFAULTS.gripper_standby_width_m, 0.02, 0.08]


def test_tool_geometry_skipped_when_no_tools():
    # tools=None -> no tool-geometry calls (the default unit path stays GPU-free).
    log = []
    picks = [PickPerception([_grasp()], workpiece=object())]
    places = [PlacePerception([_place()])]
    coord, _, _, _, world, _ = _make_coordinator(picks, places)  # tools=None
    coord.run_cycle()
    assert world.tool_widths == []


# -- robustness: empty q0, execution divergence, fault cleanup --


def test_empty_q0_faults_without_planning():
    # An empty joint-state read must FAULT (never silently plan from cuRobo's default config).
    picks = [PickPerception([_grasp()])]
    places = [PlacePerception([_place()])]
    coord, log, *_ = _make_coordinator(picks, places, joints=FakeJointStateSource([]))
    result = coord.run_cycle()
    assert not result.success
    assert result.state == CoordinatorState.FAULT
    assert "q0" in result.fault_reason
    assert not any(e[0] == "exec.send" for e in log)  # nothing was streamed


def test_execution_divergence_faults_and_stops():
    # Feedback that diverges from the planned segment end faults the cycle and stops the robot.
    log = []
    joints = FakeJointStateSource([0.0] * DOF)
    execution = RecordingExecution(log, joint_state=joints, drift=0.2)  # 0.2 rad > 0.05 tol
    picks = [PickPerception([_grasp()], workpiece=object())]
    places = [PlacePerception([_place()])]
    coord, log, *_ = _make_coordinator(picks, places, log=log, joints=joints, execution=execution)
    result = coord.run_cycle()
    assert not result.success
    assert result.state == CoordinatorState.FAULT
    assert "diverged" in result.fault_reason
    assert any(e[0] == "exec.stop" for e in log)


def test_exception_midcycle_cleans_up_attachment():
    # A planner that raises during PLACE planning (after the pick attach) must still return a
    # FAULT result (not raise) and detach the held object + stop the robot.
    class RaiseOnPlace:
        def __init__(self):
            self.calls = 0

        def plan_segment(self, goal, constraints, q0=None, axis=None):
            self.calls += 1
            if self.calls > 3:  # pick = 3 segments; the 4th call is the place transport
                raise RuntimeError("planner blew up mid-place")
            return PlanResult(success=True, trajectory=_traj(q0), metrics={"total_time": 0.02})

    picks = [PickPerception([_grasp()], workpiece=object())]
    places = [PlacePerception([_place()])]
    coord, log, _, _, world, _ = _make_coordinator(picks, places, planner=RaiseOnPlace())
    result = coord.run_cycle()
    assert not result.success
    assert result.state == CoordinatorState.FAULT
    assert "blew up" in result.fault_reason
    # The workpiece attached during the pick is released; the controller is stopped.
    assert world.attached is False
    assert ("world.detach",) in log
    assert any(e[0] == "exec.stop" for e in log)
