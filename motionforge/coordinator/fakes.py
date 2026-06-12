"""Reusable fakes for the coordinator (tests, the Isaac twin bring-up, demos).

GPU-free. ``FakeGripper`` records actuation + barrier order; ``ScriptedPerception`` returns
preset perceptions on successive calls (drives fallback/recapture); ``RecordingExecution``
records the trajectories streamed to the controller.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from motionforge.coordinator.interfaces import (
    GripperActuator,
    PerceptionSource,
    PickPerception,
    PlacePerception,
)
from motionforge.execution.adapter import ExecutionAdapter
from motionforge.types import JointTrajectory, ToolAction


class FakeGripper(GripperActuator):
    def __init__(self, log: Optional[list] = None) -> None:
        self.log = log if log is not None else []
        self.width: Optional[float] = None
        self.commands: List[ToolAction] = []

    def command(self, action: ToolAction) -> None:
        self.commands.append(action)
        self.log.append(("gripper.command", action.grip.width_m, action.blocking))
        # Non-blocking actuations complete in the background; the barrier is wait().
        if action.blocking:
            self.width = action.grip.width_m

    def wait(self) -> None:
        self.log.append(("gripper.wait",))
        if self.commands:
            self.width = self.commands[-1].grip.width_m


class RecordingExecution(ExecutionAdapter):
    """Execution adapter that records streamed trajectories and the final q (no socket).

    Optionally drives a joint-state sink (the coordinator reads feedback from there after each
    segment for execution verification). ``drift`` adds a per-joint-0 offset to the reported
    end config to simulate the robot NOT reaching the planned target (divergence → FAULT).
    """

    def __init__(self, log: Optional[list] = None, joint_state=None, drift: float = 0.0) -> None:
        self.log = log if log is not None else []
        self.sent: List[JointTrajectory] = []
        self._last_q: List[float] = []
        self._joint_state = joint_state
        self._drift = drift

    def send_trajectory(self, traj: JointTrajectory):
        self.sent.append(traj)
        self.log.append(("exec.send", len(traj)))
        if traj.points:
            self._last_q = list(traj.points[-1][0])
            if self._drift:
                self._last_q[0] += self._drift
            if self._joint_state is not None:
                self._joint_state.set(self._last_q)
        return {"waypoints": len(traj)}

    def read_joint_state(self) -> List[float]:
        return list(self._last_q)

    def stop(self) -> None:
        self.log.append(("exec.stop",))


class ScriptedPerception(PerceptionSource):
    """Returns the i-th preset perception on the i-th call (last one repeats)."""

    def __init__(
        self,
        picks: Sequence[PickPerception],
        places: Sequence[PlacePerception],
    ) -> None:
        self._picks = list(picks)
        self._places = list(places)
        self.pick_calls = 0
        self.place_calls = 0

    def perceive_pick(self) -> PickPerception:
        p = self._picks[min(self.pick_calls, len(self._picks) - 1)]
        self.pick_calls += 1
        return p

    def perceive_place(self) -> PlacePerception:
        p = self._places[min(self.place_calls, len(self._places) - 1)]
        self.place_calls += 1
        return p
