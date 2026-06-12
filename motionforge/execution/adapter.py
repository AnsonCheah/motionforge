"""OEM-agnostic Execution Adapter interface (SPEC §5.6).

The boundary is fixed so EGM / ros2_control adapters drop in later without touching upstream.
``send_trajectory`` consumes our dense :class:`JointTrajectory`; concrete adapters down-sample
and stream it to the controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from motionforge.types import JointTrajectory


@dataclass
class Waypoint:
    """A controller waypoint: joint target + motion blending + timing.

    ``dt_s`` is the planned time to reach this waypoint from the previous one, carried over
    from cuRobo's time-parameterization so the controller (or the sim) can honor the planned
    cadence instead of running at a fixed rate. ``speed``/``zone`` are reserved for the real
    RAPID ``speeddata``/``zonedata`` mapping.
    """

    q: List[float]
    speed: float = 1.0   # speeddata scalar
    zone: float = 0.01   # zonedata corner-blend radius (m); >0 lets look-ahead blend corners
    dt_s: float = 0.0    # planned time delta from the previous waypoint (s)


class ExecutionAdapter(ABC):
    """Planner-as-master execution boundary."""

    @abstractmethod
    def send_trajectory(self, traj: JointTrajectory):
        """Execute a trajectory; returns an implementation-defined handle."""

    @abstractmethod
    def read_joint_state(self) -> List[float]:
        """Return the latest joint feedback (the trajectory start ``q0`` before motion)."""

    @abstractmethod
    def stop(self) -> None:
        """Abort motion and clear the controller buffer."""


def trajectory_to_waypoints(
    traj: JointTrajectory,
    kept_indices: Sequence[int],
    speed: float = 1.0,
    zone: float = 0.01,
) -> List[Waypoint]:
    """Build controller waypoints from the kept (down-sampled) trajectory indices.

    Each waypoint's ``dt_s`` is the planned time elapsed since the previous KEPT waypoint
    (point ``time_from_start`` deltas), preserving cuRobo's time-parameterization through the
    down-sampling. The first kept waypoint's ``dt_s`` is its own ``time_from_start``.
    """
    waypoints: List[Waypoint] = []
    prev_t = 0.0
    for i in kept_indices:
        t = float(traj.points[i][3])
        waypoints.append(Waypoint(q=list(traj.points[i][0]), speed=speed, zone=zone, dt_s=t - prev_t))
        prev_t = t
    return waypoints
