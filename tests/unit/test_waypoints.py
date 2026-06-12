"""Unit tests for trajectory → controller-waypoint conversion incl. timing (SPEC §5.6)."""

import pytest

from motionforge.execution.adapter import Waypoint, trajectory_to_waypoints
from motionforge.types import JointTrajectory


def _traj(times):
    pts = [([float(i)] * 2, [0.0] * 2, [0.0] * 2, float(t)) for i, t in enumerate(times)]
    return JointTrajectory(joint_names=["a", "b"], points=pts)


def test_dt_s_is_delta_since_previous_kept_waypoint():
    traj = _traj([0.0, 0.1, 0.25, 0.3, 0.7])
    wps = trajectory_to_waypoints(traj, kept_indices=[0, 1, 2, 3, 4])
    assert [round(w.dt_s, 6) for w in wps] == [0.0, 0.1, 0.15, 0.05, 0.4]


def test_dt_s_spans_downsampled_gaps():
    # When intermediate points are dropped, dt accumulates across the gap to the kept point.
    traj = _traj([0.0, 0.1, 0.25, 0.3, 0.7])
    wps = trajectory_to_waypoints(traj, kept_indices=[0, 2, 4])
    assert [round(w.dt_s, 6) for w in wps] == [0.0, 0.25, 0.45]
    # Total planned time is preserved regardless of which points are kept.
    assert sum(w.dt_s for w in wps) == pytest.approx(0.7)


def test_speed_and_zone_passthrough():
    traj = _traj([0.0, 0.2])
    wps = trajectory_to_waypoints(traj, kept_indices=[0, 1], speed=2.0, zone=0.03)
    assert all(isinstance(w, Waypoint) for w in wps)
    assert wps[0].speed == 2.0 and wps[0].zone == 0.03
    assert wps[1].dt_s == pytest.approx(0.2)
