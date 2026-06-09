"""Unit tests for error-bounded waypoint down-sampling (SPEC §5.6)."""

import numpy as np

from motionforge.execution.downsample import downsample_waypoints, interpolation_error


def test_straight_line_keeps_only_endpoints():
    pts = [[float(i), float(2 * i)] for i in range(20)]  # collinear
    kept = downsample_waypoints(pts, max_joint_error=1e-6)
    assert kept == [0, 19]


def test_curved_trajectory_reduces_within_bound():
    t = np.linspace(0, np.pi, 60)
    pts = np.stack([np.sin(t), np.cos(t), 0.5 * t], axis=1).tolist()
    tol = 0.02
    kept = downsample_waypoints(pts, max_joint_error=tol)
    assert 2 <= len(kept) < len(pts)  # actually reduced
    assert kept[0] == 0 and kept[-1] == len(pts) - 1  # endpoints preserved
    assert interpolation_error(pts, kept) <= tol + 1e-9  # bound honored


def test_tighter_tolerance_keeps_more():
    t = np.linspace(0, np.pi, 60)
    pts = np.stack([np.sin(t), np.cos(t)], axis=1).tolist()
    loose = downsample_waypoints(pts, 0.05)
    tight = downsample_waypoints(pts, 0.005)
    assert len(tight) >= len(loose)


def test_short_trajectories_pass_through():
    assert downsample_waypoints([], 0.1) == []
    assert downsample_waypoints([[0.0, 0.0]], 0.1) == [0]
    assert downsample_waypoints([[0.0], [1.0]], 0.1) == [0, 1]
