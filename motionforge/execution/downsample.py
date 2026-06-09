"""Curvature/error-bounded waypoint down-sampling (SPEC §5.6 step 1).

Reduce a dense cuRobo trajectory to a small set of waypoints such that linearly
interpolating between consecutive kept waypoints stays within ``max_joint_error`` of the
original at every original sample (greedy, joint-space). Endpoints are always kept.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _segment_within(P: np.ndarray, a: int, b: int, tol: float) -> bool:
    """True if the straight segment P[a]→P[b] keeps every intermediate point within ``tol``."""
    if b - a < 2:
        return True
    seg = P[b] - P[a]
    span = b - a
    for k in range(a + 1, b):
        interp = P[a] + seg * ((k - a) / span)
        if float(np.max(np.abs(P[k] - interp))) > tol:
            return False
    return True


def downsample_waypoints(positions: Sequence[Sequence[float]], max_joint_error: float) -> List[int]:
    """Return kept indices into ``positions`` (a dense list of dof-vectors)."""
    P = np.asarray(positions, dtype=float)
    n = len(P)
    if n <= 2:
        return list(range(n))
    kept = [0]
    anchor = 0
    for i in range(2, n):
        if not _segment_within(P, anchor, i, max_joint_error):
            kept.append(i - 1)
            anchor = i - 1
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return kept


def interpolation_error(positions: Sequence[Sequence[float]], kept_indices: Sequence[int]) -> float:
    """Max joint error when linearly interpolating the original samples between kept waypoints."""
    P = np.asarray(positions, dtype=float)
    idx = sorted(kept_indices)
    max_err = 0.0
    for s, e in zip(idx[:-1], idx[1:]):
        span = e - s
        for k in range(s, e + 1):
            t = 0.0 if span == 0 else (k - s) / span
            interp = P[s] + (P[e] - P[s]) * t
            max_err = max(max_err, float(np.max(np.abs(P[k] - interp))))
    return max_err
