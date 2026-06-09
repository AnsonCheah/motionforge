"""Phase 4 integration test: ABB socket adapter ↔ in-process fake RAPID server (SPEC §5.6).

Proves the ring-buffer look-ahead (≥ N points buffered ahead — no per-point stall), correct
joint feedback (q0 on connect, final q after motion), and full consumption.
"""

from dataclasses import replace

import numpy as np
import pytest

from motionforge.config import DEFAULTS
from motionforge.execution.abb_socket import AbbSocketAdapter
from motionforge.execution.fake_rapid_server import FakeRapidServer
from motionforge.types import JointTrajectory

pytestmark = pytest.mark.integration

HOME_Q = [0.1, -1.2, 1.0, -0.5, 0.3, 0.0]


def _zigzag_trajectory(n=24, dof=6):
    """A corner-at-every-point trajectory so down-sampling keeps all points (worst case)."""
    points = []
    for i in range(n):
        q = [HOME_Q[j] + 0.02 * ((-1) ** i) + 0.001 * i for j in range(dof)]
        points.append((q, [0.0] * dof, [0.0] * dof, 0.05 * i))
    return JointTrajectory(joint_names=[f"j{j}" for j in range(dof)], points=points)


def test_loopback_buffered_streaming_and_feedback():
    config = replace(DEFAULTS, waypoint_buffer_depth=5)
    server = FakeRapidServer(home_q=HOME_Q, consume_dt=0.01)
    port = server.start()
    adapter = AbbSocketAdapter("127.0.0.1", port, config=config, max_joint_error=1e-6)
    try:
        # q0 from the controller on connect (Joint State Source).
        q0 = adapter.connect()
        assert np.allclose(q0, HOME_Q)

        traj = _zigzag_trajectory(n=24)
        handle = adapter.send_trajectory(traj)
        kept = len(handle["kept_indices"])
        assert kept == 24  # zigzag keeps every point

        # Look-ahead: the controller held a full buffer (≥ buffer_depth) — no per-point stall.
        assert server.max_buffer_depth >= min(config.waypoint_buffer_depth, kept) - 1
        assert server.max_buffer_depth <= config.waypoint_buffer_depth  # flow control caps it
        assert server.max_buffer_depth >= 2  # definitively not one-at-a-time blocking

        # Everything executed; final feedback is the last waypoint.
        assert server.consumed_count == kept
        assert np.allclose(adapter.read_joint_state(), traj.points[-1][0])
    finally:
        adapter.close()
        server.stop()


def test_connect_reports_q0_before_motion():
    server = FakeRapidServer(home_q=HOME_Q)
    port = server.start()
    adapter = AbbSocketAdapter("127.0.0.1", port)
    try:
        q0 = adapter.connect()
        assert np.allclose(q0, HOME_Q)
        assert np.allclose(adapter.read_joint_state(), HOME_Q)
    finally:
        adapter.close()
        server.stop()
