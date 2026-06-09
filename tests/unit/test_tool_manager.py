"""Unit tests for the Tool & Gripper Manager (SPEC §5.4) — pure, no GPU."""

import numpy as np
import pytest

from motionforge.geometry import Pose
from motionforge.tools import ToolManager, parallel_jaw_geom_fn, vacuum_geom_fn
from motionforge.types import GripConfig, ToolDescriptor

QZ90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def _jaw_tool():
    return ToolDescriptor(
        tool_id="jaw",
        tcp_pose=Pose([0, 0, 0.12], [1, 0, 0, 0]),
        collision_geom_fn=parallel_jaw_geom_fn(),
        actuation_iface="socket://gripper",
        payload_kg=0.5,
    )


def test_register_and_active_default():
    m = ToolManager()
    m.register(_jaw_tool())
    assert m.active_id == "jaw"
    assert m.active.tool_id == "jaw"
    assert np.allclose(m.active_tcp().position, [0, 0, 0.12])


def test_set_active_unknown_raises():
    m = ToolManager()
    m.register(_jaw_tool())
    with pytest.raises(KeyError):
        m.set_active("nope")


def test_collision_geom_reflects_commanded_width():
    m = ToolManager()
    m.register(_jaw_tool())
    narrow = m.collision_geom(GripConfig(width_m=0.02))
    wide = m.collision_geom(GripConfig(width_m=0.08))
    # SPEC §5.4: geometry tracks the actual commanded width (never a fixed worst-case).
    assert narrow.data["width"] == 0.02
    assert wide.data["width"] == 0.08
    # The open/standby (wide) envelope separates the jaws further than the closed one.
    narrow_sep = narrow.data["jaws"][0]["offset"][0]
    wide_sep = wide.data["jaws"][0]["offset"][0]
    assert wide_sep > narrow_sep
    assert narrow.frame == "tcp"


def test_grasp_transform_roundtrip():
    m = ToolManager()
    m.register(_jaw_tool())
    grasp = Pose([0.5, 0.0, 0.3], QZ90)
    obj = Pose([0.5, 0.1, 0.3], QZ90)
    t = m.grasp_transform(obj, grasp)
    # Re-composing recovers the object pose (object expressed in the TCP frame).
    assert grasp.multiply(t).approx_equal(obj, atol=1e-9)


def test_actuation_uses_active_tool():
    m = ToolManager()
    m.register(_jaw_tool())
    action = m.actuation(GripConfig(width_m=0.02), blocking=True)
    assert action.tool_id == "jaw"
    assert action.blocking is True
    assert action.grip.width_m == 0.02


def test_vacuum_geom_fn_is_cylinder():
    m = ToolManager()
    m.register(
        ToolDescriptor(
            tool_id="vac",
            tcp_pose=Pose([0, 0, 0.05], [1, 0, 0, 0]),
            collision_geom_fn=vacuum_geom_fn(radius=0.02, length=0.04),
            actuation_iface="socket://vac",
            payload_kg=0.3,
        )
    )
    body = m.collision_geom(GripConfig(width_m=0.0, mode="vacuum_on"))
    assert "cylinder" in body.data
    assert body.data["cylinder"]["radius"] == 0.02
