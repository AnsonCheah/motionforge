"""Phase 6 ROS smoke test: PlannerNode bring-up, params, service (no GPU).

Marked ``ros``; skipped when rclpy isn't importable. Exercises node init, parameter →
Config mapping, parameter overrides, and the run_cycle service — without building the GPU
planner (that's lazy, on first run_cycle).
"""

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter  # noqa: E402

from motionforge_ros.planner_node import PlannerNode  # noqa: E402

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_brings_up_with_default_params(ros_context):
    node = PlannerNode()
    try:
        assert node.get_parameter("robot_yaml").value == "ur10e.yml"
        assert node.config.robot_yaml == "ur10e.yml"
        assert node.config.base_frame == "base_link"
        assert node.config.tcp_frame == "tool0"
        assert node.config.recapture_cap == 3
        services = [s for s, _ in node.get_service_names_and_types()]
        assert any(s.endswith("run_cycle") for s in services)
    finally:
        node.destroy_node()


def test_param_override_flows_into_config(ros_context):
    node = PlannerNode(
        parameter_overrides=[
            Parameter("recapture_cap", Parameter.Type.INTEGER, 5),
            Parameter("robot_yaml", Parameter.Type.STRING, "abb_irb1200.yml"),
        ]
    )
    try:
        assert node.config.recapture_cap == 5
        assert node.config.robot_yaml == "abb_irb1200.yml"
    finally:
        node.destroy_node()


def test_run_cycle_without_io_reports_error(ros_context):
    node = PlannerNode()
    try:
        from std_srvs.srv import Trigger

        resp = node._on_run_cycle(Trigger.Request(), Trigger.Response())
        # Without set_io(...) the lazy build fails gracefully (no crash, success=False).
        assert resp.success is False
        assert "set_io" in resp.message
    finally:
        node.destroy_node()
