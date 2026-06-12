"""Phase 3 GPU tests: TCP tool frame, attach-frame correctness, gripper collision geometry.

Covers the breaking points: goals planned to the real TCP (not the flange); the held object
actually lands on the tool (regression for the phantom-attach bug); the gripper's own
width-dependent geometry participates in collision while not self-colliding with the arm.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from curobo.scene import Cuboid  # noqa: E402

from motionforge.collision import CollisionWorldManager  # noqa: E402
from motionforge.geometry import Pose  # noqa: E402
from motionforge.planner import MotionPlannerAdapter  # noqa: E402
from motionforge.tools import parallel_jaw_geom_fn  # noqa: E402
from motionforge.tools.tool_manager import collision_body_to_cuboid_specs  # noqa: E402
from motionforge.types import GripConfig  # noqa: E402

from tests.gpu.conftest import TEST_CONFIG  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

TCP_OFFSET = Pose([0.0, 0.0, 0.15], [1.0, 0.0, 0.0, 0.0])  # 0.15 m past tool0 along +Z


@pytest.fixture(scope="module")
def tool_adapter():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = MotionPlannerAdapter(
        config=TEST_CONFIG, tcp_offset=TCP_OFFSET,
        collision_cache={"cuboid": 8},
        attached_object_spheres=64, tool_spheres=32,
    )
    a.warmup()
    return a


def _q(adapter, deltas):
    q = np.array(adapter.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return q.tolist()


def _active_spheres(adapter, q):
    """World-frame [N,4] (x,y,z,r) collision spheres with positive radius at config ``q``."""
    state = adapter.planner.compute_kinematics(adapter.to_joint_state(q))
    sph = state.robot_spheres.reshape(-1, 4).detach().cpu().numpy()
    return sph[sph[:, 3] > 0.0]


def test_tool_frames_is_tcp(tool_adapter):
    assert tool_adapter.tool_frames == ["tcp"]


def test_tcp_offset_shifts_tool_frame(tool_adapter, mf_planner):
    # mf_planner is the plain UR10e (tool_frames=["tool0"]) — its FK is the flange reference.
    assert mf_planner.tool_frames == ["tool0"]
    for deltas in ([0, 0, 0], [0.2, -0.1, 0.0], [0.0, 0.3, -0.2]):
        q = _q(tool_adapter, deltas)
        tcp = tool_adapter.tcp_pose_at(q)
        flange = mf_planner.tcp_pose_at(q)
        expected = flange.multiply(TCP_OFFSET)
        assert tcp.approx_equal(expected, atol=1e-4), f"tcp {tcp.position} != {expected.position}"


def test_plan_to_tcp_goal_reaches(tool_adapter):
    q_target = _q(tool_adapter, [0.25, -0.2, 0.2])
    goal = tool_adapter.tcp_pose_at(q_target)
    result = tool_adapter.plan_free(goal, max_attempts=5)
    assert result.success
    q_end = result.trajectory.points[-1][0]
    fk_end = tool_adapter.tcp_pose_at(q_end)
    assert np.allclose(fk_end.position, goal.position, atol=1e-2)


def test_attach_object_lands_on_tool(tool_adapter):
    # The held workpiece must sit at the TCP, not a phantom ~1 m away (the prior bug).
    q = tool_adapter.default_q0
    tcp = tool_adapter.tcp_pose_at(q)
    tcp_pos = np.asarray(tcp.position)

    before = _active_spheres(tool_adapter, q)
    d_before = np.linalg.norm(before[:, :3] - tcp_pos, axis=1).min()

    cube = Cuboid(name="part", pose=[*tcp.position.tolist(), 1, 0, 0, 0], dims=[0.05, 0.05, 0.05])
    tool_adapter.attach_object([cube], q_grasp=q)
    try:
        after = _active_spheres(tool_adapter, q)
        d_after = np.linalg.norm(after[:, :3] - tcp_pos, axis=1).min()
        # Before attach the nearest active sphere is an arm/wrist sphere; after attach the
        # workpiece spheres sit right at the TCP.
        assert d_before > 0.05, f"unexpected sphere near TCP pre-attach: {d_before:.3f}"
        assert d_after < 0.03, f"attached object did not land on the TCP: {d_after:.3f}"
        # Planning still succeeds with the object attached (self-collision ignored).
        assert tool_adapter.plan_free(tool_adapter.tcp_pose_at(_q(tool_adapter, [0.2, -0.15, 0.2]))).success
    finally:
        tool_adapter.detach_object()
    assert tool_adapter.attached_link_name is None


def test_tool_geometry_blocks_and_selfcollision_ignored(tool_adapter):
    q = tool_adapter.default_q0
    tcp = tool_adapter.tcp_pose_at(q)
    # A wide (open/standby) jaw set: each jaw sits half_sep laterally from the TCP axis.
    jaw_fn = parallel_jaw_geom_fn(jaw_length=0.04, jaw_thickness=0.02, jaw_height=0.05)
    body = jaw_fn(GripConfig(width_m=0.08))
    half_sep = abs(body.data["jaws"][0]["offset"][0])  # (0.08 + 0.02)/2 = 0.05

    world = CollisionWorldManager(tool_adapter)
    tool_adapter.attach_tool(
        [Cuboid(name=s["name"], pose=s["pose"], dims=s["dims"])
         for s in collision_body_to_cuboid_specs(body)]
    )
    try:
        # Self-collision is ignored: the wide jaws at the TCP do not block free planning.
        assert tool_adapter.plan_free(tool_adapter.tcp_pose_at(_q(tool_adapter, [0.2, -0.15, 0.2]))).success
        assert tool_adapter.tool_attached

        # Place a small static obstacle exactly where the +x jaw is (lateral to the TCP), where
        # the bare flange/arm is NOT. With tool geometry attached, the start state collides.
        x_world = tcp.rotation_matrix() @ np.array([1.0, 0.0, 0.0])  # tool +x in base frame
        jaw_center = np.asarray(tcp.position) + half_sep * x_world
        obstacle = Cuboid(name="post", pose=[*jaw_center.tolist(), 1, 0, 0, 0], dims=[0.03, 0.03, 0.03])

        world.set_static(cuboids=[obstacle])
        world.commit()
        # Jaw overlaps the post -> start in collision -> planning fails.
        assert not tool_adapter.plan_free(tcp, max_attempts=1).success

        # Detach the tool geometry: the bare arm clears the post (it's lateral, not on the arm).
        tool_adapter.detach_tool()
        assert tool_adapter.plan_free(tool_adapter.tcp_pose_at(_q(tool_adapter, [0.1, -0.1, 0.15]))).success
    finally:
        tool_adapter.detach_tool()
        world.set_static()
        world.commit()
