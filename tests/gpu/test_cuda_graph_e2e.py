"""Phase 7 GPU test: the FULL pipeline under captured CUDA graphs (the production config).

The node runs ``use_cuda_graph=True`` while every other GPU test runs with graphs OFF, so this
file closes that gap: it proves runtime mutations (voxel commits, attach/detach, per-segment
tool-pose-criteria swaps) are visible under graph replay, and MEASURES per-segment plan times
so the <1 s budget is verified for the real config. If a mutation kind ever goes stale here,
add it to ``MotionPlannerAdapter.GRAPH_RESET_KINDS`` (and re-measure the budget).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from dataclasses import replace  # noqa: E402

from curobo._src.geom.types import VoxelGrid  # noqa: E402
from curobo.scene import Cuboid  # noqa: E402

from motionforge.collision import CollisionWorldManager  # noqa: E402
from motionforge.coordinator import CoordinatorState, TaskCoordinator  # noqa: E402
from motionforge.coordinator.fakes import FakeGripper, RecordingExecution, ScriptedPerception  # noqa: E402
from motionforge.coordinator.interfaces import PickPerception, PlacePerception  # noqa: E402
from motionforge.geometry import Pose  # noqa: E402
from motionforge.joint_state import FakeJointStateSource  # noqa: E402
from motionforge.planner import MotionPlannerAdapter  # noqa: E402
from motionforge.tools import ToolManager, parallel_jaw_geom_fn  # noqa: E402
from motionforge.types import GraspCandidate, GripConfig, PlaceCandidate, ToolDescriptor  # noqa: E402

from tests.gpu.conftest import TEST_CONFIG  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DIMS = (1.0, 1.0, 1.0)
VS = 0.02
BIN_CENTER = (0.5, 0.0, 0.3)
TRAY_CENTER = (0.1, 0.4, 0.3)

# Graphs ON — the production config (vs TEST_CONFIG which disables them).
GRAPH_CONFIG = replace(TEST_CONFIG, use_cuda_graph=True, warmup_iterations=2)


def _empty_grid(center=BIN_CENTER):
    n = [round(d / VS) for d in DIMS]
    feat = torch.full(tuple(n), 1.0, dtype=torch.float16, device="cuda:0")
    return VoxelGrid(name="esdf", pose=[*center, 1, 0, 0, 0], dims=list(DIMS), voxel_size=VS,
                     feature_tensor=feat, feature_dtype=torch.float16)


def _box_grid(center, box_center, half=(0.1, 0.1, 0.1)):
    n = [round(d / VS) for d in DIMS]
    ix, iy, iz = (torch.arange(k, device="cuda:0", dtype=torch.float32) for k in n)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    wx = center[0] + (gx - (n[0] - 1) / 2) * VS
    wy = center[1] + (gy - (n[1] - 1) / 2) * VS
    wz = center[2] + (gz - (n[2] - 1) / 2) * VS
    dx = (wx - box_center[0]).abs() - half[0]
    dy = (wy - box_center[1]).abs() - half[1]
    dz = (wz - box_center[2]).abs() - half[2]
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.stack([dx, dy, dz], -1).max(-1).values.clamp(max=0)
    return VoxelGrid(name="esdf", pose=[*center, 1, 0, 0, 0], dims=list(DIMS), voxel_size=VS,
                     feature_tensor=(outside + inside).to(torch.float16), feature_dtype=torch.float16)


@pytest.fixture(scope="module")
def graph_adapter():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    adapter = MotionPlannerAdapter(
        config=GRAPH_CONFIG,
        collision_cache={"cuboid": 8, "voxel": {"layers": 2, "dims": list(DIMS), "voxel_size": VS}},
        attached_object_spheres=64, tool_spheres=32,
        tcp_offset=Pose([0.0, 0.0, 0.15], [1.0, 0.0, 0.0, 0.0]),  # realistic config (graphs ON)
    )
    adapter.warmup()  # captures CUDA graphs (enable_graph follows use_cuda_graph)
    return adapter


def _q(adapter, deltas):
    q = np.array(adapter.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return adapter.tcp_pose_at(q.tolist())


def test_full_pick_and_place_cycle_with_graphs(graph_adapter, capsys):
    adapter = graph_adapter
    world = CollisionWorldManager(adapter)
    tools = ToolManager()
    tools.register(ToolDescriptor("jaw", Pose([0, 0, 0.0], [1, 0, 0, 0]), parallel_jaw_geom_fn(),
                                  actuation_iface="socket://gripper", payload_kg=0.5))

    grasp_pose = _q(adapter, [0.25, -0.25, 0.2])
    place_pose = _q(adapter, [-0.25, -0.25, 0.2])
    grasp = GraspCandidate(grasp_pose, [0, 0, -1], 0.05, "jaw", GripConfig(0.02, force=30.0))
    place = PlaceCandidate(place_pose, [0, 0, -1], 0.05, "jaw", GripConfig(0.08, mode="outward"))
    workpiece = Cuboid(name="part", pose=[*grasp_pose.position.tolist(), 1, 0, 0, 0],
                       dims=[0.04, 0.04, 0.04])

    perception = ScriptedPerception(
        picks=[PickPerception([grasp], bin_voxels=_empty_grid(BIN_CENTER), workpiece=workpiece)],
        places=[PlacePerception([place], tray_voxels=_empty_grid(TRAY_CENTER))],
    )
    joints = FakeJointStateSource(adapter.default_q0)
    execution = RecordingExecution(joint_state=joints)
    coord = TaskCoordinator(
        planner=adapter, world=world, tools=tools, perception=perception,
        gripper=FakeGripper(), execution=execution,
        joint_state_source=joints, config=GRAPH_CONFIG,
    )

    result = coord.run_cycle()

    # Under captured graphs the whole pipeline (voxel commits + attach/detach + per-segment
    # criteria swaps) still produces a collision-free cycle to DONE.
    assert result.success, f"cycle failed in {result.state} ({result.fault_reason})"
    assert result.state == CoordinatorState.DONE
    assert len(execution.sent) == 6
    assert adapter.attached_link_name is None

    # MEASUREMENT GATE (SPEC §8/§11): per-segment plan budget under the production config.
    with capsys.disabled():
        print(f"\n[cuda-graph] per-segment plan times (s): "
              f"{[round(t, 4) for t in result.plan_times_s]}  max={result.max_plan_time_s:.4f}")
    assert result.max_plan_time_s < 1.0, (
        f"per-segment plan budget exceeded under graphs: {result.max_plan_time_s:.3f}s "
        "(if a mutation forced re-capture, reconsider GRAPH_RESET_KINDS / the graph decision)"
    )


def test_voxel_commit_visible_under_graphs(graph_adapter):
    # A voxel obstacle committed AFTER graph capture must be enforced (mutation not stale).
    adapter = graph_adapter
    world = CollisionWorldManager(adapter)
    tcp = adapter.tcp_pose_at()

    world.set_voxel_layer("pick_bin", _empty_grid(BIN_CENTER))
    world.commit()
    assert adapter.plan_free(_q(adapter, [0.2, -0.2, 0.2]), max_attempts=3).success

    world.set_voxel_layer("pick_bin", _box_grid(BIN_CENTER, tuple(tcp.position.tolist())))
    world.commit()
    assert not adapter.plan_free(tcp, max_attempts=1).success

    world.clear_voxel_layer("pick_bin")
    world.commit()


def test_attach_visible_under_graphs(graph_adapter):
    # An attach committed AFTER graph capture must propagate to the kinematics the planner's
    # captured graphs read (link_spheres are written in-place; replay sees the update).
    adapter = graph_adapter
    q = adapter.default_q0
    tcp = adapter.tcp_pose_at(q)
    tcp_pos = np.asarray(tcp.position)

    def nearest_active_sphere_dist():
        st = adapter.planner.compute_kinematics(adapter.to_joint_state(q))
        sph = st.robot_spheres.reshape(-1, 4).detach().cpu().numpy()
        active = sph[sph[:, 3] > 0.0]
        return np.linalg.norm(active[:, :3] - tcp_pos, axis=1).min()

    bare = nearest_active_sphere_dist()  # bare arm: nearest sphere is an arm/wrist sphere
    cube = Cuboid(name="part", pose=[*tcp.position.tolist(), 1, 0, 0, 0], dims=[0.05, 0.05, 0.05])
    adapter.attach_object([cube], q_grasp=q)
    try:
        attached = nearest_active_sphere_dist()
        # The attached body sits right at the TCP — clearly nearer than any bare-arm sphere.
        assert attached < 0.02 and attached < bare - 0.02, f"bare={bare:.3f} attached={attached:.3f}"
        # The plan engine still produces a valid trajectory with the body attached.
        assert adapter.plan_free(_q(adapter, [0.2, -0.15, 0.2])).success
    finally:
        adapter.detach_object()
    assert nearest_active_sphere_dist() == pytest.approx(bare, abs=1e-4)  # detach restored bare arm
