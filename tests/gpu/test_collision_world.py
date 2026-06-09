"""Phase 2 GPU tests: CollisionWorldManager (SceneCfg + ESDF mapper + attach) on UR10e.

Mirrors cuRobo's ``test_motion_planner_esdf.py`` synthetic-ESDF patterns. All voxel layers
use a common grid (1×1×1 m @ 2 cm, centered at (0.5, 0, 0.3)) so one warmed planner with a
pre-sized voxel scene serves every test.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from curobo._src.geom.types import VoxelGrid  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402

from motionforge.collision import CollisionWorldManager  # noqa: E402
from motionforge.collision.world_manager import PICK_BIN_LAYER, PLACE_TRAY_LAYER  # noqa: E402
from motionforge.geometry import Pose  # noqa: E402
from motionforge.perception.camera import build_camera_observation  # noqa: E402
from motionforge.planner import MotionPlannerAdapter  # noqa: E402
from motionforge.types import PlanResult  # noqa: E402

from tests.gpu.conftest import TEST_CONFIG  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

GRID_DIMS = (1.0, 1.0, 1.0)
GRID_VS = 0.02
GRID_CENTER = (0.5, 0.0, 0.3)


def _empty_grid():
    n = [round(d / GRID_VS) for d in GRID_DIMS]
    feat = torch.full(tuple(n), 1.0, dtype=torch.float16, device="cuda:0")
    return VoxelGrid(
        name="esdf",
        pose=[*GRID_CENTER, 1.0, 0.0, 0.0, 0.0],
        dims=list(GRID_DIMS),
        voxel_size=GRID_VS,
        feature_tensor=feat,
        feature_dtype=torch.float16,
    )


def _box_grid(box_center, half=(0.06, 0.06, 0.06)):
    n = [round(d / GRID_VS) for d in GRID_DIMS]
    ix = torch.arange(n[0], device="cuda:0", dtype=torch.float32)
    iy = torch.arange(n[1], device="cuda:0", dtype=torch.float32)
    iz = torch.arange(n[2], device="cuda:0", dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    wx = GRID_CENTER[0] + (gx - (n[0] - 1) / 2.0) * GRID_VS
    wy = GRID_CENTER[1] + (gy - (n[1] - 1) / 2.0) * GRID_VS
    wz = GRID_CENTER[2] + (gz - (n[2] - 1) / 2.0) * GRID_VS
    dx = (wx - box_center[0]).abs() - half[0]
    dy = (wy - box_center[1]).abs() - half[1]
    dz = (wz - box_center[2]).abs() - half[2]
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.stack([dx, dy, dz], -1).max(-1).values.clamp(max=0)
    return VoxelGrid(
        name="esdf",
        pose=[*GRID_CENTER, 1.0, 0.0, 0.0, 0.0],
        dims=list(GRID_DIMS),
        voxel_size=GRID_VS,
        feature_tensor=(outside + inside).to(torch.float16),
        feature_dtype=torch.float16,
    )


@pytest.fixture(scope="module")
def world():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Initial empty voxel scene sizes the collision buffers; attached_object link enables attach.
    adapter = MotionPlannerAdapter(
        config=TEST_CONFIG,
        scene=Scene(voxel=[_empty_grid()]),
        attached_object_spheres=64,
    )
    adapter.warmup()
    return adapter, CollisionWorldManager(adapter)


def _reachable_goal(adapter, deltas):
    q = np.array(adapter.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return adapter.tcp_pose_at(q.tolist())


# ── Scene assembly + update_world ──


def test_build_scene_includes_layers_and_static(world):
    adapter, mgr = world
    mgr.set_static(cuboids=[Cuboid(name="table", pose=[0.4, 0, -0.05, 1, 0, 0, 0], dims=[0.8, 1.2, 0.05])])
    mgr.set_voxel_layer(PICK_BIN_LAYER, _empty_grid())
    scene = mgr.build_scene()
    assert scene.voxel is not None and len(scene.voxel) == 1
    assert scene.voxel[0].name == PICK_BIN_LAYER
    assert scene.cuboid is not None and len(scene.cuboid) == 1
    mgr.clear_voxel_layer(PICK_BIN_LAYER)
    mgr.set_static()  # reset so other tests start clean


def test_plan_free_after_empty_voxel_commit(world):
    adapter, mgr = world
    mgr.set_voxel_layer(PICK_BIN_LAYER, _empty_grid())
    mgr.commit()
    result = adapter.plan_free(_reachable_goal(adapter, [0.3, -0.2, 0.15]))
    assert isinstance(result, PlanResult)
    assert result.success
    mgr.clear_voxel_layer(PICK_BIN_LAYER)


def test_plan_with_box_obstacle_layer_runs(world):
    adapter, mgr = world
    # Box ESDF offset to the side; planner still returns a typed result.
    mgr.set_voxel_layer(PICK_BIN_LAYER, _box_grid(box_center=(0.5, 0.3, 0.3)))
    mgr.commit()
    result = adapter.plan_free(_reachable_goal(adapter, [0.25, -0.25, 0.2]), max_attempts=3)
    assert isinstance(result, PlanResult)
    mgr.clear_voxel_layer(PICK_BIN_LAYER)


# ── Multiple ESDF layers must compose (not mask each other) ──


def test_voxel_layers_merge_into_one_obstacle(world):
    adapter, mgr = world
    box = _box_grid(box_center=(0.5, 0.0, 0.3))
    mgr.set_voxel_layer(PICK_BIN_LAYER, box)
    mgr.set_voxel_layer(PLACE_TRAY_LAYER, _empty_grid())  # all free; must NOT mask the box
    scene = mgr.build_scene()
    # cuRobo honors only one voxel grid, so the layers are merged into a single grid...
    assert scene.voxel is not None and len(scene.voxel) == 1
    merged = scene.voxel[0].feature_tensor.float()
    # ...and the union (min signed distance) preserves the box's negative interior.
    assert float(merged.min()) < 0.0
    assert torch.allclose(merged, box.feature_tensor.float())
    mgr.clear_voxel_layer(PICK_BIN_LAYER)
    mgr.clear_voxel_layer(PLACE_TRAY_LAYER)


def test_empty_layer_does_not_mask_box_obstacle(world):
    adapter, mgr = world
    # Box centred on the default TCP so the start config is inside the obstacle.
    tcp = adapter.tcp_pose_at()
    box = _box_grid(box_center=tuple(tcp.position.tolist()), half=(0.10, 0.10, 0.10))
    mgr.set_voxel_layer(PICK_BIN_LAYER, box)
    mgr.set_voxel_layer(PLACE_TRAY_LAYER, _empty_grid())  # committed after the box
    mgr.commit()
    # With the merge fix the box is enforced (start in collision) -> planning fails.
    # Before the fix, the empty layer masked the box and this would spuriously succeed.
    result = adapter.plan_free(tcp, max_attempts=1)
    assert not result.success
    mgr.clear_voxel_layer(PICK_BIN_LAYER)
    mgr.clear_voxel_layer(PLACE_TRAY_LAYER)
    mgr.commit()


# ── Native warp ESDF mapper (depth → TSDF → ESDF → collision) ──


def test_mapper_integrate_compute_esdf_and_plan(world):
    adapter, mgr = world
    mapper = CollisionWorldManager.make_mapper(
        extent_meters_xyz=GRID_DIMS,
        voxel_size=GRID_VS,
        esdf_voxel_size=GRID_VS,
        esdf_extent_meters_xyz=GRID_DIMS,
        grid_center=GRID_CENTER,
    )
    # Synthetic depth: a camera 0.7 m above a flat surface at the grid centre, looking down.
    h = w = 48
    depth = np.full((h, w), 0.7, dtype=np.float32)
    intr = np.array([[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1.0]], dtype=np.float32)
    cam_pose = Pose([GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2] + 0.7], [0.0, 1.0, 0.0, 0.0])
    obs = build_camera_observation(depth, intr, cam_pose)

    grid = mgr.integrate_layer(PICK_BIN_LAYER, obs, mapper)
    assert grid.feature_tensor is not None
    assert grid.name == PICK_BIN_LAYER

    mgr.commit()
    result = adapter.plan_free(_reachable_goal(adapter, [0.2, -0.2, 0.25]))
    assert isinstance(result, PlanResult)
    mgr.clear_voxel_layer(PICK_BIN_LAYER)


# ── Attached held object ──


def test_attach_detach_roundtrip(world):
    adapter, mgr = world
    assert not mgr.attached
    # A small cube near the TCP at the default config.
    tcp = adapter.tcp_pose_at()
    cube = Cuboid(name="held", pose=[*tcp.position.tolist(), 1, 0, 0, 0], dims=[0.05, 0.05, 0.05])
    mgr.attach_object([cube], q_grasp=adapter.default_q0)
    assert mgr.attached
    assert adapter.attached_link_name == "attached_object"

    # Planning still works with the object attached.
    result = adapter.plan_free(_reachable_goal(adapter, [0.2, -0.15, 0.2]))
    assert isinstance(result, PlanResult)

    mgr.detach_object()
    assert not mgr.attached


def test_detach_without_attach_is_noop(world):
    adapter, mgr = world
    mgr.detach_object()
    assert not mgr.attached
