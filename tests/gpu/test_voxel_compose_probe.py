"""Probe: do multiple ESDF ``VoxelGrid``s COMPOSE in cuRobo's collision world?

Two behaviors are pinned here (both verified against curobov2 source):

1. **Batched path composes.** ``VoxelData.load_batch(grids, env)`` loads every grid with
   its own pose/dims and enables all of them (data_voxel.py:402-470); the warp collision
   kernel then iterates all enabled grids. Grids may sit at DIFFERENT world poses — this is
   the basis for per-ROI ``pick_bin`` / ``place_tray`` layers at different cell locations.

2. **Scene path replaces.** ``update_world(Scene)`` → ``load_from_scene_cfg`` calls
   ``add_obstacle`` once per grid (data_scene.py:441-443), and the VoxelGrid branch calls
   ``load_batch([single_grid])`` (data_scene.py:299) which REPLACES the environment's whole
   voxel set — so only the last grid of a Scene survives ("last one wins"). This is why
   ``CollisionWorldManager.commit()`` must push voxel layers in one batched call instead of
   listing them in the Scene. If upstream cuRobo ever fixes this, the characterization test
   below starts failing and we can drop the workaround.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from curobo._src.geom.types import VoxelGrid  # noqa: E402
from curobo.scene import Scene  # noqa: E402

from motionforge.planner import MotionPlannerAdapter  # noqa: E402

from tests.gpu.conftest import TEST_CONFIG  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DIMS = (0.6, 0.6, 0.6)
VS = 0.02


def _grid(name, center, box_center=None, half=(0.06, 0.06, 0.06)):
    """A VoxelGrid centered at ``center``; optionally containing a box obstacle (true SDF)."""
    n = [round(d / VS) for d in DIMS]
    if box_center is None:
        feat = torch.full(tuple(n), 1.0, dtype=torch.float16, device="cuda:0")
    else:
        ix, iy, iz = (torch.arange(k, device="cuda:0", dtype=torch.float32) for k in n)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
        wx = center[0] + (gx - (n[0] - 1) / 2.0) * VS
        wy = center[1] + (gy - (n[1] - 1) / 2.0) * VS
        wz = center[2] + (gz - (n[2] - 1) / 2.0) * VS
        dx = (wx - box_center[0]).abs() - half[0]
        dy = (wy - box_center[1]).abs() - half[1]
        dz = (wz - box_center[2]).abs() - half[2]
        outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
        inside = torch.stack([dx, dy, dz], -1).max(-1).values.clamp(max=0)
        feat = (outside + inside).to(torch.float16)
    return VoxelGrid(
        name=name,
        pose=[*center, 1.0, 0.0, 0.0, 0.0],
        dims=list(DIMS),
        voxel_size=VS,
        feature_tensor=feat,
        feature_dtype=torch.float16,
    )


@pytest.fixture(scope="module")
def probe():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    adapter = MotionPlannerAdapter(
        config=TEST_CONFIG,
        collision_cache={"cuboid": 4, "voxel": {"layers": 2, "dims": list(DIMS), "voxel_size": VS}},
    )
    adapter.warmup()

    def goal_at(deltas):
        q = np.array(adapter.default_q0, dtype=float)
        q[: len(deltas)] += np.array(deltas, dtype=float)
        return adapter.tcp_pose_at(q.tolist())

    goal_a = goal_at([0.25, -0.25, 0.2])
    goal_b = goal_at([-0.25, -0.25, 0.2])
    return adapter, goal_a, goal_b


def _voxels(adapter):
    return adapter.planner.scene_collision_checker.data.voxels


def test_batched_load_composes_grids_with_different_poses(probe):
    adapter, goal_a, goal_b = probe
    pa, pb = tuple(goal_a.position.tolist()), tuple(goal_b.position.tolist())

    # Baseline: two EMPTY grids at the two ROI poses — both goals are reachable. This rules
    # out unreachability as the cause of any failure below.
    _voxels(adapter).load_batch([_grid("layer_a", center=pa), _grid("layer_b", center=pb)], 0)
    try:
        assert adapter.plan_free(goal_a, max_attempts=3).success
        assert adapter.plan_free(goal_b, max_attempts=3).success
    finally:
        _voxels(adapter).load_batch([], 0)

    # Now put an obstacle in EACH grid (at different world poses) and load both at once.
    grid_a = _grid("layer_a", center=pa, box_center=pa)
    grid_b = _grid("layer_b", center=pb, box_center=pb)
    _voxels(adapter).load_batch([grid_a, grid_b], 0)
    try:
        # Goal inside the FIRST grid's obstacle: must fail.
        assert not adapter.plan_free(goal_a, max_attempts=1).success
        # Goal inside the SECOND grid's obstacle: must ALSO fail — proves the second slot
        # participates in the collision query (composition), not last-one-wins.
        assert not adapter.plan_free(goal_b, max_attempts=1).success
    finally:
        _voxels(adapter).load_batch([], 0)


def test_scene_path_replaces_voxel_grids(probe):
    """Characterization of the cuRobo bug our commit() works around (see module docstring).

    A Scene listing [obstacle_grid, empty_grid] leaves only the LAST grid active, so the
    obstacle is silently dropped and a plan into it (wrongly) succeeds. If this test fails
    after a cuRobo upgrade, the Scene path started composing — the batched-push workaround
    in CollisionWorldManager.commit() can then be retired.
    """
    adapter, goal_a, _goal_b = probe
    pa = tuple(goal_a.position.tolist())
    grid_a = _grid("layer_a", center=pa, box_center=pa)
    grid_empty = _grid("layer_b", center=(pa[0], pa[1], pa[2] + 1.0))

    adapter.update_world(Scene(voxel=[grid_a, grid_empty]))
    try:
        result = adapter.plan_free(goal_a, max_attempts=1)
        assert result.success, (
            "Scene voxel path now composes (upstream fix?) — retire the batched-push "
            "workaround in CollisionWorldManager.commit()"
        )
    finally:
        _voxels(adapter).load_batch([], 0)
