"""Collision World Manager (SPEC §5.2).

Maintains the cuRobo collision world for a :class:`MotionPlannerAdapter`:
  - persistent static ``Cuboid`` / ``Mesh`` cell geometry,
  - per-cycle ESDF ``VoxelGrid`` layers (``pick_bin``, ``place_tray``) from the native warp
    ``Mapper`` (depth → TSDF → ESDF), composed in one collision query,
  - the attached held-object body on the tool frame.

torch/curobo are imported lazily so the module stays importable without a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from motionforge.planner.motion_planner import MotionPlannerAdapter

PICK_BIN_LAYER = "pick_bin"
PLACE_TRAY_LAYER = "place_tray"


class CollisionWorldManager:
    """Owns scene assembly + ESDF integration + attach/detach for one planner."""

    def __init__(self, adapter: MotionPlannerAdapter) -> None:
        self._adapter = adapter
        self._cuboids: list = []
        self._meshes: list = []
        self._voxel_layers: Dict[str, object] = {}

    # -- static cell geometry (persistent) --

    def set_static(self, cuboids: Optional[Sequence] = None, meshes: Optional[Sequence] = None) -> None:
        self._cuboids = list(cuboids or [])
        self._meshes = list(meshes or [])

    # -- per-cycle ESDF voxel layers --

    def set_voxel_layer(self, name: str, grid) -> None:
        """Add/replace a named ESDF ``VoxelGrid`` layer (re-perceived each cycle)."""
        grid.name = name  # ensure a unique, stable name within the scene
        self._voxel_layers[name] = grid

    def clear_voxel_layer(self, name: str) -> None:
        self._voxel_layers.pop(name, None)

    def layers(self) -> List[str]:
        return list(self._voxel_layers.keys())

    def build_scene(self):
        """Assemble a cuRobo ``SceneCfg`` from static geometry + voxel layers.

        cuRobo honors only ONE voxel grid per scene — a second ``VoxelGrid`` overwrites the
        first (the last one wins) rather than composing — so all ESDF layers are merged into a
        single grid here via :meth:`_merge_voxel_layers`. This restores the SPEC §5.2 intent of
        composing ``pick_bin`` + ``place_tray`` in one collision query; without it, an all-free
        layer committed after the bin would mask the bin's obstacles.
        """
        from curobo.scene import Scene  # Scene == SceneCfg

        kwargs = {}
        if self._cuboids:
            kwargs["cuboid"] = self._cuboids
        if self._meshes:
            kwargs["mesh"] = self._meshes
        if self._voxel_layers:
            kwargs["voxel"] = [self._merge_voxel_layers()]
        return Scene(**kwargs)

    def _merge_voxel_layers(self):
        """Union all ESDF layers into one ``VoxelGrid`` (element-wise MIN signed distance).

        The union of obstacles is the distance to the nearest surface across all layers, i.e.
        the per-voxel minimum of the signed distances (negative = inside an obstacle). Layers
        must share grid geometry (pose/dims/voxel_size); differing grids would need resampling
        onto a common grid, which is not supported yet.
        """
        import numpy as np
        import torch
        from curobo._src.geom.types import VoxelGrid

        grids = list(self._voxel_layers.values())
        if len(grids) == 1:
            return grids[0]

        base = grids[0]
        merged = base.feature_tensor.clone()
        for g in grids[1:]:
            if (
                g.feature_tensor.shape != base.feature_tensor.shape
                or not np.isclose(g.voxel_size, base.voxel_size)
                or not np.allclose(g.dims, base.dims)
                or not np.allclose(g.pose, base.pose)
            ):
                raise NotImplementedError(
                    "voxel ESDF layers must share pose/dims/voxel_size to merge; differing "
                    "grids require resampling onto a common grid (not implemented)"
                )
            merged = torch.minimum(merged, g.feature_tensor)

        return VoxelGrid(
            name="esdf_merged", pose=list(base.pose), dims=list(base.dims),
            voxel_size=base.voxel_size, feature_tensor=merged, feature_dtype=base.feature_dtype,
        )

    def commit(self) -> None:
        """Push the assembled scene into the planner's collision world."""
        self._adapter.update_world(self.build_scene())

    # -- native warp ESDF mapper --

    @staticmethod
    def make_mapper(
        extent_meters_xyz: Tuple[float, float, float],
        voxel_size: float = 0.01,
        esdf_voxel_size: Optional[float] = None,
        esdf_extent_meters_xyz: Optional[Tuple[float, float, float]] = None,
        grid_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        num_cameras: int = 1,
        device: str = "cuda:0",
        **kwargs,
    ):
        """Create a native warp ``Mapper`` for a per-cycle ESDF layer (SPEC §5.2)."""
        import torch
        from curobo._src.perception.mapper import Mapper, MapperCfg

        cfg = MapperCfg(
            extent_meters_xyz=tuple(extent_meters_xyz),
            voxel_size=voxel_size,
            esdf_voxel_size=esdf_voxel_size if esdf_voxel_size is not None else voxel_size,
            extent_esdf_meters_xyz=(
                tuple(esdf_extent_meters_xyz)
                if esdf_extent_meters_xyz is not None
                else tuple(extent_meters_xyz)
            ),
            grid_center=torch.tensor(grid_center, dtype=torch.float32),
            num_cameras=num_cameras,
            device=device,
            **kwargs,
        )
        return Mapper(cfg)

    def integrate_layer(self, name: str, observation, mapper, reset: bool = True):
        """Integrate one base-frame depth observation into ``name``'s ESDF layer.

        Per cycle we ``reset`` then integrate (re-perceive); for a sliding window pass
        ``reset=False``. Returns the produced ``VoxelGrid`` (also stored as the layer).
        """
        if reset:
            mapper.reset()
        mapper.integrate(observation)
        grid = mapper.compute_esdf()
        self.set_voxel_layer(name, grid)
        return grid

    # -- attached held object --

    def attach_object(
        self,
        obstacles,
        q_grasp: Optional[Sequence[float]] = None,
        world_objects_pose_offset=None,
        disable_obstacle_names: Optional[List[str]] = None,
        num_spheres: Optional[int] = None,
    ) -> None:
        """Attach the held workpiece to the tool on grasp confirmation (SPEC §5.2)."""
        self._adapter.attach_object(
            obstacles,
            q_grasp=q_grasp,
            world_objects_pose_offset=world_objects_pose_offset,
            disable_obstacle_names=disable_obstacle_names,
            num_spheres=num_spheres,
        )

    def detach_object(self) -> None:
        """Remove the held-object body on release."""
        self._adapter.detach_object()

    @property
    def attached(self) -> bool:
        return self._adapter.attached_link_name is not None
