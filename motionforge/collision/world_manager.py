"""Collision World Manager (SPEC §5.2).

Maintains the cuRobo collision world for a :class:`MotionPlannerAdapter`:
  - persistent static ``Cuboid`` / ``Mesh`` cell geometry,
  - per-cycle ESDF ``VoxelGrid`` layers (``pick_bin``, ``place_tray``) from the native warp
    ``Mapper`` (depth → TSDF → ESDF), composed in one collision query,
  - the attached held-object body on the tool frame.

**Voxel composition (verified).** cuRobo's ``Scene`` voxel path REPLACES rather than composes:
``update_world(Scene(voxel=[a, b]))`` keeps only the last grid (``load_from_scene_cfg`` →
``add_obstacle`` → ``load_batch([single])``, which clears the env's voxel set each call). The
batched ``VoxelData.load_batch([a, b], env)`` DOES compose, loading every grid at its own pose
and enabling all of them (proven in ``tests/gpu/test_voxel_compose_probe.py``). So static
bodies go through ``update_world`` while ESDF layers are pushed together via the adapter's
``load_voxel_layers`` — ROI layers may sit at different cell poses but share dims/voxel_size
(the cache spec, see :class:`~motionforge.collision.grid_spec.ESDFGridSet`).

torch/curobo are imported lazily so the module stays importable without a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from motionforge.collision.grid_spec import ESDFGridSet
from motionforge.planner.motion_planner import MotionPlannerAdapter

PICK_BIN_LAYER = "pick_bin"
PLACE_TRAY_LAYER = "place_tray"


class CollisionWorldManager:
    """Owns scene assembly + ESDF integration + attach/detach for one planner."""

    def __init__(
        self, adapter: MotionPlannerAdapter, grid_set: Optional[ESDFGridSet] = None
    ) -> None:
        self._adapter = adapter
        self._grid_set = grid_set
        self._cuboids: list = []
        self._meshes: list = []
        self._voxel_layers: Dict[str, object] = {}

    @property
    def grid_set(self) -> Optional[ESDFGridSet]:
        return self._grid_set

    # -- static cell geometry (persistent) --

    def set_static(self, cuboids: Optional[Sequence] = None, meshes: Optional[Sequence] = None) -> None:
        self._cuboids = list(cuboids or [])
        self._meshes = list(meshes or [])

    # -- per-cycle ESDF voxel layers --

    def set_voxel_layer(self, name: str, grid) -> None:
        """Add/replace a named ESDF ``VoxelGrid`` layer (re-perceived each cycle)."""
        if self._grid_set is not None:
            self._grid_set.validate_grid(name, grid)  # fail loudly now, not mid-cycle in cuRobo
        grid.name = name  # ensure a unique, stable name within the scene
        self._voxel_layers[name] = grid

    def clear_voxel_layer(self, name: str) -> None:
        self._voxel_layers.pop(name, None)

    def layers(self) -> List[str]:
        return list(self._voxel_layers.keys())

    def voxel_grids(self) -> List[object]:
        """The current ESDF layers, in insertion order (one cuRobo ``load_batch``)."""
        return list(self._voxel_layers.values())

    def build_static_scene(self):
        """Assemble a cuRobo ``SceneCfg`` from static cuboids/meshes ONLY.

        Voxel layers are intentionally excluded: they are pushed via the adapter's batched
        ``load_voxel_layers`` because the ``Scene`` voxel path replaces instead of composing
        (see the module docstring).
        """
        from curobo.scene import Scene  # Scene == SceneCfg

        kwargs = {}
        if self._cuboids:
            kwargs["cuboid"] = self._cuboids
        if self._meshes:
            kwargs["mesh"] = self._meshes
        return Scene(**kwargs)

    def commit(self) -> None:
        """Push static geometry then all ESDF layers into the planner's collision world.

        Order matters: ``update_world`` clears the whole env (including voxels), so static
        bodies are loaded first and the voxel layers are batch-loaded after.
        """
        self._adapter.update_world(self.build_static_scene())
        self._adapter.load_voxel_layers(self.voxel_grids())

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

    # -- gripper collision geometry (width-dependent) --

    def attach_tool_geometry(self, body, num_spheres: Optional[int] = None) -> None:
        """Attach the gripper's own collision geometry (a width-dependent ``CollisionBody``).

        Converts the TCP-frame primitive into cuRobo ``Cuboid``s and attaches them to the
        planner's ``tool_collision`` link. Re-call whenever the commanded width changes so the
        planner always sees the ACTUAL gripper envelope per segment (SPEC §5.4).
        """
        from curobo.scene import Cuboid

        from motionforge.tools.tool_manager import collision_body_to_cuboid_specs

        specs = collision_body_to_cuboid_specs(body)
        cuboids = [Cuboid(name=s["name"], pose=s["pose"], dims=s["dims"]) for s in specs]
        self._adapter.attach_tool(cuboids, num_spheres=num_spheres)

    def detach_tool_geometry(self) -> None:
        self._adapter.detach_tool()

    @property
    def tool_attached(self) -> bool:
        return self._adapter.tool_attached
