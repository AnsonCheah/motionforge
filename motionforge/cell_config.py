"""Cell config (SPEC §4, §5.8) — per-deployment scene-as-data, not code.

A declarative YAML manifest + referenced geometry assets describing one robot cell: the
static collision world, camera registry, ESDF ROIs, and tool library. Loaded once at
startup; the planner stack derives from it instead of assembling the scene in code, so a new
layout is an authoring task, not a coding task.

Pure data/CPU — ``yaml`` and ``curobo`` are imported lazily inside the functions that need
them, so the dataclasses and the loader stay importable without a GPU and the curobo
conversion (:func:`static_bodies_to_curobo`) is the only GPU-adjacent boundary.

``collision_geom_fn`` (a callable) is not YAML-serializable, so tools reference a named
builder in :data:`GEOM_REGISTRY` with parameters; the loader resolves it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from motionforge.geometry import Pose
from motionforge.perception.frame_adapter import CameraInfo
from motionforge.tools import ToolManager, parallel_jaw_geom_fn, vacuum_geom_fn
from motionforge.types import CollisionBody, ToolDescriptor

Vec3 = Tuple[float, float, float]

#: Named gripper-geometry builders referenced by the manifest (``geometry.type``). Each maps
#: ``params`` (kwargs) -> a ``collision_geom_fn`` (``GripConfig -> CollisionBody``).
GEOM_REGISTRY: Dict[str, Callable[..., Callable]] = {
    "parallel_jaw": parallel_jaw_geom_fn,
    "vacuum": vacuum_geom_fn,
}


# -- dataclasses (SPEC §4) --


@dataclass
class CameraEntry:
    camera_id: str
    mount: str                 # "fixed" | "eih"
    extrinsic: Pose            # fixed: cam->base ; eih: cam->mount_link
    role: str = "aux"          # "pick" | "place" | "aux"
    intrinsics: Optional[List[List[float]]] = None  # 3x3 camera matrix


@dataclass
class ROIEntry:
    name: str                  # "pick_bin" | "place_tray"
    origin: Pose               # base-frame ESDF grid origin (center)
    extent_m: Vec3             # voxel-grid bounds
    voxel_size_m: float


@dataclass
class StaticBody:
    name: str
    body: CollisionBody        # kind in {"primitive","mesh"}
    transform: Pose            # base-frame placement
    asset_path: Optional[str] = None  # for kind=="mesh": path to mesh/USD asset


@dataclass
class CellConfig:
    robot_base_frame: str
    robot_mount: Pose
    cameras: List[CameraEntry] = field(default_factory=list)
    static_bodies: List[StaticBody] = field(default_factory=list)
    rois: List[ROIEntry] = field(default_factory=list)
    tools: List[ToolDescriptor] = field(default_factory=list)


# -- YAML parsing helpers --


def _pose(d: Optional[dict]) -> Pose:
    """Parse a ``{position: [x,y,z], quaternion: [w,x,y,z]}`` block (defaults to identity)."""
    if d is None:
        return Pose.identity()
    pos = d.get("position", [0.0, 0.0, 0.0])
    quat = d.get("quaternion", [1.0, 0.0, 0.0, 0.0])
    return Pose(pos, quat)


def _resolve_geom_fn(geometry: dict) -> Callable:
    gtype = geometry.get("type")
    if gtype not in GEOM_REGISTRY:
        raise ValueError(
            f"unknown tool geometry type {gtype!r}; known: {sorted(GEOM_REGISTRY)}"
        )
    return GEOM_REGISTRY[gtype](**geometry.get("params", {}))


def _static_body(d: dict) -> StaticBody:
    kind = d.get("kind", "primitive")
    transform = _pose(d.get("transform"))
    asset_path = d.get("asset_path")
    if kind == "mesh":
        data = {"asset_path": asset_path}
    elif kind == "primitive":
        # Inline primitive params (e.g. a cuboid's dims).
        data = {k: d[k] for k in ("dims",) if k in d}
    else:
        raise ValueError(f"static body {d.get('name')!r}: kind must be 'primitive' or 'mesh', got {kind!r}")
    return StaticBody(
        name=d["name"],
        body=CollisionBody(kind=kind, data=data, frame="base"),
        transform=transform,
        asset_path=asset_path,
    )


def _tool(d: dict) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id=d["tool_id"],
        tcp_pose=_pose(d.get("tcp_pose")),
        collision_geom_fn=_resolve_geom_fn(d["geometry"]),
        actuation_iface=d.get("actuation_iface", ""),
        payload_kg=float(d.get("payload_kg", 0.0)),
    )


def load_cell_config(path: str) -> CellConfig:
    """Load a cell-config YAML manifest. Asset paths are resolved relative to the manifest."""
    import yaml

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    base_dir = os.path.dirname(os.path.abspath(path))

    static_bodies = []
    for sd in data.get("static_bodies", []):
        body = _static_body(sd)
        if body.asset_path and not os.path.isabs(body.asset_path):
            body.asset_path = os.path.normpath(os.path.join(base_dir, body.asset_path))
            body.body.data["asset_path"] = body.asset_path
        static_bodies.append(body)

    cameras = [
        CameraEntry(
            camera_id=c["camera_id"],
            mount=c["mount"],
            extrinsic=_pose(c.get("extrinsic")),
            role=c.get("role", "aux"),
            intrinsics=c.get("intrinsics"),
        )
        for c in data.get("cameras", [])
    ]
    rois = [
        ROIEntry(
            name=r["name"],
            origin=_pose(r.get("origin")),
            extent_m=tuple(r["extent_m"]),
            voxel_size_m=float(r["voxel_size_m"]),
        )
        for r in data.get("rois", [])
    ]
    tools = [_tool(t) for t in data.get("tools", [])]

    return CellConfig(
        robot_base_frame=data.get("robot_base_frame", "base_link"),
        robot_mount=_pose(data.get("robot_mount")),
        cameras=cameras,
        static_bodies=static_bodies,
        rois=rois,
        tools=tools,
    )


# -- derivations into the planner stack --


def static_bodies_to_curobo(bodies: List[StaticBody]):
    """Convert static bodies to cuRobo ``Cuboid`` / ``Mesh`` lists (lazy curobo import).

    Returns ``(cuboids, meshes)``. Primitive bodies become ``Cuboid``s placed by their
    base-frame transform; mesh bodies become ``Mesh``es referencing ``asset_path``.
    """
    from curobo.scene import Cuboid, Mesh

    cuboids, meshes = [], []
    for b in bodies:
        pose = b.transform.to_curobo_list()  # [x,y,z,qw,qx,qy,qz]
        if b.body.kind == "primitive":
            dims = b.body.data.get("dims")
            if dims is None:
                raise ValueError(f"static body {b.name!r}: primitive needs 'dims'")
            cuboids.append(Cuboid(name=b.name, pose=pose, dims=list(dims)))
        elif b.body.kind == "mesh":
            if not b.asset_path:
                raise ValueError(f"static body {b.name!r}: mesh needs 'asset_path'")
            meshes.append(Mesh(name=b.name, pose=pose, file_path=b.asset_path))
        else:
            raise ValueError(f"static body {b.name!r}: unsupported kind {b.body.kind!r}")
    return cuboids, meshes


def cameras_to_registry(cameras: List[CameraEntry]) -> List[CameraInfo]:
    """Map cell-config cameras onto :class:`CameraInfo` rows for the frame adapter."""
    return [
        CameraInfo(
            camera_id=c.camera_id,
            mount=c.mount,
            extrinsic=c.extrinsic,
            role=c.role,
            intrinsics=c.intrinsics,
        )
        for c in cameras
    ]


def tools_to_manager(tools: List[ToolDescriptor]) -> ToolManager:
    """Build a :class:`ToolManager` from the cell-config tool library."""
    tm = ToolManager()
    for t in tools:
        tm.register(t)
    return tm


def grid_set_from_rois(rois: List[ROIEntry]):
    """Build an :class:`ESDFGridSet` from the cell-config ROIs (single source of truth)."""
    from motionforge.collision.grid_spec import ESDFGridSet

    return ESDFGridSet.from_rois(rois)
