"""ESDF grid spec — the single source of truth for per-cycle voxel layer geometry.

One ``ROIGridSpec`` per ESDF region of interest (``pick_bin``, ``place_tray``, ...). An
``ESDFGridSet`` collects them and derives everything that must agree on grid geometry:

  - the cuRobo collision **cache** sizing (``collision_cache``),
  - the warp ``Mapper`` construction kwargs per ROI (``mapper_kwargs``),
  - early validation that a produced ``VoxelGrid`` matches the cache (``validate_grid``).

**Why one shared dims/voxel_size across ROIs.** cuRobo's voxel collision cache allocates a
single feature buffer of ``n_voxels`` (from the cache's dims/voxel_size) per layer slot, and
``VoxelData.load_batch`` rejects any grid whose voxel count exceeds that buffer. Layers may
sit at DIFFERENT world poses (verified: ``tests/gpu/test_voxel_compose_probe.py`` —
``load_batch`` composes grids at distinct poses), but they share dims/voxel_size so the cache
is unambiguous and the per-cycle grids never overflow it.

Pure CPU/data — no torch import at module load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ROIGridSpec:
    """One ESDF region of interest. ``center`` is the base-frame grid center."""

    name: str
    center: Vec3
    dims: Vec3          # meters; SHARED across all ROIs in a set
    voxel_size: float   # meters; SHARED across all ROIs in a set

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", tuple(float(c) for c in self.center))
        object.__setattr__(self, "dims", tuple(float(d) for d in self.dims))
        object.__setattr__(self, "voxel_size", float(self.voxel_size))
        if len(self.center) != 3 or len(self.dims) != 3:
            raise ValueError(f"ROIGridSpec {self.name!r}: center and dims must be length 3")
        if self.voxel_size <= 0:
            raise ValueError(f"ROIGridSpec {self.name!r}: voxel_size must be > 0")


@dataclass(frozen=True)
class ESDFGridSet:
    """All ESDF ROI layers for a deployment; enforces shared dims/voxel_size."""

    rois: Tuple[ROIGridSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rois", tuple(self.rois))
        if not self.rois:
            raise ValueError("ESDFGridSet needs at least one ROI")
        names = [r.name for r in self.rois]
        if len(set(names)) != len(names):
            raise ValueError(f"ESDFGridSet ROI names must be unique, got {names}")
        base = self.rois[0]
        for r in self.rois[1:]:
            if r.dims != base.dims or r.voxel_size != base.voxel_size:
                raise ValueError(
                    "all ESDF ROIs must share dims/voxel_size (cuRobo voxel cache is sized "
                    f"once): {base.name}={base.dims}@{base.voxel_size} vs "
                    f"{r.name}={r.dims}@{r.voxel_size}"
                )

    # -- shared geometry --

    @property
    def dims(self) -> Vec3:
        return self.rois[0].dims

    @property
    def voxel_size(self) -> float:
        return self.rois[0].voxel_size

    def names(self) -> List[str]:
        return [r.name for r in self.rois]

    def get(self, name: str) -> ROIGridSpec:
        for r in self.rois:
            if r.name == name:
                return r
        raise KeyError(f"no ROI named {name!r}; have {self.names()}")

    # -- derivations --

    def collision_cache(self) -> Dict[str, object]:
        """Voxel portion of a cuRobo ``collision_cache`` (combine with cuboid/mesh counts)."""
        return {"layers": len(self.rois), "dims": list(self.dims), "voxel_size": self.voxel_size}

    def mapper_kwargs(self, name: str) -> Dict[str, object]:
        """Kwargs for :meth:`CollisionWorldManager.make_mapper` for one ROI."""
        roi = self.get(name)
        return {
            "extent_meters_xyz": roi.dims,
            "voxel_size": roi.voxel_size,
            "esdf_voxel_size": roi.voxel_size,
            "esdf_extent_meters_xyz": roi.dims,
            "grid_center": roi.center,
        }

    def validate_grid(self, name: str, grid) -> None:
        """Raise early if a produced ``VoxelGrid`` won't fit the cache for ``name``.

        Checks dims/voxel_size against the shared spec (pose is free to differ per ROI).
        """
        roi = self.get(name)
        g_dims = tuple(round(float(d), 6) for d in grid.dims)
        if g_dims != tuple(round(d, 6) for d in roi.dims):
            raise ValueError(
                f"voxel layer {name!r}: dims {g_dims} != ROI spec {roi.dims}; the cuRobo voxel "
                f"cache is sized for {roi.dims}@{roi.voxel_size}"
            )
        if round(float(grid.voxel_size), 9) != round(roi.voxel_size, 9):
            raise ValueError(
                f"voxel layer {name!r}: voxel_size {grid.voxel_size} != ROI spec {roi.voxel_size}"
            )

    # -- construction from CellConfig ROIs (Phase 2) --

    @staticmethod
    def from_rois(rois: Sequence) -> "ESDFGridSet":
        """Build from CellConfig ``ROIEntry`` objects (duck-typed: ``name``, ``origin.position``,
        ``extent_m``, ``voxel_size_m``)."""
        specs = [
            ROIGridSpec(
                name=r.name,
                center=tuple(r.origin.position),
                dims=tuple(r.extent_m),
                voxel_size=float(r.voxel_size_m),
            )
            for r in rois
        ]
        return ESDFGridSet(tuple(specs))
