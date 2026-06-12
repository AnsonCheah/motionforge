"""Unit tests for the ESDF grid spec (single source of truth for voxel layer geometry)."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from motionforge.collision.grid_spec import ESDFGridSet, ROIGridSpec


def _set():
    return ESDFGridSet(
        (
            ROIGridSpec("pick_bin", center=(0.5, -0.3, 0.3), dims=(0.6, 0.6, 0.6), voxel_size=0.01),
            ROIGridSpec("place_tray", center=(0.5, 0.3, 0.3), dims=(0.6, 0.6, 0.6), voxel_size=0.01),
        )
    )


def test_shared_dims_enforced():
    with pytest.raises(ValueError, match="share dims/voxel_size"):
        ESDFGridSet(
            (
                ROIGridSpec("a", (0, 0, 0), (0.6, 0.6, 0.6), 0.01),
                ROIGridSpec("b", (1, 0, 0), (0.5, 0.5, 0.5), 0.01),  # different dims
            )
        )


def test_shared_voxel_size_enforced():
    with pytest.raises(ValueError, match="share dims/voxel_size"):
        ESDFGridSet(
            (
                ROIGridSpec("a", (0, 0, 0), (0.6, 0.6, 0.6), 0.01),
                ROIGridSpec("b", (1, 0, 0), (0.6, 0.6, 0.6), 0.02),  # different voxel size
            )
        )


def test_unique_names_enforced():
    with pytest.raises(ValueError, match="unique"):
        ESDFGridSet(
            (
                ROIGridSpec("dup", (0, 0, 0), (0.6, 0.6, 0.6), 0.01),
                ROIGridSpec("dup", (1, 0, 0), (0.6, 0.6, 0.6), 0.01),
            )
        )


def test_empty_set_rejected():
    with pytest.raises(ValueError, match="at least one ROI"):
        ESDFGridSet(())


def test_bad_voxel_size_rejected():
    with pytest.raises(ValueError, match="voxel_size must be > 0"):
        ROIGridSpec("a", (0, 0, 0), (0.6, 0.6, 0.6), 0.0)


def test_collision_cache_derivation():
    cache = _set().collision_cache()
    assert cache == {"layers": 2, "dims": [0.6, 0.6, 0.6], "voxel_size": 0.01}


def test_mapper_kwargs_roundtrip():
    kw = _set().mapper_kwargs("place_tray")
    assert kw["extent_meters_xyz"] == (0.6, 0.6, 0.6)
    assert kw["esdf_extent_meters_xyz"] == (0.6, 0.6, 0.6)
    assert kw["voxel_size"] == 0.01
    assert kw["esdf_voxel_size"] == 0.01
    assert kw["grid_center"] == (0.5, 0.3, 0.3)


def test_mapper_kwargs_unknown_roi_raises():
    with pytest.raises(KeyError, match="no ROI named"):
        _set().mapper_kwargs("nope")


@dataclass
class _StubGrid:
    dims: Tuple[float, float, float]
    voxel_size: float


def test_validate_grid_accepts_matching():
    # Different pose is fine; only dims/voxel_size are checked.
    _set().validate_grid("pick_bin", _StubGrid(dims=(0.6, 0.6, 0.6), voxel_size=0.01))


def test_validate_grid_rejects_wrong_voxel_size():
    with pytest.raises(ValueError, match="voxel_size"):
        _set().validate_grid("pick_bin", _StubGrid(dims=(0.6, 0.6, 0.6), voxel_size=0.02))


def test_validate_grid_rejects_wrong_dims():
    with pytest.raises(ValueError, match="dims"):
        _set().validate_grid("pick_bin", _StubGrid(dims=(0.5, 0.6, 0.6), voxel_size=0.01))


def test_shared_geometry_accessors():
    s = _set()
    assert s.dims == (0.6, 0.6, 0.6)
    assert s.voxel_size == 0.01
    assert s.names() == ["pick_bin", "place_tray"]
    assert s.get("place_tray").center == (0.5, 0.3, 0.3)


def test_from_rois_duck_typed():
    @dataclass
    class _Origin:
        position: np.ndarray

    @dataclass
    class _ROIEntry:
        name: str
        origin: _Origin
        extent_m: Tuple[float, float, float]
        voxel_size_m: float

    rois = [
        _ROIEntry("pick_bin", _Origin(np.array([0.5, -0.3, 0.3])), (0.6, 0.6, 0.6), 0.01),
        _ROIEntry("place_tray", _Origin(np.array([0.5, 0.3, 0.3])), (0.6, 0.6, 0.6), 0.01),
    ]
    gs = ESDFGridSet.from_rois(rois)
    assert gs.names() == ["pick_bin", "place_tray"]
    assert gs.get("pick_bin").center == (0.5, -0.3, 0.3)
    assert gs.voxel_size == 0.01
