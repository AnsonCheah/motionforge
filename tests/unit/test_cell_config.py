"""Unit tests for the CellConfig YAML loader and its derivations (CPU, no GPU)."""

import textwrap

import numpy as np
import pytest

from motionforge.cell_config import (
    CellConfig,
    cameras_to_registry,
    grid_set_from_rois,
    load_cell_config,
    tools_to_manager,
)
from motionforge.types import CollisionBody, GripConfig


MANIFEST = textwrap.dedent(
    """
    robot_base_frame: base_link
    robot_mount:
      position: [0.0, 0.0, 0.1]
      quaternion: [1.0, 0.0, 0.0, 0.0]
    cameras:
      - camera_id: cam_pick
        mount: fixed
        role: pick
        extrinsic:
          position: [0.5, -0.3, 1.0]
          quaternion: [0.0, 1.0, 0.0, 0.0]
        intrinsics: [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
      - camera_id: cam_eih
        mount: eih
        role: place
        extrinsic:
          position: [0.0, 0.05, 0.0]
          quaternion: [1.0, 0.0, 0.0, 0.0]
    static_bodies:
      - name: table
        kind: primitive
        dims: [1.2, 1.0, 0.05]
        transform:
          position: [0.5, 0.0, -0.025]
          quaternion: [1.0, 0.0, 0.0, 0.0]
      - name: fixture
        kind: mesh
        asset_path: assets/fixture.obj
        transform:
          position: [0.5, 0.0, 0.05]
          quaternion: [1.0, 0.0, 0.0, 0.0]
    rois:
      - name: pick_bin
        origin:
          position: [0.5, -0.3, 0.3]
          quaternion: [1.0, 0.0, 0.0, 0.0]
        extent_m: [0.6, 0.6, 0.6]
        voxel_size_m: 0.01
      - name: place_tray
        origin:
          position: [0.5, 0.3, 0.3]
          quaternion: [1.0, 0.0, 0.0, 0.0]
        extent_m: [0.6, 0.6, 0.6]
        voxel_size_m: 0.01
    tools:
      - tool_id: jaw
        tcp_pose:
          position: [0.0, 0.0, 0.15]
          quaternion: [1.0, 0.0, 0.0, 0.0]
        geometry:
          type: parallel_jaw
          params:
            jaw_length: 0.04
            jaw_thickness: 0.02
        actuation_iface: "socket://gripper"
        payload_kg: 0.5
    """
)


@pytest.fixture
def manifest_path(tmp_path):
    p = tmp_path / "cell.yml"
    p.write_text(MANIFEST)
    return str(p)


def test_load_basic_fields(manifest_path):
    cell = load_cell_config(manifest_path)
    assert isinstance(cell, CellConfig)
    assert cell.robot_base_frame == "base_link"
    assert np.allclose(cell.robot_mount.position, [0.0, 0.0, 0.1])
    assert len(cell.cameras) == 2
    assert len(cell.static_bodies) == 2
    assert len(cell.rois) == 2
    assert len(cell.tools) == 1


def test_camera_parsing(manifest_path):
    cell = load_cell_config(manifest_path)
    pick = next(c for c in cell.cameras if c.camera_id == "cam_pick")
    assert pick.mount == "fixed"
    assert pick.role == "pick"
    assert np.allclose(pick.extrinsic.position, [0.5, -0.3, 1.0])
    assert pick.intrinsics is not None and len(pick.intrinsics) == 3


def test_static_body_kinds(manifest_path):
    cell = load_cell_config(manifest_path)
    table = next(b for b in cell.static_bodies if b.name == "table")
    fixture = next(b for b in cell.static_bodies if b.name == "fixture")
    assert table.body.kind == "primitive"
    assert table.body.data["dims"] == [1.2, 1.0, 0.05]
    assert fixture.body.kind == "mesh"
    # Asset path resolves relative to the manifest directory.
    assert fixture.asset_path.endswith("assets/fixture.obj")
    assert "/" in fixture.asset_path  # absolute/normalized, not the bare relative ref


def test_tool_geometry_resolves_to_callable(manifest_path):
    cell = load_cell_config(manifest_path)
    tool = cell.tools[0]
    assert tool.tool_id == "jaw"
    assert np.allclose(tool.tcp_pose.position, [0.0, 0.0, 0.15])
    body = tool.collision_geom_fn(GripConfig(width_m=0.06))
    assert isinstance(body, CollisionBody)
    assert body.frame == "tcp"
    # Jaw separation tracks the commanded width.
    assert abs(body.data["width"] - 0.06) < 1e-9


def test_unknown_geometry_type_raises(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text(
        textwrap.dedent(
            """
            robot_base_frame: base_link
            tools:
              - tool_id: x
                geometry:
                  type: laser_tweezers
            """
        )
    )
    with pytest.raises(ValueError, match="unknown tool geometry type"):
        load_cell_config(str(bad))


def test_bad_static_body_kind_raises(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text(
        textwrap.dedent(
            """
            robot_base_frame: base_link
            static_bodies:
              - name: weird
                kind: hologram
            """
        )
    )
    with pytest.raises(ValueError, match="kind must be"):
        load_cell_config(str(bad))


def test_grid_set_from_rois(manifest_path):
    cell = load_cell_config(manifest_path)
    gs = grid_set_from_rois(cell.rois)
    assert gs.names() == ["pick_bin", "place_tray"]
    assert gs.voxel_size == 0.01
    assert gs.get("place_tray").center == (0.5, 0.3, 0.3)


def test_grid_set_mismatched_voxel_size_raises(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text(
        textwrap.dedent(
            """
            robot_base_frame: base_link
            rois:
              - name: pick_bin
                origin: {position: [0.5, -0.3, 0.3]}
                extent_m: [0.6, 0.6, 0.6]
                voxel_size_m: 0.01
              - name: place_tray
                origin: {position: [0.5, 0.3, 0.3]}
                extent_m: [0.6, 0.6, 0.6]
                voxel_size_m: 0.02
            """
        )
    )
    cell = load_cell_config(str(bad))
    with pytest.raises(ValueError, match="share dims/voxel_size"):
        grid_set_from_rois(cell.rois)


def test_cameras_to_registry(manifest_path):
    cell = load_cell_config(manifest_path)
    infos = cameras_to_registry(cell.cameras)
    assert [c.camera_id for c in infos] == ["cam_pick", "cam_eih"]
    # CameraInfo.__post_init__ reshapes intrinsics to 3x3.
    pick = infos[0]
    assert pick.intrinsics.shape == (3, 3)


def test_tools_to_manager(manifest_path):
    cell = load_cell_config(manifest_path)
    tm = tools_to_manager(cell.tools)
    assert tm.active_id == "jaw"
    assert np.allclose(tm.active_tcp().position, [0.0, 0.0, 0.15])


def test_example_manifest_loads():
    # The shipped example must stay loadable (it documents the format).
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cell = load_cell_config(os.path.join(repo_root, "configs", "cell_example.yml"))
    assert cell.rois and cell.tools and cell.static_bodies
    grid_set_from_rois(cell.rois)  # ROIs are consistent
