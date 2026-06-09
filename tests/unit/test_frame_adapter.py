"""Unit tests for the Perception Frame Adapter (SPEC §5.1) — pure geometry, no GPU."""

import numpy as np
import pytest

from motionforge.geometry import Pose
from motionforge.perception import CameraInfo, PerceptionFrameAdapter

QZ90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
INTRINSICS = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


def test_camera_info_validates_mount():
    with pytest.raises(ValueError):
        CameraInfo("c", mount="bogus", extrinsic=Pose.identity())


def test_camera_info_reshapes_intrinsics():
    cam = CameraInfo("c", "fixed", Pose.identity(), intrinsics=INTRINSICS.reshape(-1))
    assert cam.intrinsics.shape == (3, 3)


def test_fixed_camera_pose_is_extrinsic():
    ext = Pose([0.8, 0.0, 1.2], QZ90)
    cam = CameraInfo("pick", "fixed", ext, role="pick")
    adapter = PerceptionFrameAdapter()
    assert adapter.camera_pose_base(cam).approx_equal(ext)


def test_eih_camera_composes_fk_and_extrinsic():
    # base<-tool0 from FK (mock); tool0<-camera extrinsic 0.1 m along tool +Z.
    base_tool = Pose([0.4, 0.0, 0.5], [1, 0, 0, 0])
    extrinsic = Pose([0.0, 0.0, 0.1], [1, 0, 0, 0])
    adapter = PerceptionFrameAdapter(fk_fn=lambda q: base_tool)
    cam = CameraInfo("eih", "eih", extrinsic, role="pick")
    out = adapter.camera_pose_base(cam, capture_q=[0, 0, 0, 0, 0, 0])
    assert np.allclose(out.position, [0.4, 0.0, 0.6], atol=1e-9)


def test_eih_camera_applies_fk_rotation_to_extrinsic():
    # base<-tool0 rotated +90 about Z at [0.4,0,0.5]; extrinsic offset 0.1 m along tool +X.
    base_tool = Pose([0.4, 0.0, 0.5], QZ90)
    extrinsic = Pose([0.1, 0.0, 0.0], [1, 0, 0, 0])
    adapter = PerceptionFrameAdapter(fk_fn=lambda q: base_tool)
    cam = CameraInfo("eih", "eih", extrinsic)
    out = adapter.camera_pose_base(cam, capture_q=[0.1] * 6)
    # tool +X rotated into base +Y -> camera lands at y = 0.5.
    assert np.allclose(out.position, [0.4, 0.1, 0.5], atol=1e-9)


def test_eih_requires_fk_and_capture_q():
    cam = CameraInfo("eih", "eih", Pose.identity())
    no_fk = PerceptionFrameAdapter()
    with pytest.raises(ValueError):
        no_fk.camera_pose_base(cam, capture_q=[0] * 6)
    with_fk = PerceptionFrameAdapter(fk_fn=lambda q: Pose.identity())
    with pytest.raises(ValueError):
        with_fk.camera_pose_base(cam, capture_q=None)


def test_registry_and_by_role():
    adapter = PerceptionFrameAdapter()
    adapter.register(CameraInfo("p", "fixed", Pose.identity(), role="pick"))
    adapter.register(CameraInfo("q", "fixed", Pose.identity(), role="place"))
    assert adapter.get("p").role == "pick"
    assert [c.camera_id for c in adapter.by_role("place")] == ["q"]
