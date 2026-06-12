"""Phase 8 GPU test: robot self-filter removes the robot's own body from depth (SPEC §5.2)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from motionforge.geometry import Pose  # noqa: E402
from motionforge.perception.camera import build_camera_observation  # noqa: E402
from motionforge.perception.self_filter import RobotSelfFilter  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DOWN = [0.0, 1.0, 0.0, 0.0]  # camera +Z -> world -Z (looking straight down)


def _largest_sphere(adapter, q):
    """World-frame (center, radius) of the robot's biggest active collision sphere at ``q``."""
    st = adapter.planner.compute_kinematics(adapter.to_joint_state(q))
    sph = st.robot_spheres.reshape(-1, 4).detach().cpu().numpy()
    sph = sph[sph[:, 3] > 0.0]
    return sph[np.argmax(sph[:, 3])]


def test_self_filter_masks_robot_keeps_background(mf_planner):
    q = mf_planner.default_q0
    cx, cy, cz, r = _largest_sphere(mf_planner, q)

    # Camera 1 m directly above that sphere, looking straight down.
    cam_pose = Pose([cx, cy, cz + 1.0], DOWN)
    H = W = 64
    intr = np.array([[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1.0]], dtype=np.float32)
    # Center block lands ON the sphere (depth 1.0); the border lands ~1 m BELOW it and offset
    # laterally (depth 2.0) — clear of the whole arm.
    depth = np.full((H, W), 2.0, dtype=np.float32)
    c0, c1 = H // 2 - 8, H // 2 + 8
    depth[c0:c1, c0:c1] = 1.0

    obs = build_camera_observation(depth, intr, cam_pose)
    sf = RobotSelfFilter(robot_yaml="ur10e.yml", distance_threshold=0.05, use_cuda_graph=False)
    mask, filtered = sf.get_mask(obs, q)

    mask_np = mask.detach().cpu().numpy().reshape(H, W)
    filt_np = filtered.detach().cpu().numpy().reshape(H, W)

    # The robot is detected (center pixels) but the background (border) is kept.
    assert mask_np.any(), "self-filter masked nothing — robot not detected"
    assert not mask_np.all(), "self-filter masked the whole image — background lost"
    # Center block is the robot; a far corner is background.
    assert mask_np[H // 2, W // 2]
    assert not mask_np[0, 0]
    # Filtered depth is zeroed exactly where masked and unchanged elsewhere.
    assert np.allclose(filt_np[mask_np], 0.0)
    assert np.allclose(filt_np[~mask_np], depth[~mask_np], atol=1e-2)


def test_self_filter_empty_when_robot_out_of_view(mf_planner):
    q = mf_planner.default_q0
    # A camera far to the side looking AWAY from the robot sees only a distant flat plane.
    cam_pose = Pose([3.0, 3.0, 1.0], DOWN)
    H = W = 48
    intr = np.array([[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1.0]], dtype=np.float32)
    depth = np.full((H, W), 1.0, dtype=np.float32)  # flat plane 1 m below, far from the arm

    obs = build_camera_observation(depth, intr, cam_pose)
    sf = RobotSelfFilter(robot_yaml="ur10e.yml", distance_threshold=0.05, use_cuda_graph=False)
    mask, _filtered = sf.get_mask(obs, q)
    assert not mask.detach().cpu().numpy().any(), "masked pixels with the robot out of view"
