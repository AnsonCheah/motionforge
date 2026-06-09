"""Depth → cuRobo CameraObservation, and robot self-filtering of depth (SPEC §5.1–5.2).

torch/curobo are imported lazily so the module is importable without a GPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from motionforge.geometry import Pose


def build_camera_observation(
    depth,
    intrinsics,
    camera_pose_base: Pose,
    rgb=None,
    device: str = "cuda:0",
):
    """Build a batched cuRobo ``CameraObservation`` (leading camera dim) in the base frame.

    Args:
        depth: ``(H, W)`` depth image in **metres** (np array or torch tensor).
        intrinsics: ``(3, 3)`` camera matrix.
        camera_pose_base: base←camera pose (from :class:`PerceptionFrameAdapter`).
        rgb: optional ``(H, W, 3)`` uint8 image.
        device: CUDA device.
    """
    import torch
    from curobo._src.types.camera import CameraObservation
    from curobo._src.types.pose import Pose as CuPose

    depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
    if depth_t.ndim == 2:
        depth_t = depth_t.unsqueeze(0)  # (1, H, W)
    intr_t = torch.as_tensor(np.asarray(intrinsics), device=device, dtype=torch.float32)
    if intr_t.ndim == 2:
        intr_t = intr_t.unsqueeze(0)
    pos = torch.as_tensor(camera_pose_base.position, device=device, dtype=torch.float32).view(1, 3)
    quat = torch.as_tensor(camera_pose_base.quaternion, device=device, dtype=torch.float32).view(1, 4)

    if rgb is None:
        h, w = int(depth_t.shape[-2]), int(depth_t.shape[-1])
        rgb_t = torch.zeros((1, h, w, 3), device=device, dtype=torch.uint8)
    else:
        rgb_t = torch.as_tensor(rgb, device=device, dtype=torch.uint8)
        if rgb_t.ndim == 3:
            rgb_t = rgb_t.unsqueeze(0)

    return CameraObservation(
        depth_image=depth_t,
        rgb_image=rgb_t,
        intrinsics=intr_t,
        pose=CuPose(position=pos, quaternion=quat),
        depth_to_meter=1.0,  # depth is already in metres
    )


def mask_robot_depth(depth, robot_mask, invalid: float = 0.0):
    """Self-filter: zero the robot's region in the depth image before integration (SPEC §5.2).

    ``robot_mask`` is a boolean array (True where the robot occupies the image). Works on
    numpy arrays or torch tensors.
    """
    try:
        out = depth.clone()  # torch
    except AttributeError:
        out = np.array(depth, copy=True)  # numpy
    out[robot_mask] = invalid
    return out
