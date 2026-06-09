"""Perception Frame Adapter (SPEC §5.1) — normalize every camera to the robot base frame.

Pure geometry (numpy, CPU). Fixed camera: base←camera is the static extrinsic. Eye-in-hand
(EIH): base←camera = FK(capture config) ∘ (mount_link←camera extrinsic). The FK function is
injected (e.g. ``MotionPlannerAdapter.tcp_pose_at``) so this module stays GPU-free and
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from motionforge.geometry import Pose


@dataclass
class CameraInfo:
    """One camera in the registry (SPEC §5.1).

    ``extrinsic`` meaning depends on ``mount``:
      - ``"fixed"``: base←camera (static, in the robot base frame).
      - ``"eih"``: mount_link←camera (camera pose relative to the FK'd mounting link,
        e.g. ``tool0``); combined with FK at capture time.
    """

    camera_id: str
    mount: str  # "fixed" | "eih"
    extrinsic: Pose
    role: str = "aux"  # "pick" | "place" | "aux"
    intrinsics: Optional[np.ndarray] = None  # 3x3 camera matrix

    def __post_init__(self) -> None:
        if self.mount not in ("fixed", "eih"):
            raise ValueError(f"mount must be 'fixed' or 'eih', got {self.mount!r}")
        if self.intrinsics is not None:
            self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64).reshape(3, 3)


class PerceptionFrameAdapter:
    """Holds the camera registry and resolves base-frame camera poses.

    Args:
        fk_fn: ``q -> Pose`` giving base←(mounting link) at a capture config. Required only
            for EIH cameras.
    """

    def __init__(self, fk_fn: Optional[Callable[[Sequence[float]], Pose]] = None) -> None:
        self._fk = fk_fn
        self._cameras: Dict[str, CameraInfo] = {}

    def register(self, camera: CameraInfo) -> None:
        self._cameras[camera.camera_id] = camera

    def get(self, camera_id: str) -> CameraInfo:
        return self._cameras[camera_id]

    def by_role(self, role: str) -> list[CameraInfo]:
        return [c for c in self._cameras.values() if c.role == role]

    def camera_pose_base(
        self, camera: CameraInfo, capture_q: Optional[Sequence[float]] = None
    ) -> Pose:
        """Return base←camera as a :class:`Pose`."""
        if camera.mount == "fixed":
            return camera.extrinsic
        # EIH: base<-camera = (base<-mount_link via FK) ∘ (mount_link<-camera extrinsic)
        if self._fk is None:
            raise ValueError(f"EIH camera {camera.camera_id!r} needs an FK function")
        if capture_q is None:
            raise ValueError(f"EIH camera {camera.camera_id!r} needs the capture joint config")
        base_mount = self._fk(capture_q)
        return base_mount.multiply(camera.extrinsic)

    def camera_pose_base_by_id(
        self, camera_id: str, capture_q: Optional[Sequence[float]] = None
    ) -> Pose:
        return self.camera_pose_base(self._cameras[camera_id], capture_q)
