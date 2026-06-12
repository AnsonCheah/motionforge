"""Robot self-filter (SPEC §5.1–§5.2) — zero the robot's own body out of depth before ESDF.

Wraps curobov2's native ``RobotSegmenter``: given a base-frame depth observation and the
capture-time joint config, it masks the pixels whose back-projected points fall within
``distance_threshold`` of the robot's collision spheres and returns the depth with those pixels
zeroed. Integrating the filtered depth keeps the arm from writing itself into the bin/tray
ESDF (which would phantom-block the planner — see the place sequence, arm in the tray camera's
view).

torch/curobo are imported lazily so the module stays importable without a GPU.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


class RobotSelfFilter:
    """Self-filter backed by ``curobo`` ``RobotSegmenter`` for one robot.

    Args:
        robot_yaml: bundled cuRobo robot config (the PLAIN yaml — independent of any injected
            tcp/attached_object links, which don't affect the arm's own silhouette).
        distance_threshold: a depth point within this distance (m) of a robot collision sphere
            is treated as robot and removed.
        use_cuda_graph: capture the masking op (off by default to match the test config).
        device: CUDA device string.
    """

    def __init__(
        self,
        robot_yaml: str = "ur10e.yml",
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = False,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from curobo._src.perception.robot_segmenter import RobotSegmenter
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.types.robot import RobotCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import join_path
        from curobo._src.util_file import load_yaml
        from curobo.content import get_robot_configs_path

        self._torch = torch
        self._device = torch.device(device)
        device_cfg = DeviceCfg(device=self._device, dtype=torch.float32)

        # Build the segmenter directly (not via from_robot_file) so ops run in float32 — its
        # default bfloat16 ops conflict with the float32 kinematics ("expected float32, got
        # bfloat16"). Plain robot yaml: the arm silhouette is independent of injected links.
        robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_yaml))
        robot_cfg = RobotCfg.create(robot_dict, device_cfg=device_cfg)
        self._seg = RobotSegmenter(
            Kinematics(robot_cfg.kinematics),
            distance_threshold=distance_threshold,
            use_cuda_graph=use_cuda_graph,
            ops_dtype=torch.float32,
        )

    @property
    def joint_names(self):
        return list(self._seg._kinematics.joint_names)

    def _joint_state(self, q: Sequence[float]):
        from curobo.types import JointState

        torch = self._torch
        pos = torch.as_tensor(q, device=self._device, dtype=torch.float32).view(1, -1)
        return JointState.from_position(pos, joint_names=self.joint_names)

    def get_mask(self, observation, q: Sequence[float]) -> Tuple[object, object]:
        """Return ``(mask, filtered_depth)`` for a base-frame ``CameraObservation`` at config ``q``.

        ``mask`` is True where the robot occupies the image; ``filtered_depth`` is the depth
        with those pixels zeroed.
        """
        return self._seg.get_robot_mask(observation, self._joint_state(q))

    def filter_observation(self, observation, q: Sequence[float]):
        """Return a new ``CameraObservation`` with the robot's pixels removed from the depth,
        ready for ``CollisionWorldManager.integrate_layer`` (SPEC §5.2)."""
        from curobo._src.types.camera import CameraObservation

        _mask, filtered_depth = self.get_mask(observation, q)
        return CameraObservation(
            depth_image=filtered_depth,
            rgb_image=observation.rgb_image,
            intrinsics=observation.intrinsics,
            pose=observation.pose,
            depth_to_meter=observation.depth_to_meter,
        )
