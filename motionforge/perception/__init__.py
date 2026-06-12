"""Perception layer: camera registry, base-frame normalization, depth→CameraObservation."""

from motionforge.perception.frame_adapter import CameraInfo, PerceptionFrameAdapter
from motionforge.perception.self_filter import RobotSelfFilter

__all__ = ["CameraInfo", "PerceptionFrameAdapter", "RobotSelfFilter"]
