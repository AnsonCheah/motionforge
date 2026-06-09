"""Execution layer: OEM-agnostic adapter + ABB raw-socket impl + fake RAPID server.

SPEC §5.6. The adapter boundary (``send_trajectory`` / ``read_joint_state`` / ``stop``) is
fixed so EGM / ros2_control adapters drop in later. The MVP impl streams down-sampled
waypoints into a RAPID ring buffer keeping ``waypoint_buffer_depth`` points ahead so the
controller blends corners (no per-point stall).
"""

from motionforge.execution.adapter import ExecutionAdapter, Waypoint
from motionforge.execution.downsample import downsample_waypoints, interpolation_error

__all__ = [
    "ExecutionAdapter",
    "Waypoint",
    "downsample_waypoints",
    "interpolation_error",
]
