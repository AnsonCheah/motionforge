"""motionforge — GPU motion planning pipeline for bin picking.

ROS-agnostic core library embedding cuRobo (v0.8.0 / "curobov2"). See SPEC.md for the
build specification and CONTEXT.md for the rationale. ROS2 nodes (under ``src/``) wrap
these modules; everything here is importable and testable without ROS.
"""

from motionforge import config, geometry, types

__all__ = ["config", "geometry", "types"]
__version__ = "0.1.0"
