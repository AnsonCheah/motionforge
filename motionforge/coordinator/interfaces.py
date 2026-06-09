"""Coordinator collaborator interfaces (SPEC §5.5, §6).

Perception (vision upstream) and gripper actuation are abstracted so the coordinator can be
driven by real hardware, the Isaac twin, or fakes. The planner / collision world / tool
manager / execution adapter are the concrete modules from earlier phases (duck-typed here).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from motionforge.types import GraspCandidate, PlaceCandidate, ToolAction


@dataclass
class PickPerception:
    """Vision output for the pick phase (base frame)."""

    grasps: List[GraspCandidate]
    bin_voxels: Any = None       # cuRobo VoxelGrid for the pick_bin ESDF layer (optional)
    workpiece: Any = None        # cuRobo Cuboid/Mesh attached to the TCP on grasp (optional)


@dataclass
class PlacePerception:
    """Vision output for the place phase (re-perceived every cycle)."""

    places: List[PlaceCandidate]
    tray_voxels: Any = None      # cuRobo VoxelGrid for the place_tray ESDF layer (optional)


class PerceptionSource(ABC):
    @abstractmethod
    def perceive_pick(self) -> PickPerception: ...

    @abstractmethod
    def perceive_place(self) -> PlacePerception: ...


class GripperActuator(ABC):
    """Decoupled tool actuation with sync barriers (SPEC §2.4)."""

    @abstractmethod
    def command(self, action: ToolAction) -> None:
        """Issue an actuation (non-blocking start)."""

    @abstractmethod
    def wait(self) -> None:
        """Barrier: block until the last commanded actuation completes."""
