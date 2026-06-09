"""Task Coordinator (SPEC §5.5): the pick-and-place state machine."""

from motionforge.coordinator.interfaces import (
    GripperActuator,
    PerceptionSource,
    PickPerception,
    PlacePerception,
)
from motionforge.coordinator.state_machine import (
    CoordinatorState,
    CycleResult,
    TaskCoordinator,
)

__all__ = [
    "TaskCoordinator",
    "CoordinatorState",
    "CycleResult",
    "PerceptionSource",
    "GripperActuator",
    "PickPerception",
    "PlacePerception",
]
