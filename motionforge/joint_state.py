"""Joint State Source (SPEC §5.7) — provide the current ``q0`` to the planner.

MVP: parse joint feedback from the RAPID socket server (the :class:`AbbSocketAdapter` also
implements this interface). A :class:`FakeJointStateSource` backs unit tests and the sim path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class JointStateSource(ABC):
    @abstractmethod
    def read_joint_state(self) -> List[float]:
        """Return the current joint configuration ``q0``."""


class FakeJointStateSource(JointStateSource):
    """A settable in-memory joint-state source for tests and the sim path."""

    def __init__(self, q: Sequence[float]) -> None:
        self._q = list(q)

    def read_joint_state(self) -> List[float]:
        return list(self._q)

    def set(self, q: Sequence[float]) -> None:
        self._q = list(q)
