"""SE3 / quaternion geometry helpers (numpy, CPU — no torch, no GPU).

Conventions (match cuRobo and SPEC §4):
    - position: (3,) metres, in the robot **base frame** unless stated.
    - quaternion: (4,) ``wxyz``, unit norm.
    - cuRobo pose list format is ``[x, y, z, qw, qx, qy, qz]`` (see ``Pose.to_curobo_list``).

A ``Pose`` is a rigid transform T = (R(quat), position). ``a.multiply(b)`` composes them
as matrices (``T_a @ T_b``): if ``b`` is expressed in ``a``'s local frame, the result
applies ``b`` *within* that frame — this matches cuRobo's ``Pose.multiply`` (used by
``plan_grasp`` to offset a grasp by an approach vector in the tool frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

Vec3 = Union[Sequence[float], np.ndarray]
Quat = Union[Sequence[float], np.ndarray]

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def axis_to_vector(axis: str) -> np.ndarray:
    """Map a principal-axis string ``"x"|"y"|"z"`` to a unit vector (matches cuRobo)."""
    key = axis.lower()
    if key not in _AXES:
        raise ValueError(f"Invalid axis: {axis!r}, must be 'x', 'y', or 'z'")
    return np.array(_AXES[key], dtype=np.float64)


def as_vec3(v: Vec3) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"expected a length-3 vector, got shape {arr.shape}")
    return arr


def normalize(v: Vec3) -> np.ndarray:
    arr = as_vec3(v)
    n = np.linalg.norm(arr)
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return arr / n


def quat_normalize(q: Quat) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"expected a length-4 quaternion, got shape {arr.shape}")
    n = np.linalg.norm(arr)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    arr = arr / n
    # Canonicalize sign so w >= 0 (a quaternion and its negation are the same rotation).
    if arr[0] < 0:
        arr = -arr
    return arr


def quat_conjugate(q: Quat) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(-1)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_multiply(q1: Quat, q2: Quat) -> np.ndarray:
    """Hamilton product of two ``wxyz`` quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(-1)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_rotate_vector(q: Quat, v: Vec3) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` (wxyz)."""
    qarr = np.asarray(q, dtype=np.float64).reshape(-1)
    w = qarr[0]
    u = qarr[1:]
    vec = as_vec3(v)
    t = 2.0 * np.cross(u, vec)
    return vec + w * t + np.cross(u, t)


@dataclass
class Pose:
    """SE3 rigid transform. ``position`` (3,) metres, ``quaternion`` (4,) wxyz unit."""

    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.quaternion = quat_normalize(self.quaternion)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Pose":
        """Build from cuRobo's ``[x, y, z, qw, qx, qy, qz]`` list format."""
        v = list(values)
        if len(v) != 7:
            raise ValueError(f"expected 7 values [x,y,z,qw,qx,qy,qz], got {len(v)}")
        return cls(np.array(v[:3]), np.array(v[3:]))

    def to_curobo_list(self) -> list[float]:
        """Return ``[x, y, z, qw, qx, qy, qz]`` (cuRobo ``Pose.from_list`` format)."""
        return [*self.position.tolist(), *self.quaternion.tolist()]

    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def multiply(self, other: "Pose") -> "Pose":
        """Compose transforms: ``self ∘ other`` (``T_self @ T_other``)."""
        pos = self.position + quat_rotate_vector(self.quaternion, other.position)
        quat = quat_multiply(self.quaternion, other.quaternion)
        return Pose(pos, quat)

    def inverse(self) -> "Pose":
        q_inv = quat_conjugate(self.quaternion)
        pos = -quat_rotate_vector(q_inv, self.position)
        return Pose(pos, q_inv)

    def transform_point(self, point: Vec3) -> np.ndarray:
        return self.position + quat_rotate_vector(self.quaternion, point)

    def approx_equal(self, other: "Pose", atol: float = 1e-8) -> bool:
        pos_ok = bool(np.allclose(self.position, other.position, atol=atol))
        # Quaternions are sign-canonicalized in __post_init__, so a direct compare is safe.
        quat_ok = bool(np.allclose(self.quaternion, other.quaternion, atol=atol))
        return pos_ok and quat_ok


def offset_along_axis(pose: Pose, axis_unit_base: Vec3, distance_m: float) -> Pose:
    """Shift ``pose`` by ``distance_m`` along a **base-frame** unit axis; keep orientation."""
    a = normalize(axis_unit_base)
    return Pose(pose.position + distance_m * a, pose.quaternion.copy())


def pre_grasp_pose(grasp_pose: Pose, approach_axis_base: Vec3, standoff_m: float) -> Pose:
    """Pre-grasp = grasp offset by ``standoff_m`` along **-approach_axis** (SPEC §5.3).

    ``approach_axis`` is a base-frame unit vector (usually tool +Z). Orientation is kept.
    """
    return offset_along_axis(grasp_pose, approach_axis_base, -abs(standoff_m))


def grasp_transform(object_pose: Pose, grasp_pose: Pose) -> Pose:
    """Object pose expressed in the TCP/grasp frame: ``inv(grasp) ∘ object`` (SPEC §5.4).

    This is the ``object_pose ⊖ grasp_pose`` used to place the attached body relative to
    the TCP after a confirmed grasp.
    """
    return grasp_pose.inverse().multiply(object_pose)
