"""Default configuration parameters (SPEC §11) and robot/planner defaults.

All values are overridable: construct ``Config()`` and pass overrides, or use
``dataclasses.replace(cfg, plan_time_budget_s=0.5)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    # --- SPEC §11 pipeline parameters ---
    standoff_m: float = 0.05  # -> grasp_approach_offset; placeholder until vision supplies per-candidate
    grasp_approach_axis: str = "z"  # tool-frame principal axis for approach
    grasp_approach_tstep_fraction: float = 0.6  # when the approach constraint activates
    min_clearance_m: float = 0.01  # hard filter
    #: vacuum-carry orientation lock, SPEC pose-cost order [rx,ry,rz, x,y,z] (orientation first).
    hold_vec_weight_vacuum_carry: Tuple[float, float, float, float, float, float] = (
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    )
    plan_time_budget_s: float = 1.0  # per planning call
    recapture_cap: int = 3  # before FAULT
    waypoint_buffer_depth: int = 5  # RAPID look-ahead points
    esdf_voxel_size: float = 0.005  # mapper voxel size (SPEC range 0.005–0.01); tune per part scale
    #: Post-execution check: max |feedback - planned-end| per joint before FAULT (SPEC §5.7).
    exec_joint_tol_rad: float = 0.05

    # --- segment-builder defaults (not in SPEC §11; overridable) ---
    gripper_standby_width_m: float = 0.08  # open/standby width during approach (must clear neighbors)
    retract_m: float = 0.05  # straight-line lift/retract distance after grasp/place

    # --- Robot / planner defaults (UR10e MVP; swap to an ABB IRB yml later, config-only) ---
    robot_yaml: str = "ur10e.yml"  # bundled cuRobo config; 6-DOF
    base_frame: str = "base_link"
    tcp_frame: str = "tool0"  # cuRobo tool/EE frame for ur10e.yml
    num_ik_seeds: int = 16
    num_trajopt_seeds: int = 2
    max_goalset: int = 8  # ranked-candidate axis capacity (num_goalset)
    use_cuda_graph: bool = True
    warmup_iterations: int = 5


DEFAULTS = Config()
