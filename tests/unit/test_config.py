"""Unit tests for motionforge.config defaults (SPEC §11)."""

import dataclasses

from motionforge.config import DEFAULTS, Config


def test_spec_section_11_defaults():
    c = Config()
    assert c.standoff_m == 0.05
    assert c.grasp_approach_axis == "z"
    assert c.grasp_approach_tstep_fraction == 0.6
    assert c.min_clearance_m == 0.01
    assert c.hold_vec_weight_vacuum_carry == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert c.plan_time_budget_s == 1.0
    assert c.recapture_cap == 3
    assert c.waypoint_buffer_depth == 5
    assert 0.005 <= c.esdf_voxel_size <= 0.01


def test_robot_defaults_are_ur10e_6dof():
    c = Config()
    assert c.robot_yaml == "ur10e.yml"
    assert c.base_frame == "base_link"
    assert c.tcp_frame == "tool0"


def test_config_is_frozen_and_overridable_via_replace():
    c = Config()
    c2 = dataclasses.replace(c, plan_time_budget_s=0.5, robot_yaml="abb_irb1200.yml")
    assert c2.plan_time_budget_s == 0.5
    assert c2.robot_yaml == "abb_irb1200.yml"
    # Original is unchanged (frozen).
    assert c.plan_time_budget_s == 1.0
    try:
        c.plan_time_budget_s = 2.0  # type: ignore[misc]
        assert False, "Config should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_defaults_singleton():
    assert isinstance(DEFAULTS, Config)
