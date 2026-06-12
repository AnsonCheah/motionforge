"""motionforge planner ROS2 node (SPEC §3, §5) — the integration fabric.

Thin rclpy wrapper: ROS parameters → :class:`motionforge.config.Config`, TF for fixed/EIH
frame management, and a ``run_cycle`` Trigger service that drives the :class:`TaskCoordinator`.
The heavy GPU collaborators (cuRobo planner, collision world, ABB socket) are built lazily on
the first cycle so node bring-up stays light and testable without a GPU. Perception and gripper
are deployment-specific and injected via :meth:`set_io`.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import rclpy
import tf2_ros
from rclpy.node import Node
from std_srvs.srv import Trigger

from motionforge.config import DEFAULTS, Config

# Config fields exposed as ROS parameters (type inferred from the default).
_PARAM_FIELDS: List[str] = [
    "robot_yaml", "base_frame", "tcp_frame", "grasp_approach_axis",
    "standoff_m", "min_clearance_m", "plan_time_budget_s",
    "recapture_cap", "waypoint_buffer_depth", "esdf_voxel_size",
]


class PlannerNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("motionforge_planner", **kwargs)

        for field in _PARAM_FIELDS:
            self.declare_parameter(field, getattr(DEFAULTS, field))
        self.declare_parameter("rapid_host", "127.0.0.1")
        self.declare_parameter("rapid_port", 11000)
        # Per-deployment cell config (SPEC §5.8). Empty = no static world / tools (a planner
        # that sees no fixed obstacles — unsafe for real runs; warned at build time).
        self.declare_parameter("cell_config", "")

        self.config: Config = self._config_from_params()

        # TF: fixed vs eye-in-hand frame handling (SPEC §5.1).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._coordinator = None
        self._perception = None
        self._gripper = None
        self.frame_adapter = None  # built from cell_config cameras (SPEC §5.1)

        self.run_cycle_srv = self.create_service(Trigger, "run_cycle", self._on_run_cycle)
        self.get_logger().info(
            f"motionforge planner up (robot={self.config.robot_yaml}, "
            f"base={self.config.base_frame}, tcp={self.config.tcp_frame})"
        )

    # -- configuration --

    def _config_from_params(self) -> Config:
        values = {f: self.get_parameter(f).value for f in _PARAM_FIELDS}
        return dataclasses.replace(DEFAULTS, **values)

    def set_io(self, perception, gripper) -> None:
        """Inject the perception source and gripper actuator (deployment-specific)."""
        self._perception = perception
        self._gripper = gripper

    # -- run-cycle service --

    def _on_run_cycle(self, request, response):
        try:
            coordinator = self._ensure_coordinator()
            result = coordinator.run_cycle()
            response.success = bool(result.success)
            response.message = (
                f"state={result.state.value} recaptures={result.recaptures} "
                f"max_plan_time_s={result.max_plan_time_s:.3f} {result.fault_reason}".strip()
            )
        except Exception as exc:  # noqa: BLE001 — surface wiring/runtime errors to the caller
            response.success = False
            response.message = f"error: {exc}"
        return response

    def _ensure_coordinator(self):
        if self._coordinator is None:
            self._coordinator = self._build_coordinator()
        return self._coordinator

    def _build_coordinator(self):
        """Wire the GPU planning stack + ABB execution (lazy; ~15 s warmup on first call).

        Perception and gripper must be set via :meth:`set_io` (the vision pipeline and gripper
        driver are deployment-specific). Returns a ready :class:`TaskCoordinator`.
        """
        if self._perception is None or self._gripper is None:
            raise RuntimeError("call set_io(perception, gripper) before run_cycle")

        # Imported here so node bring-up doesn't require torch/cuRobo.
        from motionforge.collision import CollisionWorldManager, ESDFGridSet, ROIGridSpec
        from motionforge.coordinator import TaskCoordinator
        from motionforge.execution.abb_socket import AbbSocketAdapter
        from motionforge.planner import MotionPlannerAdapter
        from motionforge.tools import ToolManager

        cell = self._load_cell_config()

        # ESDF grid geometry is the single source of truth for the voxel cache + mappers,
        # derived from the CellConfig ROIs (or a 1 m default per ROI when none are authored).
        vs = self.config.esdf_voxel_size
        if cell is not None and cell.rois:
            from motionforge.cell_config import grid_set_from_rois

            grid_set = grid_set_from_rois(cell.rois)
        else:
            grid_set = ESDFGridSet(
                (
                    ROIGridSpec("pick_bin", (0.5, -0.3, 0.3), (1.0, 1.0, 1.0), vs),
                    ROIGridSpec("place_tray", (0.5, 0.3, 0.3), (1.0, 1.0, 1.0), vs),
                )
            )

        # The active tool's TCP offset becomes the planned tool frame (immutable per planner
        # build); its width-dependent geometry uses a dedicated sphere budget.
        tcp_offset = cell.tools[0].tcp_pose if (cell is not None and cell.tools) else None

        collision_cache = {"cuboid": 30, "mesh": 30, "voxel": grid_set.collision_cache()}
        planner = MotionPlannerAdapter(
            config=self.config, collision_cache=collision_cache,
            attached_object_spheres=64, tcp_offset=tcp_offset, tool_spheres=32,
        )
        planner.warmup()
        world = CollisionWorldManager(planner, grid_set=grid_set)

        # Static collision world + camera registry + tool library from the cell config.
        tools = ToolManager()
        if cell is not None:
            from motionforge.cell_config import (
                cameras_to_registry,
                static_bodies_to_curobo,
                tools_to_manager,
            )
            from motionforge.perception.frame_adapter import PerceptionFrameAdapter

            cuboids, meshes = static_bodies_to_curobo(cell.static_bodies)
            world.set_static(cuboids=cuboids, meshes=meshes)
            world.commit()  # establish the static free space before any cycle
            if not cuboids and not meshes:
                self.get_logger().warning(
                    "cell_config has NO static bodies — the planner sees no fixed obstacles; "
                    "unmodeled cell structure is treated as free space (CONTEXT §3)"
                )
            self.frame_adapter = PerceptionFrameAdapter(fk_fn=planner.tcp_pose_at)
            for cam in cameras_to_registry(cell.cameras):
                self.frame_adapter.register(cam)
            if cell.tools:
                tools = tools_to_manager(cell.tools)
        else:
            self.get_logger().warning(
                "no cell_config parameter set — running with an EMPTY static world and no tool "
                "library; set the 'cell_config' parameter to a manifest for a real deployment"
            )

        host = self.get_parameter("rapid_host").value
        port = int(self.get_parameter("rapid_port").value)
        execution = AbbSocketAdapter(host, port, config=self.config)
        execution.connect()  # also the Joint State Source (SPEC §5.7)

        return TaskCoordinator(
            planner=planner, world=world, tools=tools,
            perception=self._perception, gripper=self._gripper,
            execution=execution, joint_state_source=execution, config=self.config,
        )

    def _load_cell_config(self):
        """Load the cell config manifest if the 'cell_config' parameter is set, else None."""
        path = self.get_parameter("cell_config").value
        if not path:
            return None
        from motionforge.cell_config import load_cell_config

        cell = load_cell_config(path)
        self.get_logger().info(
            f"cell_config loaded: {len(cell.static_bodies)} static bodies, "
            f"{len(cell.cameras)} cameras, {len(cell.rois)} ROIs, {len(cell.tools)} tools"
        )
        return cell


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
