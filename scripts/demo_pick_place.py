"""Headless pick-and-place demo with cuRobo's native Viser visualizer (browser, no Isaac).

Plans a full cycle through the real TaskCoordinator (UR10e) against a static table + a
pick_bin ESDF obstacle + a place_tray layer, then animates the planned trajectories — robot,
collision world, target frames, and the held workpiece — in cuRobo's ViserVisualizer.

Run:   pixi run -e planner demo            # then open the printed http://<host>:8080 URL
       pixi run -e planner demo -- --once  # play the cycle once and exit

Viser serves a web UI; no local display needed (works over SSH — forward port 8080).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch

from curobo._src.geom.types import VoxelGrid
from curobo._src.types.content_path import ContentPath
from curobo._src.types.pose import Pose as CuPose
from curobo.scene import Cuboid
from curobo.viewer import ViserVisualizer

from motionforge.collision import CollisionWorldManager
from motionforge.config import DEFAULTS
from motionforge.coordinator import TaskCoordinator
from motionforge.coordinator.fakes import FakeGripper, RecordingExecution, ScriptedPerception
from motionforge.coordinator.interfaces import PickPerception, PlacePerception
from motionforge.geometry import Pose
from motionforge.joint_state import FakeJointStateSource
from motionforge.planner import MotionPlannerAdapter
from motionforge.tools import ToolManager, parallel_jaw_geom_fn
from motionforge.types import GraspCandidate, GripConfig, PlaceCandidate, ToolDescriptor

DIMS = (1.0, 1.0, 1.0)
VS = 0.02
CENTER = (0.5, 0.0, 0.3)
TCP_OFFSET = Pose([0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])  # jaw TCP, 0.2 m past tool0 (+Z)


def _empty_grid(name="esdf") -> VoxelGrid:
    n = [round(d / VS) for d in DIMS]
    feat = torch.full(tuple(n), 1.0, dtype=torch.float16, device="cuda:0")
    return VoxelGrid(name=name, pose=[*CENTER, 1, 0, 0, 0], dims=list(DIMS), voxel_size=VS,
                     feature_tensor=feat, feature_dtype=torch.float16)


def _box_grid(box_center, half=(0.07, 0.07, 0.12), name="esdf") -> VoxelGrid:
    """An ESDF voxel grid with a single box obstacle (the 'bin contents')."""
    n = [round(d / VS) for d in DIMS]
    ix, iy, iz = (torch.arange(k, device="cuda:0", dtype=torch.float32) for k in n)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    wx = CENTER[0] + (gx - (n[0] - 1) / 2) * VS
    wy = CENTER[1] + (gy - (n[1] - 1) / 2) * VS
    wz = CENTER[2] + (gz - (n[2] - 1) / 2) * VS
    dx = (wx - box_center[0]).abs() - half[0]
    dy = (wy - box_center[1]).abs() - half[1]
    dz = (wz - box_center[2]).abs() - half[2]
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.stack([dx, dy, dz], -1).max(-1).values.clamp(max=0)
    return VoxelGrid(name=name, pose=[*CENTER, 1, 0, 0, 0], dims=list(DIMS), voxel_size=VS,
                     feature_tensor=(outside + inside).to(torch.float16), feature_dtype=torch.float16)


def _cu_pose(p: Pose) -> CuPose:
    return CuPose(
        position=torch.tensor([p.position.tolist()], device="cuda:0", dtype=torch.float32),
        quaternion=torch.tensor([p.quaternion.tolist()], device="cuda:0", dtype=torch.float32),
    )


def plan_cycle(adapter):
    """Run a full pick-and-place through the coordinator; return (result, world, scene)."""
    world = CollisionWorldManager(adapter)

    # Top-down poses in the robot BASE frame (tool +Z -> world -Z).
    DOWN = [0.0, 1.0, 0.0, 0.0]
    grasp_pose = Pose([1.0, 0.0, 0.30], DOWN)   # in the bin: +x side of the robot, y = 0
    place_pose = Pose([0.0, 0.5, 0.30], DOWN)   # +y side of the robot, x = 0
    grasp = GraspCandidate(grasp_pose, [0, 0, -1], 0.06, "jaw", GripConfig(0.02, force=30.0))
    place = PlaceCandidate(place_pose, [0, 0, -1], 0.06, "jaw", GripConfig(0.08, mode="outward"))
    workpiece = Cuboid(name="part", pose=[*grasp_pose.position.tolist(), 1, 0, 0, 0],
                       dims=[0.05, 0.05, 0.05])

    # Obstacle in the +x/+y quadrant, between grasp(+x, 0) and place(0, +y) so the transport
    # must route around it. Kept inside the pick_bin grid (centered at CENTER, y in [-0.5, 0.5])
    # so it is both perceived by the planner AND rendered.
    bin_box = _box_grid(box_center=(0.50, 0.50, 0.3), half=(0.07, 0.07, 0.15))

    perception = ScriptedPerception(
        picks=[PickPerception([grasp], bin_voxels=bin_box, workpiece=workpiece)],
        places=[PlacePerception([place], tray_voxels=_empty_grid())],
    )
    tools = ToolManager()
    tools.register(ToolDescriptor("jaw", TCP_OFFSET, parallel_jaw_geom_fn(),
                                  "socket://gripper", payload_kg=0.5))
    joints = FakeJointStateSource(adapter.default_q0)
    coord = TaskCoordinator(
        planner=adapter, world=world, tools=tools, perception=perception,
        gripper=FakeGripper(), execution=RecordingExecution(joint_state=joints),
        joint_state_source=joints, config=adapter._cfg,
    )

    # Wall-clock planning time (execution is fake/instant, so this is ~all planning).
    t0 = time.perf_counter()
    result = coord.run_cycle()
    plan_wall_s = time.perf_counter() - t0
    solves = result.plan_times_s
    print(f"  planning wall time: {plan_wall_s:.3f}s over {len(solves)} plan calls "
          f"(cuRobo solve: sum={sum(solves):.3f}s, max={result.max_plan_time_s:.3f}s)")

    return result, world, grasp_pose, place_pose, bin_box


def occupied_voxel_points(grid, device_cfg) -> np.ndarray:
    """World-frame centers of occupied (on/inside-surface) voxels, for point-cloud rendering."""
    xyzr = grid.create_xyzr_tensor(transform_to_origin=True, device_cfg=device_cfg)
    feat = grid.feature_tensor.reshape(-1).float()
    mask = feat < (0.5 * grid.voxel_size)  # signed ESDF: <= ~0 is on/inside the obstacle
    return xyzr[mask][:, :3].detach().cpu().numpy()


def animate(viz, adapter, result, grasp_pose, place_pose, dt, stride):
    """Play approach→grasp→lift→transport→place→retract; the part follows the TCP while held."""
    pick, place = result.pick_plan, result.place_plan
    # (label, trajectory, held?) — the part is held from the lift through the place.
    phases = [
        ("approach", pick.approach, False),
        ("grasp", pick.grasp, False),
        ("lift", pick.retract, True),
        ("transport", place.transport, True),
        ("place", place.place, True),
        ("retract", place.retract, False),
    ]
    part = viz._server.scene.add_box("/workpiece", color=(220, 120, 40), dimensions=(0.05, 0.05, 0.05))
    part.position = np.array(grasp_pose.position)

    for label, traj, held in phases:
        if traj is None:
            continue
        print(f"  ▶ {label:9s} ({len(traj)} waypoints, held={held})")
        for i in range(0, len(traj), stride):
            q = traj.points[i][0]
            viz.set_joint_positions(
                torch.tensor(q, device="cuda:0", dtype=torch.float32), adapter.joint_names
            )
            if held:
                tcp = adapter.tcp_pose_at(q)  # part rides with the TCP
                part.position = np.array(tcp.position)
                part.wxyz = np.array(tcp.quaternion)
            time.sleep(dt)
        if not held:  # park the part at the grasp (pre-grip) or place (post-release)
            anchor = place_pose if label == "retract" else grasp_pose
            part.position = np.array(anchor.position)
            part.wxyz = np.array(anchor.quaternion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dt", type=float, default=0.02, help="seconds per animation frame")
    ap.add_argument("--stride", type=int, default=2, help="show every Nth waypoint")
    ap.add_argument("--once", action="store_true", help="play once then exit")
    args = ap.parse_args()

    config = replace(DEFAULTS, use_cuda_graph=False, warmup_iterations=2)
    print("Building UR10e planner (warmup ~10-15 s)...")
    # Pre-allocate cuboid + mesh + voxel buffers so update_world() can compose the static
    # table with the per-cycle pick_bin / place_tray ESDF layers.
    collision_cache = {
        "cuboid": 10, "mesh": 10,
        "voxel": {"layers": 2, "dims": list(DIMS), "voxel_size": VS},
    }
    adapter = MotionPlannerAdapter(config=config, collision_cache=collision_cache,
                                   attached_object_spheres=64, tcp_offset=TCP_OFFSET,
                                   tool_spheres=32)
    adapter.warmup()

    print("Planning pick-and-place cycle...")
    result, _world, grasp_pose, place_pose, bin_box = plan_cycle(adapter)
    print(f"  cycle: success={result.success} state={result.state.value} "
          f"max_plan_time={result.max_plan_time_s:.3f}s recaptures={result.recaptures}")
    if not result.success:
        print(f"  FAILED: {result.fault_reason}")
        return 1

    viz = ViserVisualizer(content_path=ContentPath(robot_config_file=config.robot_yaml),
                          device_cfg=adapter.device_cfg, connect_port=args.port,
                          add_control_frames=False)
    # Render the pick_bin ESDF obstacle as the occupied-voxel point cloud the planner sees.
    pts = occupied_voxel_points(bin_box, adapter.device_cfg)
    if len(pts):
        viz.add_point_cloud(pts, colors=[120, 120, 130], point_size=VS, name="/obstacles/pick_bin")
    else:
        lo = [c - 0.5 * d for c, d in zip(CENTER, DIMS)]
        hi = [c + 0.5 * d for c, d in zip(CENTER, DIMS)]
        print(f"  WARNING: obstacle produced 0 occupied voxels — likely outside the grid bounds "
              f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}].")
    viz.add_frame("/targets/grasp", _cu_pose(grasp_pose), scale=0.12)
    viz.add_frame("/targets/place", _cu_pose(place_pose), scale=0.12)

    print(f"\n  Viser UI: http://localhost:{args.port}  (Ctrl-C to stop)\n")
    try:
        while True:
            animate(viz, adapter, result, grasp_pose, place_pose, args.dt, args.stride)
            if args.once:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
