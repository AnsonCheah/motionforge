"""Validate cuRobo (curobov2 / v0.8.0) on the target GPU: warmup + a REAL plan.

Importing cuRobo is not enough — warp-lang's PTX JIT on Blackwell (sm_120) only runs
during an actual plan. Mirrors curobov2's getting_started/motion_planning example.
Exits non-zero on failure so it can gate `pixi run -e planner setup`.
"""
import sys, time, torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

print("torch:", torch.__version__, "| device:", torch.cuda.get_device_name(0))

config = MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml")
planner = MotionPlanner(config)

t = time.time()
planner.warmup(enable_graph=True, num_warmup_iterations=5)   # compiles CUDA graphs + warp kernels on sm_120
print("warmup: %.1fs (one-time)" % (time.time() - t))

q_start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)
goal = GoalToolPose(
    tool_frames=planner.tool_frames,
    position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
    quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
)

t = time.time()
result = planner.plan_pose(goal, q_start)
wall = time.time() - t

ok = result is not None and bool(result.success.any())
solve_ms = float(getattr(result, "total_time", 0.0)) * 1e3 if ok else 0.0
print("plan_pose: success=%s  wall=%.4fs  solve_total_time=%.1f ms" % (ok, wall, solve_ms))
if ok:
    interp = result.get_interpolated_plan()
    print("  trajectory waypoints:", interp.position.shape[-2])
if not ok:
    print("curobov2 VALIDATION FAILED", file=sys.stderr)
    sys.exit(1)
print("curobov2 OK")
