#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Fix conda's split CUDA layout so torch's cpp_extension finds headers/libs, and
# target sm_120 (Blackwell). See cuda_env.sh.
source "$HERE/cuda_env.sh"

# Pin cuRobo to v0.8.0 ("curobov2"): flat-architecture rewrite (impl under curobo/_src)
# with a NATIVE warp depth->ESDF perception mapper (no external nvblox) and the modern
# warp API. Public API: curobo.motion_planner.{MotionPlanner,MotionPlannerCfg},
# curobo.types.{GoalToolPose,JointState,Pose}; plan_pose / plan_grasp. NOTE: this is the
# new API — the old v0.7.x classic API (PoseCostMetric, wrap.reacher.motion_gen) is gone.
CUROBO_REF="${CUROBO_REF:-v0.8.0}"

mkdir -p external && cd external
if [ ! -d curobo/.git ]; then
  git clone --depth 1 --branch "$CUROBO_REF" https://github.com/NVlabs/curobo.git curobo
else
  # Reconcile an existing checkout to the pinned ref (e.g. a prior main clone).
  git -C curobo fetch --depth 1 origin tag "$CUROBO_REF"
  git -C curobo checkout -q "$CUROBO_REF"
fi
cd curobo
# Patch: curobov2 v0.8.0 still calls the removed `wp.torch.device_from_torch` in its mesh
# paths (mesh_extractor.py voxel->mesh, geom/data/data_mesh.py), which breaks on warp >=1.12
# ("module 'warp' has no attribute 'torch'") — the function is now top-level wp.device_from_torch.
# Without this, mesh-extraction + scene-collision tests fail. Idempotent; skips missing files.
for _pf in curobo/_src/geom/data/data_mesh.py curobo/_src/perception/mapper/mesh_extractor.py; do
  [ -f "$_pf" ] && sed -i 's/wp\.torch\.device_from_torch/wp.device_from_torch/g' "$_pf"
done
# Build cuRobo's CUDA kernels against the env's torch (no build isolation).
pip install -e . --no-build-isolation
