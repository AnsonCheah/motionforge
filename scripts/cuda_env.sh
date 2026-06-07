# CUDA build/runtime env for the pixi planner env.
#
# AUTO-SOURCED on every planner activation via [feature.planner.activation] in pixi.toml,
# so CUDA_HOME / CPATH / TORCH_CUDA_ARCH_LIST are always set for `pixi run -e planner ...`
# and `pixi shell -e planner` — you do NOT need to source it by hand. The build scripts
# also source it directly (standalone use); the guard below makes that idempotent.
#
# WHY it's needed: conda-forge's cuda-toolkit puts headers/libs under a split layout
#   $CONDA_PREFIX/targets/x86_64-linux/{include,lib}
# which is NOT on the compiler's default search path. Without this, torch's cpp_extension
# (cuRobo build) and curobolib's runtime cuda.core JIT can't find cuda_runtime.h.

# Safe no-op if not inside an env (must not abort activation), and skip if already applied
# for this prefix (avoids duplicate CPATH/LIBRARY_PATH entries when sourced more than once).
if [ -z "${CONDA_PREFIX:-}" ] || [ "${_MOTIONFORGE_CUDA_ENV:-}" = "${CONDA_PREFIX:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
export _MOTIONFORGE_CUDA_ENV="$CONDA_PREFIX"

_cuda_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
_cuda_lib="$CONDA_PREFIX/targets/x86_64-linux/lib"

export CUDA_HOME="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CPATH="${_cuda_inc}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${_cuda_lib}:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_cuda_lib}:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Blackwell (RTX 50xx) = sm_120. FORCE this (unconditional `=`, not `:-`): conda's
# cuda-nvcc activation pre-sets a broad TORCH_CUDA_ARCH_LIST that includes "10.1",
# which this torch (2.11 cu128) rejects ("Unknown CUDA arch (10.1)"). Edit for other GPUs.
export TORCH_CUDA_ARCH_LIST="12.0"
