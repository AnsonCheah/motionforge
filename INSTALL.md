# INSTALL.md — motionforge environment setup

Two isolated environments in one reproducible pixi workspace:

- **`planner`** — ROS2 (RoboStack **Jazzy**) + PyTorch (cu128) + **cuRobo v0.8.0 ("curobov2")**. Dev and runtime. Perception collision uses curobov2's **native warp ESDF mapper** — no nvblox.
- **`sim`** — Isaac Sim 6.0. The validation twin.

They do **not** share Python packages. They talk over DDS on a shared `ROS_DOMAIN_ID`, which is transport-level and environment-agnostic. That separation is deliberate: Isaac Sim 6.0 pins exact dependency versions that collide with the cu128 planner stack, so co-installing them is the thing that breaks. Both envs run Python 3.12, but are pinned per-feature so either can move independently. Rationale lives in `CONTEXT.md`. The *why* behind each version pin is in §6.

This is a Blackwell-class setup (RTX 50xx, **sm_120**); everything assumes **CUDA 12.8 + cu128 torch**.

---

## 1. Prerequisites (system level)

- **Linux**, Ubuntu **24.04** (glibc 2.35+; `ldd --version`). On 24.04 Isaac Sim 6.0 auto-loads its internal ROS2 **Jazzy** — which is why the planner uses RoboStack Jazzy (§4).
- **NVIDIA driver** new enough for CUDA 12.8 (`nvidia-smi`). You do **not** need a system CUDA install; the toolkit comes from conda into the `planner` env (only the driver is needed).
- **pixi** installed: `curl -fsSL https://pixi.sh/install.sh | bash`, then restart the shell.

---

## 2. Helper scripts (in `scripts/`)

The pixi tasks call four scripts (already in the repo):

- `scripts/check_gpu.py` — Phase-0 gate: runs an actual kernel on sm_120 (`is_available()` alone is not enough).
- `scripts/cuda_env.sh` — **sourced by the build/validate steps.** Fixes conda's split CUDA layout (`$CONDA_PREFIX/targets/x86_64-linux/{include,lib}` not on the compiler path) by exporting `CUDA_HOME` + `CPATH`/`LIBRARY_PATH`, and **forces `TORCH_CUDA_ARCH_LIST=12.0`** (conda's cuda-nvcc preset includes `10.1`, which torch 2.11 rejects).
- `scripts/build_curobo.sh` — clones cuRobo pinned at `v0.8.0`, sources `cuda_env.sh`, `pip install -e . --no-build-isolation` (compiles `curobolib` against the env's torch).
- `scripts/check_curobo.py` — validates curobov2: `MotionPlanner.warmup()` + a real `plan_pose` on the GPU (exercises the warp / `cuda.core` JIT on sm_120; an import alone won't).

---

## 3. Install and build

```bash
# 1. Resolve + create the planner env from the lock (ROS Jazzy + cu128 torch + toolchain).
#    NOTE: the lockfile is monolithic, so this still solves the `sim` env too.
pixi install -e planner

# 2. Phase-0 gate — STOP if this fails. Must print a kernel running on sm_120.
pixi run -e planner check-gpu

# 3. Build cuRobo (curobov2) and validate with a real plan.
#    setup = check-gpu -> build-curobo(v0.8.0) -> check-curobo(plan_pose).
pixi run -e planner setup

# 4. (Separate, large/slow) the Isaac validation twin.
pixi install -e sim
pixi run -e sim isaacsim        # accepts EULA, downloads assets the first time
```

Interactive shells: `pixi shell -e planner` / `pixi shell -e sim`.

A successful `setup` ends with something like:

```
torch: 2.11.0+cu128 | device: NVIDIA GeForce RTX 5080
warmup: ~14s (one-time)
plan_pose: success=True  solve_total_time=~30 ms
curobov2 OK
```

---

## 4. The ROS2 <-> Isaac bridge

Both environments inherit `ROS_DOMAIN_ID=42` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. The planner node (`planner`) and Isaac's ROS2 bridge (`sim`) discover each other over DDS with no shared Python env. The planner distro is **Jazzy** to match Isaac Sim 6.0 on Ubuntu 24.04 (it auto-loads internal Jazzy libs). The match isn't strictly required — DDS keys on message IDL, not the distro label — but matching avoids cross-distro type-hash edge cases. On a 22.04 host (where Isaac auto-loads Humble), switch the `robostack-jazzy` channel and `ros-jazzy-*` names in `pixi.toml` to Humble.

---

## 5. Validate before building on top

These are the failure modes specific to this stack. Confirm each early.

1. **torch kernel on sm_120** — covered by `check-gpu`. A tensor op must run, not just `is_available()`.
2. **curobov2's runtime JIT** — the most likely thing to break on Blackwell. `curobolib` compiles its CUDA kernels at first use via NVIDIA `cuda.core` (NVRTC), and the planner also JIT-compiles warp kernels. `check-curobo` runs a real `plan_pose`, which exercises both — do not trust a bare import.
3. **CUDA toolkit vs torch** — the conda `cuda-toolkit` is 12.8 to match torch's cu128. A mismatch compiles fine then fails at runtime with "no kernel image available."
4. **GLIBC 2.35+** for Isaac, on the host.

---

## 6. Reproducibility & why each pin exists

- Commit `pixi.lock` alongside `pixi.toml`. That is the actual pin; the toml is intent. `.pixi/` is disposable (regenerated from the lock); **never delete `pixi.lock`**.
- cuRobo is pinned to `v0.8.0` in `scripts/build_curobo.sh` so the source build reproduces (the lockfile doesn't cover source builds).
- The non-obvious pins, each one a real failure we hit on this stack:

| Pin / fix | Where | Why |
|---|---|---|
| `libc = glibc 2.35` | `[system-requirements]` | else pixi assumes ~2.28 and rejects Isaac's `manylinux_2_35` wheels |
| both envs Python `3.12`, pinned per-feature | per-feature `python` | pinned per-feature so either env can move independently; a shared pin couples both |
| `setuptools <82` (sim) | sim feature | isaacsim-core 6.0 pins torch==2.11.0, whose wheel needs setuptools<82 (conda installs 82.x) |
| `index-strategy=unsafe-best-match` (sim) | `[feature.sim.pypi-options]` | Isaac 6.0 deps span pypi.org + pypi.nvidia.com; sim-scoped so the planner env is untouched |
| `tinyobjloader==2.0.0rc13` (sim) | sim feature | a pre-release transitive dep; uv only honors it if declared directly |
| ROS **Jazzy** | ros feature | Isaac 6.0 auto-loads Jazzy on Ubuntu 24.04; RoboStack Jazzy now ships numpy 2 / py3.12 builds so the planner runs `numpy >=2` |
| `gxx = 13.*` | planner deps | CUDA 12.8 caps the host compiler at gcc < 14 |
| `cuda_env.sh` (CUDA `targets/` layout + `TORCH_CUDA_ARCH_LIST=12.0`) | build scripts | conda CUDA headers aren't on the compiler path; cuda-nvcc presets a `10.1` arch torch rejects |
| cuRobo `v0.8.0` (curobov2) | build_curobo.sh | classic v0.7.x is a different API; curobov2 has native ESDF (no nvblox) |
| `cuda-core[cu12]` | planner pypi-deps | curobov2's curobolib JIT backend (no pre-built pybind) |
| `warp-lang >=1.12` | planner pypi-deps | curobov2 uses the modern `wp.device_from_torch` API |

A teammate runs `pixi install -e planner` + `pixi run -e planner setup` and gets the same validated stack.

---

## 7. Repository layout

```
motionforge/
  pixi.toml           # this workspace (two environments)
  pixi.lock           # committed pin
  INSTALL.md          # this file
  SPEC.md             # build specification (curobov2)
  CONTEXT.md          # design rationale, caveats, glossary
  scripts/            # check_gpu.py, cuda_env.sh, build_curobo.sh, check_curobo.py
  external/           # cloned source build: curobo/  (gitignored)
  src/                # your ROS2 packages (built from SPEC.md)
```

`external/` and `.pixi/` are gitignored; they hold the upstream clone and the solved env (regenerated).

---

## 8. Troubleshooting quick hits

- **`CUDA error: no kernel image is available`** — toolkit/torch CUDA mismatch, or torch is not a cu128 build. Re-check `check-gpu`.
- **`Unknown CUDA arch (10.1)`** — `TORCH_CUDA_ARCH_LIST` from conda's cuda-nvcc activation leaked in; `cuda_env.sh` forces `12.0`. Make sure the build/validate steps source it.
- **`compiler version is greater than the maximum required by CUDA 12.8`** — host gcc ≥ 14; the `gxx = 13.*` pin fixes it (re-solve if you changed it).
- **`No module named 'cuda.core'`** — curobov2's curobolib backend; ensure `cuda-core` (extras `cu12`) is installed (`pixi install -e planner`).
- **`module 'warp' has no attribute 'torch'`** — that's the *old* v0.7.x failure; on curobov2 keep `warp-lang >=1.12`.
- **cuRobo build can't find torch / cuda_runtime.h** — the build must `source scripts/cuda_env.sh` and use `--no-build-isolation`. `build-curobo` does both.
- **Isaac Sim pip resolve errors** — known churn; pin a specific 5.x patch and retry, or fall back to the pre-built binary installer for `sim`.
- **`rosdep` does nothing** — pixi does not support rosdep. Add ROS packages with `pixi add ros-jazzy-<pkg>`.
- **nvblox** — not used and won't build on CUDA 12.8/Blackwell; curobov2's native mapper replaces it. Don't re-add it.
