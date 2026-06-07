# Getting Started — motionforge

Stand up the two-environment workspace: a GPU motion planner (cuRobo) and an Isaac Sim
validation twin, both reproducible with `pixi`. This is the happy path — for the rationale
behind every version pin and a full troubleshooting catalogue, see
[INSTALL.md](../INSTALL.md); for the system design, [SPEC.md](../SPEC.md) and
[CONTEXT.md](../CONTEXT.md).

## The two environments

| Env | Stack | Python | Use |
|---|---|---|---|
| `planner` | ROS2 Jazzy (RoboStack) + PyTorch cu128 + **cuRobo v0.8.0 (curobov2)** | 3.12 | dev + runtime |
| `sim` | **Isaac Sim 6.0** | 3.12 | validation twin |

They never share Python packages — they talk over DDS on a shared `ROS_DOMAIN_ID`. Validated
on **Ubuntu 24.04 + RTX 5080** (Blackwell, sm_120), driver 595, CUDA 12.8.

---

## 1. Prerequisites

- **Ubuntu 24.04** (glibc ≥ 2.35 — check with `ldd --version`).
- **NVIDIA driver** new enough for CUDA 12.8 on a Blackwell / RTX 50xx GPU (`nvidia-smi`).
  You do **not** need a system CUDA install — the toolkit comes from conda into the env.
- **pixi**: `curl -fsSL https://pixi.sh/install.sh | bash`, then restart your shell.

```bash
git clone <your-repo-url> motionforge && cd motionforge
```

---

## 2. Planner environment (cuRobo)

```bash
pixi install -e planner       # solve + download ROS Jazzy + cu128 torch + toolchain
                              # NOTE: the lock is monolithic, so this also solves `sim`.
pixi run -e planner setup     # check-gpu -> build cuRobo v0.8.0 -> validate a real plan
```

`setup` runs three stages and ends with a real motion plan. Success looks like:

```
torch: 2.11.0+cu128 | device: NVIDIA GeForce RTX 5080
warmup: ~15s (one-time)
plan_pose: success=True  solve_total_time=~30 ms
curobov2 OK
```

`build-curobo` clones cuRobo into `external/` and compiles its kernels for sm_120 — the
first run is slow. If `check-gpu` fails, **stop**: torch isn't running on the GPU (see the
[INSTALL.md](../INSTALL.md) troubleshooting section).

---

## 3. Sim environment (Isaac Sim 6.0)

```bash
pixi install -e sim           # large download (tens of GB)
pixi run -e sim check-sim     # headless launch + renderer smoke on your GPU
```

Success:

```
SimulationApp (headless) up in ~14s
app.update x60 OK
```

Launch the GUI with `pixi run -e sim isaacsim` (needs a display; first run downloads assets).

> Isaac Sim **5.1** crashes on Blackwell + driver 595 (`librtx.scenedb`); **6.0** fixes it —
> which is why the `sim` env pins 6.0 (and Python 3.12).

---

## 4. Daily use

```bash
pixi shell -e planner            # interactive shell; CUDA env is auto-sourced
pixi run -e planner check-curobo # re-validate cuRobo (warmup + a real plan)
pixi run -e planner test-curobo  # safety check: cuRobo's tests, per-file isolated
```

CUDA (`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`, …) is **auto-sourced** on every planner
activation — you never source `cuda_env.sh` by hand. Verify:

```bash
pixi run -e planner bash -c 'echo $CUDA_HOME'   # -> .../.pixi/envs/planner
```

### Task reference

| Task | Env | What it does |
|---|---|---|
| `check-gpu` | planner | torch runs an actual kernel on sm_120 |
| `build-curobo` | planner | compile cuRobo v0.8.0 from source |
| `check-curobo` | planner | warmup + a real `plan_pose` |
| `test-curobo` | planner | cuRobo's tests, **per-file isolated** (`test-curobo <path>` to narrow scope) |
| `setup` | planner | check-gpu → build-curobo → check-curobo |
| `check-sim` | sim | headless Isaac Sim render smoke |
| `isaacsim` | sim | launch the Isaac Sim GUI |

> **Running cuRobo's tests:** always use `test-curobo`, not a bare `pytest` over the whole
> tree. The full suite in one process corrupts a shared CUDA context and cascades into
> thousands of false failures; `test-curobo` runs each file in a fresh process. Files that
> `SKIP` are fine (they need optional deps like `usd-core` that live in the sim env).

---

## 5. Reproduce / verify from scratch

`.pixi/` (the solved env) and `external/` (the cuRobo clone) are disposable and gitignored;
`pixi.lock` is the pin — **never delete it**.

```bash
rm -rf .pixi external/curobo     # keep pixi.lock
pixi install -e planner
pixi run -e planner setup        # re-clones cuRobo + rebuilds + validates
```

Removing `.pixi` alone still requires a `pixi run -e planner build-curobo` afterward — the
editable cuRobo install record lives in `.pixi`. For a true "can a teammate reproduce this?"
check, remove **both** and run `setup`.

---

## More

- [INSTALL.md](../INSTALL.md) — why every version is pinned, plus a troubleshooting catalogue.
- [SPEC.md](../SPEC.md) — the motion-planning pipeline design.
- [CONTEXT.md](../CONTEXT.md) — rationale, caveats, glossary.
