# CONTEXT.md — Design Rationale and Background

> Companion to `SPEC.md`. This file explains *why* the system is shaped the way it is, the tradeoffs behind each decision, the caveats a builder must respect, and the vocabulary. It is not a build instruction; it is the reasoning a reviewer or future maintainer needs.

---

## 1. Problem statement

A robotic bin-picking cell needs motion planning that takes a robot start pose and a set of target poses (grasp, then place), perceives the scene each cycle, and produces collision-free trajectories against the scene, the robot itself, and the held workpiece. Tools and grip configurations change at runtime. The first deployment uses a **structured-light** camera (one capture per phase, 2–6 s capture latency), an **ABB** robot, and rigid parts, but the design must generalize across robot OEMs and extend to live/reactive operation later. The cell already runs production bin picking; this pipeline is the motion-planning layer, with the vision pipeline upstream owning grasp/place pose selection.

## 2. Key decisions and rationale

### 2.1 cuRobo as an embedded library inside a ROS2 node; no MoveIt2
cuRobo is a standalone, GPU-accelerated Python motion-generation library (IK, collision, trajectory optimization, constrained planning). We embed it directly — specifically **v0.8.0 ("curobov2")**, whose flat-architecture rewrite adds a native warp depth→ESDF perception mapper (no external nvblox). cuMotion is its MoveIt2 plugin wrapper.

- **Why not MoveIt2/cuMotion:** our collision world is cuRobo's native fused world (meshes + ESDF voxels + cuboids), our constraints use cuRobo's native pose-cost metric and `plan_grasp`, and our execution is a custom ABB socket adapter, not a MoveIt2 controller. Adopting MoveIt2 would force a second collision representation (its PlanningScene), would not cleanly expose cuRobo's constraint features, and would be bypassed at execution anyway. Net: heavy dependency, little gain.
- **Why ROS2 still:** TF (essential for fixed vs eye-in-hand frame handling and multiple cameras), node lifecycle, inter-module topics/services/actions, parameters, and RViz. ROS2 is the integration fabric; cuRobo is the planning engine inside it.
- **Effort:** this combination is lower-effort and higher-control than MoveIt2 for our custom orchestration, and matches the "ROS2 welcome" stance.

### 2.2 GPU planner (cuRobo) over CPU planners (MoveIt2/OMPL/Tesseract)
A runtime NVIDIA GPU is available. cuRobo formulates collision-free, minimum-jerk trajectory optimization with parallel seeds and generates global motions in tens of milliseconds, far faster than CPU sampling/optimization planners. It also provides exactly the features we need natively: parallel IK, continuous (swept) collision against meshes/voxels/depth, constrained planning, and a path to MPC for the future reactive mode. This makes the < 1 s budget trivial and gives diverse-then-rank essentially for free.

### 2.3 Hybrid collision world: persistent static meshes + per-cycle ESDF voxels
The bin contents shift every cycle, but support structures and conveyor are static. cuRobo composes cuboids, meshes, and voxel ESDF grids in one collision query, so we load the static cell once as meshes and re-perceive the bin (and, separately, the place tray) into per-cycle ESDF `VoxelGrid` layers from base-frame depth. curobov2's native warp mapper builds these (block-sparse TSDF → ESDF) entirely on-GPU — replacing nvblox, which we abandoned because its 2024 fork does not build on the CUDA 12.8 / Blackwell stack. The mapper supports EMA-like decay, giving a clean path to reactive dynamic obstacles later without restructuring.

### 2.4 Segment model with sync barriers (decoupled tool actuation)
Tool actuation does not need true synchronization with arm motion. Instead the task is an ordered list of motion segments, each carrying its own constraints and optional pre/post tool actions with blocking barriers. Example: the arm begins moving toward the pick while the gripper independently moves to a standby width; before the final straight-line approach the coordinator waits for "standby reached"; at the grasp pose it waits for "gripped." The same decoupled pattern covers linear/rotary tool actuation.

- This collapses what could have been coordinated-motion planning into sequencing logic. It unifies tool sync, the orientation-hold-on-carry constraint, collision-aware place, and straight-line approach/retract under one structure.
- The pick-place phase template is a configurable template, not hardcoded waypoints. Paths and target selection within it are computed dynamically (satisfies the "dynamic paths, not fixed waypoints/sequence" requirement). Full task-level planning (TAMP) is out of scope.

### 2.5 Cartesian goal sets with planner-owned IK (not pre-solved joint goals)
A joint goal is unambiguous, but the targets are natively Cartesian (vision emits TCP poses) and pre-solving IK to one configuration would lock an arm branch that may be suboptimal or colliding. Feeding the planner the Cartesian goal lets it enumerate IK branches and pick the branch yielding the cheapest collision-free path. The grasp/place orientation (including yaw) is exact; the only configurable freedom is the *approach direction* (usually tool +Z, but allowed off-axis), which is a constraint-frame choice, not a goal relaxation. All relaxation/symmetry handling lives upstream in vision.

### 2.6 N-diverse-then-rank, not optimize-one
Optimizing a single seed risks local minima (wrong homotopy class, e.g. wrong side of an obstacle) with no fallback. Evaluating many seeds in parallel and ranking escapes local minima, yields a ranked set for fallback, and supports the ranked-candidate requirement. On GPU the extra compute is negligible, and it is how cuRobo already operates. Candidate priority follows the vision ranking; within a candidate, the planner returns the best seed.

### 2.7 Execution adapter; ABB raw socket for MVP (no EGM)
"Planner as master" with a fixed, OEM-agnostic Execution Adapter boundary. The MVP robot has no EGM option, so the adapter v1 uses a raw TCP socket to a RAPID server.

- **What raw socket gives:** waypoint following with controller-side interpolation. This is the long-proven ROS-Industrial `abb_driver` pattern, explicitly deemed sufficient for pick-and-place and relatively slow motion.
- **The critical detail:** the RAPID server must buffer several points ahead and use corner zones so its look-ahead blends corners. Sending points one-at-a-time blocking makes the robot decelerate to every point (stutter). The ring buffer is the heart of the adapter.
- **What it gives up vs EGM:** ownership of time-parameterization, high-rate reference tracking, and smooth mid-motion replanning. None are needed for static one-shot picks.
- **Later:** EGM (250 Hz reference streaming, owns timing) or ros2_control adapters drop in behind the same boundary when reactive/dynamic operation or precise timing is needed. The existing asyncio TCP socket manager (fixed-header framing, decorator service registration) is reused on the PC side.

### 2.8 Place-side re-perception every cycle
Vision guarantees a collision-free *final place pose*, not a collision-free *path*, and placed parts tend to roll from their intended pose. So the tray is re-perceived each cycle into its own ESDF `VoxelGrid` layer (native warp mapper), and the planner owns the place-approach collision against that fresh layer plus the held object. Place candidates are a ranked set (often length 1), with the same fallback ladder as grasp.

### 2.9 Robot base frame as the canonical frame
There are pick, place, and possibly auxiliary cameras, sometimes arm-mounted for both phases. Normalizing everything to the robot base frame keeps one consistent planning frame regardless of how many cameras exist or how they are mounted. The Perception Frame Adapter handles fixed extrinsics and eye-in-hand (extrinsic × FK at capture).

### 2.10 Declarative cell config — scene as data, not code
For a package deployed across multiple cells, the static scene (cell geometry, camera extrinsics, robot mount, bin/tray ROIs, tool library) must be **data, not code**. Originally these were assembled in code, so a new layout required coding-level work — the actual deployability gap (Isaac Sim being "optional validation" was never the issue). The fix is a declarative per-deployment `CellConfig` artifact, loaded once at startup (SPEC §4, §5.8).

- **Adapter-boundary philosophy, reused.** This mirrors the Execution Adapter (§2.7): future scene-source adapters (touch-probe, scan, CAD import) sit behind one common base-frame representation (`CellConfig` / `CollisionBody`), so the planner is agnostic to how geometry was captured. A cell can mix sources by region.
- **Why YAML manifest + asset refs (not USD-only).** Keep the semantic, safety-relevant fields (camera roles, ROIs, transforms) human-diffable and runtime-loadable today; let geometry assets (mesh/USD) come from any 3D editor. A pure-USD scene is heavier, not git-diffable, and awkward for that metadata.
- **The editor is reused, not built.** A bespoke 3D editor is *more* code, not less. Isaac/Blender/FreeCAD already import meshes, create primitives, and place them with gizmos — they author the cell config via the asset format. This is Isaac's concrete *optional* commissioning role; it never becomes mandatory (footprint: ~tens of GB + RTX + a separate env per site).
- **Alignment invariant.** The one thing present and trustworthy at every site is the robot's FK / base frame, so **TCP touch-registration** is the single universal mechanism for aligning any non-base-native source to the base. Calibration is not "extra": EIH runtime perception already owes hand-eye calibration, so a commissioning scan reuses it — the real limit is camera *coverage*, not calibration.

## 3. Caveats and risks (must respect)

- **Dense clutter is an open problem for voxel-based collision.** cuRobo's own docs state camera-perceived collision-free planning works well in *sparse* environments. A bin of intertwined parts is dense. Division of labor: vision owns "can I reach into the clutter" (it supplies a reachable grasp + standoff); the planner owns gross free-space motion plus approach/retract-line clearance against the cell and the bin's outer geometry. Do not expect the planner to thread dense clutter from voxels alone.
- **Software collision avoidance is not functional safety.** Any shared-workspace or dynamic-scene operation requires safety-rated hardware (safety scanners, safety-rated monitored stop). EGM in particular bypasses the controller's own collision avoidance, so the planner must own collision when EGM is later used.
- **Static-world coverage is a safety concern.** Free space must be *positively established, not assumed*: a structural obstacle missing from the `CellConfig` static world and outside the per-cycle ESDF ROIs is treated as free space, and the arm will route straight through it. Unmodeled regions should flag/block, not default to free. Human measurement in a 3D editor is exactly where this error enters, which is why the coverage-verification gate (committed world overlaid against a fresh empty-cell capture; flag missed and phantom obstacles) is the net. Online robot jogging stays on the teach pendant — the GUI's online role is visualization, not commanding.
- **Perception mapping is per-cycle.** Re-integrate the native warp mapper each cycle (clear / re-integrate, or move the ESDF origin for a sliding window). Unlike the old nvblox path there is no in-process re-init CUDA quirk — it is native warp. Still validate the mapper + planner integration early on the target GPU.
- **Constraints are soft costs.** curobov2's pose-cost metric (the held-orientation/partial-pose term) is added as a cost, so trajectories will not land exactly on the offset pose and orientation hold is approximate; tune weights. Linear approach is easiest axis-aligned; off-axis approach requires aligning the constraint reference frame. The grasp approach itself is encapsulated by `plan_grasp`.
- **Socket timing fidelity.** Waypoint-following timing is owned by the RAPID interpolator, not the planner. Acceptable for one-shot static picks; revisit when precise timing or replanning is needed.
- **Deformable rescan completeness deferred.** A post-grasp rescan of a held part may miss occluded regions, leaving collision gaps. MVP uses rigid-mesh attach; deformable handling is a later capability with this known limitation flagged.

## 4. Extensibility roadmap

1. **Reactive / live operation (stereo):** add a local layer mirroring the mobile-robot global+local split — cuRobo MPC (`ModelPredictiveControl`, public in curobov2) re-optimizing against a decaying ESDF dynamic layer (the native mapper's EMA decay) fed at sensor rate. Global plan + local correction.
2. **Execution interfaces:** EGM adapter (planner owns 250 Hz reference), ros2_control adapter, EtherNet/IP joint read.
3. **Extra DOF:** rail/turntable/7-DOF added as joints in the planned chain (config-only for planning); the real cost is coordinated real-time execution across arm and external-axis drives, especially across separate controllers.
4. **Learning-based:** learned cost/sampler or a VLA terminal manipulation skill for the in-clutter grasp, with cuRobo/classical validity as the gatekeeper on any learned output.
5. **Tool-change motion** generation with off-cycle mating-point calibration.
6. **Multi-arm / multi-robot** shared-workspace collision.
7. **Commissioning / scene capture:** scene-source adapters (TCP touch-probe, EIH/fixed scan, CAD import) + a reused 3D editor + the coverage-verification gate, all behind the `CellConfig` boundary (the data model itself ships first).

## 5. Glossary

- **TCP** — Tool Center Point; the controlled frame at the tool, updated when the tool changes.
- **EIH / ETH** — Eye-in-hand (camera on the arm/flange) / Eye-to-hand (fixed camera).
- **SL** — Structured light; one capture per phase, 2–6 s latency, dense and accurate.
- **ESDF** — Euclidean Signed Distance Field; per-voxel distance to nearest obstacle, used for collision cost and gradients.
- **Native ESDF mapper** — curobov2's on-GPU warp pipeline (`curobo._src.perception.mapper`) building block-sparse TSDF → ESDF from depth; replaces nvblox (not used here — its 2024 fork won't build on CUDA 12.8). **nvblox** itself is NVIDIA's standalone GPU TSDF/ESDF library; viable only via its Docker, as a ROS node (isaac_ros_nvblox), or a newer pairing — none of which fit our in-process embed.
- **Pose-cost metric / `plan_grasp`** — curobov2's constrained-planning mechanisms. The pose-cost metric (`offset_position` / `tstep_fraction` / `linear_axis`) locks chosen linear/angular axes (e.g. hold orientation). `plan_grasp` fuses pre-grasp offset + linear approach + grasp + lift into one call over a `GoalToolPose` goalset of candidates.
- **Homotopy class** — a family of paths that can be continuously deformed into each other without crossing an obstacle; "different ways around" an obstacle. Diverse seeds explore different classes.
- **Manipulability** — Yoshikawa index √det(J·Jᵀ); how far a configuration is from a singularity. Low manipulability means a given Cartesian velocity needs large joint velocities. Distinct from smoothness (jerk), which does not minimize joint speed.
- **EGM** — ABB Externally Guided Motion; UDP/protobuf real-time reference streaming at ~250 Hz. Provides no path planning and bypasses controller safety. Optional licensed feature (absent on the MVP robot).
- **RWS** — ABB Robot Web Services; REST/WebSocket for state, I/O, program control (not real-time motion).
- **EtherNet/IP (EIP)** — industrial fieldbus; a possible future channel for joint feedback from the controller.
- **TAMP** — Task and Motion Planning; jointly planning the discrete task sequence and the motions. Out of scope; we use a fixed pick-place template.
- **MoveIt2 / cuMotion** — ROS2 motion-planning framework / NVIDIA's cuRobo-backed MoveIt2 plugin. Deliberately not used; cuRobo is embedded directly.
- **Cell config** — the per-deployment declarative artifact (`CellConfig`, SPEC §4): a git-diffable YAML manifest (camera registry, robot mount, ROIs, tool library, static-body transforms) plus referenced geometry assets. The unit of deployment and the common base-frame representation all scene-source adapters emit into.
- **Scene-source adapter** — a pluggable producer of static-world geometry (reused 3D editor, TCP touch-probe, scan, CAD import) behind a fixed boundary, all emitting `CollisionBody` in the robot base frame. Same pattern as the Execution Adapter. Deferred.
- **TCP touch-registration** — using the robot's own FK as a metrology tool: jog the TCP to touch known points/features and record configurations to fit primitives or align imported geometry to the base frame. The universal alignment backstop because FK is the one invariant across varied sites.
- **USD** — Universal Scene Description; Pixar/NVIDIA's 3D scene/interchange format, native to Isaac Sim/Omniverse and supported by cuRobo's USD helpers. Here it is one option for the geometry *assets* a cell config references, and the bridge that lets Isaac act as an optional cell-config editor.

## 6. References

- cuRobo v0.8.0 / "curobov2": https://curobo.org and https://github.com/NVlabs/curobo (motion generation, constrained planning, world collision, native warp depth→ESDF perception mapper).
- ABB EGM application manual (3HAC073318/073319) and ROS-Industrial `abb_driver` (socket/RAPID) vs `abb_robot_driver` (EGM).
- Abuelsamen et al., "Industrial Robot Motion Planning with GPUs: Integration of cuRobo for Extended DOF Systems," arXiv:2508.04146 — extended-DOF integration, MPC, constrained orientation, pick-place benchmarking.
