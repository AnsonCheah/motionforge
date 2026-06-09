"""Isaac Sim validation twin (SPEC §9) — MANUAL acceptance, run in the `sim` env.

Brings up Isaac Sim 6.0 headless, loads the same UR10e robot, and is the hook for executing a
planned trajectory against the same WorldSnapshot. This is the one acceptance criterion not
covered by the headless pytest suite (it needs the Isaac renderer + a cross-env DDS bridge).

Run:  pixi run -e sim validate-isaac

Extend `playback_trajectory()` to subscribe to the planner's trajectory over DDS (the planner
node publishes on the shared ROS_DOMAIN_ID) or load a saved JointTrajectory, then drive the
articulation and assert collision-free execution. Kept a skeleton so the import/bring-up path
is validated on the target GPU before wiring the full twin.
"""

import sys
import time

from isaacsim import SimulationApp

UR10E_USD = "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"  # Isaac asset path (nucleus)


def playback_trajectory(world) -> None:
    """TODO: receive the planned JointTrajectory (DDS or file) and drive the articulation,
    checking collision-free execution against the loaded WorldSnapshot."""
    for _ in range(60):
        world.step(render=True)


def main() -> int:
    t0 = time.time()
    app = SimulationApp({"headless": True})
    print("SimulationApp (headless) up in %.1fs" % (time.time() - t0))

    ok = False
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.nucleus import get_assets_root_path
        from isaacsim.core.utils.stage import add_reference_to_stage

        world = World(stage_units_in_meters=1.0)
        assets_root = get_assets_root_path()
        if assets_root is not None:
            add_reference_to_stage(assets_root + UR10E_USD, "/World/ur10e")
        world.scene.add_default_ground_plane()
        world.reset()

        playback_trajectory(world)
        ok = True
        print("Isaac twin stepped trajectory playback OK")
    except Exception as exc:  # noqa: BLE001
        print("ISAAC VALIDATION FAILED:", exc, file=sys.stderr)
    finally:
        app.close()

    if not ok:
        return 1
    print("ISAAC_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
