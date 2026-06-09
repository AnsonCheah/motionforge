"""Phase 5 GPU acceptance test (SPEC §9): full pick-and-place on UR10e, headless.

Real MotionPlanner + CollisionWorldManager against a pick_bin ESDF layer + re-perceived
place_tray ESDF layer, with fake perception/gripper/execution. Asserts a collision-free cycle
to DONE and end-to-end planning under the 1 s budget.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from curobo._src.geom.types import VoxelGrid  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402

from motionforge.collision import CollisionWorldManager  # noqa: E402
from motionforge.coordinator import CoordinatorState, TaskCoordinator  # noqa: E402
from motionforge.coordinator.fakes import FakeGripper, RecordingExecution, ScriptedPerception  # noqa: E402
from motionforge.coordinator.interfaces import PickPerception, PlacePerception  # noqa: E402
from motionforge.joint_state import FakeJointStateSource  # noqa: E402
from motionforge.planner import MotionPlannerAdapter  # noqa: E402
from motionforge.tools import ToolManager, parallel_jaw_geom_fn  # noqa: E402
from motionforge.types import GraspCandidate, GripConfig, PlaceCandidate, ToolDescriptor  # noqa: E402
from motionforge.geometry import Pose  # noqa: E402

from tests.gpu.conftest import TEST_CONFIG  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DIMS = (1.0, 1.0, 1.0)
VS = 0.02
CENTER = (0.5, 0.0, 0.3)


def _empty_grid():
    n = [round(d / VS) for d in DIMS]
    feat = torch.full(tuple(n), 1.0, dtype=torch.float16, device="cuda:0")
    return VoxelGrid(name="esdf", pose=[*CENTER, 1, 0, 0, 0], dims=list(DIMS), voxel_size=VS,
                     feature_tensor=feat, feature_dtype=torch.float16)


@pytest.fixture(scope="module")
def e2e():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    adapter = MotionPlannerAdapter(
        config=TEST_CONFIG, scene=Scene(voxel=[_empty_grid()]), attached_object_spheres=64
    )
    adapter.warmup()
    return adapter


def _from_config(adapter, deltas):
    q = np.array(adapter.default_q0, dtype=float)
    q[: len(deltas)] += np.array(deltas, dtype=float)
    return adapter.tcp_pose_at(q.tolist())


def test_full_pick_and_place_cycle(e2e):
    adapter = e2e
    world = CollisionWorldManager(adapter)
    tools = ToolManager()
    tools.register(
        ToolDescriptor("jaw", Pose([0, 0, 0.0], [1, 0, 0, 0]), parallel_jaw_geom_fn(),
                       actuation_iface="socket://gripper", payload_kg=0.5)
    )

    grasp_pose = _from_config(adapter, [0.25, -0.25, 0.2])
    place_pose = _from_config(adapter, [-0.25, -0.25, 0.2])
    # Top-down approach: approach_axis points along the motion into the target (downward).
    grasp = GraspCandidate(grasp_pose, [0, 0, -1], 0.05, "jaw", GripConfig(0.02, force=30.0))
    place = PlaceCandidate(place_pose, [0, 0, -1], 0.05, "jaw", GripConfig(0.08, mode="outward"))
    workpiece = Cuboid(name="part", pose=[*grasp_pose.position.tolist(), 1, 0, 0, 0],
                       dims=[0.04, 0.04, 0.04])

    perception = ScriptedPerception(
        picks=[PickPerception([grasp], bin_voxels=_empty_grid(), workpiece=workpiece)],
        places=[PlacePerception([place], tray_voxels=_empty_grid())],
    )
    execution = RecordingExecution()
    coord = TaskCoordinator(
        planner=adapter, world=world, tools=tools, perception=perception,
        gripper=FakeGripper(), execution=execution,
        joint_state_source=FakeJointStateSource(adapter.default_q0), config=TEST_CONFIG,
    )

    result = coord.run_cycle()

    assert result.success, f"cycle failed in {result.state} ({result.fault_reason})"
    assert result.state == CoordinatorState.DONE
    assert len(execution.sent) == 6  # approach, grasp, lift, transport, place, retract
    # SPEC §9: end-to-end planning call < 1 s (steady-state, post-warmup).
    assert result.max_plan_time_s < 1.0
    # Held object was attached then detached over the cycle.
    assert adapter.attached_link_name is None
