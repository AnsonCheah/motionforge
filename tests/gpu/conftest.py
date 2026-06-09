"""Shared fixtures for GPU (cuRobo) tests.

Imports are guarded so this collects cleanly on a machine without torch/curobo (the whole
module is skipped). All GPU tests require CUDA and are marked ``gpu``.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from motionforge.config import DEFAULTS  # noqa: E402
from motionforge.planner import MotionPlannerAdapter  # noqa: E402

# A faster planner config for tests: skip CUDA-graph capture (~14 s) and trim warmup.
TEST_CONFIG = replace(
    DEFAULTS,
    use_cuda_graph=False,
    num_ik_seeds=16,
    num_trajopt_seeds=2,
    warmup_iterations=2,
)


@pytest.fixture(scope="session")
def mf_planner():
    """A warmed UR10e MotionPlannerAdapter shared across GPU tests (free space)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    adapter = MotionPlannerAdapter(config=TEST_CONFIG)
    adapter.warmup()
    return adapter
