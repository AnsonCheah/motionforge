"""ABB raw-socket Execution Adapter (SPEC §5.6) — the MVP planner-as-master executor.

Down-samples the dense trajectory, then streams waypoints into the RAPID ring buffer keeping
``waypoint_buffer_depth`` points buffered ahead (credit-based flow control: the controller
emits a ``CONSUMED`` credit per executed waypoint). This keeps the controller's look-ahead full
so it blends corners with ``zonedata`` instead of decelerating to every point. Also serves as
the MVP Joint State Source (SPEC §5.7), reporting feedback ``q``.

Runs its asyncio socket on a private loop thread so the public API stays synchronous.
"""

from __future__ import annotations

from typing import List, Optional

from motionforge.config import DEFAULTS, Config
from motionforge.execution._loop import LoopThread
from motionforge.execution.adapter import ExecutionAdapter, trajectory_to_waypoints
from motionforge.execution.framing import (
    FrameDecoder,
    MessageType,
    decode_json,
    encode_json,
)
from motionforge.execution.downsample import downsample_waypoints
from motionforge.joint_state import JointStateSource
from motionforge.types import JointTrajectory


class AbbSocketAdapter(ExecutionAdapter, JointStateSource):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11000,
        config: Config = DEFAULTS,
        max_joint_error: float = 0.05,
        speed: float = 1.0,
        zone: float = 0.01,
    ) -> None:
        self._host = host
        self._port = port
        self._buffer_depth = config.waypoint_buffer_depth
        self._max_joint_error = max_joint_error
        self._speed = speed
        self._zone = zone

        self._lt: Optional[LoopThread] = None
        self._reader = None
        self._writer = None
        self._reader_task = None
        self._dec = FrameDecoder()

        self._latest_q: Optional[List[float]] = None
        self._q_event = None
        self._credits = None
        self._expected: Optional[int] = None
        self._consumed = 0
        self._done_event = None

    # -- lifecycle --

    def connect(self) -> List[float]:
        """Connect and return the controller's start config ``q0``."""
        import asyncio

        self._lt = LoopThread()

        async def _aconnect():
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
            self._q_event = asyncio.Event()
            self._credits = asyncio.Semaphore(self._buffer_depth)
            self._reader_task = asyncio.create_task(self._read_loop())
            await asyncio.wait_for(self._q_event.wait(), timeout=5.0)

        self._lt.run(_aconnect())
        return self.read_joint_state()

    def close(self) -> None:
        if self._lt is None:
            return

        async def _aclose():
            if self._reader_task is not None:
                self._reader_task.cancel()
            if self._writer is not None:
                self._writer.close()

        try:
            self._lt.run(_aclose(), timeout=2.0)
        except Exception:
            pass
        self._lt.stop()
        self._lt = None

    # -- ExecutionAdapter --

    def send_trajectory(self, traj: JointTrajectory):
        return self._lt.run(self._asend(traj), timeout=30.0)

    def read_joint_state(self) -> List[float]:
        return list(self._latest_q) if self._latest_q is not None else []

    def stop(self) -> None:
        self._lt.run(self._asend_simple(MessageType.STOP), timeout=5.0)

    # -- async internals --

    async def _read_loop(self) -> None:
        while True:
            data = await self._reader.read(4096)
            if not data:
                break
            self._dec.feed(data)
            for mtype, payload in self._dec:
                obj = decode_json(payload)
                if mtype == MessageType.FEEDBACK:
                    self._latest_q = obj["q"]
                    if self._q_event is not None:
                        self._q_event.set()
                elif mtype == MessageType.CONSUMED:
                    self._latest_q = obj["q"]
                    self._consumed += 1
                    if self._credits is not None:
                        self._credits.release()
                    if (
                        self._expected is not None
                        and self._consumed >= self._expected
                        and self._done_event is not None
                    ):
                        self._done_event.set()

    async def _asend(self, traj: JointTrajectory):
        import asyncio

        positions = [list(p[0]) for p in traj.points]
        kept = downsample_waypoints(positions, self._max_joint_error)
        waypoints = trajectory_to_waypoints(traj, kept, self._speed, self._zone)

        self._expected = len(waypoints)
        self._consumed = 0
        self._done_event = asyncio.Event()

        for idx, wp in enumerate(waypoints):
            await self._credits.acquire()  # keep <= buffer_depth points in the ring buffer
            self._writer.write(
                encode_json(
                    MessageType.WAYPOINT,
                    {"q": wp.q, "speed": wp.speed, "zone": wp.zone, "dt_s": wp.dt_s, "index": idx},
                )
            )
            await self._writer.drain()

        self._writer.write(encode_json(MessageType.DONE, {}))
        await self._writer.drain()
        if self._expected > 0:
            await self._done_event.wait()
        return {"waypoints": len(waypoints), "kept_indices": kept}

    async def _asend_simple(self, mtype: MessageType) -> None:
        self._writer.write(encode_json(mtype, {}))
        await self._writer.drain()
