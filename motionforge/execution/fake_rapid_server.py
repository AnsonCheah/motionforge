"""In-process fake RAPID socket server (SPEC §5.6) for tests — no robot needed.

Models the controller side: a ring buffer fed by the PC adapter, a motion task that consumes
one waypoint per ``consume_dt`` (simulating ``MoveAbsJ`` with zones), and joint feedback on the
same channel. Emits a ``CONSUMED`` credit per waypoint so the adapter keeps the buffer filled
ahead (look-ahead) instead of stalling at every point. Records ``max_buffer_depth`` so a test
can prove the look-ahead actually happened.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence

from motionforge.execution._loop import LoopThread
from motionforge.execution.framing import (
    FrameDecoder,
    MessageType,
    decode_json,
    encode_json,
)


class FakeRapidServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        home_q: Optional[Sequence[float]] = None,
        consume_dt: float = 0.01,
    ) -> None:
        self.host = host
        self.port = port
        self._home_q = list(home_q) if home_q is not None else [0.0] * 6
        self._consume_dt = consume_dt

        self.current_q: List[float] = list(self._home_q)
        self.received_count = 0
        self.consumed_count = 0
        self.max_buffer_depth = 0
        self.received_dts: List[float] = []  # per-waypoint planned dt_s, in consume order

        self._lt: Optional[LoopThread] = None
        self._server: Optional[asyncio.AbstractServer] = None

    def start(self) -> int:
        self._lt = LoopThread()

        async def _start():
            self._server = await asyncio.start_server(self._handle, self.host, self.port)
            self.port = self._server.sockets[0].getsockname()[1]

        self._lt.run(_start())
        return self.port

    def stop(self) -> None:
        if self._lt is None:
            return

        async def _stop():
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()

        try:
            self._lt.run(_stop(), timeout=2.0)
        except Exception:
            pass
        self._lt.stop()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        dec = FrameDecoder()
        buffer: asyncio.Queue = asyncio.Queue()
        stream_done = asyncio.Event()

        # Report q0 on connect (the Joint State Source start config).
        writer.write(encode_json(MessageType.FEEDBACK, {"q": self.current_q}))
        await writer.drain()

        consumer = asyncio.create_task(self._consume(buffer, writer, stream_done))
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                dec.feed(data)
                for mtype, payload in dec:
                    obj = decode_json(payload)
                    if mtype == MessageType.WAYPOINT:
                        await buffer.put(obj)
                        self.received_count += 1
                        self.max_buffer_depth = max(self.max_buffer_depth, buffer.qsize())
                    elif mtype == MessageType.DONE:
                        stream_done.set()
                    elif mtype == MessageType.STOP:
                        while not buffer.empty():
                            buffer.get_nowait()
                        stream_done.set()
        finally:
            stream_done.set()
            await consumer
            writer.close()

    async def _consume(
        self, buffer: asyncio.Queue, writer: asyncio.StreamWriter, stream_done: asyncio.Event
    ) -> None:
        # Let the adapter's initial burst fill the look-ahead buffer before motion starts.
        await asyncio.sleep(self._consume_dt)
        while True:
            if buffer.empty():
                if stream_done.is_set():
                    return
                await asyncio.sleep(self._consume_dt)
                continue
            wp = buffer.get_nowait()
            # Honor the planned per-waypoint timing when supplied (fall back to the fixed rate
            # for legacy frames without dt_s).
            dt = wp.get("dt_s")
            self.received_dts.append(float(dt) if dt is not None else self._consume_dt)
            await asyncio.sleep(dt if dt else self._consume_dt)  # simulate motion to the waypoint
            self.current_q = list(wp["q"])
            self.consumed_count += 1
            writer.write(
                encode_json(MessageType.CONSUMED, {"index": wp.get("index", -1), "q": self.current_q})
            )
            await writer.drain()
