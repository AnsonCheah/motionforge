"""Fixed-header TCP framing for the PC↔RAPID socket (SPEC §5.6).

Frame = 6-byte header ``>I H`` (payload length, message type) + JSON payload. A streaming
:class:`FrameDecoder` accumulates bytes and yields complete frames. Pure stdlib — no GPU.
"""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Iterator, Tuple

_HEADER = struct.Struct(">IH")  # payload length (uint32), message type (uint16)


class MessageType(IntEnum):
    WAYPOINT = 1   # client→server: {"q":[...], "speed":float, "zone":float, "index":int}
    CONSUMED = 2   # server→client: {"index":int, "q":[...]}  (motion advanced one waypoint = a credit)
    FEEDBACK = 3   # server→client: {"q":[...]}               (current joint state, incl. q0 on connect)
    DONE = 4       # client→server: {}                        (end of trajectory stream)
    STOP = 5       # client→server: {}                        (abort)


def encode_frame(msg_type: MessageType, payload: bytes = b"") -> bytes:
    return _HEADER.pack(len(payload), int(msg_type)) + payload


def encode_json(msg_type: MessageType, obj) -> bytes:
    return encode_frame(msg_type, json.dumps(obj).encode("utf-8"))


def decode_json(payload: bytes):
    return json.loads(payload.decode("utf-8")) if payload else {}


class FrameDecoder:
    """Accumulates bytes; iterate to drain all currently-complete frames."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def __iter__(self) -> Iterator[Tuple[MessageType, bytes]]:
        while True:
            if len(self._buf) < _HEADER.size:
                return
            length, mtype = _HEADER.unpack_from(self._buf, 0)
            total = _HEADER.size + length
            if len(self._buf) < total:
                return
            payload = bytes(self._buf[_HEADER.size:total])
            del self._buf[:total]
            yield MessageType(mtype), payload
