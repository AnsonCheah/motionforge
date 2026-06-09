"""Unit tests for the fixed-header TCP framing (SPEC §5.6)."""

from motionforge.execution.framing import (
    FrameDecoder,
    MessageType,
    decode_json,
    encode_json,
)


def test_roundtrip_single_frame():
    blob = encode_json(MessageType.WAYPOINT, {"q": [1.0, 2.0], "speed": 0.5})
    dec = FrameDecoder()
    dec.feed(blob)
    frames = list(dec)
    assert len(frames) == 1
    mtype, payload = frames[0]
    assert mtype == MessageType.WAYPOINT
    assert decode_json(payload) == {"q": [1.0, 2.0], "speed": 0.5}


def test_multiple_frames_in_one_feed():
    blob = encode_json(MessageType.FEEDBACK, {"q": [0]}) + encode_json(MessageType.DONE, {})
    dec = FrameDecoder()
    dec.feed(blob)
    frames = list(dec)
    assert [m for m, _ in frames] == [MessageType.FEEDBACK, MessageType.DONE]


def test_partial_feed_waits_for_completion():
    blob = encode_json(MessageType.CONSUMED, {"index": 3, "q": [9.0]})
    dec = FrameDecoder()
    dec.feed(blob[:4])  # header not yet complete
    assert list(dec) == []
    dec.feed(blob[4:])  # rest arrives
    frames = list(dec)
    assert len(frames) == 1
    assert decode_json(frames[0][1]) == {"index": 3, "q": [9.0]}


def test_empty_payload_decodes_to_empty_dict():
    dec = FrameDecoder()
    dec.feed(encode_json(MessageType.STOP, {}))
    (_, payload), = list(dec)
    assert decode_json(payload) == {}
