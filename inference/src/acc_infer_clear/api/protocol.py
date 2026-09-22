"""Public NDJSON request/response definitions, independent of model execution.

open: id, optional seed/emotion; text: id,text; end/cancel/release: id;
run/tick/drain: no request fields. Audio responses use signed PCM16 LE base64.
The CLI dispatches these messages sequentially to the runtime owner.
"""
import base64
import json
import math

FIELDS = {
    "open": ({"op", "id"}, {"seed", "emotion"}),
    "text": ({"op", "id", "text"}, set()),
    "end": ({"op", "id"}, set()),
    "cancel": ({"op", "id"}, set()),
    "release": ({"op", "id"}, set()),
    "run": ({"op"}, set()),
    "tick": ({"op"}, set()),
    "drain": ({"op"}, set()),
}


def parse_event(line):
    event = json.loads(line)
    if not isinstance(event, dict) or event.get("op") not in FIELDS:
        raise ValueError("Expected an object with a supported op")
    required, optional = FIELDS[event["op"]]
    if not required <= event.keys() or set(event) - required - optional:
        raise ValueError(f"Invalid fields for {event['op']}")
    if "id" in event and (not isinstance(event["id"], str) or not event["id"]):
        raise ValueError("Request id must be a nonempty string")
    if "text" in event and not isinstance(event["text"], str):
        raise ValueError("Text delta must be a string")
    if "seed" in event and (isinstance(event["seed"], bool) or not isinstance(event["seed"], int)
                            or not 0 <= event["seed"] < 2**63):
        raise ValueError("Seed must be an integer in [0, 2**63)")
    if "emotion" in event:
        value = event["emotion"]
        if (not isinstance(value, list) or len(value) != 8
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
            raise ValueError("Emotion must contain eight finite numbers")
    return event


def audio_event(event):
    chunk = event["chunk"]
    pcm = chunk["pcm"].astype("<i2", copy=False).tobytes()
    return dict(type="audio", id=event["request_id"], index=chunk["index"],
                sample_rate=22050, sample_start=chunk["sample_start"],
                sample_end=chunk["sample_end"], pcm_s16le=base64.b64encode(pcm).decode(),
                complete=event["complete"])
