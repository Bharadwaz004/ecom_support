"""Fan-out bus for MCP protocol trace events.

emit() is synchronous and never awaits, so instrumenting a code path cannot change its
timing or introduce a suspension point. Each viewer gets its own bounded queue; a viewer
that stops draining loses events rather than applying backpressure to the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
from typing import Any, AsyncIterator, Literal

# Roughly a few seconds of a busy trace. Past this a viewer is not keeping up and the
# useful thing to preserve is the newest events, not the oldest.
MAX_QUEUE = 250
PREVIEW_CHARS = 600

Direction = Literal["out", "in"]
Kind = Literal["initialize", "tools/list", "tools/call", "result", "llm", "error"]

_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
_sequence = itertools.count(1)
_dropped = 0


def preview(value: Any, limit: int = PREVIEW_CHARS) -> Any:
    """Shorten a payload for the wire. Strings are cut; everything else is JSON-cut."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [+{len(value) - limit} chars]"
    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)[:limit]
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + f"... [+{len(encoded) - limit} chars]"


def emit(
    direction: Direction,
    kind: Kind,
    payload: Any,
    ms: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Push one event to every viewer. Returns the event so callers can log it too."""
    global _dropped
    event = {
        "seq": next(_sequence),
        "ts": time.time(),
        "direction": direction,
        "kind": kind,
        "payload": payload,
        "ms": round(ms, 1) if ms is not None else None,
        "req": request_id,
    }
    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            _dropped += 1  # this viewer only; other viewers and the agent are unaffected
    return event


@contextlib.asynccontextmanager
async def subscribe() -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
    """Register a viewer queue for the lifetime of the block."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_QUEUE)
    _subscribers.add(queue)
    try:
        yield queue
    finally:
        _subscribers.discard(queue)


def stats() -> dict[str, int]:
    return {"viewers": len(_subscribers), "dropped": _dropped}
