"""Server-sent events.

The Rust proxy parses these frames and forwards them to the UI over a Tauri
Channel, so the wire format has to stay boringly standard: `event:` + `data:`
with a single JSON object per frame.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Proxies that buffer would defeat the whole point of streaming.
    "X-Accel-Buffering": "no",
}


def frame(event: str, data: Any) -> str:
    if isinstance(data, BaseModel):
        payload = data.model_dump_json()
    else:
        payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)
