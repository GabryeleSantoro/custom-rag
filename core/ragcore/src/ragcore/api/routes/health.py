from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ragcore import __version__
from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.schemas import Health, ProcessStatus, StreamEnvelope

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
def get_health(request: Request, config: ConfigDep, store: StoreDep) -> Health:
    uptime = time.monotonic() - request.app.state.started_at

    # ragcore reports only what it can see. The Rust shell owns the model
    # servers and merges its own view of them into what the UI renders.
    processes = [
        ProcessStatus(
            name="ragcore",
            role="ragcore",
            state="ready",
            port=config.port,
            uptime_s=round(uptime, 1),
            detail="serving fixture data",
        ),
    ]

    return Health(
        status="ok",
        version=__version__,
        dev_mode=config.dev_mode,
        uptime_s=round(uptime, 1),
        stub=True,
        processes=processes,
        index=store.index_stats(),
        models_ready=True,
    )


@router.get("/schema/events", response_model=StreamEnvelope)
def event_schema() -> StreamEnvelope:
    """Shape reference for the SSE routes. Always empty; the schema is the point."""
    return StreamEnvelope()
