"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragcore import __version__
from ragcore.api.routes import (
    chats,
    connections,
    documents,
    evals,
    health,
    jobs,
    models,
    query,
    settings,
    sources,
)
from ragcore.config import Config
from ragcore.stub.hub import HubClient
from ragcore.stub.jobs import JobManager
from ragcore.stub.store import Store

# Everything else needs the session token the Rust shell generated at spawn.
PUBLIC_PATHS = {"/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.started_at = time.monotonic()
        app.state.store = Store(config)
        app.state.jobs = JobManager()
        app.state.hub = HubClient(config)
        try:
            yield
        finally:
            await app.state.jobs.shutdown()
            await app.state.hub.aclose()

    app = FastAPI(
        title="ragcore",
        version=__version__,
        summary="Local RAG core: ingestion, retrieval and grounded generation.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if config.token and request.url.path not in PUBLIC_PATHS:
            header = request.headers.get("authorization", "")
            presented = header.removeprefix("Bearer ").strip()
            if presented != config.token:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    for router in (
        health.router,
        sources.router,
        documents.router,
        jobs.router,
        query.router,
        chats.router,
        connections.router,
        models.router,
        settings.router,
        evals.router,
    ):
        app.include_router(router)

    return app
