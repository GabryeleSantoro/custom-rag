from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import JobsDep
from ragcore.api.schemas import Job, Ok
from ragcore.api.sse import frame, sse_response

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Without traffic, an idle SSE connection looks indistinguishable from a dead
# one to anything in between. A comment frame keeps it demonstrably alive.
HEARTBEAT_S = 15.0


@router.get("", response_model=list[Job])
def list_jobs(jobs: JobsDep) -> list[Job]:
    return list(jobs.jobs.values())


@router.get("/stream")
async def stream_jobs(jobs: JobsDep):
    async def events() -> AsyncIterator[str]:
        subscription = jobs.subscribe()
        try:
            while True:
                try:
                    job = await asyncio.wait_for(anext(subscription), timeout=HEARTBEAT_S)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                except StopAsyncIteration:
                    return
                yield frame("job", job)
        finally:
            with contextlib.suppress(Exception):
                await subscription.aclose()

    return sse_response(events())


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, jobs: JobsDep) -> Job:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.post("/{job_id}/cancel", response_model=Ok)
def cancel_job(job_id: str, jobs: JobsDep) -> Ok:
    if job_id not in jobs.jobs:
        raise HTTPException(404, "job not found")
    return Ok(ok=jobs.cancel(job_id))
