from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import JobsDep, StoreDep
from ragcore.api.schemas import Job, Ok, Source, SourceCreate

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=list[Source])
def list_sources(store: StoreDep) -> list[Source]:
    return list(store.sources.values())


@router.post("", response_model=Source, status_code=201)
async def add_source(payload: SourceCreate, store: StoreDep, jobs: JobsDep) -> Source:
    if payload.project_id is not None and payload.project_id not in store.projects:
        raise HTTPException(404, "project not found")
    source = store.add_source(payload)
    documents = store.ingest_source(source.id)

    # The scan is instant on fixtures; the job exists so the UI sees the same
    # progress shape it will see once parsing and embedding are real.
    job = jobs.create(
        "index",
        f"Indexing {source.path}",
        total=max(len(documents), 1),
        source_id=source.id,
    )
    jobs.start(
        job,
        jobs.staged_runner([("Scanning", 0.6), ("Parsing", 1.4), ("Embedding", 1.6)]),
    )
    return source


@router.delete("/{source_id}", response_model=Ok)
def remove_source(source_id: str, store: StoreDep) -> Ok:
    if source_id not in store.sources:
        raise HTTPException(404, "source not found")
    store.remove_source(source_id)
    return Ok()


@router.post("/{source_id}/rescan", response_model=Job)
async def rescan(source_id: str, store: StoreDep, jobs: JobsDep) -> Job:
    if source_id not in store.sources:
        raise HTTPException(404, "source not found")
    documents = store.ingest_source(source_id)
    job = jobs.create(
        "index", "Rescanning source", total=max(len(documents), 1), source_id=source_id
    )
    return jobs.start(job, jobs.staged_runner([("Scanning", 0.5), ("Embedding", 1.0)]))
