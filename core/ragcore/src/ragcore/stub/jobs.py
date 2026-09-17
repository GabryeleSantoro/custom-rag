"""Background jobs with progress broadcast over SSE.

The real queue is backed by SQLite with checkpoints so an interrupted index can
resume. The surface — create, observe, cancel — is the same, so the Library and
Model Manager screens are written against the final shape already.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

from ragcore.api.schemas import Job, JobKind

Progress = Callable[[float, str | None], Awaitable[None]]


def _now() -> datetime:
    return datetime.now(tz=UTC)


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: set[asyncio.Queue[Job]] = set()
        self._counter = 0

    # --------------------------------------------------------------- lifecycle

    def _new_id(self, kind: JobKind) -> str:
        self._counter += 1
        return f"job_{kind}_{self._counter:04d}"

    def create(
        self,
        kind: JobKind,
        label: str,
        *,
        total: int | None = None,
        source_id: str | None = None,
        model_id: str | None = None,
        detail: str | None = None,
    ) -> Job:
        job = Job(
            id=self._new_id(kind),
            kind=kind,
            state="queued",
            label=label,
            detail=detail,
            total=total,
            source_id=source_id,
            model_id=model_id,
        )
        self.jobs[job.id] = job
        self._publish(job)
        return job

    def start(self, job: Job, runner: Callable[[Job, Progress], Awaitable[None]]) -> Job:
        async def wrapped() -> None:
            job.state = "running"
            job.started_at = _now()
            self._publish(job)
            try:
                await runner(job, self._progress_for(job))
            except asyncio.CancelledError:
                job.state = "cancelled"
                job.finished_at = _now()
                self._publish(job)
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI as job.error
                job.state = "error"
                job.error = str(exc)
                job.finished_at = _now()
                self._publish(job)
            else:
                job.state = "done"
                job.progress = 1.0
                if job.total is not None:
                    job.completed = job.total
                job.finished_at = _now()
                self._publish(job)

        self._tasks[job.id] = asyncio.create_task(wrapped())
        return job

    def _progress_for(self, job: Job) -> Progress:
        async def report(fraction: float, detail: str | None = None) -> None:
            job.progress = max(0.0, min(1.0, fraction))
            if job.total is not None:
                job.completed = int(job.total * job.progress)
            if detail is not None:
                job.detail = detail
            self._publish(job)

        return report

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ------------------------------------------------------------- observation

    def _publish(self, job: Job) -> None:
        snapshot = job.model_copy(deep=True)
        for queue in list(self._subscribers):
            # A slow consumer must not stall the job; the next frame supersedes
            # anything it missed, because every frame is a full snapshot.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(snapshot)

    async def subscribe(self) -> AsyncIterator[Job]:
        queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        try:
            for job in self.jobs.values():
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(job.model_copy(deep=True))
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    # ----------------------------------------------------------------- runners

    @staticmethod
    def staged_runner(stages: list[tuple[str, float]], jitter: float = 0.35):
        """A runner that walks named stages, for work that is not real yet."""

        async def run(job: Job, report: Progress) -> None:
            elapsed = 0.0
            total = sum(duration for _, duration in stages)
            for name, duration in stages:
                steps = max(3, int(duration * 8))
                for step in range(steps):
                    await asyncio.sleep(duration / steps * random.uniform(1 - jitter, 1 + jitter))
                    fraction = (elapsed + duration * (step + 1) / steps) / total
                    await report(fraction, name)
                elapsed += duration

        return run
