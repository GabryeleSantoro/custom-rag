"""Background jobs: the lifecycle the Library and Model Manager render."""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi.testclient import TestClient
from ragcore.stub.jobs import JobManager

# --------------------------------------------------------------------- manager


async def _drain(manager: JobManager, job_id: str) -> None:
    """Wait for a started job's task to settle, however it settles."""
    with contextlib.suppress(asyncio.CancelledError):
        await manager._tasks[job_id]


async def _next(subscription, timeout: float = 2.0):
    """One frame, or a failure rather than a hung suite."""
    return await asyncio.wait_for(anext(subscription), timeout)


async def test_a_created_job_starts_queued() -> None:
    job = JobManager().create("index", "Indexing /corpus", total=12, source_id="src_1")

    assert (job.state, job.progress, job.completed) == ("queued", 0, 0)
    assert (job.total, job.source_id) == (12, "src_1")


async def test_job_ids_are_unique_and_name_their_kind() -> None:
    manager = JobManager()
    first = manager.create("index", "one")
    second = manager.create("download", "two")

    assert first.id != second.id
    assert first.id.startswith("job_index_")
    assert second.id.startswith("job_download_")


async def test_a_job_that_finishes_reports_done_and_a_full_count() -> None:
    manager = JobManager()
    job = manager.create("reindex", "Re-indexing", total=8)

    async def runner(_job, report) -> None:
        await report(0.5, "Halfway")

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert job.state == "done"
    assert (job.progress, job.completed) == (1.0, 8)
    assert job.started_at is not None and job.finished_at is not None


async def test_a_runner_that_raises_surfaces_the_message_instead_of_crashing() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing")

    async def runner(_job, _report) -> None:
        raise RuntimeError("disk full")

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert job.state == "error"
    assert job.error == "disk full"
    assert job.finished_at is not None


async def test_progress_updates_completed_against_the_total() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", total=200)
    seen: list[int] = []

    async def runner(_job, report) -> None:
        await report(0.25, "Parsing")
        seen.append(job.completed)

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert seen == [50]


async def test_progress_outside_zero_to_one_is_clamped() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", total=10)
    seen: list[float] = []

    async def runner(_job, report) -> None:
        await report(-3.0, None)
        seen.append(job.progress)
        await report(9.0, None)
        seen.append(job.progress)

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert seen == [0.0, 1.0]


async def test_a_progress_report_without_a_detail_keeps_the_previous_one() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", detail="from /corpus")

    async def runner(_job, report) -> None:
        await report(0.5, None)

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert job.detail == "from /corpus"


async def test_cancelling_a_running_job_marks_it_cancelled() -> None:
    manager = JobManager()
    job = manager.create("download", "Downloading")

    async def runner(_job, _report) -> None:
        await asyncio.sleep(30)

    manager.start(job, runner)
    await asyncio.sleep(0)  # let the task reach the sleep

    assert manager.cancel(job.id) is True
    await _drain(manager, job.id)
    assert job.state == "cancelled"
    assert job.finished_at is not None


async def test_cancelling_a_finished_job_reports_that_nothing_happened() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing")

    async def runner(_job, _report) -> None:
        return None

    manager.start(job, runner)
    await _drain(manager, job.id)

    assert manager.cancel(job.id) is False


async def test_cancelling_an_unknown_job_is_false_not_an_error() -> None:
    assert JobManager().cancel("job_index_9999") is False


async def test_a_subscriber_gets_the_jobs_that_already_exist() -> None:
    """A screen that opens mid-run must not wait for the next frame to show anything."""
    manager = JobManager()
    existing = manager.create("index", "Indexing")

    subscription = manager.subscribe()
    replayed = await _next(subscription)
    await subscription.aclose()

    assert replayed.id == existing.id


async def test_a_subscriber_sees_every_state_change() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", total=4)
    subscription = manager.subscribe()
    assert (await _next(subscription)).state == "queued"  # the replay

    async def runner(_job, report) -> None:
        await report(0.5, "Parsing")

    manager.start(job, runner)
    await _drain(manager, job.id)

    states = [(await _next(subscription)).state for _ in range(3)]
    await subscription.aclose()

    assert states == ["running", "running", "done"]


async def test_published_frames_are_snapshots_not_live_references() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", total=10)
    subscription = manager.subscribe()
    frame = await _next(subscription)

    job.progress = 0.9

    assert frame.progress == 0
    await subscription.aclose()


async def test_a_closed_subscription_stops_receiving() -> None:
    manager = JobManager()
    manager.create("index", "Indexing")
    subscription = manager.subscribe()
    await _next(subscription)
    assert len(manager._subscribers) == 1

    await subscription.aclose()

    assert manager._subscribers == set()


async def test_shutdown_cancels_everything_still_running() -> None:
    manager = JobManager()
    job = manager.create("download", "Downloading")

    async def runner(_job, _report) -> None:
        await asyncio.sleep(30)

    manager.start(job, runner)
    await asyncio.sleep(0)
    await manager.shutdown()

    assert job.state == "cancelled"


async def test_the_staged_runner_walks_its_stages_in_order() -> None:
    manager = JobManager()
    job = manager.create("index", "Indexing", total=100)
    details: list[str] = []

    async def report(fraction: float, detail: str | None) -> None:
        if detail and detail not in details:
            details.append(detail)
        assert 0.0 < fraction <= 1.0000001

    await JobManager.staged_runner([("Scanning", 0.01), ("Embedding", 0.01)])(job, report)

    assert details == ["Scanning", "Embedding"]


# ---------------------------------------------------------------------- routes


def test_listing_jobs_is_empty_before_anything_runs(client: TestClient) -> None:
    assert client.get("/jobs").json() == []


def test_a_started_job_shows_up_in_the_listing(client: TestClient) -> None:
    job = client.post("/index/rebuild", json={"full": True}).json()

    listed = client.get("/jobs").json()

    assert [j["id"] for j in listed] == [job["id"]]
    assert listed[0]["kind"] == "reindex"


def test_a_job_can_be_fetched_by_id(client: TestClient) -> None:
    job = client.post("/index/rebuild", json={"full": True}).json()

    assert client.get(f"/jobs/{job['id']}").json()["label"] == "Full re-index"


def test_an_unknown_job_id_is_a_404(client: TestClient) -> None:
    assert client.get("/jobs/job_index_9999").status_code == 404
    assert client.post("/jobs/job_index_9999/cancel").status_code == 404


def test_cancelling_a_running_job_reports_ok(client: TestClient) -> None:
    job = client.post("/index/rebuild", json={"full": True}).json()

    assert client.post(f"/jobs/{job['id']}/cancel").json() == {"ok": True}


def test_the_job_stream_replays_what_already_exists(client: TestClient) -> None:
    job = client.post("/index/rebuild", json={"full": True}).json()

    with client.stream("GET", "/jobs/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        # Buffering here would defeat live progress in the UI.
        assert response.headers["x-accel-buffering"] == "no"
        payload = None
        for raw in response.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                break

    assert payload is not None
    assert payload["id"] == job["id"]
