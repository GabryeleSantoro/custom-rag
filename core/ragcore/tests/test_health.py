"""Health and the event-schema stub the UI generates its types from."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_the_index_it_is_serving(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["version"]
    assert body["stub"] is True
    assert body["models_ready"] is True
    assert body["index"]["chunks"] > 0
    assert body["index"]["embed_dim"] == 1024


def test_health_lists_ragcore_as_a_supervised_process(client: TestClient) -> None:
    [process] = client.get("/health").json()["processes"]

    assert (process["name"], process["role"], process["state"]) == ("ragcore", "ragcore", "ready")
    assert process["uptime_s"] >= 0


def test_uptime_only_moves_forward(client: TestClient) -> None:
    first = client.get("/health").json()["uptime_s"]

    assert client.get("/health").json()["uptime_s"] >= first


def test_health_tracks_the_index_as_it_changes(client: TestClient) -> None:
    assert client.get("/health").json()["index"]["documents"] > 0

    client.post("/settings/wipe", json={"confirm": "DELETE"})

    empty = client.get("/health").json()["index"]
    assert (empty["documents"], empty["chunks"]) == (0, 0)
    assert empty["last_indexed_at"] is None


def test_the_event_schema_endpoint_exists_for_its_shape_alone(client: TestClient) -> None:
    body = client.get("/schema/events").json()

    assert all(value is None for value in body.values())


def test_every_sse_frame_shape_reaches_openapi(client: TestClient) -> None:
    """`bun run gen:types` derives the UI's frame types from this one model."""
    envelope = client.get("/openapi.json").json()["components"]["schemas"]["StreamEnvelope"]

    assert {"start", "mode", "sources", "token", "citations", "done", "error"} <= set(
        envelope["properties"]
    )
