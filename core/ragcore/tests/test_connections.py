"""Contract tests for the reachability probe run before generation."""

from __future__ import annotations

import httpx

_LM_STUDIO = {
    "name": "LM Studio",
    "kind": "openai-compatible",
    "base_url": "http://localhost:1234/v1",
    "model_id": "qwen3-8b-instruct",
}


def test_probe_reports_ok_when_the_model_is_offered(client, monkeypatch) -> None:
    async def fake_get(self, url, headers=None):
        assert url == "http://localhost:1234/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen3-8b-instruct"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    response = client.post("/connections/test", json={"connection_id": connection_id})

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "ok": True,
        "reachable": True,
        "model_found": True,
        "streaming": True,
        "tokens_per_second": None,
        "latency_ms": result["latency_ms"],
        "error": None,
    }


def test_probe_reports_unreachable_when_the_endpoint_refuses_the_connection(
    client, monkeypatch
) -> None:
    async def fake_get(self, url, headers=None):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    response = client.post("/connections/test", json={"connection_id": connection_id})

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is False
    assert result["reachable"] is False
    assert "ConnectError" in result["error"]


def test_probe_rejects_an_unknown_connection_id(client) -> None:
    response = client.post("/connections/test", json={"connection_id": "missing"})

    assert response.status_code == 404
