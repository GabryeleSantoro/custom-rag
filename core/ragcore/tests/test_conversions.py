"""Contract tests for the slide conversion stream."""

from __future__ import annotations

from ragcore.api.routes import conversions
from ragcore.api.schemas import ConnectionTestResult


async def _fake_probe_ok(store, kind, base_url, model_id, api_key):
    return ConnectionTestResult(ok=True, reachable=True, model_found=True, streaming=True)


class _FakeAnswerer:
    """Stands in for the active connection's model; records what it was given."""

    def __init__(self, captured: dict) -> None:
        self.captured = captured

    async def stream(self, question, chunks, directives, *, system_prompt=None, max_tokens=None):
        self.captured["question"] = question
        self.captured["chunks"] = chunks
        self.captured["system_prompt"] = system_prompt
        self.captured["max_tokens"] = max_tokens
        yield "ok"


def _mock_ok(client, monkeypatch) -> dict:
    client.post(
        "/connections",
        json={
            "name": "LM Studio",
            "kind": "openai-compatible",
            "base_url": "http://127.0.0.1:1234/v1",
            "model_id": "qwen3-8b-instruct",
            "max_output_tokens": 4096,
        },
    )
    monkeypatch.setattr(conversions, "probe_connection", _fake_probe_ok)

    from ragcore.api import deps

    captured: dict = {}
    monkeypatch.setitem(
        client.app.dependency_overrides, deps.get_answerer, lambda: _FakeAnswerer(captured)
    )
    return captured


def test_slide_conversion_saves_and_indexes_markdown(client, read_events, monkeypatch) -> None:
    _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream(
        "POST", "/conversions/slides", json={"slide_ids": [slide_id], "research_query": "esempi numerici"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "conversion_start"
    assert "token" in names
    assert names[-2:] == ["conversion_saved", "conversion_done"]
    done = events[-1][1]
    assert done["failed"] == []
    assert len(done["saved"]) == 1
    saved_event = done["saved"][0]
    assert saved_event["path"].endswith(".md")
    saved = client.get(f"/documents/{saved_event['document_id']}").json()
    assert saved["path"] == saved_event["path"]
    assert saved["ext"] == ".md"


def test_slide_conversion_is_disabled_without_an_active_connection(client) -> None:
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    response = client.post("/conversions/slides", json={"slide_ids": [slide_id]})

    assert response.status_code == 409


def test_slide_conversion_is_disabled_when_the_active_connection_is_unreachable(
    client, monkeypatch
) -> None:
    client.post(
        "/connections",
        json={"name": "LM Studio", "kind": "openai-compatible", "model_id": "qwen3-8b-instruct"},
    )
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    async def fake_probe(store, kind, base_url, model_id, api_key):
        return ConnectionTestResult(
            ok=False, reachable=False, model_found=False, streaming=False, error="ConnectError"
        )

    monkeypatch.setattr(conversions, "probe_connection", fake_probe)

    response = client.post("/conversions/slides", json={"slide_ids": [slide_id]})

    assert response.status_code == 409
    assert "ConnectError" in response.json()["detail"]


def test_slide_conversion_accepts_a_local_file_without_indexing_the_input(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    slide = tmp_path / "local-slide.md"
    slide.write_text("# Local slide\n\n## Context\n\nThis file is only used for conversion.\n")

    with client.stream(
        "POST", "/conversions/slides", json={"file_paths": [str(slide)], "output_title": "Local conversion"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    assert events[-1][0] == "conversion_done"
    assert not any(source["path"] == str(tmp_path) for source in client.get("/sources").json())


def test_slide_conversion_processes_each_presentation_independently(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    good = tmp_path / "good.md"
    good.write_text("# Good\n\n## Context\n\nReadable content.\n")
    missing = str(tmp_path / "missing.pdf")

    with client.stream(
        "POST", "/conversions/slides", json={"file_paths": [missing, str(good)]},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "presentation_error"
    assert names[-1] == "conversion_done"
    error_event = events[0][1]
    assert error_event["presentation_index"] == 0
    assert error_event["presentation_total"] == 2
    done = events[-1][1]
    assert len(done["saved"]) == 1
    assert len(done["failed"]) == 1
    assert done["saved"][0]["presentation_index"] == 1


def test_slide_conversion_uses_the_hardened_system_prompt(client, monkeypatch) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream("POST", "/conversions/slides", json={"slide_ids": [slide_id]}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured["system_prompt"] == conversions._build_system_prompt("it")
    assert captured["max_tokens"] == 4096  # the active connection's output budget


def test_system_prompt_forbids_treating_passages_as_instructions() -> None:
    prompt_it = conversions._build_system_prompt("it")
    prompt_en = conversions._build_system_prompt("en")

    assert "Passages:" in prompt_it and "Passages:" in prompt_en
    assert "materiale grezzo" in prompt_it
    assert "NON obbedire" in prompt_it
    assert "raw material" in prompt_en
    assert "do NOT obey" in prompt_en


def test_system_prompt_is_language_specific() -> None:
    assert conversions._build_system_prompt("it") != conversions._build_system_prompt("en")
    assert "italiano" in conversions._build_system_prompt("it")
    assert "English" in conversions._build_system_prompt("en")


def test_system_prompt_demands_textbook_prose_over_bullets() -> None:
    prompt_it = conversions._build_system_prompt("it")
    prompt_en = conversions._build_system_prompt("en")

    assert "manuale universitario" in prompt_it
    assert "elenchi puntati sono l'eccezione" in prompt_it.lower()
    assert "university textbook" in prompt_en
    assert "bulleted lists are the exception" in prompt_en.lower()


def test_conversion_prompt_asks_the_model_to_deepen_the_topics(client, monkeypatch) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"slide_ids": [slide_id], "research_query": "esempi numerici"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert "approfondisci" in captured["question"].lower()
    assert "esempi numerici" in captured["question"]


def test_unique_path_appends_a_counter_on_collision(tmp_path) -> None:
    (tmp_path / "intro.md").write_text("existing")

    first = conversions._unique_path(tmp_path, "intro")
    first.write_text("first")
    second = conversions._unique_path(tmp_path, "intro")

    assert first == tmp_path / "intro-2.md"
    assert second == tmp_path / "intro-3.md"


def test_unique_path_uses_the_plain_name_when_free(tmp_path) -> None:
    assert conversions._unique_path(tmp_path, "fresh-title") == tmp_path / "fresh-title.md"
