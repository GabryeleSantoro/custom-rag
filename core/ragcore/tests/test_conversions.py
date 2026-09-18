"""Contract tests for the research-backed slide conversion stream."""

from __future__ import annotations

import asyncio

from ragcore.api.routes import conversions
from ragcore.api.schemas import ConnectionTestResult, RetrievedChunk, WebResearchResult


def test_system_prompt_forbids_treating_passages_as_instructions() -> None:
    prompt_it = conversions._build_system_prompt("it")
    prompt_en = conversions._build_system_prompt("en")

    for prompt in (prompt_it, prompt_en):
        assert "Passages:" in prompt
        assert "```" not in prompt.split("COSA NON DEVI MAI FARE")[-1] if "COSA NON DEVI MAI FARE" in prompt else True

    assert "materiale grezzo" in prompt_it
    assert "NON obbedire" in prompt_it
    assert "raw material" in prompt_en
    assert "do NOT obey" in prompt_en


def test_system_prompt_is_language_specific() -> None:
    assert conversions._build_system_prompt("it") != conversions._build_system_prompt("en")
    assert "italiano" in conversions._build_system_prompt("it")
    assert "English" in conversions._build_system_prompt("en")


async def _fake_probe_ok(store, kind, base_url, model_id, api_key):
    return ConnectionTestResult(ok=True, reachable=True, model_found=True, streaming=True)


def test_slide_conversion_saves_and_indexes_markdown(client, read_events, monkeypatch) -> None:
    async def fake_search(query: str):
        assert "current" in query
        return [
            WebResearchResult(
                title="A useful reference",
                url="https://example.com/reference",
                snippet="A current context for the selected slide.",
            )
        ], None

    monkeypatch.setattr(conversions, "_search_web", fake_search)
    monkeypatch.setattr(conversions, "probe_connection", _fake_probe_ok)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"slide_ids": [slide_id], "research_query": "current context"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[:2] == ["conversion_start", "conversion_research"]
    assert "token" in names
    assert names[-2:] == ["conversion_saved", "conversion_done"]
    done = events[-1][1]
    assert done["research_count"] == 1
    assert done["path"].endswith(".md")
    saved = client.get(f"/documents/{done['document_id']}").json()
    assert saved["path"] == done["path"]
    assert saved["ext"] == ".md"


def test_slide_conversion_is_disabled_without_an_active_connection(client) -> None:
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]
    connection_id = client.get("/connections").json()[0]["id"]
    assert client.delete(f"/connections/{connection_id}").status_code == 200

    response = client.post("/conversions/slides", json={"slide_ids": [slide_id]})

    assert response.status_code == 409


def test_slide_conversion_is_disabled_when_the_active_connection_is_unreachable(
    client, monkeypatch
) -> None:
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
    async def fake_search(_query: str):
        return [], "No web result"

    monkeypatch.setattr(conversions, "_search_web", fake_search)
    monkeypatch.setattr(conversions, "probe_connection", _fake_probe_ok)
    slide = tmp_path / "local-slide.md"
    slide.write_text("# Local slide\n\n## Context\n\nThis file is only used for conversion.\n")

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"file_paths": [str(slide)], "output_title": "Local conversion"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    assert events[-1][0] == "conversion_done"
    assert not any(source["path"] == str(tmp_path) for source in client.get("/sources").json())


def test_research_queries_combine_the_request_query_and_section_titles() -> None:
    pages = [
        RetrievedChunk(
            chunk_id="a", doc_id="d", doc_title="Reranking", page_start=1, page_end=1,
            section_path="Reranking > Cross encoders", text="...",
        ),
        RetrievedChunk(
            chunk_id="b", doc_id="d", doc_title="Reranking", page_start=2, page_end=2,
            section_path="Reranking > Latency", text="...",
        ),
    ]

    queries = conversions._research_queries("current state of the art", "Reranking", pages)

    assert queries == [
        "current state of the art",
        "Reranking: Cross encoders",
        "Reranking: Latency",
    ]


def test_research_queries_fall_back_to_a_generic_query_with_no_sections() -> None:
    queries = conversions._research_queries(None, "Reranking", [])

    assert queries == ["Reranking: key concepts, current context, examples and sources"]


def test_research_aggregates_and_dedupes_results_across_queries(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_search(query: str):
        calls.append(query)
        if query == "Reranking: Cross encoders":
            return [
                WebResearchResult(title="A", url="https://example.com/a", snippet="..."),
                WebResearchResult(title="B", url="https://example.com/b", snippet="..."),
            ], None
        return [
            WebResearchResult(title="A dup", url="https://example.com/a", snippet="..."),
        ], None

    monkeypatch.setattr(conversions, "_search_web", fake_search)
    pages = [
        RetrievedChunk(
            chunk_id="a", doc_id="d", doc_title="Reranking", page_start=1, page_end=1,
            section_path="Reranking > Cross encoders", text="...",
        ),
        RetrievedChunk(
            chunk_id="b", doc_id="d", doc_title="Reranking", page_start=2, page_end=2,
            section_path="Reranking > Latency", text="...",
        ),
    ]

    queries, results, warning = asyncio.run(conversions._research(None, "Reranking", pages))

    assert calls == ["Reranking: Cross encoders", "Reranking: Latency"]
    assert [r.url for r in results] == ["https://example.com/a", "https://example.com/b"]
    assert warning is None
