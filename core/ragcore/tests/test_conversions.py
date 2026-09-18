"""Contract tests for the research-backed slide conversion stream."""

from __future__ import annotations

from ragcore.api.routes import conversions
from ragcore.api.schemas import WebResearchResult


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


def test_slide_conversion_accepts_a_local_file_without_indexing_the_input(
    client, read_events, monkeypatch, tmp_path
) -> None:
    async def fake_search(_query: str):
        return [], "No web result"

    monkeypatch.setattr(conversions, "_search_web", fake_search)
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
