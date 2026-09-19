"""Chat history: sessions, projects, messages, and the 404s in between."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_a_new_session_starts_empty_and_untitled(client: TestClient) -> None:
    session = client.post("/chats", json={}).json()

    assert session["title"] == "New chat"
    assert session["message_count"] == 0
    assert session["pinned"] is False
    assert session["project_id"] is None
    assert client.get(f"/chats/{session['id']}/messages").json() == []


def test_a_session_can_be_created_with_a_title_and_a_document_scope(
    client: TestClient,
) -> None:
    doc_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    session = client.post("/chats", json={"title": "Reranking", "scope_doc_id": doc_id}).json()

    assert session["title"] == "Reranking"
    assert session["scope_doc_id"] == doc_id


def test_creating_a_session_in_an_unknown_project_is_a_404(client: TestClient) -> None:
    assert client.post("/chats", json={"project_id": "project_nope"}).status_code == 404


def test_pinned_sessions_sort_ahead_of_the_rest(client: TestClient) -> None:
    first = client.post("/chats", json={"title": "First"}).json()
    client.post("/chats", json={"title": "Second"})
    client.patch(f"/chats/{first['id']}", json={"pinned": True})

    listed = client.get("/chats").json()

    assert listed[0]["id"] == first["id"]


def test_asking_a_question_records_both_turns_and_titles_the_session(
    client: TestClient,
) -> None:
    with client.stream("POST", "/query", json={"q": "How does reranking improve retrieval?"}) as r:
        list(r.iter_lines())

    session = client.get("/chats").json()[0]
    messages = client.get(f"/chats/{session['id']}/messages").json()

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert session["title"] == "How does reranking improve retrieval?"[:48]
    assert session["message_count"] == 2
    assert messages[1]["citations"], "the assistant turn keeps its citations"


def test_an_unknown_session_is_a_404_everywhere(client: TestClient) -> None:
    assert client.get("/chats/chat_nope").status_code == 404
    assert client.get("/chats/chat_nope/messages").status_code == 404
    assert client.patch("/chats/chat_nope", json={"title": "x"}).status_code == 404
    assert client.delete("/chats/chat_nope").status_code == 404


def test_renaming_a_session_sticks(client: TestClient) -> None:
    session = client.post("/chats", json={}).json()

    renamed = client.patch(f"/chats/{session['id']}", json={"title": "Retrieval notes"}).json()

    assert renamed["title"] == "Retrieval notes"
    assert client.get(f"/chats/{session['id']}").json()["title"] == "Retrieval notes"


def test_a_blank_rename_is_ignored_rather_than_wiping_the_title(client: TestClient) -> None:
    session = client.post("/chats", json={"title": "Keep me"}).json()

    assert client.patch(f"/chats/{session['id']}", json={"title": "   "}).json()["title"] == (
        "Keep me"
    )


def test_deleting_a_session_takes_its_messages_with_it(client: TestClient) -> None:
    with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
        list(response.iter_lines())
    session_id = client.get("/chats").json()[0]["id"]

    assert client.delete(f"/chats/{session_id}").json() == {"ok": True}

    assert client.get(f"/chats/{session_id}/messages").status_code == 404
    assert client.get("/chats").json() == []


# -------------------------------------------------------------------- projects


def test_a_project_defaults_to_using_global_knowledge(client: TestClient) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()

    assert project["use_global_sources"] is True
    assert project["pinned"] is False


def test_a_project_with_a_blank_name_gets_a_placeholder(client: TestClient) -> None:
    assert client.post("/chats/projects", json={"name": "   "}).json()["name"] == (
        "Untitled project"
    )


def test_pinned_projects_sort_first(client: TestClient) -> None:
    first = client.post("/chats/projects", json={"name": "A"}).json()
    client.post("/chats/projects", json={"name": "B"})
    client.patch(f"/chats/projects/{first['id']}", json={"pinned": True})

    assert client.get("/chats/projects").json()[0]["id"] == first["id"]


def test_an_unknown_project_is_a_404(client: TestClient) -> None:
    assert client.patch("/chats/projects/project_nope", json={"name": "x"}).status_code == 404
    assert client.delete("/chats/projects/project_nope").status_code == 404


def test_a_blank_project_rename_is_ignored(client: TestClient) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()

    assert client.patch(f"/chats/projects/{project['id']}", json={"name": " "}).json()["name"] == (
        "Research"
    )


def test_a_session_can_be_moved_into_and_out_of_a_project(client: TestClient) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    session = client.post("/chats", json={}).json()

    moved = client.patch(f"/chats/{session['id']}", json={"project_id": project["id"]}).json()
    assert moved["project_id"] == project["id"]

    detached = client.patch(f"/chats/{session['id']}", json={"project_id": None}).json()
    assert detached["project_id"] is None


def test_moving_a_session_into_an_unknown_project_is_a_404(client: TestClient) -> None:
    session = client.post("/chats", json={}).json()

    response = client.patch(f"/chats/{session['id']}", json={"project_id": "project_nope"})

    assert response.status_code == 404


def test_a_patch_that_omits_project_id_leaves_it_alone(client: TestClient) -> None:
    """`None` means detach; absent means untouched — the two must not collapse."""
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    session = client.post("/chats", json={"project_id": project["id"]}).json()

    patched = client.patch(f"/chats/{session['id']}", json={"pinned": True}).json()

    assert patched["project_id"] == project["id"]


def test_deleting_a_project_keeps_its_folders_and_chats(client: TestClient, tmp_path) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "brief.md").write_text("Project-only material.")
    source = client.post(
        "/sources",
        json={"path": str(folder), "include_globs": ["**/*.md"], "project_id": project["id"]},
    ).json()
    session = client.post("/chats", json={"project_id": project["id"]}).json()

    client.delete(f"/chats/projects/{project['id']}")

    assert client.get(f"/chats/{session['id']}").json()["project_id"] is None
    remaining = {s["id"]: s for s in client.get("/sources").json()}
    assert source["id"] in remaining, "the folder stays indexed as global knowledge"
    assert remaining[source["id"]]["project_id"] is None
