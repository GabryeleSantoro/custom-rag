"""Watched folders: adding, rescanning, removing, and what leaves the index with them."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

DEFAULT_FILES = {"one.md": "# One\n\nReranking reorders candidates.\n"}


def a_folder(tmp_path: Path, name: str = "notes", files: dict[str, str] | None = None) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, body in (files or DEFAULT_FILES).items():
        (folder / filename).write_text(body)
    return folder


def test_the_fixture_corpus_is_seeded_as_a_source(client: TestClient) -> None:
    sources = client.get("/sources").json()

    assert len(sources) == 1
    assert sources[0]["document_count"] > 0
    assert sources[0]["indexed_count"] == sources[0]["document_count"]


def test_adding_a_folder_indexes_its_files(client: TestClient, tmp_path: Path) -> None:
    folder = a_folder(tmp_path, files={"a.md": "# A\n\nFirst.\n", "b.md": "# B\n\nSecond.\n"})

    source = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()

    assert source["document_count"] == 2
    assert source["last_scan_at"] is not None
    assert client.get("/documents", params={"source_id": source["id"]}).json()["total"] == 2


def test_adding_a_folder_starts_an_indexing_job(client: TestClient, tmp_path: Path) -> None:
    folder = a_folder(tmp_path)

    source = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()

    jobs = [j for j in client.get("/jobs").json() if j["source_id"] == source["id"]]
    assert [j["kind"] for j in jobs] == ["index"]
    assert jobs[0]["label"].endswith(str(folder))


def test_exclude_globs_keep_files_out_of_the_index(client: TestClient, tmp_path: Path) -> None:
    folder = a_folder(tmp_path, files={"keep.md": "# Keep\n\nA.\n", "draft.md": "# Draft\n\nB.\n"})

    source = client.post(
        "/sources",
        json={
            "path": str(folder),
            "include_globs": ["**/*.md"],
            "exclude_globs": ["**/draft.md"],
        },
    ).json()

    titles = [
        d["title"]
        for d in client.get("/documents", params={"source_id": source["id"]}).json()["items"]
    ]
    assert titles == ["Keep"]


def test_adding_a_folder_that_does_not_exist_indexes_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    source = client.post("/sources", json={"path": str(tmp_path / "ghost")}).json()

    assert source["document_count"] == 0


def test_a_source_can_belong_to_a_project(client: TestClient, tmp_path: Path) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    folder = a_folder(tmp_path)

    source = client.post(
        "/sources",
        json={"path": str(folder), "include_globs": ["**/*.md"], "project_id": project["id"]},
    ).json()

    assert source["project_id"] == project["id"]


def test_adding_a_source_to_an_unknown_project_is_a_404(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/sources", json={"path": str(a_folder(tmp_path)), "project_id": "project_nope"}
    )

    assert response.status_code == 404


def test_rescanning_picks_up_a_file_added_since(client: TestClient, tmp_path: Path) -> None:
    folder = a_folder(tmp_path)
    source = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()
    (folder / "two.md").write_text("# Two\n\nChunking splits documents.\n")

    job = client.post(f"/sources/{source['id']}/rescan").json()

    assert job["kind"] == "index"
    assert job["total"] == 2
    assert client.get("/documents", params={"source_id": source["id"]}).json()["total"] == 2


def test_rescanning_an_unknown_source_is_a_404(client: TestClient) -> None:
    assert client.post("/sources/src_nope/rescan").status_code == 404
    assert client.delete("/sources/src_nope").status_code == 404


def test_removing_a_source_removes_its_documents(client: TestClient, tmp_path: Path) -> None:
    folder = a_folder(tmp_path)
    source = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()
    before = client.get("/documents").json()["total"]

    assert client.delete(f"/sources/{source['id']}").json() == {"ok": True}

    assert client.get("/documents").json()["total"] == before - 1
    assert source["id"] not in [s["id"] for s in client.get("/sources").json()]


def test_removing_a_source_takes_its_chunks_out_of_retrieval(
    client: TestClient, tmp_path: Path, read_events
) -> None:
    folder = a_folder(
        tmp_path, files={"cobalt.md": "# Cobalt\n\n## Launch\n\nThe cobalt launch shipped.\n"}
    )
    source = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()
    with client.stream("POST", "/query", json={"q": "cobalt launch shipped"}) as response:
        found = dict(read_events(response))["sources"]["chunks"]
    assert any(chunk["doc_id"] == "cobalt" for chunk in found)

    client.delete(f"/sources/{source['id']}")

    with client.stream("POST", "/query", json={"q": "cobalt launch shipped"}) as response:
        after = dict(read_events(response))["sources"]["chunks"]
    assert all(chunk["doc_id"] != "cobalt" for chunk in after)


def test_sources_and_removals_survive_a_restart(client, tmp_path) -> None:
    from fastapi.testclient import TestClient
    from ragcore.api.app import create_app

    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "kept.md").write_text("# Kept\n\nStays in the library.")
    (folder / "gone.md").write_text("# Gone\n\nRemoved by the user.")
    source = client.post("/sources", json={"path": str(folder)}).json()
    client.delete("/documents/gone")
    before = {d["id"] for d in client.get("/documents").json()["items"]}

    with TestClient(create_app(client.app.state.config)) as restarted:
        restarted.headers["Authorization"] = client.headers["Authorization"]
        after = {d["id"] for d in restarted.get("/documents").json()["items"]}
        sources = [s["id"] for s in restarted.get("/sources").json()]
        # A rescan honours the removal too.
        restarted.post(f"/sources/{source['id']}/rescan")
        rescanned = {d["id"] for d in restarted.get("/documents").json()["items"]}

    assert "kept" in after and "gone" not in after
    assert after == before == rescanned
    # The sample docs are seeded once, not again on every start.
    assert sources.count(source["id"]) == 1 and len(sources) == 2
    # The sample source is saved by placeholder, not by its temp-dir path.
    assert '"<bundled-fixtures>"' in (tmp_path / "library.json").read_text()
