"""User folders group documents in the library without touching files on disk."""

from __future__ import annotations

from fastapi.testclient import TestClient
from ragcore.api.app import create_app


def _doc_ids(client) -> list[str]:
    return [d["id"] for d in client.get("/documents").json()["items"]]


def test_a_document_is_filed_into_one_folder_and_the_library_filters_by_it(client) -> None:
    first, second, *_ = _doc_ids(client)
    papers = client.post("/folders", json={"name": "Papers"}).json()
    notes = client.post("/folders", json={"name": "Notes"}).json()

    client.put(f"/documents/{first}/folder", json={"folder_id": papers["id"]})
    client.put(f"/documents/{second}/folder", json={"folder_id": papers["id"]})
    # Moving keeps a document in exactly one folder.
    client.put(f"/documents/{second}/folder", json={"folder_id": notes["id"]})

    filed = client.get("/documents", params={"folder_id": papers["id"]}).json()
    assert [d["id"] for d in filed["items"]] == [first]
    by_name = {f["name"]: f["doc_ids"] for f in client.get("/folders").json()}
    assert by_name == {"Notes": [second], "Papers": [first]}

    client.put(f"/documents/{second}/folder", json={"folder_id": None})
    assert client.get("/documents", params={"folder_id": notes["id"]}).json()["total"] == 0


def test_folders_survive_a_restart(client) -> None:
    doc = _doc_ids(client)[0]
    folder = client.post("/folders", json={"name": "Exam"}).json()
    client.put(f"/documents/{doc}/folder", json={"folder_id": folder["id"]})
    client.patch(f"/folders/{folder['id']}", json={"name": "Exam prep"})

    with TestClient(create_app(client.app.state.config)) as restarted:
        restarted.headers["Authorization"] = client.headers["Authorization"]
        [reloaded] = restarted.get("/folders").json()

    assert reloaded["name"] == "Exam prep"
    assert reloaded["doc_ids"] == [doc]


def test_deleting_a_folder_keeps_its_documents(client) -> None:
    doc = _doc_ids(client)[0]
    folder = client.post("/folders", json={"name": "Temp"}).json()
    client.put(f"/documents/{doc}/folder", json={"folder_id": folder["id"]})

    assert client.delete(f"/folders/{folder['id']}").status_code == 200
    assert client.get("/folders").json() == []
    assert doc in _doc_ids(client)


def test_bad_input_is_rejected(client) -> None:
    doc = _doc_ids(client)[0]

    assert client.post("/folders", json={"name": "   "}).status_code == 422
    assert client.put(f"/documents/{doc}/folder", json={"folder_id": "nope"}).status_code == 404
    assert client.put("/documents/nope/folder", json={"folder_id": None}).status_code == 404
