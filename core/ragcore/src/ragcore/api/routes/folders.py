"""User-made folders for organising the library. Membership only; nothing moves on disk."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import StoreDep
from ragcore.api.schemas import Folder, FolderAssignment, FolderInput, Ok
from ragcore.ports import StorePort

router = APIRouter(tags=["folders"])


def _visible(folder: Folder, store: StorePort) -> Folder:
    # Documents removed from the index since they were filed simply drop out.
    return folder.model_copy(
        update={"doc_ids": [d for d in folder.doc_ids if d in store.documents]}
    )


def _get(folder_id: str, store: StorePort) -> Folder:
    folder = store.folders.get(folder_id)
    if folder is None:
        raise HTTPException(404, "folder not found")
    return folder


def _name(payload: FolderInput) -> str:
    name = payload.name.strip()
    if not name:
        raise HTTPException(422, "folder name cannot be empty")
    return name


@router.get("/folders", response_model=list[Folder])
def list_folders(store: StoreDep) -> list[Folder]:
    folders = sorted(store.folders.values(), key=lambda f: f.name.lower())
    return [_visible(f, store) for f in folders]


@router.post("/folders", response_model=Folder, status_code=201)
def create_folder(payload: FolderInput, store: StoreDep) -> Folder:
    folder = Folder(
        id=store.new_id("folder"), name=_name(payload), created_at=datetime.now(tz=UTC)
    )
    store.folders[folder.id] = folder
    store.save_folders()
    return folder


@router.patch("/folders/{folder_id}", response_model=Folder)
def rename_folder(folder_id: str, payload: FolderInput, store: StoreDep) -> Folder:
    folder = _get(folder_id, store)
    folder.name = _name(payload)
    store.save_folders()
    return _visible(folder, store)


@router.delete("/folders/{folder_id}", response_model=Ok)
def delete_folder(folder_id: str, store: StoreDep) -> Ok:
    """The documents stay in the library; only the grouping goes."""
    _get(folder_id, store)
    store.folders.pop(folder_id)
    store.save_folders()
    return Ok()


@router.put("/documents/{doc_id}/folder", response_model=Ok)
def move_document(doc_id: str, payload: FolderAssignment, store: StoreDep) -> Ok:
    """A document sits in at most one folder, like a file."""
    if doc_id not in store.documents:
        raise HTTPException(404, "document not found")
    if payload.folder_id is not None:
        _get(payload.folder_id, store)
    for folder in store.folders.values():
        if doc_id in folder.doc_ids:
            folder.doc_ids.remove(doc_id)
    if payload.folder_id is not None:
        store.folders[payload.folder_id].doc_ids.append(doc_id)
    store.save_folders()
    return Ok()
