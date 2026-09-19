from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import StoreDep
from ragcore.api.schemas import AppSettings, AppSettingsPatch, Ok, WipeRequest

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=AppSettings)
def get_settings(store: StoreDep) -> AppSettings:
    return store.settings


@router.patch("", response_model=AppSettings)
def patch_settings(payload: AppSettingsPatch, store: StoreDep) -> AppSettings:
    # The value off the model, not off model_dump: dumping would turn the nested
    # retrieval/performance models into plain dicts, which the retriever then
    # attribute-accesses and blows up on.
    for field in payload.model_dump(exclude_none=True):
        setattr(store.settings, field, getattr(payload, field))
    return store.settings


@router.post("/wipe", response_model=Ok)
def wipe(payload: WipeRequest, store: StoreDep) -> Ok:
    if payload.confirm != "DELETE":
        raise HTTPException(400, 'confirm must be the literal string "DELETE"')

    store.sources.clear()
    store.documents.clear()
    store.loaded.clear()
    store.sessions.clear()
    store.messages.clear()
    store.rebuild_index()
    if not payload.keep_connections:
        store.connections.clear()
        store.settings.active_connection_id = None
    store.settings.onboarded = False
    return Ok()
