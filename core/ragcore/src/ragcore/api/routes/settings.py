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
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(store.settings, field, value)
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
