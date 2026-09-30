from __future__ import annotations

from fastapi import APIRouter

from ragcore.api.deps import StoreDep
from ragcore.api.errors import api_error
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
    if payload.active_connection_id is not None:
        store.save_connections()
    store.save_settings()
    return store.settings


@router.post("/wipe", response_model=Ok)
def wipe(payload: WipeRequest, store: StoreDep) -> Ok:
    if payload.confirm != "DELETE":
        raise api_error(400, "wipe_confirm_required", 'confirm must be the literal string "DELETE"')

    store.wipe(keep_connections=payload.keep_connections)
    return Ok()
