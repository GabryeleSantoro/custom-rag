from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from ragcore.api.deps import StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import (
    ChatMessage,
    ChatProject,
    ChatProjectCreate,
    ChatProjectPatch,
    ChatSession,
    ChatSessionCreate,
    ChatSessionPatch,
    Ok,
)

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=list[ChatSession])
def list_sessions(store: StoreDep) -> list[ChatSession]:
    return sorted(store.sessions.values(), key=lambda s: (s.pinned, s.updated_at), reverse=True)


@router.get("/projects", response_model=list[ChatProject])
def list_projects(store: StoreDep) -> list[ChatProject]:
    return sorted(store.projects.values(), key=lambda p: (p.pinned, p.updated_at), reverse=True)


@router.post("/projects", response_model=ChatProject, status_code=201)
def create_project(payload: ChatProjectCreate, store: StoreDep) -> ChatProject:
    return store.create_project(payload.name)


@router.patch("/projects/{project_id}", response_model=ChatProject)
def update_project(project_id: str, payload: ChatProjectPatch, store: StoreDep) -> ChatProject:
    project = store.projects.get(project_id)
    if project is None:
        raise api_error(404, "project_not_found", "project not found")
    if payload.name is not None:
        project.name = payload.name.strip() or project.name
    if payload.pinned is not None:
        project.pinned = payload.pinned
    if payload.use_global_sources is not None:
        project.use_global_sources = payload.use_global_sources
    project.updated_at = datetime.now(tz=UTC)
    store.save_chats()
    return project


@router.delete("/projects/{project_id}", response_model=Ok)
def delete_project(project_id: str, store: StoreDep) -> Ok:
    if project_id not in store.projects:
        raise api_error(404, "project_not_found", "project not found")
    store.projects.pop(project_id)
    # Project folders remain available as global knowledge when their project
    # is removed; deleting a project must not delete files from the index.
    for source in store.sources.values():
        if source.project_id == project_id:
            source.project_id = None
    for session in store.sessions.values():
        if session.project_id == project_id:
            session.project_id = None
    store.save_chats()
    store.save_library()
    return Ok()


@router.post("", response_model=ChatSession, status_code=201)
def create_session(payload: ChatSessionCreate, store: StoreDep) -> ChatSession:
    if payload.project_id is not None and payload.project_id not in store.projects:
        raise api_error(404, "project_not_found", "project not found")
    return store.create_session(payload.title, payload.scope_doc_id, payload.project_id)


@router.get("/{session_id}", response_model=ChatSession)
def get_session(session_id: str, store: StoreDep) -> ChatSession:
    session = store.sessions.get(session_id)
    if session is None:
        raise api_error(404, "session_not_found", "session not found")
    return session


@router.get("/{session_id}/messages", response_model=list[ChatMessage])
def get_messages(session_id: str, store: StoreDep) -> list[ChatMessage]:
    if session_id not in store.sessions:
        raise api_error(404, "session_not_found", "session not found")
    return store.messages.get(session_id, [])


@router.delete("/{session_id}", response_model=Ok)
def delete_session(session_id: str, store: StoreDep) -> Ok:
    if session_id not in store.sessions:
        raise api_error(404, "session_not_found", "session not found")
    store.sessions.pop(session_id)
    store.messages.pop(session_id, None)
    store.save_chats()
    return Ok()


@router.patch("/{session_id}", response_model=ChatSession)
def update_session(session_id: str, payload: ChatSessionPatch, store: StoreDep) -> ChatSession:
    session = store.sessions.get(session_id)
    if session is None:
        raise api_error(404, "session_not_found", "session not found")
    if payload.title is not None and payload.title.strip():
        session.title = payload.title
    if "project_id" in payload.model_fields_set:
        if payload.project_id is not None and payload.project_id not in store.projects:
            raise api_error(404, "project_not_found", "project not found")
        session.project_id = payload.project_id
    if payload.pinned is not None:
        session.pinned = payload.pinned
    store.save_chats()
    return session
