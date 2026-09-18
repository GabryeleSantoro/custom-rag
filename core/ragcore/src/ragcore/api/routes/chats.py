from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import StoreDep
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
        raise HTTPException(404, "project not found")
    if payload.name is not None:
        project.name = payload.name.strip() or project.name
    if payload.pinned is not None:
        project.pinned = payload.pinned
    project.updated_at = datetime.now(tz=UTC)
    return project


@router.delete("/projects/{project_id}", response_model=Ok)
def delete_project(project_id: str, store: StoreDep) -> Ok:
    if project_id not in store.projects:
        raise HTTPException(404, "project not found")
    store.projects.pop(project_id)
    for session in store.sessions.values():
        if session.project_id == project_id:
            session.project_id = None
    return Ok()


@router.post("", response_model=ChatSession, status_code=201)
def create_session(payload: ChatSessionCreate, store: StoreDep) -> ChatSession:
    if payload.project_id is not None and payload.project_id not in store.projects:
        raise HTTPException(404, "project not found")
    return store.create_session(payload.title, payload.scope_doc_id, payload.project_id)


@router.get("/{session_id}", response_model=ChatSession)
def get_session(session_id: str, store: StoreDep) -> ChatSession:
    session = store.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@router.get("/{session_id}/messages", response_model=list[ChatMessage])
def get_messages(session_id: str, store: StoreDep) -> list[ChatMessage]:
    if session_id not in store.sessions:
        raise HTTPException(404, "session not found")
    return store.messages.get(session_id, [])


@router.delete("/{session_id}", response_model=Ok)
def delete_session(session_id: str, store: StoreDep) -> Ok:
    if session_id not in store.sessions:
        raise HTTPException(404, "session not found")
    store.sessions.pop(session_id)
    store.messages.pop(session_id, None)
    return Ok()


@router.patch("/{session_id}", response_model=ChatSession)
def update_session(session_id: str, payload: ChatSessionPatch, store: StoreDep) -> ChatSession:
    session = store.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if payload.title is not None and payload.title.strip():
        session.title = payload.title
    if "project_id" in payload.model_fields_set:
        if payload.project_id is not None and payload.project_id not in store.projects:
            raise HTTPException(404, "project not found")
        session.project_id = payload.project_id
    if payload.pinned is not None:
        session.pinned = payload.pinned
    return session
