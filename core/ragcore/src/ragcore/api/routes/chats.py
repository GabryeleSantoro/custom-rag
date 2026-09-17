from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import StoreDep
from ragcore.api.schemas import ChatMessage, ChatSession, ChatSessionCreate, Ok

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=list[ChatSession])
def list_sessions(store: StoreDep) -> list[ChatSession]:
    return sorted(store.sessions.values(), key=lambda s: s.updated_at, reverse=True)


@router.post("", response_model=ChatSession, status_code=201)
def create_session(payload: ChatSessionCreate, store: StoreDep) -> ChatSession:
    return store.create_session(payload.title, payload.scope_doc_id)


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
def rename_session(session_id: str, payload: ChatSessionCreate, store: StoreDep) -> ChatSession:
    session = store.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if payload.title:
        session.title = payload.title
    return session
