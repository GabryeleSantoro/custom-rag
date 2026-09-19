"""Chooses and assembles a backend for one process lifetime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from ragcore.api.schemas import RetrievedChunk
from ragcore.config import Config
from ragcore.ports import AnswerEngine, StorePort


@dataclass(frozen=True, slots=True)
class Backend:
    store: StorePort
    jobs: object
    hub: object
    answerer: AnswerEngine


class StubAnswerEngine:
    """Scripted until the user activates a connection; then that connection answers."""

    def __init__(self, store: StorePort) -> None:
        self.store = store

    def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        directives: set[str],
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        from ragcore.stub.answers import llm_stream, scripted_stream

        connection = self.store.active_connection()
        if connection is None:
            return scripted_stream(question, chunks, directives)
        return llm_stream(
            connection,
            self.store.secrets.get(connection.id),
            question,
            chunks,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )


def build_backend(config: Config) -> Backend:
    if config.backend == "stub":
        from ragcore.stub.hub import HubClient
        from ragcore.stub.jobs import JobManager
        from ragcore.stub.store import Store

        store = Store(config)
        return Backend(
            store=store,
            jobs=JobManager(),
            hub=HubClient(config),
            answerer=StubAnswerEngine(store),
        )
    raise ValueError(f"unknown backend: {config.backend!r}")
