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
    """Scripted by default; streams from a real model when RAGCORE_LLM is set."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def stream(
        self, question: str, chunks: list[RetrievedChunk], directives: set[str]
    ) -> AsyncIterator[str]:
        from ragcore.stub.answers import llm_stream, scripted_stream

        if self.config.llm_base_url:
            return llm_stream(self.config, question, chunks)
        return scripted_stream(question, chunks, directives)


def build_backend(config: Config) -> Backend:
    if config.backend == "stub":
        from ragcore.stub.hub import HubClient
        from ragcore.stub.jobs import JobManager
        from ragcore.stub.store import Store

        return Backend(
            store=Store(config),
            jobs=JobManager(),
            hub=HubClient(config),
            answerer=StubAnswerEngine(config),
        )
    raise ValueError(f"unknown backend: {config.backend!r}")
