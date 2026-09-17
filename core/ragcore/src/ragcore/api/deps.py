"""Request-scoped access to the process-wide singletons."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ragcore.config import Config
from ragcore.ports import AnswerEngine, StorePort


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_store(request: Request) -> StorePort:
    return request.app.state.store


def get_jobs(request: Request) -> object:
    return request.app.state.jobs


def get_answerer(request: Request) -> AnswerEngine:
    return request.app.state.answerer


ConfigDep = Annotated[Config, Depends(get_config)]
StoreDep = Annotated[StorePort, Depends(get_store)]
JobsDep = Annotated[object, Depends(get_jobs)]
AnswererDep = Annotated[AnswerEngine, Depends(get_answerer)]
