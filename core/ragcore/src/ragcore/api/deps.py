"""Request-scoped access to the process-wide singletons."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ragcore.config import Config
from ragcore.stub.jobs import JobManager
from ragcore.stub.store import Store


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


ConfigDep = Annotated[Config, Depends(get_config)]
StoreDep = Annotated[Store, Depends(get_store)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]
