from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ragcore.api.errors import ApiError, api_error, api_error_handler


def _app() -> TestClient:
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)

    @app.get("/coded")
    def coded():
        raise api_error(404, "thing_missing", "thing 7 is missing", id="7")

    @app.get("/plain")
    def plain():
        from fastapi import HTTPException

        raise HTTPException(409, "plain text")

    return TestClient(app)


def test_a_coded_error_keeps_the_english_detail_and_adds_code_and_params() -> None:
    response = _app().get("/coded")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "thing 7 is missing",
        "code": "thing_missing",
        "params": {"id": "7"},
    }


def test_an_uncoded_error_is_left_alone() -> None:
    response = _app().get("/plain")

    assert response.status_code == 409
    assert response.json() == {"detail": "plain text"}


def test_a_real_route_reports_its_code(client) -> None:
    response = client.get("/chats/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "session not found"
    assert response.json()["code"] == "session_not_found"
