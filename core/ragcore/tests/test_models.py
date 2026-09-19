"""Model manager: inventory, hardware profile, hub browsing, download, activation."""

from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient
from ragcore.api.schemas import HubFile, HubModel, HubModelDetail
from ragcore.stub.jobs import JobManager


class FakeHub:
    """Stands in for `HubClient`; records what the routes asked it for."""

    def __init__(self, *, files: list[HubFile] | None = None, raises: Exception | None = None):
        self.files = files if files is not None else [
            HubFile(path="model-Q4_K_M.gguf", size_bytes=400_000_000, quant="Q4_K_M", fit="ram")
        ]
        self.raises = raises
        self.calls: list[tuple] = []

    def _model(self, repo_id: str = "Qwen/Qwen3-Embedding-0.6B-GGUF") -> HubModel:
        return HubModel(id=repo_id, name=repo_id.split("/")[-1], gguf=True)

    async def search(self, *, role, query, sort, limit):
        self.calls.append(("search", role, query, sort, limit))
        if self.raises:
            raise self.raises
        return [self._model()]

    async def recommended(self, role):
        self.calls.append(("recommended", role))
        if self.raises:
            raise self.raises
        return [self._model()]

    async def detail(self, repo_id, role=None):
        self.calls.append(("detail", repo_id, role))
        if self.raises:
            raise self.raises
        return HubModelDetail(model=self._model(repo_id), files=self.files)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def hub(client: TestClient) -> FakeHub:
    fake = FakeHub()
    client.app.state.hub = fake
    return fake


@pytest.fixture
def instant_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the staged runner's sleeps; the stages themselves are tested in test_jobs."""

    def immediate(stages, jitter: float = 0.35):
        async def run(job, report) -> None:
            for name, _ in stages:
                await report(1.0, name)

        return run

    monkeypatch.setattr(JobManager, "staged_runner", staticmethod(immediate))


def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["state"] in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ------------------------------------------------------------------- inventory


def test_the_shipped_models_are_installed_and_active(client: TestClient) -> None:
    inventory = client.get("/models").json()

    roles = {model["role"] for model in inventory["installed"]}
    assert roles == {"embedding", "reranking"}
    assert inventory["active_embedding"] is not None
    assert inventory["active_reranking"] is not None
    assert all(model["shipped"] for model in inventory["installed"])


def test_the_inventory_is_ordered_by_role_then_name(client: TestClient) -> None:
    installed = client.get("/models").json()["installed"]

    assert [m["role"] for m in installed] == sorted(m["role"] for m in installed)


# -------------------------------------------------------------------- hardware


def test_hardware_reports_what_the_shell_measured(client: TestClient) -> None:
    """The conftest config says 16 GB of RAM, no VRAM, CPU backend."""
    hardware = client.get("/models/hardware").json()

    assert hardware["ram_mb"] == 16384
    assert hardware["vram_mb"] == 0
    assert hardware["gpu_backend"] == "cpu"
    assert hardware["profile"] == "cpu"
    assert hardware["cpu_count"] >= 1


@pytest.mark.parametrize(
    ("vram_mb", "gpu_backend", "profile"),
    [(24576, "cuda", "gpu"), (4096, "metal", "balanced"), (0, "cpu", "cpu")],
)
def test_the_profile_follows_the_vram_budget(
    client: TestClient, vram_mb: int, gpu_backend: str, profile: str
) -> None:
    client.app.state.config.vram_mb = vram_mb
    client.app.state.config.gpu_backend = gpu_backend

    assert client.get("/models/hardware").json()["profile"] == profile


# ------------------------------------------------------------------------- hub


def test_hub_search_passes_the_filters_through(client: TestClient, hub: FakeHub) -> None:
    response = client.get(
        "/models/hub/search", params={"role": "embedding", "q": "qwen", "sort": "likes", "limit": 5}
    )

    assert response.status_code == 200
    assert hub.calls == [("search", "embedding", "qwen", "likes", 5)]
    assert response.json()["role"] == "embedding"
    assert response.json()["query"] == "qwen"


def test_hub_search_rejects_a_sort_the_hub_does_not_offer(client: TestClient) -> None:
    assert client.get("/models/hub/search", params={"sort": "vibes"}).status_code == 422


def test_hub_search_caps_the_page_size(client: TestClient) -> None:
    assert client.get("/models/hub/search", params={"limit": 500}).status_code == 422


def test_hub_recommended_needs_a_role(client: TestClient) -> None:
    assert client.get("/models/hub/recommended").status_code == 422


def test_hub_recommended_asks_for_that_roles_picks(client: TestClient, hub: FakeHub) -> None:
    assert client.get("/models/hub/recommended", params={"role": "reranking"}).status_code == 200
    assert hub.calls == [("recommended", "reranking")]


def test_hub_detail_accepts_a_slashed_repo_id(client: TestClient, hub: FakeHub) -> None:
    response = client.get("/models/hub/Qwen/Qwen3-Embedding-0.6B-GGUF")

    assert response.status_code == 200
    assert hub.calls == [("detail", "Qwen/Qwen3-Embedding-0.6B-GGUF", None)]
    assert response.json()["files"][0]["path"] == "model-Q4_K_M.gguf"


def test_an_unreachable_hub_becomes_a_502_not_a_crash(client: TestClient) -> None:
    client.app.state.hub = FakeHub(
        raises=httpx.ConnectError("no route", request=httpx.Request("GET", "https://hf.co"))
    )

    for path, params in (
        ("/models/hub/search", {}),
        ("/models/hub/recommended", {"role": "embedding"}),
        ("/models/hub/Qwen/thing", {}),
    ):
        response = client.get(path, params=params)
        assert response.status_code == 502, path
        assert "Hugging Face unreachable" in response.json()["detail"]


def test_a_hub_404_is_forwarded_with_its_status(client: TestClient) -> None:
    request = httpx.Request("GET", "https://huggingface.co/api/models/nobody/nothing")
    client.app.state.hub = FakeHub(
        raises=httpx.HTTPStatusError("404", request=request, response=httpx.Response(404))
    )

    response = client.get("/models/hub/nobody/nothing")

    assert response.status_code == 404


# -------------------------------------------------------------------- download


def test_downloading_a_new_embedder_without_accepting_the_reindex_is_refused(
    client: TestClient, hub: FakeHub
) -> None:
    response = client.post(
        "/models/download",
        json={
            "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",
            "filename": "model-Q4_K_M.gguf",
            "role": "embedding",
            "activate": True,
        },
    )

    assert response.status_code == 409
    assert "accept_reindex" in response.json()["detail"]
    assert hub.calls == [], "the refusal must come before any network call"


def test_downloading_an_embedder_without_activating_it_is_allowed(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    response = client.post(
        "/models/download",
        json={
            "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",
            "filename": "model-Q4_K_M.gguf",
            "role": "embedding",
        },
    )

    assert response.status_code == 200


def test_downloading_a_file_the_repo_does_not_have_is_a_404(
    client: TestClient, hub: FakeHub
) -> None:
    response = client.post(
        "/models/download",
        json={"repo_id": "Qwen/whatever", "filename": "absent.gguf", "role": "reranking"},
    )

    assert response.status_code == 404
    assert "absent.gguf" in response.json()["detail"]


def test_a_download_job_carries_the_file_size_as_its_total(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={"repo_id": "Qwen/whatever", "filename": "model-Q4_K_M.gguf", "role": "reranking"},
    ).json()

    assert job["kind"] == "download"
    assert job["total"] == 400_000_000
    assert job["detail"] == "from Qwen/whatever"


def test_a_finished_download_installs_the_model(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={"repo_id": "Qwen/whatever", "filename": "model-Q4_K_M.gguf", "role": "reranking"},
    ).json()
    assert wait_for_job(client, job["id"])["state"] == "done"

    installed = {m["id"]: m for m in client.get("/models").json()["installed"]}
    model = installed["qwen_whatever_model-q4_k_m.gguf"]
    assert model["role"] == "reranking"
    assert model["name"] == "model-Q4_K_M"
    assert model["quant"] == "Q4_K_M"
    assert model["shipped"] is False
    assert model["active"] is False


def test_a_download_that_activates_deactivates_the_previous_model_for_that_role(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={
            "repo_id": "Qwen/whatever",
            "filename": "model-Q4_K_M.gguf",
            "role": "reranking",
            "activate": True,
        },
    ).json()
    wait_for_job(client, job["id"])

    inventory = client.get("/models").json()
    assert inventory["active_reranking"] == "qwen_whatever_model-q4_k_m.gguf"
    # Swapping the reranker must not disturb the embedder.
    assert inventory["active_embedding"] == "embed-qwen3-0.6b"
    active = [m["id"] for m in inventory["installed"] if m["active"]]
    assert len(active) == 2


# ------------------------------------------------------------------ activation


def test_activating_an_unknown_model_is_a_404(client: TestClient) -> None:
    assert client.post("/models/nope/activate").status_code == 404


def test_switching_the_embedder_needs_the_reindex_accepted(client: TestClient) -> None:
    response = client.post("/models/embed-qwen3-0.6b/activate")

    assert response.status_code == 409
    assert "accept_reindex" in response.json()["detail"]


def test_accepting_the_reindex_activates_the_embedder(client: TestClient) -> None:
    response = client.post(
        "/models/embed-qwen3-0.6b/activate", params={"accept_reindex": "true"}
    )

    assert response.status_code == 200
    assert response.json()["active"] is True


def test_activating_a_reranker_needs_no_confirmation(client: TestClient) -> None:
    """Reranker vectors are not stored, so swapping one does not invalidate the index."""
    assert client.post("/models/rerank-qwen3-0.6b/activate").status_code == 200


def test_activation_is_exclusive_within_a_role(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={"repo_id": "Qwen/whatever", "filename": "model-Q4_K_M.gguf", "role": "reranking"},
    ).json()
    wait_for_job(client, job["id"])

    client.post("/models/qwen_whatever_model-q4_k_m.gguf/activate")

    rerankers = [m for m in client.get("/models").json()["installed"] if m["role"] == "reranking"]
    assert [m["id"] for m in rerankers if m["active"]] == ["qwen_whatever_model-q4_k_m.gguf"]


# -------------------------------------------------------------------- deletion


def test_deleting_an_unknown_model_is_a_404(client: TestClient) -> None:
    assert client.delete("/models/nope").status_code == 404


def test_a_shipped_model_cannot_be_removed(client: TestClient) -> None:
    response = client.delete("/models/rerank-qwen3-0.6b")

    assert response.status_code == 409
    assert "Shipped" in response.json()["detail"]


def test_the_active_model_for_a_role_cannot_be_removed(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={
            "repo_id": "Qwen/whatever",
            "filename": "model-Q4_K_M.gguf",
            "role": "reranking",
            "activate": True,
        },
    ).json()
    wait_for_job(client, job["id"])

    response = client.delete("/models/qwen_whatever_model-q4_k_m.gguf")

    assert response.status_code == 409
    assert "Activate another" in response.json()["detail"]


def test_an_inactive_downloaded_model_can_be_removed(
    client: TestClient, hub: FakeHub, instant_jobs: None
) -> None:
    job = client.post(
        "/models/download",
        json={"repo_id": "Qwen/whatever", "filename": "model-Q4_K_M.gguf", "role": "reranking"},
    ).json()
    wait_for_job(client, job["id"])

    assert client.delete("/models/qwen_whatever_model-q4_k_m.gguf").json() == {"ok": True}
    assert "qwen_whatever_model-q4_k_m.gguf" not in [
        m["id"] for m in client.get("/models").json()["installed"]
    ]
