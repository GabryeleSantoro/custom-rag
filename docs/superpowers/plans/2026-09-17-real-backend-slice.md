# Real Backend Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ragcore` stub backend with a real one along a single narrow path — real embeddings, a real LanceDB index, real hybrid retrieval with reranking, real PDF ingestion — runnable headless, with the frozen HTTP contract unchanged.

**Architecture:** A Protocol seam (`ports.py`) plus a factory (`backend.py`) lets a stub backend and a real backend satisfy the same API. The real backend stores vectors and text in LanceDB, application state in SQLite, and talks to two `llama-server` processes over HTTP for embedding and reranking. The six backend-agnostic contract tests run against both backends, which is how we prove the contract never moved; the seven that assert stub-only semantics (dev directives, seeded connections, model inventory, eval sets) stay on the stub, since those subsystems are out of this slice's scope.

**Tech Stack:** Python 3.13, FastAPI, httpx, LanceDB, PyArrow, NumPy, pypdfium2, SQLite (stdlib `sqlite3`), pytest, llama.cpp `llama-server`.

**Spec:** `docs/superpowers/specs/2026-09-17-real-backend-slice-design.md`

## Global Constraints

- Python `>=3.13`. Dependencies go on the `ragcore` package (`core/ragcore/pyproject.toml`), not the workspace root.
- Ruff: `line-length = 100`, `target-version = "py313"`, lint rules `["E", "F", "I", "UP", "B", "SIM"]`. Run `uv run ruff check .` and `uv run ruff format .` before every commit.
- pytest: `asyncio_mode = "auto"`, `testpaths = ["core/ragcore/tests"]`.
- **No `torch` in `ragcore` dependencies, ever.** The S0 spike installs it in a throwaway virtualenv outside the project.
- **No AGPL dependencies.** Specifically never PyMuPDF or `pymupdf4llm`. PDF parsing is `pypdfium2` (Apache/BSD).
- `EMBED_DIM = 1024`. RRF constant `k = 60`.
- **The HTTP contract is frozen.** Do not edit `core/ragcore/src/ragcore/api/schemas.py`, do not change SSE frame order (`start` → `mode` → `sources` → `token`* → `citations` → `done`, with `error` replacing everything from the point it occurs), do not change route paths or status codes. If a task appears to require a schema change, stop and escalate.
- **No Rust changes in this slice.** `apps/desktop/` is untouched.
- Embedder is started with `--embedding --pooling last`. Qwen3 embedding models pool from the last token; mean pooling silently degrades recall.
- Commit after every task. Conventional Commits style subjects.

---

### Task 1: Extract the citation helpers into a shared module

`api/routes/query.py:33` imports `extract_citations` and `parse_directives` from `ragcore.stub.answers`. Both are pure and backend-agnostic, so they must not live in `stub/` once a second backend exists. This task moves them with no behavior change.

**Files:**
- Create: `core/ragcore/src/ragcore/citations.py`
- Create: `core/ragcore/tests/test_citations.py`
- Modify: `core/ragcore/src/ragcore/stub/answers.py` (delete the moved functions, import them instead)
- Modify: `core/ragcore/src/ragcore/api/routes/query.py:33-38` (import from the new module)

**Interfaces:**
- Consumes: `RetrievedChunk`, `Citation`, `Grounding` from `ragcore.api.schemas`.
- Produces:
  - `MARKER: re.Pattern[str]`
  - `extract_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[list[Citation], int, Grounding]`
  - `parse_directives(question: str) -> tuple[str, set[str]]`

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_citations.py`:

```python
"""Citation parsing is contract-critical: a marker nothing retrieved must never render."""

from __future__ import annotations

from datetime import UTC, datetime

from ragcore.api.schemas import RetrievedChunk
from ragcore.citations import extract_citations, parse_directives


def chunk(doc_id: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c_{doc_id}_{page}",
        doc_id=doc_id,
        doc_title=doc_id.title(),
        page_start=page,
        page_end=page,
        text="Reranking reorders candidates with a cross-encoder.",
        rerank_score=0.9,
    )


def test_marker_backed_by_a_retrieved_chunk_is_kept() -> None:
    citations, dropped, grounding = extract_citations(
        "Rerankers reorder results [reranking:1] and improve precision [reranking:2].",
        [chunk("reranking", 1), chunk("reranking", 2)],
    )

    assert [c.marker for c in citations] == ["[reranking:1]", "[reranking:2]"]
    assert dropped == 0
    assert grounding == "ok"


def test_marker_for_an_unretrieved_document_is_dropped() -> None:
    citations, dropped, grounding = extract_citations(
        "As shown in [invented:7], the answer is yes.", [chunk("reranking", 1)]
    )

    assert citations == []
    assert dropped == 1
    assert grounding == "none"


def test_repeated_marker_is_emitted_once() -> None:
    citations, dropped, _ = extract_citations(
        "First [reranking:1]. Again [reranking:1].", [chunk("reranking", 1)]
    )

    assert len(citations) == 1
    assert dropped == 0


def test_directives_are_stripped_from_the_question() -> None:
    question, directives = parse_directives("!slow what is reranking?")

    assert question == "what is reranking?"
    assert "slow" in directives
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_citations.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.citations'`

- [ ] **Step 3: Create the module by moving the code**

Create `core/ragcore/src/ragcore/citations.py`. Move `MARKER`, `extract_citations` and `parse_directives` out of `stub/answers.py` verbatim — do not retype them, do not "improve" them. The file header:

```python
"""Citation marker parsing and dev directives.

Backend-agnostic: both the stub and the real backend produce answers containing
`[doc_id:page]` markers, and both must drop any marker that no retrieved passage
backs. Lives outside `stub/` because the API layer imports it directly.
"""

from __future__ import annotations

import re

from ragcore.api.schemas import Citation, Grounding, RetrievedChunk

MARKER = re.compile(r"\[([A-Za-z0-9_.\-]+):(\d+)\]")
```

Then the two moved function bodies, unchanged.

- [ ] **Step 4: Update the two importers**

In `core/ragcore/src/ragcore/stub/answers.py`, delete the moved definitions (`MARKER`, `extract_citations`, `parse_directives`). `MARKER`'s only use was inside `extract_citations`, which is leaving, so drop the `import re` too if nothing else in the file uses it — `ruff check` will tell you. Do **not** re-export the moved names: `backend.py` (Task 2) imports `llm_stream`/`scripted_stream` from this module and the citation helpers from `ragcore.citations`, so a re-export would be dead weight that Task 11 has to clean up again.

In `core/ragcore/src/ragcore/api/routes/query.py`, change the import block at line 33 to:

```python
from ragcore.citations import extract_citations, parse_directives
from ragcore.stub.answers import llm_stream, scripted_stream
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — 4 new citation tests plus the 13 existing contract tests.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore/src/ragcore/citations.py core/ragcore/tests/test_citations.py core/ragcore/src/ragcore/stub/answers.py core/ragcore/src/ragcore/api/routes/query.py
git commit -m "refactor: move citation parsing out of the stub package"
```

---

### Task 2: Add the backend seam

Introduce the Protocols and the factory, and route the existing stub through them. Nothing changes behaviorally; afterwards the API layer no longer names `stub` anywhere.

**Files:**
- Create: `core/ragcore/src/ragcore/ports.py`
- Create: `core/ragcore/src/ragcore/backend.py`
- Create: `core/ragcore/tests/test_backend.py`
- Modify: `core/ragcore/src/ragcore/config.py` (add `backend` field, env fallback)
- Modify: `core/ragcore/src/ragcore/cli.py` (add `--backend`)
- Modify: `core/ragcore/src/ragcore/api/app.py:26-45` (lifespan builds a `Backend`)
- Modify: `core/ragcore/src/ragcore/api/deps.py` (resolve from `app.state.backend`, add `AnswererDep`)
- Modify: `core/ragcore/src/ragcore/api/routes/query.py` (use the injected answerer)
- Modify: `core/ragcore/src/ragcore/api/routes/models.py:22` (hub comes from the backend)

**Interfaces:**
- Consumes: `Config`, the stub classes `Store`, `JobManager`, `HubClient`.
- Produces:
  - `ports.StorePort`, `ports.RetrieverPort`, `ports.AnswerEngine` (Protocols)
  - `backend.Backend` — frozen dataclass with fields `store`, `jobs`, `hub`, `answerer`
  - `backend.build_backend(config: Config) -> Backend`
  - `deps.AnswererDep`
  - `Config.backend: Literal["stub", "real"]`, default `"stub"`

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_backend.py`:

```python
"""The seam itself: the factory honours the flag and the app never names a backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragcore.backend import build_backend
from ragcore.config import Config


def config_for(tmp_path: Path, backend: str) -> Config:
    return Config(host="127.0.0.1", port=0, token="", data_dir=tmp_path, backend=backend)


def test_stub_is_the_default(tmp_path: Path) -> None:
    assert Config(data_dir=tmp_path).backend == "stub"


def test_factory_builds_the_stub_backend(tmp_path: Path) -> None:
    built = build_backend(config_for(tmp_path, "stub"))

    assert type(built.store).__module__.startswith("ragcore.stub")
    assert built.answerer is not None


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend(config_for(tmp_path, "nonsense"))
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_backend.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.backend'`

- [ ] **Step 3: Write the Protocols**

Create `core/ragcore/src/ragcore/ports.py`:

```python
"""The seam between the API layer and whatever is answering it.

The API layer was written against the stub's shape, so these Protocols describe
that shape rather than an idealised one. A real backend has to fit the existing
routes; the routes are contract and do not bend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ragcore.api.schemas import (
    AppSettings,
    ChatMessage,
    ChatSession,
    Connection,
    Document,
    DocumentContent,
    EvalSet,
    IndexStats,
    InstalledModel,
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    Source,
    SourceCreate,
    StageLatency,
)


@runtime_checkable
class RetrieverPort(Protocol):
    def search(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        """Returns the kept chunks, per-stage latency, and the candidate count."""
        ...


@runtime_checkable
class StorePort(Protocol):
    settings: AppSettings
    sources: dict[str, Source]
    documents: dict[str, Document]
    connections: dict[str, Connection]
    models: dict[str, InstalledModel]
    sessions: dict[str, ChatSession]
    messages: dict[str, list[ChatMessage]]
    eval_sets: dict[str, EvalSet]
    retriever: RetrieverPort

    def add_source(self, payload: SourceCreate) -> Source: ...
    def remove_source(self, source_id: str) -> int: ...
    def ingest_source(self, source_id: str) -> list[Document]: ...
    def rebuild_index(self) -> None: ...
    def doc_meta(self) -> dict[str, dict]: ...
    def content(self, doc_id: str) -> DocumentContent | None: ...
    def index_stats(self) -> IndexStats: ...
    def create_session(self, title: str | None, scope_doc_id: str | None = None) -> ChatSession: ...
    def append_message(self, message: ChatMessage) -> None: ...
    def new_id(self, prefix: str) -> str: ...


@runtime_checkable
class AnswerEngine(Protocol):
    def stream(
        self, question: str, chunks: list[RetrievedChunk], directives: set[str]
    ) -> AsyncIterator[str]:
        """Yields answer text pieces. Directives are stub-only and ignored by real engines."""
        ...
```

Note on `store.loaded`: `routes/documents.py` reads it in three places. Leave it off the Protocol — it is stub-internal, and Task 9 replaces those reads with `store.content()`.

- [ ] **Step 4: Write the factory**

Create `core/ragcore/src/ragcore/backend.py`:

```python
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
```

The real branch lands in Task 11. Raising for `"real"` right now is correct: the plan never leaves a half-wired branch reachable.

- [ ] **Step 5: Add the config field and the CLI flag**

In `config.py`, add to the dataclass (after `gpu_backend`):

```python
    backend: Literal["stub", "real"] = "stub"
```

Add `from typing import Literal` to the imports, and in `from_env`, add:

```python
            backend=os.getenv("RAGCORE_BACKEND", "stub"),  # type: ignore[arg-type]
```

In `cli.py`, add to the `serve` parser:

```python
    serve.add_argument("--backend", default="stub", choices=["stub", "real"])
```

and pass `backend=args.backend` into the `Config(...)` construction in `main`.

- [ ] **Step 6: Wire the app and deps**

In `api/app.py`, replace the three `ragcore.stub.*` imports with `from ragcore.backend import build_backend`, and rewrite the lifespan body:

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.started_at = time.monotonic()
        backend = build_backend(config)
        app.state.backend = backend
        app.state.store = backend.store
        app.state.jobs = backend.jobs
        app.state.hub = backend.hub
        app.state.answerer = backend.answerer
        try:
            yield
        finally:
            await backend.jobs.shutdown()
            await backend.hub.aclose()
```

In `api/deps.py`, drop the `ragcore.stub` imports, type `get_store` as `StorePort`, and add:

```python
def get_answerer(request: Request) -> AnswerEngine:
    return request.app.state.answerer


AnswererDep = Annotated[AnswerEngine, Depends(get_answerer)]
```

- [ ] **Step 7: Route the query through the injected answerer**

In `routes/query.py`, delete `from ragcore.stub.answers import llm_stream, scripted_stream`, add `AnswererDep` to the signature, and replace the stream selection block:

```python
        stream = answerer.stream(question, chunks, directives)
```

Delete the now-dead `use_llm` local. In `routes/models.py:22`, delete the `HubClient` import and read the hub from `request.app.state.hub` the way the other routes read their dependencies.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — 3 backend tests, 4 citation tests, 13 contract tests.

Then confirm the seam is real:

Run: `grep -rn "ragcore.stub" core/ragcore/src/ragcore/api/`
Expected: no output.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore/src/ragcore core/ragcore/tests/test_backend.py
git commit -m "feat: add backend protocol seam and stub/real switch"
```

---

### Task 3: S0 — the reranker gate

**This task is a gate, not a deliverable.** Its output is a number that either permits the rest of the plan or stops it. The scripts are dev tooling; the spike script is throwaway.

**Files:**
- Create: `scripts/fetch-models.sh`
- Create: `scripts/dev/serve-models.sh`
- Create: `scripts/spikes/rerank_check.py`
- Create: `docs/superpowers/notes/2026-09-17-s0-reranker-gate.md` (the recorded result)

**Interfaces:**
- Produces: two running `llama-server` instances on ports 8770 (embed) and 8771 (rerank), and a recorded verdict. Later tasks assume those ports in their dev instructions.

- [ ] **Step 1: Write the model fetch script**

Create `scripts/fetch-models.sh`:

```bash
#!/usr/bin/env bash
# Downloads the shipped GGUF models for local development.
# Production downloads go through the model hub; this is the dev shortcut.
set -euo pipefail

DEST="${RAGCORE_MODEL_DIR:-$HOME/.custom-rag/models}"
mkdir -p "$DEST"

fetch() {
  local url="$1" name="$2"
  if [[ -f "$DEST/$name" ]]; then
    echo "have $name"
    return
  fi
  echo "fetching $name"
  curl -fL --progress-bar -o "$DEST/$name" "$url"
}

fetch "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf" \
      "Qwen3-Embedding-0.6B-Q8_0.gguf"
fetch "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B-GGUF/resolve/main/Qwen3-Reranker-0.6B-Q8_0.gguf" \
      "Qwen3-Reranker-0.6B-Q8_0.gguf"

echo "models in $DEST"
ls -lh "$DEST"
```

Make it executable: `chmod +x scripts/fetch-models.sh`

If a URL 404s, list the repo's files at `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/main` and use the actual Q8_0 filename. Do not silently substitute a different quantisation — record whatever you used in the notes file in Step 6.

- [ ] **Step 2: Write the dev server script**

Create `scripts/dev/serve-models.sh`:

```bash
#!/usr/bin/env bash
# Starts the two llama-server instances ragcore's real backend talks to.
# Embedder: port 8770. Reranker: port 8771.
#
# --pooling last is not optional: Qwen3 embedding models pool from the last
# token, and mean pooling returns plausible vectors with quietly worse recall.
set -euo pipefail

DEST="${RAGCORE_MODEL_DIR:-$HOME/.custom-rag/models}"
LLAMA="${LLAMA_SERVER:-llama-server}"

"$LLAMA" -m "$DEST/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --embedding --pooling last -c 8192 -ngl 999 --port 8770 --host 127.0.0.1 &
EMBED_PID=$!

"$LLAMA" -m "$DEST/Qwen3-Reranker-0.6B-Q8_0.gguf" \
  --reranking -c 8192 -ngl 999 --port 8771 --host 127.0.0.1 &
RERANK_PID=$!

trap 'kill $EMBED_PID $RERANK_PID 2>/dev/null || true' EXIT
echo "embed pid $EMBED_PID :8770 | rerank pid $RERANK_PID :8771"
wait
```

Make it executable: `chmod +x scripts/dev/serve-models.sh`

Install `llama-server` if absent: `brew install llama.cpp`.

- [ ] **Step 3: Write the comparison spike**

Create `scripts/spikes/rerank_check.py`:

```python
"""S0 gate: does llama-server rank the way the reference implementation does?

Throwaway. Needs torch and transformers, which must NOT be added to ragcore.
Run it in a scratch virtualenv:

    python3 -m venv /tmp/s0 && /tmp/s0/bin/pip install torch transformers httpx scipy
    /tmp/s0/bin/python scripts/spikes/rerank_check.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import httpx
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

RERANK_URL = "http://127.0.0.1:8771/v1/rerank"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
TASK = "Given a web search query, retrieve relevant passages that answer the query"

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def pairs() -> list[tuple[str, str]]:
    """50 (query, passage) pairs drawn from the repo's fixture corpus."""
    docs = sorted(Path("fixtures/docs").glob("*.md"))
    passages = []
    for doc in docs:
        blocks = [b.strip() for b in doc.read_text().split("\n\n") if len(b.strip()) > 120]
        passages.extend(blocks[:4])
    queries = [
        "how does reranking improve retrieval precision",
        "what chunk size works best for embeddings",
        "how do hybrid search and BM25 combine",
        "which metrics evaluate a RAG system",
        "what makes an embedding model suitable for retrieval",
    ]
    combos = list(itertools.product(queries, passages))[:50]
    assert len(combos) == 50, f"need 50 pairs, built {len(combos)}"
    return combos


def llama_scores(data: list[tuple[str, str]]) -> list[float]:
    out: list[float] = []
    with httpx.Client(timeout=120) as client:
        for query, passage in data:
            response = client.post(
                RERANK_URL, json={"query": query, "documents": [passage], "top_n": 1}
            )
            response.raise_for_status()
            out.append(float(response.json()["results"][0]["relevance_score"]))
    return out


def reference_scores(data: list[tuple[str, str]]) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    out: list[float] = []
    for query, passage in data:
        body = f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {passage}"
        ids = tokenizer(PREFIX + body + SUFFIX, return_tensors="pt", truncation=True,
                        max_length=4096)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        both = torch.stack([logits[no_id], logits[yes_id]])
        out.append(float(torch.softmax(both, dim=0)[1]))
    return out


def main() -> int:
    data = pairs()
    mine = llama_scores(data)
    reference = reference_scores(data)

    rho = float(spearmanr(mine, reference).statistic)
    top1 = int(max(range(50), key=mine.__getitem__) == max(range(50), key=reference.__getitem__))
    agree = sum(
        (mine[i] > mine[j]) == (reference[i] > reference[j])
        for i, j in itertools.combinations(range(50), 2)
    ) / len(list(itertools.combinations(range(50), 2)))

    verdict = {
        "spearman": round(rho, 4),
        "pairwise_agreement": round(agree, 4),
        "top1_match": top1,
        "pass": rho >= 0.9 and agree >= 0.9,
    }
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Check the embedder while the servers are up**

Run:

```bash
curl -s http://127.0.0.1:8770/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["reranking reorders candidates"],"model":"qwen3-embedding"}' \
| python3 -c 'import json,sys,math; v=json.load(sys.stdin)["data"][0]["embedding"]; print("dim", len(v), "norm", round(math.sqrt(sum(x*x for x in v)),4))'
```

Expected: `dim 1024`. Record the norm — if it is not ~1.0, `models/embed.py` must L2-normalize client-side (Task 4 handles both cases, but the note tells you which path is live).

- [ ] **Step 5: Run the gate**

```bash
./scripts/fetch-models.sh
./scripts/dev/serve-models.sh &   # leave running
python3 -m venv /tmp/s0 && /tmp/s0/bin/pip install torch transformers httpx scipy
/tmp/s0/bin/python scripts/spikes/rerank_check.py
```

Expected: `"pass": true`, with `spearman >= 0.9` and `pairwise_agreement >= 0.9`.

**If it fails: STOP.** Do not start Task 4. The fallback is ONNX Runtime for the reranker, which changes the `models/` design and therefore the spec. Report the numbers and escalate.

- [ ] **Step 6: Record the verdict**

Create `docs/superpowers/notes/2026-09-17-s0-reranker-gate.md` with: the exact GGUF filenames used, the `llama-server` version (`llama-server --version`), the machine and GPU backend, the JSON verdict, and the embedding dim and norm from Step 4.

- [ ] **Step 7: Commit**

```bash
git add scripts docs/superpowers/notes
git commit -m "chore: add model fetch, dev servers and the S0 reranker gate"
```

---

### Task 4: Embedding client

**Files:**
- Create: `core/ragcore/src/ragcore/models/__init__.py`
- Create: `core/ragcore/src/ragcore/models/embed.py`
- Create: `core/ragcore/src/ragcore/models/fakes.py`
- Create: `core/ragcore/tests/test_embed.py`
- Modify: `core/ragcore/pyproject.toml` (add `numpy`)
- Modify: `pyproject.toml` (register the `requires_models` marker)

**Interfaces:**
- Produces:
  - `models.embed.EmbedClient(base_url: str, *, model: str = "qwen3-embedding", timeout: float = 120.0, normalize: bool = True)`
  - `async EmbedClient.embed(texts: list[str], *, batch_size: int = 32) -> list[list[float]]`
  - `async EmbedClient.aclose() -> None`
  - `EmbedClient.dim: int = 1024` (module constant `EMBED_DIM = 1024`)
  - `models.fakes.FakeEmbedClient()` with the same `embed`/`aclose`/`dim` surface

- [ ] **Step 1: Add dependencies and the test marker**

```bash
uv add --package ragcore numpy
```

In the workspace `pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
markers = ["requires_models: needs a live llama-server on 8770/8771"]
addopts = "-m 'not requires_models'"
```

- [ ] **Step 2: Write the failing test**

Create `core/ragcore/tests/test_embed.py`:

```python
"""The embed client's contract: batching, dimension, normalisation."""

from __future__ import annotations

import math

import pytest
from ragcore.models.embed import EMBED_DIM, EmbedClient
from ragcore.models.fakes import FakeEmbedClient


async def test_fake_client_is_deterministic_and_the_right_shape() -> None:
    client = FakeEmbedClient()
    first = await client.embed(["reranking reorders candidates"])
    second = await client.embed(["reranking reorders candidates"])

    assert len(first) == 1
    assert len(first[0]) == EMBED_DIM
    assert first == second


async def test_fake_client_returns_unit_vectors() -> None:
    [vector] = await FakeEmbedClient().embed(["chunking strategies"])

    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


async def test_fake_client_separates_different_texts() -> None:
    a, b = await FakeEmbedClient().embed(["hybrid search", "optical character recognition"])

    assert sum(x * y for x, y in zip(a, b, strict=True)) < 0.9


async def test_batches_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    seen: list[int] = []

    async def fake_post(texts: list[str]) -> list[list[float]]:
        seen.append(len(texts))
        return [[0.0] * EMBED_DIM for _ in texts]

    monkeypatch.setattr(client, "_post", fake_post)
    await client.embed([f"text {i}" for i in range(70)], batch_size=32)

    assert seen == [32, 32, 6]


@pytest.mark.requires_models
async def test_live_server_returns_1024_dimensions() -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    try:
        [vector] = await client.embed(["reranking reorders candidates"])
    finally:
        await client.aclose()

    assert len(vector) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-3)
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_embed.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.models'`

- [ ] **Step 4: Implement the client and the fake**

Create `core/ragcore/src/ragcore/models/__init__.py` (empty) and `core/ragcore/src/ragcore/models/embed.py`:

```python
"""Embedding over llama-server's OpenAI-compatible endpoint.

The server is started with `--embedding --pooling last`. Normalisation is done
here regardless of what the server returns, because the index stores unit
vectors and cosine similarity on unit vectors is a dot product.
"""

from __future__ import annotations

import httpx
import numpy as np

EMBED_DIM = 1024


class EmbedClient:
    dim = EMBED_DIM

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "qwen3-embedding",
        timeout: float = 120.0,
        normalize: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.normalize = normalize
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    async def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            out.extend(await self._post(texts[start : start + batch_size]))
        if not self.normalize:
            return out
        matrix = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    async def aclose(self) -> None:
        await self._client.aclose()
```

Create `core/ragcore/src/ragcore/models/fakes.py`:

```python
"""Deterministic stand-ins so the real backend is testable without llama-server.

Vectors are hash-derived, so the same text always lands in the same place and
different texts land apart. They carry no semantics — these are for wiring
tests, never for quality measurement.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ragcore.models.embed import EMBED_DIM


def _vector(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


class FakeEmbedClient:
    dim = EMBED_DIM

    async def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        return [_vector(text) for text in texts]

    async def aclose(self) -> None:
        return None


class FakeRerankClient:
    """Scores by literal token overlap: crude, but it reorders, which is what tests check."""

    async def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
        terms = set(query.lower().split())
        scored = [
            (i, len(terms & set(doc.lower().split())) / (len(terms) or 1))
            for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n]

    async def aclose(self) -> None:
        return None
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_embed.py -v`
Expected: PASS (4 tests; the `requires_models` one is deselected).

With servers up: `uv run pytest core/ragcore/tests/test_embed.py -v -m requires_models`
Expected: PASS (1 test).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore pyproject.toml uv.lock
git commit -m "feat: add llama-server embedding client and deterministic fakes"
```

---

### Task 5: SQLite application store

**Files:**
- Create: `core/ragcore/src/ragcore/store/__init__.py`
- Create: `core/ragcore/src/ragcore/store/meta.py`
- Create: `core/ragcore/tests/test_meta_store.py`

**Interfaces:**
- Produces: `store.meta.MetaStore(db_path: Path)` with
  - `add_source(payload: SourceCreate) -> Source`
  - `list_sources() -> dict[str, Source]`
  - `remove_source(source_id: str) -> list[str]` (returns removed doc ids)
  - `upsert_document(doc: Document) -> None`
  - `list_documents() -> dict[str, Document]`
  - `get_document(doc_id: str) -> Document | None`
  - `delete_documents(doc_ids: list[str]) -> None`
  - `sha_index() -> dict[str, str]` mapping `path -> sha256`
  - `set_sha(path: str, sha: str) -> None`
  - `wipe(*, keep_connections: bool) -> None` — clears sources, documents, shas, sessions and messages

  - `create_session(title: str | None, scope_doc_id: str | None) -> ChatSession`
  - `list_sessions() -> dict[str, ChatSession]`
  - `append_message(message: ChatMessage) -> None`
  - `list_messages() -> dict[str, list[ChatMessage]]`
  - `load_settings(default: AppSettings) -> AppSettings`
  - `save_settings(settings: AppSettings) -> None`
  - `close() -> None`

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_meta_store.py`:

```python
"""SQLite state: sources, documents, chats, settings. Survives a reopen."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ragcore.api.schemas import AppSettings, ChatMessage, Document, SourceCreate
from ragcore.store.meta import MetaStore


def a_document(doc_id: str, source_id: str, sha: str = "abc") -> Document:
    return Document(
        id=doc_id,
        source_id=source_id,
        path=f"/corpus/{doc_id}.md",
        title=doc_id,
        ext=".md",
        mime="text/markdown",
        size_bytes=1024,
        n_pages=2,
        n_chunks=4,
        status="indexed",
        mtime=datetime.now(tz=UTC),
    )


def test_source_round_trips(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus", include_globs=["**/*.md"]))

    assert store.list_sources()[source.id].path == "/corpus"
    assert store.list_sources()[source.id].include_globs == ["**/*.md"]


def test_state_survives_a_reopen(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.close()

    reopened = MetaStore(db)

    assert "doc_1" in reopened.list_documents()
    assert reopened.list_documents()["doc_1"].title == "doc_1"


def test_removing_a_source_returns_its_document_ids(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.upsert_document(a_document("doc_2", source.id))

    removed = store.remove_source(source.id)

    assert sorted(removed) == ["doc_1", "doc_2"]
    assert store.list_documents() == {}
    assert store.list_sources() == {}


def test_sha_index_maps_path_to_digest(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.set_sha("/corpus/doc_1.md", "deadbeef")

    assert store.sha_index()["/corpus/doc_1.md"] == "deadbeef"


def test_messages_are_grouped_by_session(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    session = store.create_session("first question", None)
    store.append_message(
        ChatMessage(
            id="msg_1",
            session_id=session.id,
            role="user",
            text="what is reranking",
            created_at=datetime.now(tz=UTC),
        )
    )

    assert [m.text for m in store.list_messages()[session.id]] == ["what is reranking"]


def test_settings_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    default = AppSettings(storage_path=str(tmp_path))
    settings = store.load_settings(default)
    settings.retrieval.top_k = 9
    store.save_settings(settings)
    store.close()

    assert MetaStore(db).load_settings(default).retrieval.top_k == 9
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_meta_store.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.store'`

- [ ] **Step 3: Implement**

Create `core/ragcore/src/ragcore/store/__init__.py` (empty) and `core/ragcore/src/ragcore/store/meta.py`.

Design notes for the implementer:

- Use stdlib `sqlite3` with `check_same_thread=False` and `PRAGMA journal_mode=WAL`.
- Schema, created with `CREATE TABLE IF NOT EXISTS` in `__init__`:
  - `sources(id TEXT PRIMARY KEY, json TEXT NOT NULL)`
  - `documents(id TEXT PRIMARY KEY, source_id TEXT NOT NULL, path TEXT NOT NULL, json TEXT NOT NULL)`
  - `shas(path TEXT PRIMARY KEY, sha TEXT NOT NULL)`
  - `sessions(id TEXT PRIMARY KEY, json TEXT NOT NULL)`
  - `messages(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created_at TEXT NOT NULL, json TEXT NOT NULL)`
  - `settings(key TEXT PRIMARY KEY, json TEXT NOT NULL)`
- Store Pydantic models as JSON via `model.model_dump_json()` and read back with `Model.model_validate_json(row)`. This keeps the schema stable while `schemas.py` is frozen and avoids column-per-field churn.
- `list_messages` orders by `created_at, id`.
- `new_id(prefix)` style ids: `f"{prefix}_{uuid.uuid4().hex[:10]}"`, matching the stub's format so UI-visible ids look the same.
- `add_source` sets `added_at=datetime.now(tz=UTC)` and `id=_id("src")`.
- `remove_source` deletes the source row, collects the doc ids first, deletes those document rows and their `shas` rows, and returns the ids.
- Also implement `set_sha(path: str, sha: str) -> None` — the test uses it and Task 7 needs it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_meta_store.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add SQLite application state store"
```

---

### Task 6: LanceDB vector store

**Files:**
- Create: `core/ragcore/src/ragcore/store/lance.py`
- Create: `core/ragcore/tests/test_lance_store.py`
- Modify: `core/ragcore/pyproject.toml` (add `lancedb`, `pyarrow`)

**Interfaces:**
- Produces: `store.lance.VectorStore(root: Path)` with
  - `ChunkRow` TypedDict: `chunk_id, doc_id, doc_title, page_start, page_end, section_path, text, embed_text, n_tokens, vector`
  - `add_chunks(rows: list[ChunkRow]) -> None`
  - `delete_by_doc(doc_ids: list[str]) -> None`
  - `dense(vector: list[float], k: int) -> list[tuple[str, float]]` (chunk_id, similarity)
  - `fts(query: str, k: int) -> list[tuple[str, float]]` (chunk_id, score) — Task 12 builds the index; this task leaves the method raising `NotImplementedError`
  - `get(chunk_ids: list[str]) -> list[ChunkRow]`
  - `chunk_count() -> int`
  - `read_meta() -> dict[str, str]` / `write_meta(values: dict[str, str]) -> None`
  - `EMBED_DIM` re-exported for schema construction
- Consumes: `models.embed.EMBED_DIM`

- [ ] **Step 1: Add dependencies**

```bash
uv add --package ragcore lancedb pyarrow
```

- [ ] **Step 2: Write the failing test**

Create `core/ragcore/tests/test_lance_store.py`:

```python
"""LanceDB tables: write, search, delete, and the index_meta guard."""

from __future__ import annotations

from pathlib import Path

from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.lance import VectorStore


async def rows_for(texts: list[str], doc_id: str = "doc_1") -> list[dict]:
    vectors = await FakeEmbedClient().embed(texts)
    return [
        {
            "chunk_id": f"chk_{i}",
            "doc_id": doc_id,
            "doc_title": "Reranking",
            "page_start": 1,
            "page_end": 1,
            "section_path": "Reranking > Overview",
            "text": text,
            "embed_text": text,
            "n_tokens": len(text.split()),
            "vector": vector,
        }
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]


async def test_chunks_are_written_and_counted(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    await_rows = await rows_for(["reranking reorders", "chunking splits text"])
    store.add_chunks(await_rows)

    assert store.chunk_count() == 2


async def test_dense_search_finds_the_matching_chunk(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    texts = ["reranking reorders candidates", "optical character recognition of scans"]
    store.add_chunks(await rows_for(texts))
    [query_vector] = await FakeEmbedClient().embed([texts[0]])

    hits = store.dense(query_vector, k=2)

    assert hits[0][0] == "chk_0"
    assert hits[0][1] > hits[1][1]


async def test_deleting_a_document_removes_its_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["a reranking passage"], doc_id="doc_1"))
    store.add_chunks(await rows_for(["a chunking passage"], doc_id="doc_2"))

    store.delete_by_doc(["doc_1"])

    assert store.chunk_count() == 1
    assert store.get(["chk_0"])[0]["doc_id"] == "doc_2"


def test_meta_round_trips(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.write_meta({"embed_model": "Qwen3-Embedding-0.6B-Q8_0", "embed_dim": "1024"})

    assert store.read_meta()["embed_dim"] == "1024"


def test_meta_survives_a_reopen(tmp_path: Path) -> None:
    root = tmp_path / "index"
    VectorStore(root).write_meta({"schema_version": "1"})

    assert VectorStore(root).read_meta()["schema_version"] == "1"
```

Note: `store.add_chunks(await rows_for(...))` is valid inside an `async def` test. In the first test the helper result is bound to a local first only for readability; either form is fine.

- [ ] **Step 3: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_lance_store.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.store.lance'`

- [ ] **Step 4: Implement**

Create `core/ragcore/src/ragcore/store/lance.py`:

```python
"""LanceDB tables: chunks with their vectors, plus index_meta.

`documents` lives in SQLite rather than here. LanceDB holds what needs vector
or full-text search; everything else is relational state the UI pages through.

The `parents` table from the build plan is deliberately absent: naive chunking
produces no parent/child relationship, so context packing works on children.
It arrives with real chunking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import lancedb
import pyarrow as pa

from ragcore.models.embed import EMBED_DIM

CHUNKS = "chunks"
META = "index_meta"


class ChunkRow(TypedDict):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section_path: str
    text: str
    embed_text: str
    n_tokens: int
    vector: list[float]


CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_title", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("section_path", pa.string()),
        ("text", pa.string()),
        ("embed_text", pa.string()),
        ("n_tokens", pa.int32()),
        ("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ]
)

META_SCHEMA = pa.schema([("key", pa.string()), ("value", pa.string())])


class VectorStore:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(root))
        self.chunks = self._table(CHUNKS, CHUNK_SCHEMA)
        self.meta = self._table(META, META_SCHEMA)

    def _table(self, name: str, schema: pa.Schema):
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        if rows:
            self.chunks.add(rows)

    def delete_by_doc(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        quoted = ", ".join(f"'{doc_id}'" for doc_id in doc_ids)
        self.chunks.delete(f"doc_id IN ({quoted})")

    def dense(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        hits = (
            self.chunks.search(vector, vector_column_name="vector")
            .metric("cosine")
            .limit(k)
            .select(["chunk_id"])
            .to_list()
        )
        # LanceDB returns cosine *distance*; the pipeline wants similarity.
        return [(hit["chunk_id"], 1.0 - float(hit["_distance"])) for hit in hits]

    def fts(self, query: str, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError("full-text search lands in Task 12")

    def get(self, chunk_ids: list[str]) -> list[ChunkRow]:
        if not chunk_ids:
            return []
        quoted = ", ".join(f"'{cid}'" for cid in chunk_ids)
        found = self.chunks.search().where(f"chunk_id IN ({quoted})").limit(len(chunk_ids)).to_list()
        by_id = {row["chunk_id"]: row for row in found}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def chunk_count(self) -> int:
        return self.chunks.count_rows()

    def read_meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.meta.search().limit(10_000).to_list()}

    def write_meta(self, values: dict[str, str]) -> None:
        for key, value in values.items():
            self.meta.delete(f"key = '{key}'")
        self.meta.add([{"key": key, "value": value} for key, value in values.items()])
```

If a LanceDB API call differs in the installed version (the `search().where(...)` form in `get`, or `count_rows`), consult `uv run python -c "import lancedb; help(lancedb.table.Table)"` and adapt. Keep the method signatures above exactly — later tasks are written against them.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_lance_store.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore uv.lock
git commit -m "feat: add LanceDB chunk and index_meta store"
```

---

### Task 7: Walk and parse markdown and text

**Files:**
- Create: `core/ragcore/src/ragcore/ingest/__init__.py`
- Create: `core/ragcore/src/ragcore/ingest/walk.py`
- Create: `core/ragcore/src/ragcore/ingest/parse.py`
- Create: `core/ragcore/tests/test_ingest_walk.py`
- Create: `core/ragcore/tests/test_ingest_parse.py`

**Interfaces:**
- Produces:
  - `ingest.walk.FoundFile` dataclass: `path: Path, sha256: str, size_bytes: int, mtime: datetime, ext: str, mime: str`
  - `ingest.walk.walk_source(root: Path, *, include_globs: list[str], exclude_globs: list[str], max_file_mb: int) -> list[FoundFile]`
  - `ingest.walk.sha256_of(path: Path) -> str`
  - `ingest.parse.ParsedPage` dataclass: `page: int, section_path: str, text: str`
  - `ingest.parse.ParsedDoc` dataclass: `title: str, pages: list[ParsedPage], needs_ocr: bool`
  - `ingest.parse.parse(path: Path) -> ParsedDoc` — dispatches on suffix; raises `UnsupportedFormat` otherwise
  - `ingest.parse.UnsupportedFormat(Exception)`

- [ ] **Step 1: Write the failing walk test**

Create `core/ragcore/tests/test_ingest_walk.py`:

```python
"""Walking a source folder: globs, size cap, stable digests."""

from __future__ import annotations

from pathlib import Path

from ragcore.ingest.walk import sha256_of, walk_source


def build_corpus(root: Path) -> None:
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "reranking.md").write_text("# Reranking\n\nCross-encoders reorder.")
    (root / "notes" / "chunking.txt").write_text("Chunking splits documents.")
    (root / "notes" / "ignore.log").write_text("noise")
    (root / "notes" / "huge.md").write_text("x" * (2 * 1024 * 1024))


def test_include_globs_select_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=100)

    assert sorted(f.path.name for f in found) == ["huge.md", "reranking.md"]


def test_exclude_globs_win(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(
        tmp_path, include_globs=["**/*"], exclude_globs=["**/*.log", "**/huge.md"], max_file_mb=100
    )

    assert sorted(f.path.name for f in found) == ["chunking.txt", "reranking.md"]


def test_size_cap_skips_large_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=1)

    assert [f.path.name for f in found] == ["reranking.md"]


def test_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    target = tmp_path / "notes" / "reranking.md"
    before = sha256_of(target)
    assert before == sha256_of(target)

    target.write_text("# Reranking\n\nChanged.")

    assert sha256_of(target) != before


def test_mime_is_derived_from_the_extension(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.txt"], exclude_globs=[], max_file_mb=100)

    assert found[0].mime == "text/plain"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_ingest_walk.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.ingest'`

- [ ] **Step 3: Implement the walker**

Create `core/ragcore/src/ragcore/ingest/__init__.py` (empty) and `core/ragcore/src/ragcore/ingest/walk.py`:

```python
"""Finding files worth indexing, and fingerprinting them.

The digest is what makes re-indexing a no-op: a file whose sha256 is unchanged
is skipped without being parsed or embedded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MIME_BY_EXT = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}

READ_CHUNK = 1024 * 1024


@dataclass(slots=True)
class FoundFile:
    path: Path
    sha256: str
    size_bytes: int
    mtime: datetime
    ext: str
    mime: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def walk_source(
    root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    max_file_mb: int,
) -> list[FoundFile]:
    cap = max_file_mb * 1024 * 1024
    included: set[Path] = set()
    for pattern in include_globs:
        included.update(p for p in root.glob(pattern) if p.is_file())
    for pattern in exclude_globs:
        included.difference_update(root.glob(pattern))

    out: list[FoundFile] = []
    for path in sorted(included):
        stat = path.stat()
        if stat.st_size > cap:
            continue
        ext = path.suffix.lower()
        out.append(
            FoundFile(
                path=path,
                sha256=sha256_of(path),
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                ext=ext,
                mime=MIME_BY_EXT.get(ext, "application/octet-stream"),
            )
        )
    return out
```

- [ ] **Step 4: Write the failing parse test**

Create `core/ragcore/tests/test_ingest_parse.py`:

```python
"""Parsing to pages. Page numbers are a correctness requirement: citations carry them."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragcore.ingest.parse import UnsupportedFormat, parse


def test_markdown_splits_into_pages_on_level_two_headings(tmp_path: Path) -> None:
    source = tmp_path / "reranking.md"
    source.write_text(
        "# Reranking\n\nIntro text.\n\n"
        "## Cross encoders\n\nThey score pairs.\n\n"
        "## Latency\n\nThey are slower.\n"
    )

    parsed = parse(source)

    assert parsed.title == "Reranking"
    assert [p.page for p in parsed.pages] == [1, 2, 3]
    assert parsed.pages[1].section_path == "Reranking > Cross encoders"
    assert "score pairs" in parsed.pages[1].text


def test_markdown_without_headings_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "flat.md"
    source.write_text("Just a paragraph with no headings at all.")

    parsed = parse(source)

    assert [p.page for p in parsed.pages] == [1]
    assert parsed.title == "flat"


def test_plain_text_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Chunking splits documents into retrievable units.")

    parsed = parse(source)

    assert len(parsed.pages) == 1
    assert parsed.needs_ocr is False


def test_unknown_extension_raises(tmp_path: Path) -> None:
    source = tmp_path / "thing.xyz"
    source.write_text("?")

    with pytest.raises(UnsupportedFormat):
        parse(source)
```

- [ ] **Step 5: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.ingest.parse'`

- [ ] **Step 6: Implement the parser**

Create `core/ragcore/src/ragcore/ingest/parse.py`:

```python
"""File to pages.

A "page" is whatever unit a citation can point at. For PDFs that is a real
page; for markdown it is a level-two section, which is the closest honest
analogue and is what the reader view highlights against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


class UnsupportedFormat(Exception):
    pass


@dataclass(slots=True)
class ParsedPage:
    page: int
    section_path: str
    text: str


@dataclass(slots=True)
class ParsedDoc:
    title: str
    pages: list[ParsedPage]
    needs_ocr: bool = False


def _title(text: str, fallback: str) -> str:
    match = next((m for m in _HEADING.finditer(text) if len(m.group(1)) == 1), None)
    return match.group(2).strip() if match else fallback


def _markdown(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _title(text, path.stem)
    marks = [m for m in _HEADING.finditer(text) if len(m.group(1)) == 2]

    if not marks:
        return ParsedDoc(title=title, pages=[ParsedPage(1, title, text.strip())])

    pages: list[ParsedPage] = []
    preamble = text[: marks[0].start()].strip()
    if preamble:
        pages.append(ParsedPage(1, title, preamble))

    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        heading = mark.group(2).strip()
        pages.append(
            ParsedPage(
                page=len(pages) + 1,
                section_path=f"{title} > {heading}",
                text=text[mark.end() : end].strip(),
            )
        )
    return ParsedDoc(title=title, pages=pages)


def _plain(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDoc(title=path.stem, pages=[ParsedPage(1, path.stem, text.strip())])


def parse(path: Path) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _markdown(path)
    if suffix == ".txt":
        return _plain(path)
    raise UnsupportedFormat(f"no parser for {suffix!r}")
```

PDF dispatch is added in Task 14.

- [ ] **Step 7: Run both test files**

Run: `uv run pytest core/ragcore/tests/test_ingest_walk.py core/ragcore/tests/test_ingest_parse.py -v`
Expected: PASS (9 tests).

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add source walking and markdown/text parsing"
```

---

### Task 8: Chunker

**Files:**
- Create: `core/ragcore/src/ragcore/ingest/chunk.py`
- Create: `core/ragcore/tests/test_ingest_chunk.py`

**Interfaces:**
- Consumes: `ingest.parse.ParsedDoc`, `ingest.parse.ParsedPage`
- Produces:
  - `ingest.chunk.Chunk` dataclass: `chunk_id, doc_id, doc_title, page_start, page_end, section_path, text, embed_text, n_tokens, char_start, char_end`
  - `ingest.chunk.chunk_document(doc_id: str, parsed: ParsedDoc, *, target_words: int = 256, overlap_words: int = 30) -> list[Chunk]`

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_ingest_chunk.py`:

```python
"""Naive fixed-size chunking with a contextual header.

Word-based on purpose: the Qwen3 tokenizer and parent/child splitting land with
real chunking, and pulling `tokenizers` in now would be a dependency we then
have to justify twice.
"""

from __future__ import annotations

from ragcore.ingest.chunk import chunk_document
from ragcore.ingest.parse import ParsedDoc, ParsedPage


def doc_with(words: int, pages: int = 1) -> ParsedDoc:
    return ParsedDoc(
        title="Reranking",
        pages=[
            ParsedPage(page=i + 1, section_path=f"Reranking > Part {i + 1}", text=" ".join(
                f"word{j}" for j in range(words)
            ))
            for i in range(pages)
        ],
    )


def test_short_page_becomes_one_chunk() -> None:
    chunks = chunk_document("doc_1", doc_with(50))

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].n_tokens == 50


def test_long_page_splits_with_overlap() -> None:
    chunks = chunk_document("doc_1", doc_with(600), target_words=256, overlap_words=30)

    assert len(chunks) == 3
    first_tail = chunks[0].text.split()[-30:]
    second_head = chunks[1].text.split()[:30]
    assert first_tail == second_head


def test_embed_text_carries_the_contextual_header() -> None:
    [chunk] = chunk_document("doc_1", doc_with(20))

    assert chunk.embed_text.startswith("Reranking > Part 1\n")
    assert chunk.embed_text.endswith(chunk.text)


def test_chunks_never_span_pages() -> None:
    chunks = chunk_document("doc_1", doc_with(300, pages=2))

    for chunk in chunks:
        assert chunk.page_start == chunk.page_end
    assert {c.page_start for c in chunks} == {1, 2}


def test_chunk_ids_are_unique_and_prefixed_by_document() -> None:
    chunks = chunk_document("doc_1", doc_with(600))
    ids = [c.chunk_id for c in chunks]

    assert len(set(ids)) == len(ids)
    assert all(cid.startswith("doc_1:") for cid in ids)


def test_char_offsets_point_back_into_the_page() -> None:
    parsed = doc_with(600)
    chunks = chunk_document("doc_1", parsed)
    page_text = parsed.pages[0].text

    for chunk in chunks:
        assert page_text[chunk.char_start : chunk.char_end] == chunk.text


def test_empty_pages_are_skipped() -> None:
    parsed = ParsedDoc(title="Empty", pages=[ParsedPage(1, "Empty", "   ")])

    assert chunk_document("doc_1", parsed) == []
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_ingest_chunk.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.ingest.chunk'`

- [ ] **Step 3: Implement**

Create `core/ragcore/src/ragcore/ingest/chunk.py`:

```python
"""Splitting pages into retrievable units.

Naive by design: fixed word windows with overlap, one page at a time, never
crossing a page boundary because a citation names exactly one page. Heading-
aware parent/child chunking with real token counts replaces this wholesale.

`embed_text` prepends the section path. The embedder sees where a passage sits;
the LLM and the reader see only `text`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragcore.ingest.parse import ParsedDoc


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section_path: str
    text: str
    embed_text: str
    n_tokens: int
    char_start: int
    char_end: int


def _windows(count: int, target: int, overlap: int) -> list[tuple[int, int]]:
    if count <= target:
        return [(0, count)]
    step = max(target - overlap, 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < count:
        spans.append((start, min(start + target, count)))
        if start + target >= count:
            break
        start += step
    return spans


def chunk_document(
    doc_id: str,
    parsed: ParsedDoc,
    *,
    target_words: int = 256,
    overlap_words: int = 30,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in parsed.pages:
        text = page.text.strip()
        if not text:
            continue
        words = text.split()
        # Offsets are recovered by walking the original string so slicing it
        # with char_start/char_end returns the chunk verbatim.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)

        for lo, hi in _windows(len(words), target_words, overlap_words):
            char_start, char_end = offsets[lo][0], offsets[hi - 1][1]
            body = text[char_start:char_end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{page.page}:{lo}",
                    doc_id=doc_id,
                    doc_title=parsed.title,
                    page_start=page.page,
                    page_end=page.page,
                    section_path=page.section_path,
                    text=body,
                    embed_text=f"{page.section_path}\n{body}",
                    n_tokens=hi - lo,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    return chunks
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_ingest_chunk.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add naive word-window chunker with contextual headers"
```

---

### Task 9: The real store

Compose SQLite, LanceDB and ingestion into a `StorePort` implementation.

**Files:**
- Create: `core/ragcore/src/ragcore/store/real.py`
- Create: `core/ragcore/tests/test_real_store.py`
- Modify: `core/ragcore/src/ragcore/api/routes/documents.py` (replace `store.loaded` reads with `store.content()`)

**Interfaces:**
- Consumes: `MetaStore`, `VectorStore`, `walk_source`, `parse`, `chunk_document`, `FakeEmbedClient`, `RetrieverPort`
- Produces: `store.real.RealStore(config: Config, *, embedder, embed_model_id: str = "Qwen3-Embedding-0.6B-Q8_0", retriever=None)` satisfying `StorePort`, plus
  - `async RealStore.ingest_source_async(source_id: str) -> list[Document]`
  - `RealStore.wipe(*, keep_connections: bool) -> None`, `RealStore.save_settings() -> None`
  - `RealStore.vectors: VectorStore` (the factory wires the retriever against it in Task 11)
  - `RealStore.embed_model_id: str`
  - `RealStore.index_blocked: str | None` — set when `index_meta.embed_model` disagrees with the live embedder

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_real_store.py`:

```python
"""The real store end to end, with a fake embedder: ingest, re-ingest, remove, guard."""

from __future__ import annotations

from pathlib import Path

from ragcore.config import Config
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore


def corpus(root: Path) -> Path:
    docs = root / "corpus"
    docs.mkdir()
    (docs / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nCross encoders score query passage pairs.\n"
    )
    (docs / "chunking.md").write_text(
        "# Chunking\n\n## Sizes\n\nSmaller chunks retrieve precisely.\n"
    )
    return docs


def build(tmp_path: Path) -> RealStore:
    config = Config(data_dir=tmp_path, backend="real")
    return RealStore(config, embedder=FakeEmbedClient())


async def test_ingesting_a_source_indexes_its_documents(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = store.add_source_for_path(str(corpus(tmp_path)))

    documents = await store.ingest_source_async(source.id)

    assert len(documents) == 2
    assert {d.status for d in documents} == {"indexed"}
    assert store.index_stats().chunks > 0


async def test_reingesting_unchanged_files_is_a_no_op(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = store.add_source_for_path(str(corpus(tmp_path)))
    await store.ingest_source_async(source.id)
    chunks_before = store.index_stats().chunks

    again = await store.ingest_source_async(source.id)

    assert {d.status for d in again} == {"skipped"}
    assert store.index_stats().chunks == chunks_before


async def test_changed_file_is_reindexed(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    store = build(tmp_path)
    source = store.add_source_for_path(str(docs))
    await store.ingest_source_async(source.id)

    (docs / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nRewritten body with different words entirely.\n"
    )
    again = await store.ingest_source_async(source.id)

    statuses = {d.title: d.status for d in again}
    assert statuses["Reranking"] == "indexed"
    assert statuses["Chunking"] == "skipped"


async def test_removing_a_source_drops_its_chunks(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = store.add_source_for_path(str(corpus(tmp_path)))
    await store.ingest_source_async(source.id)

    removed = store.remove_source(source.id)

    assert removed == 2
    assert store.index_stats().chunks == 0
    assert store.documents == {}


async def test_document_content_is_readable_for_the_reader_view(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = store.add_source_for_path(str(corpus(tmp_path)))
    documents = await store.ingest_source_async(source.id)

    content = store.content(documents[0].id)

    assert content is not None
    assert content.n_pages == len(content.pages)
    assert all(c.page <= content.n_pages for c in content.chunks)


async def test_a_different_embed_model_blocks_the_index(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = store.add_source_for_path(str(corpus(tmp_path)))
    await store.ingest_source_async(source.id)
    store.close()

    config = Config(data_dir=tmp_path, backend="real")
    reopened = RealStore(config, embedder=FakeEmbedClient(), embed_model_id="some-other-model")

    assert reopened.index_blocked is not None
    assert "re-index" in reopened.index_blocked
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_real_store.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.store.real'`

- [ ] **Step 3: Implement**

Create `core/ragcore/src/ragcore/store/real.py`. Requirements, in order:

1. `__init__(self, config, *, embedder, embed_model_id="Qwen3-Embedding-0.6B-Q8_0", retriever=None)`:
   - `self.meta = MetaStore(config.data_dir / "app.db")`
   - `self.vectors = VectorStore(config.data_dir / "index")`
   - `self.settings = self.meta.load_settings(AppSettings(storage_path=str(config.data_dir)))`
   - dict attributes required by `StorePort` are **properties** reading through to SQLite: `sources`, `documents`, `sessions`, `messages`. `connections`, `models` and `eval_sets` are plain dicts, empty for now — connections and the hub stay stub concerns in this slice, and `routes/connections.py` already tolerates an empty mapping.
   - `self.retriever` is set by `build_backend` after construction (Task 11) or passed in.
2. Index guard: read `index_meta`. If empty, write `{"schema_version": "1", "embed_model": embed_model_id, "embed_dim": "1024", "reranker_model": "Qwen3-Reranker-0.6B-Q8_0"}` and leave `index_blocked = None`. If present and `embed_model` differs from `embed_model_id` (or `embed_dim != "1024"`), set:
   `self.index_blocked = f"Index was built with {stored!r}; this build ships {embed_model_id!r}. Re-index to continue."`
3. `add_source_for_path(path: str) -> Source` is a convenience wrapping `add_source(SourceCreate(path=path))`.
4. `ingest_source_async(source_id)`:
   - `walk_source(Path(source.path), include_globs=..., exclude_globs=..., max_file_mb=...)`
   - `known = self.meta.sha_index()`
   - For each found file: if `known.get(str(f.path)) == f.sha256`, emit the existing `Document` with `status="skipped"` and continue.
   - Otherwise: if a document already exists for that path, `self.vectors.delete_by_doc([old_id])` and `self.meta.delete_documents([old_id])` first.
   - `parsed = parse(f.path)`; on `UnsupportedFormat`, record `status="error"` with the message and continue.
   - `chunks = chunk_document(doc_id, parsed)`; `vectors = await self.embedder.embed([c.embed_text for c in chunks], batch_size=self.settings.performance.embed_batch)`
   - `self.vectors.add_chunks([...])` mapping `Chunk` → `ChunkRow`.
   - Persist the `Document` with `status="indexed"`, `n_pages=len(parsed.pages)`, `n_chunks=len(chunks)`, `indexed_at=now`, and `self.meta.set_sha(str(f.path), f.sha256)`.
   - Store the parsed pages as JSON next to the document so `content()` can serve the reader view: write them to `config.data_dir / "pages" / f"{doc_id}.json"`.
   - Rebuild `self.retriever` at the end if a retriever factory was supplied (Task 11 wires this).
5. `ingest_source(source_id)` — the synchronous `StorePort` method — cannot drive async embedding from inside a running event loop, so it raises `RuntimeError("use ingest_source_async")` and the two call sites move to the async variant. Both are already `async def` handlers in `core/ragcore/src/ragcore/api/routes/sources.py`:
   - `add_source`: `documents = store.ingest_source(source.id)` → `documents = await store.ingest_source_async(source.id)`
   - `rescan`: `documents = store.ingest_source(source_id)` → `documents = await store.ingest_source_async(source_id)`

   Give `stub/store.Store` a matching shim so both backends answer the same call:

```python
    async def ingest_source_async(self, source_id: str) -> list[Document]:
        return self.ingest_source(source_id)
```

   Add `async def ingest_source_async(self, source_id: str) -> list[Document]: ...` to `ports.StorePort` at the same time.
6. `content(doc_id)` reads the pages JSON and the chunk rows for that document, returning `DocumentContent` with `DocumentChunkRef` entries carrying `char_start`/`char_end`. Persist those offsets in the pages JSON alongside each chunk id, since `ChunkRow` does not carry them.
7. `index_stats()` returns `IndexStats` built from SQLite counts and `self.vectors.chunk_count()`.
8. `remove_source(source_id)` → `doc_ids = self.meta.remove_source(source_id)`, `self.vectors.delete_by_doc(doc_ids)`, delete the pages JSON files, return `len(doc_ids)`.
9. `rebuild_index()` clears the chunks table and the shas table, then leaves documents queued — the route already returns a job.
10. `close()` closes `MetaStore`.

- [ ] **Step 4: Move every route off stub internals**

Three routes reach into state that `RealStore` exposes as read-through properties, where in-place mutation silently does nothing. All three must go through the port.

**`routes/documents.py`** reads `store.loaded` in three places. Document listing uses `store.documents`; the content endpoint uses `store.content(doc_id)`.

**`routes/settings.py`** is the load-bearing one. `wipe` currently calls `.clear()` on `store.sources`, `store.documents`, `store.loaded`, `store.sessions` and `store.messages`. Against `RealStore` those are properties returning a fresh dict each access, so every `.clear()` would empty a throwaway and the wipe would report success while deleting nothing. Replace the body's state-clearing block with a single port call:

```python
@router.post("/wipe", response_model=Ok)
def wipe(payload: WipeRequest, store: StoreDep) -> Ok:
    if payload.confirm != "DELETE":
        raise HTTPException(400, 'confirm must be the literal string "DELETE"')

    store.wipe(keep_connections=payload.keep_connections)
    return Ok()
```

`patch_settings` has the same shape of bug in slow motion: it mutates `store.settings` in place and never persists, so on `RealStore` the change is lost at restart. Add `store.save_settings()` before it returns.

**`routes/evals.py`** uses `store.loaded.values()` at line 44 and calls `store.retriever.search(...)` synchronously at line 51 — the method Task 10 makes raise on the real retriever. Use `store.documents` / `store.content()` for the former and `await store.retriever.asearch(...)` for the latter; the handler is already `async def`.

Add to `ports.StorePort`:

```python
    def wipe(self, *, keep_connections: bool) -> None: ...
    def save_settings(self) -> None: ...
```

Implement both on `stub/store.Store` (clear its dicts; `save_settings` is a no-op) and on `RealStore` (`MetaStore.wipe` plus `VectorStore` chunk deletion; `save_settings` calls `MetaStore.save_settings`).

Confirm with:

Run: `grep -rn "store.loaded" core/ragcore/src/ragcore/api/`
Expected: no output.

Run: `grep -rn "retriever.search" core/ragcore/src/ragcore/api/`
Expected: no output.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_real_store.py -v`
Expected: PASS (6 tests).

Then confirm the stub still honours wipe through the new port method:

Run: `uv run pytest core/ragcore/tests/test_api.py -k wipe -v`
Expected: PASS (2 tests).

Run: `uv run pytest -v`
Expected: PASS — everything, including the 13 contract tests still on the stub.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add real store composing SQLite, LanceDB and ingestion"
```

---

### Task 10: Dense-only retriever and context packing

**Files:**
- Create: `core/ragcore/src/ragcore/retrieve/__init__.py`
- Create: `core/ragcore/src/ragcore/retrieve/hybrid.py`
- Create: `core/ragcore/src/ragcore/retrieve/pack.py`
- Create: `core/ragcore/tests/test_retrieve.py`

**Interfaces:**
- Consumes: `VectorStore`, `EmbedClient`/`FakeEmbedClient`, `RetrievalSettings`, `QueryFilters`
- Produces:
  - `retrieve.hybrid.HybridRetriever(vectors: VectorStore, embedder, *, reranker=None)` satisfying `RetrieverPort`
  - `retrieve.pack.pack(chunks: list[RetrievedChunk], budget_tokens: int) -> list[RetrievedChunk]`

`search` is synchronous per the Protocol but must call an async embedder. Implement `search` as a thin wrapper that runs `asearch` on the running loop via `asyncio.get_event_loop().run_until_complete` **only when no loop is running**; inside FastAPI a loop is always running, so `routes/query.py` must call `await store.retriever.asearch(...)`. Add `asearch` to `RetrieverPort` and give the stub retriever an `async def asearch` that delegates to its existing `search`. Update `routes/query.py` to await it.

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_retrieve.py`:

```python
"""Retrieval pipeline over a fake embedder: ranks, filters, budget, latency."""

from __future__ import annotations

from pathlib import Path

from ragcore.api.schemas import QueryFilters, RetrievalSettings, RetrievedChunk
from ragcore.models.fakes import FakeEmbedClient
from ragcore.retrieve.hybrid import HybridRetriever
from ragcore.retrieve.pack import pack
from ragcore.store.lance import VectorStore

TEXTS = {
    "chk_rerank": "Cross encoders rerank candidate passages by scoring pairs.",
    "chk_chunk": "Chunk size trades precision against context completeness.",
    "chk_ocr": "Optical character recognition extracts text from scanned images.",
}


async def build(tmp_path: Path) -> HybridRetriever:
    vectors = VectorStore(tmp_path / "index")
    embedder = FakeEmbedClient()
    embedded = await embedder.embed(list(TEXTS.values()))
    vectors.add_chunks(
        [
            {
                "chunk_id": cid,
                "doc_id": cid.replace("chk_", "doc_"),
                "doc_title": cid,
                "page_start": 1,
                "page_end": 1,
                "section_path": "Doc > Section",
                "text": text,
                "embed_text": text,
                "n_tokens": len(text.split()),
                "vector": vector,
            }
            for (cid, text), vector in zip(TEXTS.items(), embedded, strict=True)
        ]
    )
    return HybridRetriever(vectors, embedder)


async def test_dense_search_ranks_the_exact_text_first(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, latency, candidates = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(),
        filters=QueryFilters(),
        doc_meta={},
    )

    assert chunks[0].chunk_id == "chk_rerank"
    assert candidates >= 1
    assert latency.embed_ms > 0
    assert latency.dense_ms > 0


async def test_doc_id_filter_restricts_results(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, _ = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(),
        filters=QueryFilters(doc_ids=["doc_ocr"]),
        doc_meta={},
    )

    assert {c.doc_id for c in chunks} == {"doc_ocr"}


async def test_top_k_is_honoured(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, _ = await retriever.asearch(
        "text",
        settings=RetrievalSettings(top_k=2, min_score=0.0),
        filters=QueryFilters(),
        doc_meta={},
    )

    assert len(chunks) <= 2


async def test_empty_query_returns_nothing(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, candidates = await retriever.asearch(
        "   ", settings=RetrievalSettings(), filters=QueryFilters(), doc_meta={}
    )

    assert chunks == []
    assert candidates == 0


def a_chunk(chunk_id: str, tokens: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_title="Doc",
        page_start=1,
        page_end=1,
        text=" ".join(["word"] * tokens),
        rerank_score=score,
    )


def test_packing_stops_at_the_budget() -> None:
    packed = pack([a_chunk("a", 100, 0.9), a_chunk("b", 100, 0.8), a_chunk("c", 100, 0.7)], 250)

    assert [c.chunk_id for c in packed] == ["a", "b"]


def test_packing_keeps_score_order() -> None:
    packed = pack([a_chunk("low", 10, 0.1), a_chunk("high", 10, 0.9)], 1000)

    assert [c.chunk_id for c in packed] == ["high", "low"]


def test_packing_never_returns_nothing_when_a_chunk_fits_alone() -> None:
    packed = pack([a_chunk("huge", 5000, 0.9)], 100)

    assert [c.chunk_id for c in packed] == ["huge"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_retrieve.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.retrieve'`

- [ ] **Step 3: Implement packing**

Create `core/ragcore/src/ragcore/retrieve/__init__.py` (empty) and `core/ragcore/src/ragcore/retrieve/pack.py`:

```python
"""Filling the context budget in score order.

Parent mapping belongs here once real chunking produces parents. Today the
children are what the LLM sees, so packing is a greedy fill that always keeps
at least the best chunk, even when it alone exceeds the budget — an answer from
one over-long passage beats no answer at all.
"""

from __future__ import annotations

from ragcore.api.schemas import RetrievedChunk


def _tokens(chunk: RetrievedChunk) -> int:
    return len(chunk.text.split())


def pack(chunks: list[RetrievedChunk], budget_tokens: int) -> list[RetrievedChunk]:
    ordered = sorted(chunks, key=lambda c: -c.rerank_score)
    packed: list[RetrievedChunk] = []
    used = 0
    for chunk in ordered:
        cost = _tokens(chunk)
        if packed and used + cost > budget_tokens:
            continue
        packed.append(chunk)
        used += cost
    return packed
```

- [ ] **Step 4: Implement the retriever, dense leg only**

Create `core/ragcore/src/ragcore/retrieve/hybrid.py`:

```python
"""Retrieval: dense now, plus BM25 fusion and reranking in later tasks.

`asearch` is the real entry point. `search` exists because the RetrieverPort
Protocol was shaped by the stub, and nothing may call it from inside the event
loop.
"""

from __future__ import annotations

import time

from ragcore.api.schemas import (
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    StageLatency,
)
from ragcore.retrieve.pack import pack
from ragcore.store.lance import ChunkRow, VectorStore


def _allowed(row: ChunkRow, filters: QueryFilters, doc_meta: dict[str, dict]) -> bool:
    if filters.doc_ids and row["doc_id"] not in filters.doc_ids:
        return False
    meta = doc_meta.get(row["doc_id"], {})
    if filters.source_ids and meta.get("source_id") not in filters.source_ids:
        return False
    if filters.exts and meta.get("ext") not in filters.exts:
        return False
    if filters.langs and meta.get("lang") not in filters.langs:
        return False
    mtime = meta.get("mtime")
    if filters.after and mtime and mtime < filters.after:
        return False
    return not (filters.before and mtime and mtime > filters.before)


def _to_chunk(row: ChunkRow, *, dense_rank: int | None, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        section_path=row["section_path"] or None,
        text=row["text"],
        dense_rank=dense_rank,
        rerank_score=score,
    )


class HybridRetriever:
    def __init__(self, vectors: VectorStore, embedder, *, reranker=None) -> None:
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker

    async def asearch(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        latency = StageLatency()
        if not query.strip():
            return [], latency, 0

        t0 = time.perf_counter()
        [vector] = await self.embedder.embed([query])
        latency.embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = self.vectors.dense(vector, k=settings.dense_top_k)
        latency.dense_ms = (time.perf_counter() - t0) * 1000

        rows = {row["chunk_id"]: row for row in self.vectors.get([cid for cid, _ in hits])}
        scored = [
            _to_chunk(rows[cid], dense_rank=rank + 1, score=score)
            for rank, (cid, score) in enumerate(hits)
            if cid in rows and _allowed(rows[cid], filters, doc_meta)
        ]
        candidates = len(scored)

        # `min_score` is calibrated for the reranker's sigmoid probabilities.
        # Cosine similarities and RRF scores live near 0.0-0.05, so applying the
        # same 0.3 default to them filters every candidate and returns nothing.
        # The gate therefore belongs to the rerank stage (Task 13) and is skipped
        # whenever no reranker ran.
        kept = scored[: settings.top_k]
        t0 = time.perf_counter()
        kept = pack(kept, settings.context_token_budget)
        latency.pack_ms = (time.perf_counter() - t0) * 1000
        return kept, latency, candidates

    def search(self, query: str, **kwargs):
        raise RuntimeError("call asearch; search exists only to satisfy the Protocol")
```

Then widen the Protocol: add `async def asearch(...) -> tuple[list[RetrievedChunk], StageLatency, int]` to `ports.RetrieverPort`, and add to `stub/retrieval.Retriever`:

```python
    async def asearch(self, query: str, **kwargs):
        return self.search(query, **kwargs)
```

In `routes/query.py`, change the retrieval call to `await store.retriever.asearch(...)`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_retrieve.py -v`
Expected: PASS (7 tests).

Run: `uv run pytest -v`
Expected: PASS — the contract tests still pass on the stub through the new `asearch` path.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add dense retrieval and context packing"
```

---

### Task 11: Wire the real backend — S1 exit

The skeleton closes here: `--backend real` serves a real cited answer, and the backend-agnostic contract tests run against both backends.

**Files:**
- Create: `core/ragcore/src/ragcore/llm/__init__.py`
- Create: `core/ragcore/src/ragcore/llm/openai_compat.py`
- Modify: `core/ragcore/src/ragcore/backend.py` (the `"real"` branch)
- Modify: `core/ragcore/src/ragcore/config.py` (embed/rerank server URLs)
- Modify: `core/ragcore/src/ragcore/cli.py` (flags for those URLs)
- Modify: `core/ragcore/src/ragcore/api/routes/health.py` (stop hard-coding `stub=True`)
- Modify: `core/ragcore/tests/conftest.py` (add `stub_client`, parameterize `client` over both backends)
- Modify: `core/ragcore/tests/test_api.py` (seven tests switch to the `stub_client` fixture — parameter rename only)
- Modify: `core/ragcore/src/ragcore/stub/answers.py` (drop the moved `llm_stream`)

**Interfaces:**
- Produces:
  - `llm.openai_compat.OpenAICompatEngine(config: Config)` satisfying `AnswerEngine`
  - `llm.openai_compat.SYSTEM_PROMPT`, `build_messages(question, chunks) -> list[dict]`
  - `Config.embed_url: str = "http://127.0.0.1:8770"`, `Config.rerank_url: str = "http://127.0.0.1:8771"`
  - conftest fixture `client` parameterized `["stub", "real"]`

- [ ] **Step 1: Write the failing test**

Add to `core/ragcore/tests/test_backend.py`:

```python
async def test_factory_builds_the_real_backend(tmp_path: Path) -> None:
    built = build_backend(config_for(tmp_path, "real"))

    assert type(built.store).__module__ == "ragcore.store.real"
    assert type(built.answerer).__module__ == "ragcore.llm.openai_compat"
```

And create `core/ragcore/tests/test_real_query.py`:

```python
"""S1 exit: a real index answers a real query with the frozen frame order."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragcore.api.app import create_app
from ragcore.config import Config

TOKEN = "test-token"


@pytest.fixture
def real_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAGCORE_FAKE_MODELS", "1")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nCross encoders score query passage pairs "
        "and reorder candidates so the best passage is first.\n"
    )
    config = Config(port=0, token=TOKEN, data_dir=tmp_path / "data", backend="real")
    with TestClient(create_app(config)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        client.post("/sources", json={"path": str(corpus), "include_globs": ["**/*.md"]})
        yield client


def test_a_real_index_answers_with_the_frozen_frame_order(real_client, read_events) -> None:
    with real_client.stream("POST", "/query", json={"q": "what do cross encoders do"}) as response:
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[1] == "mode"
    assert names[2] == "sources"
    assert names[-2] == "citations"
    assert names[-1] == "done"


def test_sources_frame_carries_real_chunks(real_client, read_events) -> None:
    with real_client.stream("POST", "/query", json={"q": "cross encoders"}) as response:
        events = dict(read_events(response))

    chunks = events["sources"]["chunks"]
    assert chunks
    assert "cross encoders" in chunks[0]["text"].lower()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_real_query.py -v`
Expected: FAIL, `ValueError: unknown backend: 'real'`

- [ ] **Step 3: Move the LLM streaming into a real module**

Create `core/ragcore/src/ragcore/llm/__init__.py` (empty) and `core/ragcore/src/ragcore/llm/openai_compat.py`. Move `SYSTEM_PROMPT`, `NOT_FOUND` and the body of `llm_stream` out of `stub/answers.py`, wrapped in a class:

```python
"""Grounded generation over any OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator

from ragcore.api.schemas import RetrievedChunk
from ragcore.config import Config

SYSTEM_PROMPT = """You answer questions using only the passages provided.
Cite every claim with a marker of the form [document_id:page] taken from the
passage headers. If the passages do not contain the answer, say so plainly and
cite nothing. Do not invent document ids."""


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{c.doc_id}:{c.page_start}] {c.doc_title} — {c.section_path or ''}\n{c.text}"
        for c in chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question}"},
    ]


class OpenAICompatEngine:
    def __init__(self, config: Config) -> None:
        self.config = config

    def stream(
        self, question: str, chunks: list[RetrievedChunk], directives: set[str]
    ) -> AsyncIterator[str]:
        return self._stream(question, chunks)

    async def _stream(self, question, chunks) -> AsyncIterator[str]:
        ...  # the moved body of stub.answers.llm_stream, reading self.config
```

Keep the moved body's SSE parsing and `<think>` stripping exactly as it was. Delete `llm_stream` from `stub/answers.py` and have `StubAnswerEngine` use `OpenAICompatEngine` when `config.llm_base_url` is set.

If `config.llm_base_url` is unset, `OpenAICompatEngine._stream` yields a grounded extractive fallback built from the retrieved chunks. It is honest — every word comes from the passages — and its shape is constrained by the contract:

- **Yield one frame per sentence, never one frame for the whole answer.** `test_query_streams_frames_in_contract_order` asserts `names.count("token") > 1`.
- **Cite the top two *distinct* `(doc_id, page_start)` chunks.** That test also asserts `grounding == "ok"`, and `extract_citations` only returns `"ok"` with at least two citations and zero dropped. Two markers pointing at the same `(doc, page)` collapse to one citation and grade `"low"`.
- Emit each sentence as `f"{first_sentence(chunk.text)} [{chunk.doc_id}:{chunk.page_start}]"`.

If fewer than two chunks were retrieved, cite what there is — the grounding grade is then correctly `"low"`, and no test on the real backend asserts otherwise.

- [ ] **Step 4: Add the real branch to the factory**

In `backend.py`:

```python
    if config.backend == "real":
        from ragcore.llm.openai_compat import OpenAICompatEngine
        from ragcore.models.embed import EmbedClient
        from ragcore.models.fakes import FakeEmbedClient
        from ragcore.retrieve.hybrid import HybridRetriever
        from ragcore.stub.hub import HubClient
        from ragcore.stub.jobs import JobManager
        from ragcore.store.real import RealStore

        embedder = FakeEmbedClient() if os.getenv("RAGCORE_FAKE_MODELS") else EmbedClient(
            config.embed_url
        )
        store = RealStore(config, embedder=embedder)
        store.retriever = HybridRetriever(store.vectors, embedder)
        return Backend(
            store=store,
            jobs=JobManager(),
            hub=HubClient(config),
            answerer=OpenAICompatEngine(config),
        )
```

`RAGCORE_FAKE_MODELS` is a test affordance and must never be set by `cli.py`.

- [ ] **Step 5: Add the server URL config**

In `config.py` add `embed_url: str = "http://127.0.0.1:8770"` and `rerank_url: str = "http://127.0.0.1:8771"`, with `RAGCORE_EMBED_URL` / `RAGCORE_RERANK_URL` env fallbacks. In `cli.py` add `--embed-url` and `--rerank-url` and pass them through.

- [ ] **Step 6: Report the backend honestly in health**

In `routes/health.py`, replace `stub=True` with `stub=config.backend == "stub"` and change the `detail` string to `"serving fixture data"` for the stub and `"serving the local index"` for the real backend. If `store.index_blocked` is set, return `status="degraded"` and put the message in the ragcore process `detail`.

- [ ] **Step 7: Split and parameterize the contract tests**

The plan originally claimed all 13 contract tests would run against both backends. They cannot, and the reason is not a bug to fix — it is scope. Seven of them assert stub-only semantics:

| Test | Why it is stub-only |
|---|---|
| `test_a_citation_nothing_retrieved_is_dropped` | drives the `!badcite` dev directive |
| `test_an_answer_with_no_citation_reports_low_grounding` | drives `!nocite` |
| `test_the_error_frame_replaces_the_rest_of_the_stream` | drives `!error` |
| `test_wipe_sends_the_user_back_through_onboarding` | asserts seeded connections survive the wipe |
| `test_wipe_refuses_without_the_literal_confirmation` | asserts seeded document count |
| `test_embedder_activation_requires_accepting_the_reindex` | needs a seeded model inventory |
| `test_eval_reports_progress_then_metrics` | needs a seeded eval set |

Connections, the model hub and eval are all explicitly out of scope for this slice, and the dev directives are stub-only by design. **Do not relax a single assertion** and do not build those subsystems to make the tests pass. Split the fixtures instead.

In `conftest.py`, keep a stub-only fixture and add the parameterized one:

```python
import shutil

REPO = Path(__file__).resolve().parents[3]


def _config(tmp_path: Path, backend: str) -> Config:
    return Config(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        data_dir=tmp_path,
        dev_mode=True,
        ram_mb=16384,
        vram_mb=0,
        gpu_backend="cpu",
        backend=backend,
    )


def _seed_real_corpus(test_client: TestClient, tmp_path: Path) -> None:
    """Gives the real backend the documents the stub seeds itself with."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for source in (REPO / "fixtures" / "docs").glob("*.md"):
        shutil.copy(source, corpus / source.name)
    response = test_client.post(
        "/sources", json={"path": str(corpus), "include_globs": ["**/*.md"]}
    )
    assert response.status_code == 201, response.text


@pytest.fixture
def stub_client(tmp_path: Path) -> Iterator[TestClient]:
    """For contract tests that assert stub-only semantics: dev directives, seeded state."""
    with TestClient(create_app(_config(tmp_path, "stub"))) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield test_client


@pytest.fixture(params=["stub", "real"])
def client(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """For contract tests both backends must satisfy identically."""
    monkeypatch.setenv("RAGCORE_FAKE_MODELS", "1")
    with TestClient(create_app(_config(tmp_path, request.param))) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        if request.param == "real":
            _seed_real_corpus(test_client, tmp_path)
        yield test_client
```

Then in `test_api.py`, change the seven tests above to take `stub_client` instead of `client` — the parameter rename is the entire edit; every assertion stays exactly as written. The six that keep `client` are the ones both backends genuinely owe:

`test_health_is_public`, `test_everything_else_needs_the_session_token`, `test_openapi_is_reachable_without_a_token`, `test_documents_expose_content_the_reader_can_highlight`, `test_query_streams_frames_in_contract_order`, `test_sources_and_jobs_round_trip`.

Two of those six are the real prize: `test_documents_expose_content_the_reader_can_highlight` verifies that `text[char_start:char_end] == chunk["text"]` holds for chunks the real chunker produced against pages the real parser produced, and `test_query_streams_frames_in_contract_order` verifies the full SSE contract over a real index. If either fails, the failure is in your code, not in the test.

- [ ] **Step 8: Run everything**

Run: `uv run pytest -v`
Expected: PASS — 6 contract tests × 2 backends, 7 stub-only contract tests, plus every unit test. 19 test items from `test_api.py`.

- [ ] **Step 9: Verify by hand**

```bash
RAGCORE_FAKE_MODELS=1 uv run ragcore serve --backend real --port 8765 --data-dir /tmp/ragcore-dev &
curl -s -X POST localhost:8765/sources -H 'content-type: application/json' \
  -d '{"path":"'"$PWD"'/fixtures/docs","include_globs":["**/*.md"]}'
curl -s -N -X POST localhost:8765/query -H 'content-type: application/json' \
  -d '{"q":"what does reranking do"}'
```

Expected: SSE frames in order, a `sources` frame with real fixture text, and a `citations` frame with at least one marker.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: wire the real backend end to end (S1)"
```

---

### Task 12: Full-text search and RRF — S2 exit

**Files:**
- Modify: `core/ragcore/src/ragcore/store/lance.py` (`fts`, index creation)
- Modify: `core/ragcore/src/ragcore/retrieve/hybrid.py` (BM25 leg, fusion)
- Create: `core/ragcore/tests/test_fusion.py`
- Modify: `core/ragcore/tests/test_retrieve.py` (assert both ranks populated)

**Interfaces:**
- Produces:
  - `VectorStore.ensure_fts_index() -> None`
  - `VectorStore.fts(query: str, k: int) -> list[tuple[str, float]]` (implemented)
  - `retrieve.hybrid.rrf(ranked: list[list[str]], k: int) -> dict[str, float]`

- [ ] **Step 1: Write the failing fusion test**

Create `core/ragcore/tests/test_fusion.py`:

```python
"""Reciprocal rank fusion: the arithmetic, in isolation."""

from __future__ import annotations

import pytest
from ragcore.retrieve.hybrid import rrf


def test_a_chunk_ranked_first_by_both_legs_wins() -> None:
    fused = rrf([["a", "b", "c"], ["a", "c", "b"]], k=60)

    assert max(fused, key=fused.__getitem__) == "a"


def test_scores_match_the_formula() -> None:
    fused = rrf([["a", "b"], ["b", "a"]], k=60)

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_a_chunk_in_one_leg_only_still_scores() -> None:
    fused = rrf([["a"], ["b"]], k=60)

    assert set(fused) == {"a", "b"}
    assert fused["a"] == pytest.approx(1 / 61)


def test_empty_input_is_empty_output() -> None:
    assert rrf([], k=60) == {}
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_fusion.py -v`
Expected: FAIL, `ImportError: cannot import name 'rrf'`

- [ ] **Step 3: Implement fusion**

In `retrieve/hybrid.py`:

```python
def rrf(ranked: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal rank fusion. Rank is 1-based; k damps the head of each list."""
    fused: dict[str, float] = {}
    for leg in ranked:
        for rank, chunk_id in enumerate(leg, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused
```

- [ ] **Step 4: Implement full-text search in the store**

In `store/lance.py`, add to `__init__` after the tables are opened:

```python
        self._fts_ready = False
```

and:

```python
    def ensure_fts_index(self) -> None:
        """Builds the inverted index. Cheap to call repeatedly; LanceDB replaces it."""
        if self.chunk_count() == 0:
            return
        self.chunks.create_fts_index("text", replace=True)
        self._fts_ready = True

    def fts(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._fts_ready:
            self.ensure_fts_index()
        if not self._fts_ready:
            return []
        hits = (
            self.chunks.search(query, query_type="fts")
            .limit(k)
            .select(["chunk_id"])
            .to_list()
        )
        return [(hit["chunk_id"], float(hit.get("_score", 0.0))) for hit in hits]
```

Call `ensure_fts_index()` at the end of `add_chunks` when rows were added, and after `delete_by_doc`.

If LanceDB's FTS proves unusable in the installed version, the spec's fallback applies: keep vectors in LanceDB and build BM25 over a SQLite FTS5 table in `MetaStore`. Same `fts(query, k) -> list[tuple[str, float]]` signature, so nothing above this line changes. Record the decision in the commit message.

- [ ] **Step 5: Add the BM25 leg to the retriever**

In `HybridRetriever.asearch`, after the dense leg:

```python
        t0 = time.perf_counter()
        bm25_hits = self.vectors.fts(query, k=settings.bm25_top_k)
        latency.bm25_ms = (time.perf_counter() - t0) * 1000

        fused = rrf([[cid for cid, _ in hits], [cid for cid, _ in bm25_hits]], settings.rrf_k)
        order = sorted(fused, key=lambda cid: -fused[cid])[: settings.rerank_candidates]
```

Fetch rows for `order`, build `RetrievedChunk`s carrying `dense_rank`, `bm25_rank` and `rrf_score`, apply `_allowed`, and set `rerank_score = rrf_score` until Task 13 replaces it. `candidates` becomes `len(order)` after filtering.

- [ ] **Step 6: Strengthen the retrieval test**

Add to `core/ragcore/tests/test_retrieve.py`:

```python
async def test_both_legs_contribute_ranks(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, latency, _ = await retriever.asearch(
        "cross encoders rerank",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(),
        doc_meta={},
    )

    top = chunks[0]
    assert top.bm25_rank is not None
    assert top.rrf_score > 0
    assert latency.bm25_ms >= 0
```

The fake embedder has no semantics, so the dense leg and the BM25 leg genuinely disagree here — which is exactly the property fusion exists to exploit, and why this test is meaningful despite the fake.

- [ ] **Step 7: Run everything**

Run: `uv run pytest -v`
Expected: PASS.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add full-text leg and reciprocal rank fusion (S2)"
```

---

### Task 13: Reranking — S3 exit

**Files:**
- Create: `core/ragcore/src/ragcore/models/rerank.py`
- Create: `core/ragcore/tests/test_rerank.py`
- Modify: `core/ragcore/src/ragcore/retrieve/hybrid.py` (rerank stage)
- Modify: `core/ragcore/src/ragcore/backend.py` (build and inject the reranker)

**Interfaces:**
- Produces:
  - `models.rerank.RerankClient(base_url: str, *, model: str = "qwen3-reranker", timeout: float = 120.0)`
  - `async RerankClient.rerank(query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]` — `(original index, sigmoid score)`, sorted best first
  - `async RerankClient.aclose() -> None`
  - `models.rerank.sigmoid(x: float) -> float`

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_rerank.py`:

```python
"""The rerank client, and the fact that reranking actually reorders."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from ragcore.api.schemas import QueryFilters, RetrievalSettings
from ragcore.models.fakes import FakeRerankClient
from ragcore.models.rerank import RerankClient, sigmoid


def test_sigmoid_maps_to_the_unit_interval() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-8.0) < 0.01
    assert 0.99 < sigmoid(8.0) < 1.0


async def test_client_sorts_by_score_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RerankClient("http://127.0.0.1:8771")

    async def fake_post(query: str, documents: list[str], top_n: int) -> list[dict]:
        return [
            {"index": 0, "relevance_score": -2.0},
            {"index": 1, "relevance_score": 3.0},
            {"index": 2, "relevance_score": 1.0},
        ]

    monkeypatch.setattr(client, "_post", fake_post)
    ranked = await client.rerank("q", ["a", "b", "c"], top_n=2)

    assert [i for i, _ in ranked] == [1, 2]
    assert ranked[0][1] == pytest.approx(1 / (1 + math.exp(-3.0)))


async def test_fake_reranker_prefers_literal_overlap() -> None:
    ranked = await FakeRerankClient().rerank(
        "cross encoders", ["unrelated text here", "cross encoders score pairs"], top_n=2
    )

    assert ranked[0][0] == 1


@pytest.mark.requires_models
async def test_live_reranker_ranks_the_relevant_passage_first() -> None:
    client = RerankClient("http://127.0.0.1:8771")
    try:
        ranked = await client.rerank(
            "what do cross encoders do",
            [
                "Optical character recognition extracts text from scanned images.",
                "Cross encoders score a query and passage together and reorder candidates.",
            ],
            top_n=2,
        )
    finally:
        await client.aclose()

    assert ranked[0][0] == 1
```

Add to `core/ragcore/tests/test_retrieve.py`:

```python
async def test_reranking_changes_the_order(tmp_path: Path) -> None:
    from ragcore.models.fakes import FakeRerankClient

    plain = await build(tmp_path)
    with_rerank = await build(tmp_path)
    with_rerank.reranker = FakeRerankClient()

    settings = RetrievalSettings(min_score=0.0, top_k=3)
    before, _, _ = await plain.asearch(
        "optical character recognition", settings=settings, filters=QueryFilters(), doc_meta={}
    )
    after, latency, _ = await with_rerank.asearch(
        "optical character recognition", settings=settings, filters=QueryFilters(), doc_meta={}
    )

    assert after[0].chunk_id == "chk_ocr"
    assert [c.chunk_id for c in after] != [c.chunk_id for c in before]
    assert latency.rerank_ms > 0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_rerank.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.models.rerank'`

- [ ] **Step 3: Implement the client**

Create `core/ragcore/src/ragcore/models/rerank.py`:

```python
"""Reranking over llama-server's /v1/rerank endpoint.

llama.cpp returns raw cross-encoder logits. The pipeline's `min_score` gate is
expressed on a 0..1 scale, so scores are squashed here rather than at the gate —
one conversion, at the boundary where the raw number enters the system.
"""

from __future__ import annotations

import math

import httpx


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RerankClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "qwen3-reranker",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        response = await self._client.post(
            f"{self.base_url}/v1/rerank",
            json={"query": query, "documents": documents, "top_n": top_n, "model": self.model},
        )
        response.raise_for_status()
        return response.json()["results"]

    async def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        results = await self._post(query, documents, top_n)
        scored = [(int(r["index"]), sigmoid(float(r["relevance_score"]))) for r in results]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n]

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Add the rerank stage**

In `HybridRetriever.asearch`, after fusion and before the `min_score` gate:

```python
        if self.reranker is not None and scored:
            cap = min(settings.rerank_candidates, len(scored))
            t0 = time.perf_counter()
            ranked = await self.reranker.rerank(
                query, [c.text for c in scored[:cap]], top_n=settings.top_k
            )
            latency.rerank_ms = (time.perf_counter() - t0) * 1000
            scored = [
                scored[i].model_copy(update={"rerank_score": score}) for i, score in ranked
            ]
            # Now, and only now, the scores are on the scale min_score describes.
            scored = [c for c in scored if c.rerank_score >= settings.min_score]
```

The CPU profile's candidate cap arrives through `settings.rerank_candidates`, which `PerformanceSettings.profile` already drives in the settings route — no extra plumbing.

- [ ] **Step 5: Inject the reranker in the factory**

In `backend.py`'s real branch:

```python
        reranker = None if os.getenv("RAGCORE_FAKE_MODELS") else RerankClient(config.rerank_url)
        store.retriever = HybridRetriever(store.vectors, embedder, reranker=reranker)
```

No reranker under `RAGCORE_FAKE_MODELS`, deliberately. `FakeRerankClient` scores by literal token overlap, which is frequently below the 0.3 `min_score` default — wiring it into the default test backend would make contract tests fail for a reason that has nothing to do with the contract. Tests that exercise reranking inject `FakeRerankClient` explicitly, as `test_reranking_changes_the_order` does.

- [ ] **Step 6: Run everything**

Run: `uv run pytest -v`
Expected: PASS.

With servers up: `uv run pytest -v -m requires_models`
Expected: PASS (embed and rerank live tests).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore
git commit -m "feat: add reranking stage over llama-server (S3)"
```

---

### Task 14: PDF parsing with real page numbers

**Files:**
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (PDF dispatch)
- Create: `core/ragcore/tests/test_ingest_pdf.py`
- Modify: `core/ragcore/pyproject.toml` (add `pypdfium2`)

**Interfaces:**
- Produces: `parse()` handles `.pdf`, returning one `ParsedPage` per PDF page with `page` starting at 1 and `needs_ocr=True` when any page yields fewer than 20 characters.

- [ ] **Step 1: Add the dependency**

```bash
uv add --package ragcore pypdfium2
```

License check: `pypdfium2` is Apache-2.0/BSD-3. **Never** add PyMuPDF or `pymupdf4llm` — they are AGPL and this is a distributed product.

- [ ] **Step 2: Write the failing test**

First generate the fixture PDF once and commit it. `pypdfium2` reads PDFs but does not write them,
so build the fixture with `reportlab` in a scratch virtualenv — it never enters `ragcore` dependencies:

```bash
mkdir -p fixtures/pdf
python3 -m venv /tmp/pdfgen && /tmp/pdfgen/bin/pip install reportlab
/tmp/pdfgen/bin/python - <<'GEN'
from reportlab.pdfgen import canvas

c = canvas.Canvas("fixtures/pdf/three-pages.pdf")
for i, body in enumerate(
    [
        "Cross encoders score query passage pairs.",
        "Chunk size trades precision against completeness.",
        "Hybrid search merges dense and sparse retrieval.",
    ],
    start=1,
):
    c.drawString(72, 720, f"Page {i}")
    c.drawString(72, 700, body)
    c.showPage()
c.save()
GEN
```

Verify it before relying on it: `uv run python -c "import pypdfium2 as p; d=p.PdfDocument('fixtures/pdf/three-pages.pdf'); print(len(d), d[2].get_textpage().get_text_range()[:40])"`
Expected: `3` and text containing `Hybrid search`.

Then create `core/ragcore/tests/test_ingest_pdf.py`:

```python
"""PDF pages map one-to-one onto citation pages. Off-by-one here is a wrong citation."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragcore.ingest.parse import UnsupportedFormat, parse

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "pdf" / "three-pages.pdf"


def test_each_pdf_page_becomes_one_parsed_page() -> None:
    parsed = parse(FIXTURE)

    assert [p.page for p in parsed.pages] == [1, 2, 3]


def test_page_numbers_are_one_based_and_match_content() -> None:
    parsed = parse(FIXTURE)

    assert "Cross encoders" in parsed.pages[0].text
    assert parsed.pages[0].page == 1
    assert "Hybrid search" in parsed.pages[2].text
    assert parsed.pages[2].page == 3


def test_title_falls_back_to_the_file_stem() -> None:
    assert parse(FIXTURE).title in {"three-pages", "Page 1"}


def test_a_pdf_without_a_text_layer_is_flagged_for_ocr(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    blank = pdfium.PdfDocument.new()
    blank.new_page(595, 842)
    target = tmp_path / "scan.pdf"
    blank.save(str(target))

    parsed = parse(target)

    assert parsed.needs_ocr is True
    assert parsed.pages[0].text == ""
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_ingest_pdf.py -v`
Expected: FAIL — `UnsupportedFormat: no parser for '.pdf'`

- [ ] **Step 4: Implement PDF parsing**

In `ingest/parse.py`, add:

```python
OCR_THRESHOLD = 20


def _pdf(path: Path) -> ParsedDoc:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    pages: list[ParsedPage] = []
    needs_ocr = False
    try:
        for index in range(len(pdf)):
            textpage = pdf[index].get_textpage()
            try:
                body = textpage.get_text_range().strip()
            finally:
                textpage.close()
            if len(body) < OCR_THRESHOLD:
                needs_ocr = True
                body = "" if not body else body
            pages.append(
                ParsedPage(page=index + 1, section_path=f"{path.stem} > p{index + 1}", text=body)
            )
    finally:
        pdf.close()
    return ParsedDoc(title=path.stem, pages=pages, needs_ocr=needs_ocr)
```

and add `if suffix == ".pdf": return _pdf(path)` to `parse`. Add `".pdf"` handling to `walk.MIME_BY_EXT` (already present).

Pages flagged `needs_ocr` are kept with empty text rather than dropped: the page still exists in the document, and dropping it would shift every later page number by one, which is precisely the failure this task exists to prevent.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_ingest_pdf.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore fixtures/pdf uv.lock
git commit -m "feat: add PDF parsing with page-faithful citations"
```

---

### Task 15: Corpus run — S4 exit

**Files:**
- Create: `core/ragcore/tests/test_corpus_ingest.py`
- Modify: `core/ragcore/src/ragcore/store/real.py` (propagate `needs_ocr` into `Document.status`)

**Interfaces:**
- Produces: documents whose every page lacked a text layer get `status="skipped"` and `error="no text layer; OCR not available in this build"`.

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_corpus_ingest.py`:

```python
"""A corpus-scale run: mixed formats, clean re-run, OCR flagging."""

from __future__ import annotations

import shutil
from pathlib import Path

from ragcore.config import Config
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore

REPO = Path(__file__).resolve().parents[3]


def build_corpus(root: Path, copies: int = 10) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for source in (REPO / "fixtures" / "docs").glob("*.md"):
        for i in range(copies):
            shutil.copy(source, corpus / f"{source.stem}-{i}.md")
    shutil.copy(REPO / "fixtures" / "pdf" / "three-pages.pdf", corpus / "three-pages.pdf")
    return corpus


async def test_a_mixed_corpus_indexes_without_errors(tmp_path: Path) -> None:
    store = RealStore(Config(data_dir=tmp_path / "data", backend="real"), embedder=FakeEmbedClient())
    source = store.add_source_for_path(str(build_corpus(tmp_path)))

    documents = await store.ingest_source_async(source.id)

    assert len(documents) >= 50
    assert [d for d in documents if d.status == "error"] == []
    assert store.index_stats().chunks > len(documents)


async def test_the_second_run_is_a_no_op(tmp_path: Path) -> None:
    store = RealStore(Config(data_dir=tmp_path / "data", backend="real"), embedder=FakeEmbedClient())
    source = store.add_source_for_path(str(build_corpus(tmp_path)))
    await store.ingest_source_async(source.id)
    chunks = store.index_stats().chunks

    again = await store.ingest_source_async(source.id)

    assert {d.status for d in again} == {"skipped"}
    assert store.index_stats().chunks == chunks


async def test_a_scanned_pdf_is_skipped_with_a_reason(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    corpus = tmp_path / "scans"
    corpus.mkdir()
    blank = pdfium.PdfDocument.new()
    blank.new_page(595, 842)
    blank.save(str(corpus / "scan.pdf"))

    store = RealStore(Config(data_dir=tmp_path / "data", backend="real"), embedder=FakeEmbedClient())
    source = store.add_source_for_path(str(corpus))
    [document] = await store.ingest_source_async(source.id)

    assert document.status == "skipped"
    assert "OCR" in (document.error or "")


async def test_reader_page_references_match_the_pdf(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(REPO / "fixtures" / "pdf" / "three-pages.pdf", corpus / "three-pages.pdf")

    store = RealStore(Config(data_dir=tmp_path / "data", backend="real"), embedder=FakeEmbedClient())
    source = store.add_source_for_path(str(corpus))
    [document] = await store.ingest_source_async(source.id)
    content = store.content(document.id)

    assert content is not None
    assert content.n_pages == 3
    hybrid_page = next(p for p in content.pages if "Hybrid search" in p.text)
    assert hybrid_page.page == 3
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_corpus_ingest.py -v`
Expected: FAIL on `test_a_scanned_pdf_is_skipped_with_a_reason` — status is `"indexed"`, not `"skipped"`.

- [ ] **Step 3: Propagate the OCR flag**

In `RealStore.ingest_source_async`, after parsing:

```python
            if parsed.needs_ocr and not any(p.text.strip() for p in parsed.pages):
                document = self._document_for(found, source_id, status="skipped")
                document.error = "no text layer; OCR not available in this build"
                self.meta.upsert_document(document)
                self.meta.set_sha(str(found.path), found.sha256)
                results.append(document)
                continue
```

A PDF with *some* text pages still indexes; only a wholly image-based document is skipped. The sha is still recorded, so the second run does not re-parse it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_corpus_ingest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Measure ingestion against real models**

With servers up, record throughput for the spec's risk log:

```bash
./scripts/dev/serve-models.sh &
uv run ragcore serve --backend real --port 8765 --data-dir /tmp/ragcore-bench &
time curl -s -X POST localhost:8765/sources -H 'content-type: application/json' \
  -d '{"path":"'"$PWD"'/fixtures/docs","include_globs":["**/*.md"]}'
```

Append the wall time, document count and chunk count to `docs/superpowers/notes/2026-09-17-s0-reranker-gate.md` under a new "Throughput" heading. This is the number that sets `PerformanceSettings.embed_batch` defaults later.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore docs/superpowers/notes
git commit -m "feat: handle scanned PDFs and verify corpus-scale ingestion (S4)"
```

---

### Task 16: CLI verbs and the default flip — S5 exit

**Files:**
- Modify: `core/ragcore/src/ragcore/cli.py` (`index` and `ask` subcommands, default backend)
- Modify: `core/ragcore/src/ragcore/config.py` (default `backend="real"`)
- Create: `core/ragcore/tests/test_cli.py`
- Modify: `README.md` (how to run the slice)

**Interfaces:**
- Produces:
  - `ragcore index <path> [--data-dir DIR] [--embed-url URL]`
  - `ragcore ask "<question>" [--data-dir DIR] [--embed-url URL] [--rerank-url URL] [--top-k N]`
  - `Config.backend` default becomes `"real"`; `--backend stub` still selects the demo backend.

- [ ] **Step 1: Write the failing test**

Create `core/ragcore/tests/test_cli.py`:

```python
"""The headless entry points, driven the way a person drives them."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ragcore.cli import main
from ragcore.config import Config


@pytest.fixture(autouse=True)
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGCORE_FAKE_MODELS", "1")


def corpus(root: Path) -> Path:
    docs = root / "corpus"
    docs.mkdir()
    (docs / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nCross encoders score query passage pairs.\n"
    )
    return docs


def test_real_is_now_the_default_backend(tmp_path: Path) -> None:
    assert Config(data_dir=tmp_path).backend == "real"


def test_index_reports_what_it_indexed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["index", str(corpus(tmp_path)), "--data-dir", str(tmp_path / "data")])

    assert code == 0
    assert "1 document" in capsys.readouterr().out


def test_ask_prints_an_answer_with_a_citation(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    main(["index", str(corpus(tmp_path)), "--data-dir", str(tmp_path / "data")])
    capsys.readouterr()

    code = main(["ask", "what do cross encoders do", "--data-dir", str(tmp_path / "data")])
    out = capsys.readouterr().out

    assert code == 0
    assert "[" in out and ":" in out


def test_ask_on_an_empty_index_says_so(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["ask", "anything", "--data-dir", str(tmp_path / "empty")])

    assert code == 1
    assert "no documents" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest core/ragcore/tests/test_cli.py -v`
Expected: FAIL — `Config(...).backend == "stub"`, and `main(["index", ...])` prints help and returns 1.

- [ ] **Step 3: Implement the subcommands**

In `cli.py`, add two parsers and dispatch on `args.command`. Both run their async work with `asyncio.run`:

```python
    index = sub.add_parser("index", help="Index a folder into the local index")
    index.add_argument("path", type=Path)
    index.add_argument("--data-dir", type=Path, default=Path.home() / ".custom-rag")
    index.add_argument("--embed-url", default="http://127.0.0.1:8770")

    ask = sub.add_parser("ask", help="Ask a question against the local index")
    ask.add_argument("question")
    ask.add_argument("--data-dir", type=Path, default=Path.home() / ".custom-rag")
    ask.add_argument("--embed-url", default="http://127.0.0.1:8770")
    ask.add_argument("--rerank-url", default="http://127.0.0.1:8771")
    ask.add_argument("--top-k", type=int, default=6)
```

`_cmd_index` builds a real backend via `build_backend`, calls `add_source_for_path`, awaits `ingest_source_async`, and prints `f"{indexed} document(s), {chunks} chunk(s) indexed"` — the test asserts on `"1 document"`, so the singular must appear as a prefix of the plural form, i.e. print `"1 document(s)"`.

`_cmd_ask` builds the backend, returns `1` with `"No documents in the index."` when `index_stats().documents == 0`, otherwise awaits `retriever.asearch`, streams the answerer, prints the answer text, then prints each validated citation from `extract_citations` on its own line.

Change the `Config` dataclass default to `backend: Literal["stub", "real"] = "real"`, the `from_env` fallback to `os.getenv("RAGCORE_BACKEND", "real")`, and the `serve --backend` flag default to `"real"`.

- [ ] **Step 4: Fix the stub-dependent tests**

`test_backend.py::test_stub_is_the_default` now asserts the wrong thing. Rewrite it as `test_real_is_the_default` asserting `"real"`, and keep a test that `--backend stub` still builds the stub backend. The parameterized contract fixture already passes `backend=` explicitly, so it is unaffected.

- [ ] **Step 5: Run everything**

Run: `uv run pytest -v`
Expected: PASS — full suite, both backends.

- [ ] **Step 6: End-to-end by hand, with real models**

```bash
./scripts/dev/serve-models.sh &
uv run ragcore index ./fixtures/docs --data-dir /tmp/ragcore-real
uv run ragcore ask "what does reranking do" --data-dir /tmp/ragcore-real
```

Expected: an answer grounded in the fixtures, with at least one `[doc_id:page]` citation. This is the slice's exit condition.

- [ ] **Step 7: Document it**

Write `README.md` (currently empty): what the project is in two sentences, the `uv sync` / `scripts/fetch-models.sh` / `scripts/dev/serve-models.sh` / `ragcore index` / `ragcore ask` sequence, how to run the test suite including `-m requires_models`, and a short "what is real and what is still stubbed" list pointing at the spec.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add core/ragcore README.md
git commit -m "feat: add index/ask CLI verbs and default to the real backend (S5)"
```

---

## Verification checklist

Run before declaring the slice done:

- [ ] `uv run pytest -v` — full suite green; the six backend-agnostic contract tests green on **both** backends
- [ ] `uv run pytest -v -m requires_models` — green with both llama-servers up
- [ ] `uv run ruff check .` — clean
- [ ] `grep -rn "ragcore.stub" core/ragcore/src/ragcore/api/` — no output
- [ ] `grep -rn "pymupdf" core/ragcore/pyproject.toml uv.lock` — no output
- [ ] `uv run python -c "import ragcore, torch"` — `ModuleNotFoundError: torch`
- [ ] `ragcore index ./fixtures/docs && ragcore ask "what does reranking do"` against live models returns a cited answer
- [ ] `docs/superpowers/notes/2026-09-17-s0-reranker-gate.md` records the gate verdict and the throughput numbers
