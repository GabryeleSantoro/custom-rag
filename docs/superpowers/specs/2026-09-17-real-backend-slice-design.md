# Real Backend Slice: Design

**Date:** 2026-09-17
**Status:** approved, ready for implementation planning
**Scope:** replace the `ragcore` stub backend with a real one along a single narrow path, headless.

---

## 1. Why this, now

The repo today is a complete front half and a hollow back half:

- The Tauri desktop app is built: shell supervision, keychain, hardware detection, and every
  feature view (onboarding, library, chat, models, reader, eval, diagnostics, settings).
- The `ragcore` HTTP contract is built and frozen: all v1 routes, `schemas.py`, the SSE frame
  order, and 13 contract tests that lock it.
- Everything behind that contract is `ragcore/stub/`: in-memory dicts for the store, BM25 plus a
  TF-IDF cosine over six fixture markdown files for retrieval, and a scripted answer stream.

No embeddings exist. No vectors exist. No document has ever been parsed. Phase 0 of
`plan/custom-rag-build-plan.md` never ran, so the riskiest stack decision — whether
`llama-server --reranking` scores the Qwen3 reranker correctly — is still untested while the UI,
the schemas and the performance profiles are all already built on the assumption that it does.

This slice makes one path real through every layer and proves the stack, before widening.

## 2. Decisions

| Decision | Choice |
|---|---|
| Shape of work | Thin vertical slice, real end to end |
| Surface | Headless `ragcore` only. No Rust changes in this slice. |
| Build order | Walking skeleton, then deepen, with one risk spike first |
| Reranker | In scope. It is the plan's top named risk. |
| PDF | In scope via `pypdfium2`. OCR out of scope. |
| Chunking | Naive fixed-size for now. Heading-aware parent/child lands later. |
| PyInstaller bundle check | Out of scope. Deferred knowingly. |
| Stub coexistence | Protocol plus backend switch. Both backends satisfy the same contract tests. |

### Rejected alternatives

**Bottom-up by layer** (embed clients, then store, then ingest, then retrieve, then wire). Rejected
because the interfaces are already frozen by `schemas.py`, so layer isolation buys little, and it
defers all integration pain to the final day.

**Risk-first spikes, then bottom-up** (spike the reranker and LanceDB hybrid standalone, then
build). Partially adopted: the reranker spike is kept as step S0 because it gates a locked stack
decision. The LanceDB hybrid question is cheaper to answer inside the skeleton than beside it.

## 3. Architecture

### 3.1 The seam

`ragcore/ports.py` defines three Protocols: `StorePort`, `RetrieverPort`, `AnswerEngine`.

`ragcore/backend.py` exposes `build_backend(config) -> Backend`, a frozen dataclass holding
`store`, `retriever`, `answerer`, `hub` and `jobs`. The FastAPI lifespan in `api/app.py` calls it
and assigns the result to `app.state`; `api/deps.py` resolves request-scoped dependencies from
there instead of constructing stub classes directly.

`Config` gains `backend: Literal["stub", "real"]`, set by `--backend` or `RAGCORE_BACKEND`. The
default stays `stub` until S5.

### 3.2 Two leaks to close first

The current API layer reaches into `stub/` in two places that bypass any switch:

- `api/routes/query.py:33` imports `llm_stream`, `scripted_stream`, `extract_citations` and
  `parse_directives` from `stub.answers`.
- `api/routes/models.py:22` imports `HubClient` from `stub.hub`.

Resolution:

- `extract_citations` and `parse_directives` are pure and backend-agnostic. They move to
  `ragcore/citations.py` and are imported by both backends. No behavior change; the existing
  contract tests cover the move.
- `llm_stream` and `scripted_stream` move behind `AnswerEngine`. The real backend's engine is
  `llm_stream` promoted to `ragcore/llm/openai_compat.py`. The scripted path and the `!nocite`,
  `!badcite`, `!error` and `!slow` dev directives remain stub-only.
- `hub` stays stubbed for this slice. Model files are placed on disk by a dev script, not
  downloaded through the UI. The real hub client is a separate later chunk.

### 3.3 Packages this slice creates

```
ragcore/ports.py                        StorePort, RetrieverPort, AnswerEngine
ragcore/backend.py                      build_backend(config) -> Backend
ragcore/citations.py                    extract_citations, parse_directives (moved, shared)
ragcore/models/embed.py                 httpx client -> llama-server /v1/embeddings
ragcore/models/rerank.py                httpx client -> llama-server /v1/rerank
ragcore/store/lance.py                  LanceDB: documents, chunks, index_meta
ragcore/store/meta.py                   SQLite: sources, document metadata, chats, settings
ragcore/ingest/walk.py                  directory walk, glob filters, sha256
ragcore/ingest/parse.py                 pypdfium2, markdown, plain text
ragcore/ingest/chunk.py                 naive fixed-size split with contextual header
ragcore/retrieve/hybrid.py              dense + FTS -> RRF(k=60) -> rerank
ragcore/retrieve/pack.py                context packing to a token budget
ragcore/llm/openai_compat.py            promoted from stub/answers.py
```

### 3.4 Storage in this slice

LanceDB holds `documents`, `chunks` and `index_meta`, following the schemas in the build plan with
`EMBED_DIM = 1024`. SQLite holds sources, document status, chat sessions and settings.

The `parents` table is **not** created in this slice. Naive chunking produces no parent/child
relationship, so `retrieve/pack.py` packs child chunks directly to the token budget. Parent
mapping arrives with real chunking. `topic_nodes`, `summaries` and the entity tables are likewise
out of scope.

`index_meta` is written on first boot with `schema_version`, `embed_model`, `embed_dim` and
`reranker_model`. The migration rule from the build plan — block queries and offer a re-index when
the shipped embedder differs from the indexed one — is implemented in this slice, because getting
it wrong later means silently querying an index built by a different model.

## 4. Steps

Effort is in focused working days. Each step ends green, so any later step can bisect against it.

### S0 · Reranker gate (about 2 hours, throwaway)

`scripts/fetch-models.sh` downloads Qwen3-Embedding-0.6B-Q8_0 and Qwen3-Reranker-0.6B-Q8_0 GGUF
to `~/.custom-rag/models/`. `scripts/dev/serve-models.sh` starts two `llama-server` instances: the
embedder with `--embedding --pooling last`, the reranker with `--reranking`.

Qwen3 embedding models pool from the last token. Mean pooling returns plausible vectors with
quietly degraded recall, which is why the flag is pinned here rather than left to the default.

`scripts/spikes/rerank_check.py` scores 50 (query, passage) pairs through llama-server `/v1/rerank`
and through a `transformers` reference implementation, then compares them. `torch` is installed in
a throwaway virtualenv and never enters `ragcore` dependencies.

**Gate:** Spearman rank correlation at least 0.9 **and** top-1 agreement at least 0.9. Below that,
the fallback is ONNX Runtime for the reranker, which changes the `models/` design — stop and
re-plan rather than build on it.

The same script records the embedding dimension (expected 1024) and whether llama-server returns
L2-normalized vectors, which determines whether `models/embed.py` normalizes client-side.

### S1 · Skeleton: one file to a cited answer (about 2 days)

With `--backend real`, boot creates the SQLite schema, creates the LanceDB directory and writes
`index_meta`. Ingest one markdown file: walk, parse, split into roughly 256-word chunks, embed in
batches, write the `chunks` table with vectors. Each chunk's `embed_text` is the chunk prefixed
with `"{title} > {section_path}\n"`, as the build plan specifies; it costs nothing at this size and
keeps embed_text's meaning stable when real chunking replaces the splitter. Retrieval is vector-only top-k. Answers stream
through `openai_compat` with citation validation.

**Exit:** `POST /query` emits `start`, `mode`, `sources`, `token`*, `citations`, `done` from real
vectors, and all 13 contract tests pass against both backends.

### S2 · Hybrid retrieval (about 1 day)

Add a LanceDB full-text index on `chunks.text`. Dense top-40 and BM25 top-40 merge with reciprocal
rank fusion at k=60.

**Exit:** the two legs return demonstrably different orderings for the same query, the fused result
differs from either leg alone, and `StageLatency` is populated per stage rather than estimated.

### S3 · Reranking (about 1 day)

Candidates go to `models/rerank.py`, scores pass through a sigmoid, the `min_score` gate drops the
tail, and the top 6 survive. The CPU performance profile caps candidates at 20.

**Exit:** reranking measurably reorders results relative to raw RRF, and per-stage latency is
recorded.

### S4 · PDF and a real corpus (about 1 to 2 days)

`pypdfium2` parsing with page numbers preserved. Page fidelity is a correctness requirement, not
polish: citations carry `[doc:page]` and the reader view jumps to that page. Pages with fewer than
20 characters are flagged `needs_ocr` on the document rather than OCR'd. Re-running ingestion skips
files whose sha256 is unchanged.

**Exit:** 50 or more mixed documents index without errors, a second run is a no-op, and reader page
references land on the correct PDF page.

### S5 · Flip the default and expose the CLI (about half a day)

`--backend real` becomes the CLI default; the stub stays available behind the flag as the UI demo
mode. Add `ragcore index <path>` and `ragcore ask "<question>"` so the slice is drivable headless.

## 5. Testing

`conftest.py` gains a backend-parameterized client fixture, so the 13 contract tests run against
both backends and prove the HTTP contract is unchanged.

Real-backend tests cannot require two live `llama-server` processes. The real backend therefore
takes its embed and rerank clients by injection. Tests default to `FakeEmbedClient` (deterministic
hash-derived 1024-dimensional vectors) and `FakeRerankClient`, which keeps LanceDB, ingestion,
dedupe, RRF, packing and citation validation fully testable headless. Only `models/embed.py` and
`models/rerank.py` need live servers; those tests carry `@pytest.mark.requires_models` and are
deselected by default.

New unit tests cover: chunk boundaries and overlap, sha256 no-op re-index, RRF fusion math,
dropping a citation to an unretrieved document, and PDF page-number mapping.

**No golden set is built in this slice.** Naive chunking makes recall numbers unrepresentative of
the shipped system. The eval harness lands after real chunking, so its baseline means something.

## 6. Risks

| Risk | Handling |
|---|---|
| llama.cpp scores the Qwen3 reranker incorrectly | S0 gate, hard stop before anything depends on it |
| LanceDB full-text search is immature or awkward | Surfaces at S2. Fallback: BM25 over SQLite FTS5, LanceDB for vectors only. |
| Bulk embedding throughput on Metal is too slow | Measured at S1, sets batch size before the S4 corpus run |
| LanceDB native wheels resist PyInstaller | Knowingly deferred. Real debt: if it does not package, phase 6 gets expensive. |

## 7. Explicitly out of scope

Still stubbed or absent when this slice is done, by design:

- The Tauri app still runs against the stub backend. No Rust changes.
- Model downloads through the UI remain fake; `models/manifest.json` is not written.
- No OCR, and no docx, html or csv parsers.
- No `parents`, `topic_nodes`, `summaries` or entity tables.
- No `brain/` package: no document cards, topic tree, router or global answer path.
- No eval harness and no golden sets.
- No CI workflows, no license scanning, no packaging.

## 8. What comes after

In rough order: real chunking with the Qwen3 tokenizer and the parent/child split, then the eval
harness and golden sets against it, then the model hub and manifest, then wiring the Tauri shell to
the real backend including the two `llama-server` sidecars.
