# Custom RAG: Build Plan

A local-first, cross-platform desktop app for chatting with your documents. The app ships its own embedder and reranker; the user connects any LLM they like (local or cloud) for generation.

Stack: **Tauri 2 (React + TS) UI, Rust shell, Python sidecar for the RAG core, and `llama-server` sidecars for the shipped models.**

---

## 0. Locked decisions

| Area | Choice | Why |
|---|---|---|
| UI | Tauri 2 + React + TS + Vite | Cross-platform, small binaries |
| Shell | Rust (`src-tauri`) | Starts and restarts the sidecars, stores API keys in the OS keychain, handles updates |
| RAG core | Python 3.12 package `ragcore`, FastAPI on localhost | Mature ML ecosystem (clustering, topic labels, NER) |
| Shipped models | Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B (GGUF), with Reranker-4B as an optional download | Run everywhere, CPU fallback |
| Model runtime | Two `llama-server` processes, one with `--embedding` and one with `--reranking` | Metal, CUDA, Vulkan or CPU from one codebase |
| Vector store | LanceDB (embedded) | Vectors + full-text search in one place, no server |
| App state | SQLite | Connections, chats, jobs, settings |
| User LLM | OpenAI-compatible adapter + native Anthropic adapter | Covers local and cloud providers |
| OCR | RapidOCR (ONNX) | One cross-platform code path, Apache 2.0 |
| PDF parsing | `pypdfium2` by default; Docling as an optional download | Docling pulls in torch, which adds GBs to the bundle |

**License guard:** avoid PyMuPDF / `pymupdf4llm`. They're AGPL, which is a problem for a distributed product. Add a license check to CI from day one.

### Division of labor

| | Embedder (shipped) | Reranker (shipped) | External LLM (user's choice) |
|---|---|---|---|
| **Job** | Turns text into vectors | Scores how well each chunk answers the query | Reads the top chunks and writes the answer |
| **When** | Ingestion (bulk) and every query (one string) | Every query, on about 20–40 candidates | Every query, once or twice |
| **Runs** | Always local | Always local | Wherever the user points it |
| **Swappable?** | No: changing it means re-indexing everything | Yes, anytime | Yes, anytime |

The external LLM is kept out of ingestion, so index quality never depends on which model was connected at indexing time.

---

## 1. Repo layout

```
custom-rag/
├── apps/desktop/
│   ├── src/                     # React UI
│   │   ├── features/{library,chat,topics,settings,onboarding}/
│   │   └── lib/ipc.ts           # typed wrappers over Tauri commands/channels
│   └── src-tauri/
│       ├── src/{main.rs,sidecars.rs,keychain.rs,proxy.rs,hardware.rs}
│       ├── binaries/            # llama-server-<triple>, ragcore-<triple>
│       └── tauri.conf.json
├── core/ragcore/
│   ├── api/                     # FastAPI routes, SSE, auth token
│   ├── ingest/                  # watcher, parsers, OCR, dedupe
│   ├── chunking/                # heading-aware, parent/child, token counting
│   ├── models/                  # embed + rerank clients (llama-server HTTP)
│   ├── store/                   # LanceDB tables, migrations, index_meta
│   ├── retrieve/                # hybrid search, RRF, rerank, context packing
│   ├── llm/                     # provider adapters, prompts, citation validation
│   ├── brain/                   # doc cards, topic tree, router, map-reduce
│   ├── jobs/                    # background queue, progress events
│   └── eval/                    # golden-set runner, metrics
├── models/manifest.json         # GGUF URLs, sha256, sizes, min hardware
├── eval/golden/*.jsonl
├── scripts/                     # build sidecars, fetch llama.cpp releases
└── .github/workflows/           # 3-OS matrix: test, build, sign
```

---

## 2. Runtime architecture

- **Startup.** The Rust shell detects hardware (Metal, CUDA, Vulkan, RAM), picks a performance profile, and starts both `llama-server` instances and `ragcore`, each on a random free port. It passes a per-session auth token to `ragcore`.
- **IPC.** The UI never calls Python directly. It calls Tauri commands, and Rust proxies them to `ragcore` with the token. Streamed answers flow Python (SSE) → Rust → UI through a Tauri `Channel`.
- **Supervision.** Health checks every few seconds, restart on crash with backoff, and clean shutdown when the app closes.
- **Secrets.** Rust stores API keys in the OS keychain (`keyring` crate) and injects them per request. They are never written to SQLite or logs.
- **Performance profiles** set the defaults below; users can override them:

| Profile | Rerank candidates | Reranker | Embed batch |
|---|---|---|---|
| CPU-only | 20 | 0.6B | 8 |
| Apple Silicon / GPU ≤ 8 GB | 40 | 0.6B | 32 |
| GPU > 8 GB | 40 | 4B (optional) | 64 |

---

## 3. Data model (LanceDB)

```python
import pyarrow as pa

EMBED_DIM = 1024

index_meta = pa.schema([
    ("key", pa.string()),            # schema_version, embed_model, embed_dim, reranker_model, last_cluster_at
    ("value", pa.string()),
])

documents = pa.schema([
    ("doc_id", pa.string()),
    ("source_id", pa.string()),
    ("path", pa.string()),
    ("title", pa.string()),
    ("mime", pa.string()),
    ("sha256", pa.string()),
    ("mtime", pa.timestamp("ms")),
    ("lang", pa.string()),
    ("n_pages", pa.int32()),
    ("keywords", pa.list_(pa.string())),
    ("summary_chunk_ids", pa.list_(pa.string())),
    ("topic_id", pa.string()),
    ("vector", pa.list_(pa.float32(), EMBED_DIM)),
])

chunks = pa.schema([                  # child chunks: embedded + FTS indexed
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("parent_id", pa.string()),
    ("section_path", pa.string()),
    ("page_start", pa.int32()),
    ("page_end", pa.int32()),
    ("text", pa.string()),
    ("embed_text", pa.string()),      # contextual header + text
    ("n_tokens", pa.int32()),
    ("vector", pa.list_(pa.float32(), EMBED_DIM)),
])

parents = pa.schema([                 # sent to the LLM, never embedded
    ("parent_id", pa.string()),
    ("doc_id", pa.string()),
    ("section_path", pa.string()),
    ("page_start", pa.int32()),
    ("page_end", pa.int32()),
    ("text", pa.string()),
    ("n_tokens", pa.int32()),
])

topic_nodes = pa.schema([
    ("topic_id", pa.string()),
    ("parent_topic_id", pa.string()),
    ("level", pa.int32()),
    ("label", pa.string()),
    ("keywords", pa.list_(pa.string())),
    ("doc_ids", pa.list_(pa.string())),
    ("rep_chunk_ids", pa.list_(pa.string())),
    ("vector", pa.list_(pa.float32(), EMBED_DIM)),   # centroid
])

summaries = pa.schema([               # layer 5 cache, opt-in
    ("target_type", pa.string()),     # "document" | "topic"
    ("target_id", pa.string()),
    ("model_id", pa.string()),
    ("text", pa.string()),
    ("generated_at", pa.timestamp("ms")),
    ("stale", pa.bool_()),
])
```

The entity tables (`entities`, `entity_mentions`) are added in phase 7.

**Migration rule:** on startup, compare `index_meta.embed_model` and `embed_dim` with the shipped model. If either differs, block queries and offer a re-index.

### Global brain layers

| Layer | What it holds | Built with | LLM needed? |
|---|---|---|---|
| 1. Chunk index | Child and parent chunks, vectors, BM25 | Embedder | No |
| 2. Document cards | Metadata, doc vector, keywords, extractive summary | Embedder + math | No |
| 3. Topic tree | Hierarchical clusters, labels, representative chunks | Clustering + c-TF-IDF | No |
| 4. Entity graph (v2) | Entities and document co-mentions | Small NER model | No |
| 5. Abstractive summaries (opt-in) | Written summaries per document and per cluster | An LLM | Yes |

Layers 1–4 are built on-device, reproducible, and independent of the connected model. Layer 5 is a cache on top.

---

## 4. Phases

Effort is in focused working days. Calendar time depends on available time.

### Phase 0: Spikes and scaffolding (4–5 days)

Goal: remove the riskiest unknowns before writing real code.

- Check that `llama-server --reranking` scores the Qwen3-Reranker GGUF correctly: compare its scores against the reference `transformers` output on 50 pairs.
- Benchmark embedding and reranking throughput on CPU, Metal, and CUDA/Vulkan.
- Package a minimal `ragcore` with PyInstaller on all 3 OSes and record bundle size. This decides UMAP (which pulls in numba) versus PCA.
- Get a Tauri sidecar spawning, passing its port and token, and a Rust → Python round trip working.
- Run LanceDB hybrid search (vectors + full-text) on a small test corpus.

**Exit:** a CLI that indexes 10 PDFs and answers one question end to end on a Mac.

### Phase 1: Ingestion core (6–8 days)

- **Sources:** a folder picker plus a `watchdog` watcher, with include/exclude globs and a max file size.
- **Parsers:** PDF (`pypdfium2`), DOCX (`python-docx`), MD, TXT, HTML (`selectolax`), CSV as row groups.
- **OCR:** run RapidOCR only on pages with no text layer, for example fewer than 20 characters.
- **Dedupe and incremental updates:** skip files whose sha256 is unchanged; delete and re-insert changed ones; remove deleted ones.
- **Chunking:** headings or layout first, then 256-token children and 1024-token parents with about 12% overlap. Count tokens with the HF `tokenizers` Qwen3 tokenizer.
- **Contextual header:** prepend `"{title} > {section_path}\n"` to each chunk's `embed_text`.
- **Embedding:** batched calls to the embedder's `/v1/embeddings`, L2-normalized, and resumable after a crash via job checkpoints.
- **Job queue:** in SQLite, with progress events for the UI.
- **Language detection** (`lingua` or `fasttext-lid`) per document.

**Exit:** 1,000 mixed documents index without errors, re-running the index is a no-op, and interrupted jobs resume.

### Phase 2: Retrieval and eval harness (6–8 days)

- **Hybrid search:** dense top 40 plus BM25 top 40, merged with RRF (k=60).
- **Rerank:** candidates go to the reranker's `/v1/rerank`, then sigmoid scores, the `min_score` gate, and keep the top 6.
- **Context packing:** map children to their parents, dedupe overlapping parents, and fill the token budget in score order.
- **Filters:** by source, document type, date, and language.
- **Eval harness:**
  - Golden set format: `{"q", "expected_doc_ids", "expected_answer", "type": "local|global"}`.
  - Metrics: recall@k, MRR, nDCG@6, and a latency split per stage.
  - Command: `ragcore eval --set eval/golden/base.jsonl --report out.json`.
  - CI gate: fail if recall@6 drops by more than 2 points.
- **Build golden sets:** 50 questions on a public corpus (committed to the repo) plus a private one from personal documents.

**Exit:** recall@6 ≥ 0.85 on the base set and p50 retrieval under 1.5 s on the CPU profile.

### Phase 3: LLM connections and generation (5–6 days)

- **Adapters:**
  - OpenAI-compatible (LM Studio, Ollama, llama.cpp, vLLM, OpenAI, OpenRouter) and native Anthropic.
  - Streaming, cancellation, timeouts, and retries.
- **Connection settings:** base URL, model ID, context window, max output tokens, thinking toggle or reasoning effort, and a local/remote flag.
- **"Test connection":** checks reachability, streaming support, and roughly measures tokens per second.
- **Prompts:** a grounded-answer system prompt, a follow-up condenser, and a "not found" template.
- **Reasoning cleanup:** strip `<think>` blocks and `reasoning_content` before rendering.
- **Citation validation:**
  - Parse `[doc:page]` citations and drop any that don't match a retrieved chunk.
  - Flag answers that contain no citations at all.
- **Chat history:** a history budget, with older turns summarized by the user's LLM only when the budget overflows.

**Exit:** the same question answered correctly with valid citations via LM Studio, Ollama, and one cloud provider.

### Phase 4: Tauri UI MVP (8–10 days)

- **Onboarding:**
  - Hardware check, then download the shipped models with sha256 verification and resumable downloads.
  - Add the first LLM connection and the first source folder.
- **Library:** sources, per-document status, indexing progress, errors, and re-index/remove actions.
- **Chat:**
  - Streaming answers with clickable citations that open a side panel with the source passage and page.
  - A local/global mode badge.
  - A remote-provider warning when chunks will leave the device.
- **Settings:** connections, retrieval parameters (with an advanced section), performance profile, storage location, and a data wipe option.
- **Diagnostics:** sidecar status, logs, and index stats.

**Exit:** a non-technical tester goes from install to first cited answer with no help.

### Phase 5: Global brain (8–10 days)

- **Document cards:**
  - Doc vector = mean of chunk vectors, normalized.
  - Extractive summary = the chunks closest to that vector.
  - Keywords = TF-IDF against the whole corpus.
- **Topic tree:**
  - Reduce dimensions (UMAP, or PCA if bundle size demands it), then cluster with scikit-learn `HDBSCAN`.
  - Recurse into clusters above a size threshold, to a maximum of 3 levels.
  - Label clusters with class-based TF-IDF keywords and store representative chunks.
- **Incremental updates:** assign new documents to the nearest cluster centroid; re-cluster in the background once 15–20% of the corpus has changed; mark affected summaries stale.
- **Router:** an embedding classifier against labeled example queries, with an LLM fallback only for ambiguous scores and a manual override in the UI.
- **Global answer path:**
  - Pick relevant branches, then map over each branch (summary if cached, otherwise representative chunks), then reduce into one answer.
  - Cap at 8 map calls, show progress in the UI, and carry citations through to the final answer.
- **Topics explorer:** a tree view, documents per topic, and "ask about this topic" to scope a question to one branch.

**Exit:** global golden questions judged correct at least 70% of the time, and re-clustering 5,000 documents in under 2 minutes on a Mac.

### Phase 6: Packaging and distribution (6–8 days)

- **CI build matrix** for macOS arm64 + x64, Windows x64, and Linux x64; bundle `llama-server` per platform (Metal / CUDA+Vulkan / Vulkan+CPU).
- **Signing:**
  - macOS: every nested binary must be signed and the app notarized.
  - Windows: code-signing certificate.
  - Linux: AppImage + deb.
- **Updates:** the Tauri updater for the app; the model manifest is versioned separately from the app.
- **Crash reporting:** opt-in only, and it never sends document content.

**Exit:** signed installers that pass install/uninstall tests on clean VMs.

### Phase 7: v2 (after launch)

- **Entity graph:** GLiNER exported to ONNX (so torch isn't bundled), co-mention links, and "everything about X" queries.
- **Layer 5 summaries:** generated lazily, with an optional small "deep index" generator pack.
- **Optional downloads:** the Docling advanced-layout pack and the Reranker-4B upgrade.
- **Other:** more connectors (IMAP, Notion, Drive), multi-library workspaces, and reusing index snapshots across machines.

---

## 5. Sidecar API (v1)

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Status of the sidecar, the model servers, and the index |
| POST | `/sources` | Add a folder with its globs |
| DELETE | `/sources/{id}` | Remove a source and its documents |
| GET | `/documents` | Paged list with status |
| POST | `/index/rebuild` | Full or partial re-index |
| GET | `/jobs/{id}` | Job progress |
| POST | `/query` | SSE stream of mode, retrieved sources, answer tokens, then final citations |
| GET | `/brain/topics` | The topic tree |
| POST | `/brain/recluster` | Trigger re-clustering manually |
| POST | `/connections/test` | Probe an LLM endpoint |
| POST | `/eval/run` | Dev builds only |

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bundle size (torch, numba) | No torch in core; PCA fallback for UMAP; heavy features as optional downloads |
| Qwen3-Reranker scoring wrong in llama.cpp | Phase 0 comparison against `transformers`; fallback is ONNX Runtime for the reranker |
| Slow reranking on CPU | Fewer candidates in the CPU profile, batching, cache scores per (query, chunk) |
| Weak user models ignore citation format | Citation validation, a stricter prompt variant, a "low grounding" warning in the UI |
| Unknown context window | Per-connection setting, conservative default (8k), budget derived from it |
| Shipped embedder changes in an app update | `index_meta` check, background re-index, old index kept until the new one is ready |
| Data leaving the device | Local/remote flag, per-connection consent, embed and rerank always local |
| macOS notarization of nested binaries | Sign sidecars in CI before bundling; test notarization early in phase 6 |
| Copyleft dependencies | License scan in CI (`pip-licenses`, `cargo-deny`) |

---

## 7. Milestones

- **M1 (end of phase 2):** headless core with a CLI and eval numbers.
- **M2 (end of phase 4):** an internal alpha on a Mac.
- **M3 (end of phase 5):** feature-complete v1 with the global brain.
- **M4 (end of phase 6):** a public beta with signed installers on all three platforms.
