# Custom RAG

Applicazione desktop **local-first** per interrogare una raccolta di documenti con risposte corredate da citazioni. L'interfaccia è una app Tauri/React; il motore RAG (`ragcore`) è un sidecar Python esposto solo su localhost e mediato dalla shell Rust.

> **Stato del progetto — prototipo in sviluppo.** L'app desktop e il contratto API sono funzionanti con un corpus di fixture e un backend `stub`. Il backend persistente con ingestione e retrieval tramite modelli locali è ancora in costruzione: la roadmap completa è in [plan/custom-rag-build-plan.md](plan/custom-rag-build-plan.md).

## Cosa offre oggi

- UI desktop multipagina per onboarding, libreria, lettore, chat, modelli, impostazioni, diagnostica ed evaluation.
- API FastAPI con token di sessione, streaming SSE delle risposte, chat, fonti, documenti, job e impostazioni.
- Risposte con citazioni validate: una citazione che non punta a un passaggio recuperato viene scartata.
- Shell Rust che avvia e supervisiona `ragcore`, conserva il token fuori dalla webview e usa il portachiavi del sistema per i segreti delle connessioni.
- Primitive già testate per scansione di cartelle, parsing Markdown/TXT, chunking con contesto, embedding e persistenza LanceDB/SQLite.

## Architettura

```text
React + TypeScript (apps/desktop)
          │ Tauri commands / Channel
          ▼
Rust shell (src-tauri) ── token di sessione, keychain, supervisione
          │ HTTP localhost autenticato
          ▼
FastAPI sidecar (core/ragcore)
          │
          ├── backend stub corrente: fixture + retrieval/risposte simulate
          └── backend reale in sviluppo: LanceDB + SQLite + llama-server
```

Nella configurazione finale l'app distribuirà localmente un embedder e un reranker Qwen3; il modello generativo resterà una connessione scelta dall'utente, locale o remota e compatibile con API OpenAI/Anthropic. I documenti non vengono inviati a un provider esterno senza una connessione remota configurata dall'utente.

## Requisiti per lo sviluppo

- Python 3.13 o superiore e [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/)
- Toolchain Rust e i prerequisiti di sistema di [Tauri 2](https://v2.tauri.app/start/prerequisites/)

`llama-server` è necessario solo per gli spike e i test che usano i modelli reali; non serve per avviare l'attuale backend `stub`.

## Avvio rapido

Dal repository:

```bash
uv sync
cd apps/desktop
bun install
bun tauri dev
```

`bun tauri dev` avvia Vite e la shell Tauri. In modalità debug la shell esegue automaticamente `ragcore` tramite `uv`, su una porta locale casuale, e gli passa un token temporaneo. Non usare `bun run dev` da solo per provare tutte le funzioni: avvia soltanto la web UI, che non può contattare direttamente il sidecar.

Per avviare soltanto l'API, utile per esplorare OpenAPI e gli endpoint:

```bash
uv run --directory core/ragcore ragcore serve --port 8765
```

La documentazione interattiva sarà disponibile su `http://127.0.0.1:8765/docs`. Se si passa `--token`, tutti gli endpoint tranne `/health` e OpenAPI richiedono `Authorization: Bearer <token>`.

## Modelli locali (sviluppo)

Il repository include gli script per scaricare Qwen3-Embedding-0.6B e Qwen3-Reranker-0.6B in `~/.custom-rag/models` e per avviare due istanze di `llama-server`:

```bash
# Installa prima llama.cpp/llama-server per il tuo sistema.
./scripts/fetch-models.sh
./scripts/dev/serve-models.sh
```

Le porte previste sono `8770` per l'embedder e `8771` per il reranker. Al momento questi servizi supportano gli spike e i test live: il selettore di backend predefinito è ancora `stub`, quindi l'app desktop non li supervisiona né esegue l'indicizzazione reale end-to-end.

## Test e qualità

```bash
# Suite predefinita: esclude i test che richiedono modelli live
uv run pytest -v

# Lint Python
uv run ruff check .

# Con llama-server attivo sulle porte 8770 e 8771
uv run pytest -v -m requires_models
```

Per controllare il frontend:

```bash
cd apps/desktop
bun run typecheck
bun run build
```

## Struttura del repository

```text
apps/desktop/        App Tauri 2: React, TypeScript, Vite e shell Rust
core/ragcore/        Sidecar Python: API, ingestione, store e modelli
fixtures/docs/       Piccolo corpus usato da prototipo e test
scripts/             Download dei modelli e strumenti di sviluppo
plan/                Architettura e piano di implementazione
docs/superpowers/    Specifiche, piani e note tecniche
```

## Roadmap tecnica

Le prossime parti principali sono il backend `real`, il retrieval ibrido (dense + full-text), reranking, parsing esteso di PDF/DOCX/HTML/CSV, ingestione incrementale e packaging dei sidecar per macOS, Windows e Linux. Dettagli, decisioni di licenza e fasi di consegna sono documentati nel [build plan](plan/custom-rag-build-plan.md).
