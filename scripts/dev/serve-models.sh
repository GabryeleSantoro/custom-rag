#!/usr/bin/env bash
# Starts the two llama-server instances ragcore's real backend talks to.
# Embedder: port 8770. Reranker: port 8771.
#
# EmbeddingGemma takes its pooling from the GGUF metadata; don't override it.
# Its context is 2048 tokens (child chunks are ~256).
set -euo pipefail

DEST="${RAGCORE_MODEL_DIR:-$HOME/.custom-rag/models}"
LLAMA="${LLAMA_SERVER:-llama-server}"

"$LLAMA" -m "$DEST/embeddinggemma-300M-qat-Q4_0.gguf" \
  --embedding -c 2048 -ngl 999 --port 8770 --host 127.0.0.1 &
EMBED_PID=$!

"$LLAMA" -m "$DEST/Qwen3-Reranker-0.6B-Q8_0.gguf" \
  --reranking -c 8192 -ngl 999 --port 8771 --host 127.0.0.1 &
RERANK_PID=$!

trap 'kill $EMBED_PID $RERANK_PID 2>/dev/null || true' EXIT
echo "embed pid $EMBED_PID :8770 | rerank pid $RERANK_PID :8771"
wait
