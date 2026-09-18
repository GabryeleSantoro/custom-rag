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
