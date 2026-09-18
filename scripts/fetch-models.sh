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
# Qwen never published Qwen/Qwen3-Reranker-0.6B-GGUF (the repo 401s: it does
# not exist under that org). mradermacher's repo is a plain static quant of
# the same Qwen/Qwen3-Reranker-0.6B base model, no architecture changes.
fetch "https://huggingface.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/resolve/main/Qwen3-Reranker-0.6B.Q8_0.gguf" \
      "Qwen3-Reranker-0.6B-Q8_0.gguf"

echo "models in $DEST"
ls -lh "$DEST"
