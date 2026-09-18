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
# not exist under that org). Use ggml-org's own conversion, not a plain
# convert_hf_to_gguf.py static quant: Qwen3-Reranker is a causal LM scored by
# its yes/no logits, and llama.cpp's --reranking (LLAMA_POOLING_TYPE_RANK)
# needs the cls.output.weight classifier tensor + pooling metadata that only
# a reranker-aware conversion extracts. A plain static quant (e.g.
# mradermacher's) lacks it and produces near-zero garbage scores uncorrelated
# with relevance -- see docs/superpowers/notes/2026-09-17-s0-reranker-gate.md.
fetch "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf" \
      "Qwen3-Reranker-0.6B-Q8_0.gguf"

echo "models in $DEST"
ls -lh "$DEST"
