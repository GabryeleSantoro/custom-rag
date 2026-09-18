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
        ids = tokenizer(
            PREFIX + body + SUFFIX, return_tensors="pt", truncation=True, max_length=4096
        )
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
