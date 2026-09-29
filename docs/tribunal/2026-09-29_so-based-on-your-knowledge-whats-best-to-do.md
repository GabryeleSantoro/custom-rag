**Invocation:** /tribunal:deliberate so based on your knowledge what's best to do

## ⚖️ Tribunal Verdict

**Topic:** Given the eval results and the light-CPU constraint, what is the best concrete plan for the shipped embedder and reranker?
**Domain:** strategic / technical
**Session ID:** 73167682-559d-473f-8991-c299ec055f81
**Depth:** full
**Multi-agent:** no

**Basis:** the eval in `scripts/eval/results.md` plus the panel's prior knowledge. Prior-knowledge claims, **not verified this session**: EmbeddingGemma-300M is 768-d with Matryoshka truncation, about 2K-token context, 100+ languages, and ships under Gemma terms of use (redistribution allowed, use restrictions must be passed on); Qwen3-Embedding-0.6B is 1024-d, long-context, Apache-2.0; hybrid dense + keyword search with rank fusion usually delivers most of the retrieval gain before a reranker adds anything.

### Final Verdict
**Decision:** Do this, in order:
1. Adopt EmbeddingGemma-300M Q4 as the default embedder (child chunks are about 256 tokens, well inside its context). Set `EMBED_DIM` to its real dimension now, update `fetch-models.sh` / `serve-models.sh` / plan, and keep the `index_meta` model+dim check.
2. Confirm the Gemma license is acceptable for redistribution; if in doubt, download the model on first run instead of bundling it. If it is not acceptable, fall back to Qwen3-Embedding-0.6B (Apache-2.0, already validated).
3. Ship without a reranker by default. Rely on hybrid search (dense + keyword, rank fusion). Replace the reranker-based "not found" gate with a calibrated score threshold.
4. Keep Qwen3-Reranker-0.6B as an opt-in download, limited to about 8 candidates, and only enable it after re-measuring with tuned server flags on a harder set.
5. Before v1, grow the eval: more queries, private Italian documents, and rare-literal queries (part numbers, error codes) so the keyword and reranker stages can show their value.
**Confidence:** 34%
**Dissent:** Devil's Advocate (toward reject): switching away from the already-validated Qwen3 pair on a 24-query eval is a bet; keep Qwen3-Embedding as default until the license is confirmed and a harder eval agrees, because an embedder change after users have indexes forces a re-index. Logician (neutral): the eval supports "cheaper and not worse", not "better".

### Votes
| Persona | Position | Lean | Conf. | v_i | Rationale |
|---|---|---|---|---|---|
| Domain Expert | support | n/a | 68% | +1 | Half the parameters, best eval score, multilingual; reranker gives no measured gain at 9-18 s/query on CPU. |
| Devil's Advocate | conditional | toward_reject | 55% | -0.5 | Small eval, unverified license, and no reranker weakens "not found"; validated Qwen3 stays default until proven otherwise. |
| Systems Thinker | conditional | toward_support | 65% | +0.5 | Embedder is the irreversible choice, and it is cheapest to change now before any user index exists; reranker opt-in stays reversible. |
| Logician | conditional | neutral | 55% | 0 | The plan follows from the user's premise (small, light CPU) plus "not worse"; it does not follow that the reranker is useless. |
| Mediator | conditional | toward_support | 62% | +0.5 | Ordered steps with a licensing gate and a Qwen3 fallback let both camps proceed without a fight. |

### Reasoning Trail
- v_i: +1, -0.5, +0.5, 0, +0.5 ; w_i: 0.68, 0.55, 0.65, 0.55, 0.62 (Σw = 3.05)
- Σ(v_i·w_i) = 0.68 - 0.275 + 0.325 + 0 + 0.31 = 1.04 → v̄ = 0.341
- Weighted consensus strength = 34% (> +0.15 → lean support; < 80 so Dissent included)
- Minority (negative) weight 0.55 / 3.05 = 18% (< 35%)
- Decision-follows-math audit: support-leaning text, v̄ > 0.15 ✔

---
*The full deliberation was produced internally. Use `/tribunal:deliberate --full-log …` (or `--verbose`, or ask for a full log) on a later run to include topic analysis, panel, arguments, cross-examination, and deliberation **in the saved file**.*
