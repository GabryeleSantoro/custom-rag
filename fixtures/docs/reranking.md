# Reranking with Cross-Encoders

## Bi-encoders versus cross-encoders

A bi-encoder embeds the query and the passage separately, so passage vectors
can be computed once and reused. That is what makes vector search fast, and it
is also its weakness: the model never sees the query and the passage together,
so it cannot reason about how one answers the other.

A cross-encoder reads the pair as a single input and scores it directly. It is
far more accurate and far too slow to run over a whole corpus, so it is used as
a second stage over a few dozen candidates.

## Where it sits in the pipeline

Dense search and keyword search each return their own candidate list. The two
are merged, typically with reciprocal rank fusion, and the merged list goes to
the reranker. The reranker's scores then decide the final order and, through a
minimum-score gate, which passages are good enough to send to the generator at
all.

## Candidate count

Reranking cost is linear in the number of candidates. Forty candidates is a
reasonable default on a GPU; on CPU, twenty keeps latency tolerable. Going
above about sixty rarely changes the top six, because fusion has already put
the plausible passages near the top.

## The minimum-score gate

Without a gate, the reranker always returns its best six passages, even when
all six are irrelevant. Passing a sigmoid over the raw score and cutting below
a threshold lets the system return nothing, which is the correct answer to a
question the corpus does not cover. Reporting "not found" is a feature, and it
depends entirely on this gate existing.

## Swapping rerankers

Unlike the embedder, the reranker holds no state. Its scores are computed at
query time from text that is already stored, so a better reranker can be
dropped in at any moment with no re-indexing at all.
