# Hybrid Search and Fusion

## Two kinds of miss

Dense search fails on rare literal strings: part numbers, error codes, proper
nouns it never saw in training. Keyword search fails on paraphrase, where the
question and the passage share meaning but no words. The failures are
uncorrelated, which is exactly the condition under which combining two systems
helps.

## Reciprocal rank fusion

Fusion by score requires the two scoring scales to be comparable, and they are
not. Reciprocal rank fusion sidesteps this by using only positions: each
document scores the sum over lists of one divided by k plus its rank, with k
around sixty. A document ranked third in both lists beats one ranked first in
only one, which is the behaviour you want.

The constant k controls how much weight the very top positions carry. Small k
makes the fusion trust rank one heavily; large k flattens the curve and lets
broad agreement win.

## Filters belong before fusion

Restricting by source, language or date after retrieval throws away candidates
that were already paid for, and can empty the result list entirely. Pushing the
filter into both retrieval passes keeps the candidate count stable and makes
the latency predictable.

## Measuring the merge

Recall at the candidate count is the metric that matters for the retrieval
stage, because anything not in the candidate list cannot be recovered later.
Precision is the reranker's job. Tracking both separately tells you which stage
to fix when the end-to-end numbers move.
