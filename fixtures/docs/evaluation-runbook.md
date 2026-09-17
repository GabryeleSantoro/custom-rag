# Evaluation Runbook

## Build the golden set before tuning anything

Fifty questions with known answers is enough to stop guessing. Each entry
records the question, the document identifiers that contain the answer, a short
expected answer, and whether the question is local, meaning answerable from a
few passages, or global, meaning it depends on the corpus as a whole.

Questions should come from the corpus, not from imagination. Reading ten
documents and writing five questions each produces a far more honest set than
inventing questions and hoping the corpus covers them.

## Metrics that earn their keep

Recall at six says whether the answer reached the prompt. Mean reciprocal rank
says how high it landed. Normalised discounted cumulative gain at six accounts
for several relevant passages at once. A latency split per stage — embed,
dense, keyword, rerank, pack, first token — says where the time went, which no
single end-to-end number ever reveals.

## Regressions are the point

An eval that runs once is a report. An eval wired into continuous integration,
failing the build when recall at six drops more than two points, is a guard
rail. Retrieval quality degrades in small quiet steps: a chunking tweak here, a
filter default there. Only a gate catches that.

## Two sets, not one

A public corpus committed to the repository makes results reproducible for
anyone. A private set built from real personal documents catches the failures
that public corpora never show, especially around scanned pages, mixed
languages and inconsistent formatting. Report both; trust the private one.

## Judging global answers

Global questions have no single correct passage, so retrieval metrics do not
apply. Grade them by hand, or with a model as judge on a rubric, and accept
that seventy percent judged correct is a reasonable target rather than a
disappointment.
