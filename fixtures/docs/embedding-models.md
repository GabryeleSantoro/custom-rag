# Embedding Models

## What an embedder is for

An embedder maps text to a fixed-length vector so that semantically similar
text lands nearby. In a retrieval system it runs twice: in bulk over every
passage during ingestion, and once per query at search time. The bulk pass
dominates cost, which is why embedder throughput sets the ceiling on how fast a
corpus can be indexed.

## Choosing a dimension

Larger vectors carry more information and cost more to store and compare. A
1024-dimension vector at float32 is 4 KB per passage, so a million passages is
roughly 4 GB before any index overhead. Quantising to int8 cuts that by four
with a small recall loss that is usually acceptable below a few million
passages.

## Why the embedder cannot be swapped casually

Vectors from two different embedders are not comparable. Changing the model
means every stored vector is meaningless against new queries, so the entire
corpus has to be re-embedded. A system that ships an embedder must record which
model and which dimension produced the index, and refuse to serve queries when
the stored value disagrees with the running model.

That check is cheap and it prevents the worst class of silent failure, where an
app update quietly swaps the model and retrieval quality collapses with no
error anywhere.

## Contextual headers

A passage pulled out of a long document loses its context. Prepending the
document title and the section path to the text before embedding restores
enough of it that passages from different sections stop colliding. The header
belongs in the embedded text only; the text sent to the generator should be the
original passage, because the header is noise to a reader.

## Normalisation

Cosine similarity assumes unit-length vectors. Normalising once at write time
rather than at every comparison is both faster and less error-prone, and it
means a dot product and a cosine give the same ranking.
