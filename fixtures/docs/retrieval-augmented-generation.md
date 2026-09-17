# Retrieval-Augmented Generation

## What the pattern solves

A language model answers from weights alone, which means it answers from
whatever was true when it was trained. Retrieval-augmented generation puts a
search step in front of generation: the question is used to find passages in a
corpus the model has never seen, and those passages are placed in the prompt.
The model then writes an answer grounded in text it can quote, and the system
can show the user exactly which passage every claim came from.

The practical consequence is that the index, not the model, owns the knowledge.
Swapping the generator changes the writing; it does not change what the system
knows. That separation is why index quality and generation quality should be
measured separately.

## The three stages

Ingestion reads files, splits them into passages, and stores a vector per
passage. Retrieval turns a question into the same vector space, finds
candidates, and orders them. Generation packs the best passages into a prompt
and asks a model to answer using only those passages.

Most failures that look like "the model is bad" are retrieval failures. If the
right passage never reaches the prompt, no amount of prompt engineering
recovers it. Debugging therefore starts by asking whether the answer was even
retrievable, which is what a recall metric measures.

## Grounding and citations

An answer without a citation is an assertion. The system should parse citation
markers out of the generated text, check each one against the passages that
were actually retrieved, and drop any that do not match. An answer that
survives with no citations at all is a signal worth surfacing to the user
rather than hiding: it usually means the corpus did not contain the answer and
the model filled the gap from its weights.

## When retrieval is the wrong tool

Questions about the shape of a corpus rather than its contents are not
retrieval questions. "What are the main themes across these reports" cannot be
answered by six passages, because the answer depends on all of them. These
global questions need a summarisation pass over clusters of documents, and
routing between the two modes is its own problem.
