# Chunking Strategies

## Why fixed-size chunking underperforms

Splitting every N tokens is simple and ignores the document. It cuts tables in
half, separates a heading from the paragraph it introduces, and produces
passages that read like fragments. Retrieval still works, but the passages sent
to the generator are worse than they need to be.

## Structure first, size second

A better order is to split on the document's own structure — headings, list
boundaries, page breaks, table boundaries — and only then enforce a size limit
inside each structural unit. Most documents have more structure than a plain
text extractor exposes, which is the main argument for a layout-aware parser.

## Parent and child chunks

Small passages retrieve well because their vector is specific. Large passages
generate well because they carry context. The parent-child pattern gets both:
embed small children of roughly 256 tokens, but send the generator the parent
passage of roughly 1024 tokens that each winning child belongs to.

Deduplicating parents matters here. Three winning children often share one
parent, and packing that parent three times wastes the context budget on
repetition.

## Overlap

A small overlap, around ten to fifteen percent, stops a sentence that straddles
a boundary from being lost to both sides. Large overlaps inflate the index and
cause near-duplicate passages to crowd the top of the results.

## Counting tokens

Chunk sizes should be counted with the tokenizer the embedder actually uses.
Counting characters or words is off by enough that a nominal 256-token chunk
can overflow the model's input window, and a silently truncated passage is
indistinguishable from a badly written one.
