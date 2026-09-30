"""Reciprocal rank fusion: the arithmetic, in isolation."""

from __future__ import annotations

import pytest
from ragcore.retrieve.hybrid import rrf


def test_a_chunk_ranked_first_by_both_legs_wins() -> None:
    fused = rrf([["a", "b", "c"], ["a", "c", "b"]], k=60)

    assert max(fused, key=fused.__getitem__) == "a"


def test_scores_match_the_formula() -> None:
    fused = rrf([["a", "b"], ["b", "a"]], k=60)

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_a_chunk_in_one_leg_only_still_scores() -> None:
    fused = rrf([["a"], ["b"]], k=60)

    assert set(fused) == {"a", "b"}
    assert fused["a"] == pytest.approx(1 / 61)


def test_empty_input_is_empty_output() -> None:
    assert rrf([], k=60) == {}
