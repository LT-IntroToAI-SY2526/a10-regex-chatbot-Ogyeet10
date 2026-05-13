"""Tests for the %/_ wildcard matcher in match.py."""

from match import match


# --- Literal tokens ---------------------------------------------------------


def test_exact_literal_match():
    assert match(["hello", "world"], ["hello", "world"]) == []


def test_case_insensitive_literals():
    assert match(["When", "Was"], ["when", "WAS"]) == []


def test_literal_mismatch_returns_none():
    assert match(["hello"], ["goodbye"]) is None


# --- Underscore (single-word) wildcards -------------------------------------


def test_underscore_captures_single_word():
    assert match(["i", "love", "_"], ["i", "love", "python"]) == ["python"]


def test_underscore_does_not_span_multiple_words():
    assert match(["_", "and", "_"], ["cats", "and", "dogs"]) == ["cats", "dogs"]


def test_underscore_must_consume_one_token():
    # `_` requires exactly one word - missing tail means no match
    assert match(["i", "love", "_"], ["i", "love"]) is None


# --- Percent (multi-word) wildcards -----------------------------------------


def test_percent_captures_trailing_phrase():
    result = match(["when", "was", "%", "born"], ["When", "was", "Grace", "Hopper", "born"])
    assert result == ["Grace Hopper"]


def test_percent_at_end_grabs_rest():
    result = match(["tell", "me", "about", "%"], ["tell", "me", "about", "the", "moon"])
    assert result == ["the moon"]


def test_percent_can_capture_empty_at_end():
    # `%` at end of pattern matches zero or more, so an empty tail is fine
    result = match(["tell", "me", "about", "%"], ["tell", "me", "about"])
    assert result == [""]


def test_percent_fails_when_source_runs_out_mid_pattern():
    # Need to find "born" after the capture but the source ends first
    assert match(["when", "was", "%", "born"], ["when", "was", "alice"]) is None


def test_multiple_percent_captures():
    result = match(
        ["%", "vs", "%"], ["alice", "and", "bob", "vs", "charlie", "and", "dave"]
    )
    assert result == ["alice and bob", "charlie and dave"]


# --- Mixed -----------------------------------------------------------------


def test_underscore_and_percent_together():
    result = match(["_", "is", "%"], ["python", "is", "a", "programming", "language"])
    assert result == ["python", "a programming language"]
