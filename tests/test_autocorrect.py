"""Tests for the command-word autocorrect layer.

The exact bug we're guarding against: a typo in a command word (e.g.
"captial" instead of "capital") makes the specific pattern miss, so the
catch-all `what is %` fires and tries to look up nonsense as a Wikipedia
page. After autocorrect, the specific pattern matches and the right
handler runs.

Equally important: autocorrect must NOT mangle proper names that happen
to live in topic captures (the % wildcard).
"""

import pytest

from a10 import _autocorrect_words


# --- Command typos get fixed -----------------------------------------------


class TestAutocorrectFixesTypos:
    @pytest.mark.parametrize(
        "typed,expected_word",
        [
            # The reported bug: transposition
            ("captial",      "capital"),
            # Missing characters (high similarity ratio)
            ("populaton",    "population"),
            ("hedquarters",  "headquarters"),
            ("foundedd",     "founded"),
            # Exact matches stay unchanged
            ("founder",      "founder"),
            ("capital",      "capital"),
            # Wrong-letter substitution near end
            ("populatian",   "population"),
        ],
    )
    def test_single_typo_snaps_to_vocab_strict(self, typed, expected_word):
        result = _autocorrect_words([typed])  # default cutoff = 0.82
        assert result == [expected_word], (
            f"Expected {typed!r} -> {expected_word!r} at strict cutoff, got {result!r}"
        )

    def test_capital_typo_in_full_query(self):
        """The reported bug: 'what is the captial of france' must end up
        with 'capital' so the specific pattern matches."""
        original = "what is the captial of france".split()
        corrected = _autocorrect_words(original)
        assert corrected == "what is the capital of france".split()

    def test_aggressive_typo_only_at_looser_cutoff(self):
        """Tier 2 (cutoff=0.75) catches typos the strict pass intentionally
        misses to avoid false positives.

        Classic example: 'borm' vs 'born' have similarity ratio exactly 0.75
        (3 of 4 chars match). The strict 0.82 cutoff rejects this to avoid
        risky single-char-substitution snaps on short words; the looser
        second-tier cutoff catches it.
        """
        # Strict pass: should NOT fix
        strict = _autocorrect_words(["borm"], cutoff=0.82)
        assert strict == ["borm"], (
            f"Strict pass shouldn't fix 'borm', got {strict!r}"
        )
        # Looser pass: SHOULD fix
        looser = _autocorrect_words(["borm"], cutoff=0.75)
        assert looser == ["born"], (
            f"Looser pass should fix 'borm' -> 'born', got {looser!r}"
        )


# --- Proper names and unfamiliar words pass through untouched --------------


class TestAutocorrectPreservesContent:
    @pytest.mark.parametrize(
        "topic_word",
        [
            "France",
            "Microsoft",
            "Apple",
            "Tokyo",
            "Einstein",
            "SpaceX",
            "Linux",
            # The classic gotcha: word that's similar to a command word but
            # NOT a typo of it. "Captain" is close to "capital" but distinct.
            "Captain",
            # Single short words that might tempt the matcher
            "the",  # already in vocab - unchanged
            "of",   # already in vocab - unchanged
        ],
    )
    def test_topic_word_unchanged(self, topic_word):
        result = _autocorrect_words([topic_word])
        # Either exact preservation or case-folded form of itself - never
        # snapped to a different word.
        assert result[0].lower() == topic_word.lower(), (
            f"Autocorrect mangled {topic_word!r} into {result[0]!r}"
        )

    def test_full_query_with_proper_names_intact(self):
        """A correctly-typed query with proper names should round-trip
        unchanged."""
        original = "tell me about Captain America".split()
        corrected = _autocorrect_words(original)
        assert corrected == original

    def test_correctly_typed_query_unchanged(self):
        """No typos = no changes. We don't want any spooky rewrites."""
        original = "what is the capital of france".split()
        corrected = _autocorrect_words(original)
        assert corrected == original

    def test_common_content_word_tall_is_not_mangled(self):
        """Regression: command autocorrect must not mutate content words that
        belong to the generic NLP engine. `tall` should stay `tall`, not snap
        to `tell`."""
        corrected = _autocorrect_words("how tall is mount everest".split())
        assert corrected == "how tall is mount everest".split()

    def test_field_word_area_is_not_mangled_into_are(self):
        """Regression: `area` is a known field/content word and must never be
        rewritten by the command-scaffold autocorrect layer."""
        corrected = _autocorrect_words("what is the area of the sahara desert".split())
        assert corrected == "what is the area of the sahara desert".split()

    def test_scaffold_typo_dose_corrects_at_looser_retry_tier(self):
        """`dose` is a real English word, so the strict pass intentionally
        leaves it alone. The progressive retry tier should catch it at the
        looser 0.75 cutoff so the end-to-end query still succeeds."""
        strict = _autocorrect_words("what currency dose japan use".split(), cutoff=0.82)
        assert strict == "what currency dose japan use".split()

        looser = _autocorrect_words("what currency dose japan use".split(), cutoff=0.75)
        assert looser == "what currency does japan use".split()


# --- End-to-end through search_pa_list -------------------------------------


class TestSearchPaListThroughAutocorrect:
    @staticmethod
    def _install_fakes(monkeypatch, action_fakes):
        """Helper: replace named actions in pa_list with fakes that count
        invocations. Returns a `called` dict mapping name -> call count."""
        import a10

        called = {name: 0 for name in action_fakes}

        # Wrap each fake to bump the counter
        def make_wrapper(name, fake):
            def wrapped(matches):
                called[name] += 1
                return fake(matches)
            wrapped.__name__ = name
            return wrapped

        wrapped_by_name = {
            name: make_wrapper(name, fake) for name, fake in action_fakes.items()
        }

        new_pa_list = []
        for pat, act in a10.pa_list:
            name = act.__name__
            new_pa_list.append((pat, wrapped_by_name.get(name, act)))
        monkeypatch.setattr(a10, "pa_list", new_pa_list)
        # Vocab is cached from the OLD pa_list - reset so it picks up nothing
        # has changed structurally (this is paranoia; vocab content is the same)
        monkeypatch.setattr(a10, "_COMMAND_VOCAB", None)
        return called

    def test_typo_routes_to_specific_handler_not_catchall(self, monkeypatch):
        """The headline integration test: a typo'd `what is the captial of X`
        must end up in `capital_of`, NOT in the `what is %` catch-all
        (which would call `about_topic`)."""
        import a10

        called = self._install_fakes(
            monkeypatch,
            {
                "capital_of":  lambda m: ["Paris"],
                "about_topic": lambda m: None,
            },
        )
        result = a10.search_pa_list("what is the captial of france".split())
        assert called["capital_of"] == 1
        assert called["about_topic"] == 0
        assert result == ["Paris"]

    def test_clean_query_runs_action_exactly_once(self, monkeypatch):
        """No-typo queries should NOT trigger any retries."""
        import a10

        called = self._install_fakes(
            monkeypatch,
            {"capital_of": lambda m: ["Paris"]},
        )
        result = a10.search_pa_list("what is the capital of france".split())
        assert called["capital_of"] == 1, "Clean query should call action exactly once"
        assert result == ["Paris"]

    def test_progressive_retry_catches_looser_typo(self, monkeypatch):
        """A typo too aggressive for cutoff 0.82 (e.g. 'borm') should be
        caught by the second-tier 0.75 retry and routed to the right handler."""
        import a10

        called = self._install_fakes(
            monkeypatch,
            {
                "birth_date":  lambda m: ["1906-12-09"],
                "about_topic": lambda m: None,
            },
        )
        # "when was grace hopper borm" - "borm" only fixes at cutoff <= 0.75
        result = a10.search_pa_list("when was grace hopper borm".split())
        assert called["birth_date"] == 1, (
            f"birth_date should have fired after looser-cutoff retry (calls: {called})"
        )
        assert result == ["1906-12-09"]

    def test_retry_skipped_when_correction_doesnt_change(self, monkeypatch):
        """If a query has no typos, the corrected form is identical at every
        cutoff. Action should run AT MOST once across all retry tiers."""
        import a10

        called = self._install_fakes(
            monkeypatch,
            # Always fail, so retry logic would normally kick in
            {"capital_of": lambda m: ["No answers"]},
        )
        a10.search_pa_list("what is the capital of france".split())
        assert called["capital_of"] == 1, (
            f"Action ran {called['capital_of']} times for a no-typo query; "
            "duplicate-correction dedupe is broken"
        )

    def test_retry_bottoms_out_returning_last_failure(self, monkeypatch):
        """If every cutoff produces a soft failure, return the last attempt's
        failure result (so the user sees what happened, not silence)."""
        import a10

        # This test is specifically about the retry behavior of the exact
        # capital handler. The generic NLP engine now knows how to answer this
        # question too, so stub it out here to keep the test focused.
        monkeypatch.setattr(a10, "generic_field_query", lambda words: None)

        called = self._install_fakes(
            monkeypatch,
            {"capital_of": lambda m: ["No answers"]},
        )
        result = a10.search_pa_list("what is the captial of france".split())
        assert result == ["No answers"]
        # Action could fire once or twice depending on whether the cutoff-0.82
        # correction matches cutoff-0.75 correction (in this case they do, so
        # only one action call should happen due to dedupe). Either way, no
        # more than the number of unique corrections.
        assert called["capital_of"] >= 1
        assert called["capital_of"] <= len(a10._AUTOCORRECT_CUTOFFS)

    def test_no_match_returns_i_dont_understand(self, monkeypatch):
        """Total gibberish should still return the friendly fallback after
        all autocorrect tiers fail to find a pattern match."""
        import a10
        result = a10.search_pa_list("xyzzyx blorp grunch".split())
        assert result == ["I don't understand"]
