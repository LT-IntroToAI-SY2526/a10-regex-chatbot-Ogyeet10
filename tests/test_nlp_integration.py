"""Integration tests for the generic NLP field-query stack.

These run through `search_pa_list` end-to-end using cached real Wikipedia
HTML fixtures, so they cover the full path:

  command autocorrect -> generic NLP intent extraction -> topic resolution
  -> infobox parsing -> label matching -> final formatted answer
"""

import pytest

import a10
import nlp_engine


pytestmark = [pytest.mark.network]


@pytest.fixture
def patched_real_wikipedia(monkeypatch, wiki_html):
    """Use cached real Wikipedia pages for the generic NLP tests.

    We patch `resolve_topic` to canonical titles so these tests stay focused on
    the NLP field-query stack rather than the separate topic-typo/disambiguation
    system, which already has its own dedicated tests.
    """
    monkeypatch.setattr(a10, "get_page_html", wiki_html)

    canonical_titles = {
        "microsoft": "Microsoft",
        "apple": "Apple Inc.",
        "apple inc": "Apple Inc.",
        "spacex": "SpaceX",
        "france": "France",
        "japan": "Japan",
        "mount everest": "Mount Everest",
    }

    monkeypatch.setattr(
        a10,
        "resolve_topic",
        lambda topic: canonical_titles.get(topic.lower(), topic),
    )


class TestGenericNlpQueries:
    def test_possessive_revenue_query(self, patched_real_wikipedia):
        result = a10.search_pa_list("what is microsoft's revenue".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Revenue:[/bold cyan]")
        assert "US$" in result[0]

    def test_money_phrase_maps_to_revenue(self, patched_real_wikipedia):
        result = a10.search_pa_list("how much money does microsoft make".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Revenue:[/bold cyan]")

    def test_employee_count_query(self, patched_real_wikipedia):
        result = a10.search_pa_list("how many employees does microsoft have".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Number of employees:[/bold cyan]")
        assert any(ch.isdigit() for ch in result[0])

    def test_based_query_maps_to_headquarters(self, patched_real_wikipedia):
        result = a10.search_pa_list("where is microsoft based".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Headquarters:[/bold cyan]")
        assert "Redmond" in result[0]

    def test_fuzzy_field_name_query(self, patched_real_wikipedia):
        result = a10.search_pa_list("what is microsoft's headquaters".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Headquarters:[/bold cyan]")

    def test_currency_query(self, patched_real_wikipedia):
        result = a10.search_pa_list("what currency does japan use".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Currency:[/bold cyan]")
        assert "yen" in result[0].lower()

    def test_currency_query_with_dose_typo(self, patched_real_wikipedia):
        result = a10.search_pa_list("what currency dose japan use".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Currency:[/bold cyan]")
        assert "yen" in result[0].lower()

    def test_how_tall_query(self, patched_real_wikipedia):
        result = a10.search_pa_list("how tall is mount everest".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Elevation:[/bold cyan]")
        assert any(unit in result[0].lower() for unit in ["m", "ft"])

    def test_multiline_field_answer_formats_cleanly(self, patched_real_wikipedia):
        result = a10.search_pa_list("what products does microsoft make".split())
        assert result is not None
        assert result[0].startswith("[bold cyan]Products:[/bold cyan]")
        assert "\n- " in result[0]


class TestSemanticFallbackIntegration:
    def test_search_pa_list_can_flow_through_semantic_fallback(self, monkeypatch, patched_real_wikipedia):
        """Integration-level proof that the semantic fallback path can drive
        a user-visible answer through `search_pa_list`.

        We monkeypatch the semantic scorer to avoid a hard dependency on model
        downloads / runtime availability in CI while still testing the actual
        integration path through `a10.search_pa_list`.
        """

        def fake_semantic(field_phrase, candidates):
            candidate = next(c for c in candidates if c.label == "Revenue")
            return nlp_engine.FieldMatch(
                label=candidate.label,
                value=candidate.value,
                strategy="semantic",
                score=0.92,
                candidate_phrase="cash intake",
            )

        monkeypatch.setattr(nlp_engine, "_best_fuzzy_candidate", lambda field_phrase, candidates: None)
        monkeypatch.setattr(nlp_engine, "_semantic_best_candidate", fake_semantic)

        result = a10.search_pa_list("what is microsoft's cash intake".split())

        assert result is not None
        assert result[0].startswith("[bold cyan]Revenue:[/bold cyan]")


class TestSpecificHandlersStillWin:
    def test_birth_date_special_handler_beats_generic_engine(self, monkeypatch, patched_real_wikipedia):
        """The generic field engine should not steal queries already handled
        precisely by the hand-written special-case actions."""
        calls = {"generic": 0}

        real_generic = a10.generic_field_query

        def wrapped_generic(words):
            calls["generic"] += 1
            return real_generic(words)

        monkeypatch.setattr(a10, "generic_field_query", wrapped_generic)

        result = a10.search_pa_list("when was grace hopper born".split())

        assert result == ["[bold cyan]Birth date:[/bold cyan] 1906-12-09"]
        assert calls["generic"] == 0
