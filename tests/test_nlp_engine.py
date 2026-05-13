"""Unit tests for the hybrid NLP field-query engine."""

import nlp_engine
from nlp_engine import FieldMatch


ROWS = [
    ("Revenue", "US$281.7 billion"),
    ("Number of employees", "228,000"),
    ("Headquarters", "Redmond, Washington"),
    ("Products", "Windows\nMicrosoft 365\nXbox"),
    ("Operating income", "US$128.5 billion"),
    ("Currency", "Japanese yen"),
    ("Elevation", "8,848.86 m (29,031.7 ft)"),
]


def test_normalize_query_text_expands_common_contractions():
    assert nlp_engine.normalize_query_text("What's Microsoft's revenue?") == (
        "what is microsoft's revenue"
    )


def test_extract_intent_possessive_form():
    intent = nlp_engine.extract_field_query_intent("what is Microsoft's revenue")
    assert intent is not None
    assert intent.topic_phrase == "microsoft"
    assert intent.field_phrase == "revenue"


def test_extract_intent_of_form():
    intent = nlp_engine.extract_field_query_intent("what is the revenue of microsoft")
    assert intent is not None
    assert intent.topic_phrase == "microsoft"
    assert intent.field_phrase == "revenue"


def test_extract_intent_how_many_form():
    intent = nlp_engine.extract_field_query_intent("how many employees does microsoft have")
    assert intent is not None
    assert intent.topic_phrase == "microsoft"
    assert intent.field_phrase == "employees"


def test_extract_intent_what_field_does_topic_use():
    intent = nlp_engine.extract_field_query_intent("what currency does japan use")
    assert intent is not None
    assert intent.topic_phrase == "japan"
    assert intent.field_phrase == "currency"


def test_extract_intent_how_tall_is_topic():
    intent = nlp_engine.extract_field_query_intent("how tall is mount everest")
    assert intent is not None
    assert intent.topic_phrase == "mount everest"
    assert intent.field_phrase == "tall"


def test_extract_intent_money_phrase():
    intent = nlp_engine.extract_field_query_intent("how much money does microsoft make")
    assert intent is not None
    assert intent.topic_phrase == "microsoft"
    assert intent.field_phrase == "money"


def test_extract_intent_where_based_maps_to_headquarters():
    intent = nlp_engine.extract_field_query_intent("where is microsoft based")
    assert intent is not None
    assert intent.topic_phrase == "microsoft"
    assert intent.field_phrase == "headquarters"


def test_exact_alias_match_for_revenue():
    match = nlp_engine.match_infobox_field("revenue", ROWS)
    assert match is not None
    assert match.label == "Revenue"
    assert match.strategy == "alias"


def test_alias_match_for_employee_count():
    match = nlp_engine.match_infobox_field("employee count", ROWS)
    assert match is not None
    assert match.label == "Number of employees"


def test_alias_match_for_currency():
    match = nlp_engine.match_infobox_field("currency", ROWS)
    assert match is not None
    assert match.label == "Currency"


def test_alias_match_for_tall_maps_to_elevation():
    match = nlp_engine.match_infobox_field("tall", ROWS)
    assert match is not None
    assert match.label == "Elevation"


def test_rapidfuzz_match_for_misspelled_headquarters():
    match = nlp_engine.match_infobox_field("headquaters", ROWS)
    assert match is not None
    assert match.label == "Headquarters"
    assert match.strategy in {"alias", "rapidfuzz"}


def test_semantic_fallback_is_used_when_exact_and_fuzzy_miss(monkeypatch):
    def fake_semantic(field_phrase, candidates):
        candidate = next(c for c in candidates if c.label == "Revenue")
        return FieldMatch(
            label=candidate.label,
            value=candidate.value,
            strategy="semantic",
            score=0.91,
            candidate_phrase="money brought in",
        )

    monkeypatch.setattr(nlp_engine, "_best_fuzzy_candidate", lambda field_phrase, candidates: None)
    monkeypatch.setattr(nlp_engine, "_semantic_best_candidate", fake_semantic)

    match = nlp_engine.match_infobox_field("cash intake", ROWS)
    assert match is not None
    assert match.label == "Revenue"
    assert match.strategy == "semantic"


def test_no_intent_for_about_query():
    assert nlp_engine.extract_field_query_intent("tell me about microsoft") is None
