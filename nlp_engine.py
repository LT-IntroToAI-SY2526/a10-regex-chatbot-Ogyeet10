"""Hybrid natural-language field query engine for Wikipedia infoboxes.

This module implements a three-tier approach:

Tier 1:
  - rule-based query normalization and intent extraction
  - alias-based field matching

Tier 2:
  - spaCy tokenization / light linguistic normalization
  - RapidFuzz fuzzy label matching

Tier 3:
  - sentence-transformers semantic fallback for label selection

The engine is intentionally modular: action/HTTP code lives in `a10.py`, while
this module focuses on turning English into a `(topic, field)` intent and then
matching the requested field phrase to a row in an already-parsed infobox.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FieldQueryIntent:
    raw_query: str
    normalized_query: str
    topic_phrase: str
    field_phrase: str


@dataclass(frozen=True)
class FieldMatch:
    label: str
    value: str
    strategy: str
    score: float
    candidate_phrase: str


@dataclass(frozen=True)
class _LabelCandidate:
    label: str
    value: str
    phrases: Tuple[str, ...]


FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "revenue": (
        "revenue",
        "sales",
        "turnover",
        "money made",
        "money make",
        "money earned",
        "money earn",
        "bring in",
        "brought in",
        "income from sales",
    ),
    "operating income": (
        "operating income",
        "operating profit",
        "profit from operations",
    ),
    "net income": (
        "net income",
        "profit",
        "net profit",
        "bottom line",
        "earnings",
    ),
    "number of employees": (
        "number of employees",
        "employees",
        "employee count",
        "staff",
        "staff count",
        "workers",
        "people who work there",
        "people work at",
        "workforce",
    ),
    "headquarters": (
        "headquarters",
        "head office",
        "headquarter",
        "based",
        "located",
        "headquartered",
        "main office",
    ),
    "founders": (
        "founder",
        "founders",
        "founded by",
        "creator",
        "creators",
        "started by",
        "who started it",
    ),
    "founded": (
        "founded",
        "established",
        "started",
        "created",
        "formed",
        "inception",
        "formation",
        "founding date",
        "founding year",
    ),
    "products": (
        "products",
        "product",
        "makes",
        "makes products",
        "sells",
        "what it makes",
        "what does it make",
    ),
    "services": (
        "services",
        "service",
        "what services it offers",
        "offers",
    ),
    "brands": (
        "brands",
        "brand",
        "brand names",
    ),
    "industry": (
        "industry",
        "sector",
        "business sector",
        "what industry",
    ),
    "website": (
        "website",
        "web site",
        "site",
        "official website",
        "url",
        "homepage",
    ),
    "motto": (
        "motto",
        "slogan",
        "motto in latin",
    ),
    "alma mater": (
        "alma mater",
        "school",
        "schools",
        "education",
        "educated at",
        "attended",
        "studied at",
    ),
    "isbn": (
        "isbn",
        "book isbn",
        "isbn number",
    ),
    "currency": (
        "currency",
        "money",
        "money used",
        "what money",
        "what currency",
        "uses currency",
    ),
    "native name": (
        "native name",
        "native names",
        "local name",
        "name in native language",
        "endonym",
    ),
    "net worth": (
        "net worth",
        "worth",
        "wealth",
        "fortune",
    ),
    "official languages": (
        "language",
        "languages",
        "official language",
        "official languages",
        "spoken language",
        "spoken languages",
        "what language",
        "what languages",
        "language speak",
    ),
    "government": (
        "government",
        "type of government",
        "system of government",
        "government type",
    ),
    "religion": (
        "religion",
        "religions",
        "faith",
        "religion practiced",
        "practiced religion",
    ),
    "drives on the": (
        "drives on",
        "drive on",
        "side of the road",
        "road side",
        "driving side",
        "what side of the road",
    ),
    "colors": (
        "colors",
        "colours",
        "flag colors",
        "flag colours",
        "colors on the flag",
        "national colors",
        "national colours",
    ),
    "dissolved": (
        "dissolved",
        "dissolution",
        "ended",
        "end",
        "fell",
        "fall",
        "collapsed",
        "collapse",
        "historical era",
    ),
    "elevation": (
        "elevation",
        "height",
        "tall",
        "high",
        "altitude",
        "summit height",
        "how tall",
        "how high",
    ),
    "density": (
        "density",
        "mass density",
    ),
    "boiling point": (
        "boiling point",
        "boils at",
        "boil point",
    ),
    "melting point": (
        "melting point",
        "melts at",
        "melt point",
    ),
    "atomic number": (
        "atomic number",
        "atomic number z",
    ),
    "gravity": (
        "gravity",
        "surface gravity",
    ),
    "orbital period": (
        "orbital period",
        "orbit period",
        "sidereal period",
    ),
    "length": (
        "length",
        "total length",
        "how long",
    ),
    "diameter": (
        "diameter",
        "mean diameter",
        "equatorial diameter",
    ),
    "known satellites": (
        "moon",
        "moons",
        "satellite",
        "satellites",
        "known satellites",
        "natural satellites",
    ),
    "genres": (
        "genre",
        "genres",
        "music genre",
        "music genres",
    ),
    "capital": (
        "capital",
        "capital city",
        "capital and largest city",
    ),
    "population": (
        "population",
        "population count",
        "number of people",
        "people live in",
        "inhabitants",
    ),
    "area": (
        "area",
        "size",
        "land area",
        "total area",
    ),
    "spouse": (
        "spouse",
        "wife",
        "husband",
        "married to",
        "partner",
    ),
    "occupation": (
        "occupation",
        "job",
        "profession",
        "career",
    ),
    "nationality": (
        "nationality",
        "citizenship",
        "country of citizenship",
    ),
    "born": (
        "born",
        "birth date",
        "birthday",
        "date of birth",
    ),
    "died": (
        "died",
        "death date",
        "date of death",
        "when did they die",
    ),
    "polar radius": (
        "polar radius",
        "radius",
        "mean radius",
        "equatorial radius",
    ),
}


_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "to",
    "at",
    "for",
    "in",
    "on",
    "by",
    "with",
    "it",
    "their",
    "they",
    "his",
    "her",
    "its",
    "type",
}

_QUERY_NORMALIZATIONS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhat's\b", re.IGNORECASE), "what is"),
    (re.compile(r"\bwho's\b", re.IGNORECASE), "who is"),
    (re.compile(r"\bwhere's\b", re.IGNORECASE), "where is"),
    (re.compile(r"\bhow many people work at\b", re.IGNORECASE), "what is the number of employees of"),
)

_INTENT_PATTERNS: Tuple[Tuple[re.Pattern[str], Tuple[str, str]], ...] = (
    (re.compile(r"^(?:what is|what are|tell me|show me)\s+(?P<topic>.+?)'s\s+(?P<field>.+)$"), ("topic", "field")),
    (re.compile(r"^(?:what is|what are)\s+the\s+(?P<field>.+?)\s+of\s+(?P<topic>.+)$"), ("topic", "field")),
    (re.compile(r"^(?:what is|what are)\s+(?P<field>.+?)\s+of\s+(?P<topic>.+)$"), ("topic", "field")),
    (re.compile(r"^what\s+(?P<field>.+?)\s+does\s+(?P<topic>.+?)\s+have$"), ("topic", "field")),
    (re.compile(r"^what\s+(?P<field>.+?)\s+does\s+(?P<topic>.+?)\s+make$"), ("topic", "field")),
    (re.compile(r"^what\s+(?P<field>.+?)\s+does\s+(?P<topic>.+?)\s+use$"), ("topic", "field")),
    (re.compile(r"^what\s+side\s+of\s+the\s+road\s+does\s+(?P<topic>.+?)\s+drive\s+on$"), ("topic", "drives on the")),
    (re.compile(r"^what\s+colors\s+are\s+on\s+the\s+flag\s+of\s+(?P<topic>.+)$"), ("topic", "colors")),
    (re.compile(r"^what\s+colours\s+are\s+on\s+the\s+flag\s+of\s+(?P<topic>.+)$"), ("topic", "colors")),
    (re.compile(r"^what\s+language\s+do\s+(?:they|people)\s+speak\s+in\s+(?P<topic>.+)$"), ("topic", "official languages")),
    (re.compile(r"^what\s+(?P<field>.+?)\s+is\s+practiced\s+in\s+(?P<topic>.+)$"), ("topic", "field")),
    (re.compile(r"^what\s+(?P<field>.+?)\s+did\s+(?P<topic>.+?)\s+attend$"), ("topic", "field")),
    (re.compile(r"^how\s+(?P<field>.+?)\s+is\s+(?P<topic>.+)$"), ("topic", "field")),
    (re.compile(r"^how many\s+(?P<field>.+?)\s+does\s+(?P<topic>.+?)\s+have$"), ("topic", "field")),
    (re.compile(r"^how much\s+(?P<field>.+?)\s+does\s+(?P<topic>.+?)\s+(?:have|make|earn|bring in)$"), ("topic", "field")),
    (re.compile(r"^what\s+industry\s+is\s+(?P<topic>.+?)\s+in$"), ("topic", "industry")),
    (re.compile(r"^where\s+is\s+(?P<topic>.+?)\s+(?:based|located)$"), ("topic", "headquarters")),
    (re.compile(r"^who\s+founded\s+(?P<topic>.+)$"), ("topic", "founders")),
    (re.compile(r"^when\s+did\s+(?P<topic>.+?)\s+(?:fall|end|collapse|dissolve)$"), ("topic", "dissolved")),
)

_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def _import_spacy_module():
    """Import spaCy lazily.

    Startup optimization: importing spaCy at module import time adds noticeable
    overhead to CLI launch. We only need it if/when the generic NLP path runs,
    so load it on demand.
    """
    try:  # pragma: no cover - import availability env-dependent
        return importlib.import_module("spacy")
    except Exception:
        return None


def _import_rapidfuzz():
    """Import RapidFuzz lazily for the same startup reason as spaCy."""
    try:  # pragma: no cover - import availability env-dependent
        fuzz = importlib.import_module("rapidfuzz.fuzz")
        process = importlib.import_module("rapidfuzz.process")
        return fuzz, process
    except Exception:
        return None, None


def _import_sentence_transformer_class():
    """Import SentenceTransformer lazily.

    This is the big one: importing sentence-transformers pulls in transformers,
    torch, etc., which dominated CLI startup. Keep it completely out of the
    import path until semantic fallback is actually needed.
    """
    try:  # pragma: no cover - import availability env-dependent
        module = importlib.import_module("sentence_transformers")
        return getattr(module, "SentenceTransformer", None)
    except Exception:
        return None


def normalize_query_text(query: str) -> str:
    """Normalize superficial phrasing differences before intent extraction."""
    text = query.strip().replace("’", "'")
    text = re.sub(r"\?+$", "", text)
    text = re.sub(r"\s+", " ", text)
    for pattern, replacement in _QUERY_NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=1)
def _get_nlp():
    """Return a spaCy pipeline.

    We prefer `en_core_web_sm` if available because it includes a real
    lemmatizer. If not installed, fall back to `spacy.blank("en")` so the
    tokenizer still works without model downloads.
    """
    spacy = _import_spacy_module()
    if spacy is None:  # pragma: no cover - dependency missing in LSP env
        return None
    try:  # pragma: no cover - env-dependent
        return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except Exception:  # pragma: no cover - env-dependent
        return spacy.blank("en")


def _simple_lemma(token_text: str) -> str:
    low = token_text.lower()
    if low.endswith("ies") and len(low) > 3:
        return low[:-3] + "y"
    if low.endswith("ers") and len(low) > 4:
        return low[:-1]
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        return low[:-1]
    return low


def canonicalize_field_phrase(text: str) -> str:
    """Lightly normalize a field phrase.

    Uses spaCy tokenization when available; otherwise falls back to regex word
    splitting. We aggressively strip filler words so phrasings like
    "the number of employees" and "employees" collapse to the same key.
    """
    nlp = _get_nlp()
    tokens: List[str] = []

    if nlp is not None:
        doc = nlp(text)
        for token in doc:
            raw = token.text.strip().lower()
            if not raw or token.is_punct or raw in _STOPWORDS:
                continue
            lemma = (token.lemma_ or "").strip().lower()
            if not lemma or lemma == "-pron-":
                lemma = _simple_lemma(raw)
            tokens.append(lemma)
    else:
        for raw in re.findall(r"[a-zA-Z0-9']+", text.lower()):
            if raw in _STOPWORDS:
                continue
            tokens.append(_simple_lemma(raw))

    phrase = " ".join(tokens).strip()

    # A few high-signal semantic rewrites so natural phrasings normalize to the
    # infobox concepts we actually store.
    if phrase in {"money", "money make", "money earn", "money bring", "money brought"}:
        return "revenue"
    if phrase in {"native name", "local name", "endonym"}:
        return "native name"
    if phrase in {"language", "languages", "official language", "official languages"}:
        return "official languages"
    if phrase in {"alma mater", "education", "educated", "attend", "school", "studied"}:
        return "alma mater"
    if phrase in {"government", "government type", "system government"}:
        return "government"
    if phrase in {"religion", "faith", "religions"}:
        return "religion"
    if phrase in {"colors", "colours", "flag colors", "flag colours"}:
        return "colors"
    if phrase in {"drive side", "road side", "side road", "driving side", "side road drive"}:
        return "drives on the"
    if phrase in {"fall", "fell", "end", "collapse", "dissolve", "ended", "collapsed", "dissolved"}:
        return "dissolved"
    if phrase in {"boiling point", "boil point"}:
        return "boiling point"
    if phrase in {"melting point", "melt point"}:
        return "melting point"
    if phrase in {"atomic number"}:
        return "atomic number"
    if phrase in {"gravity", "surface gravity"}:
        return "gravity"
    if phrase in {"orbital period", "orbit period", "sidereal period"}:
        return "orbital period"
    if phrase in {"length", "total length", "long"}:
        return "length"
    if phrase in {"diameter"}:
        return "diameter"
    if phrase in {"moon", "moons", "satellite", "satellites"}:
        return "known satellites"
    if phrase in {"genre", "genres"}:
        return "genres"
    if phrase in {"tall", "high", "height", "altitude"}:
        return "elevation"
    if phrase in {"people work", "staff", "worker", "workforce"}:
        return "number of employees"
    if phrase in {"based", "located", "headquartered"}:
        return "headquarters"
    if phrase in {"profit", "bottom line", "earning"}:
        return "net income"
    return phrase


def extract_field_query_intent(query: str) -> Optional[FieldQueryIntent]:
    """Try to interpret `query` as a generic infobox field request."""
    normalized = normalize_query_text(query)

    for pattern, groups in _INTENT_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        topic_group, field_group = groups
        topic = match.group(topic_group).strip()
        field = field_group if field_group not in match.groupdict() else match.group(field_group).strip()
        if topic and field:
            return FieldQueryIntent(
                raw_query=query,
                normalized_query=normalized,
                topic_phrase=topic,
                field_phrase=field,
            )

    # Possessive fallback via tokenization, e.g. `microsoft's annual revenue`.
    nlp = _get_nlp()
    if nlp is None:
        return None

    doc = nlp(normalized)
    texts = [token.text for token in doc]
    if texts[:2] not in (["what", "is"], ["what", "are"], ["tell", "me"]):
        return None

    for idx, token in enumerate(doc):
        if token.text in {"'s", "’s"} and 2 <= idx < len(doc) - 1:
            topic = " ".join(texts[2:idx]).strip()
            field = " ".join(texts[idx + 1 :]).strip()
            if topic and field:
                return FieldQueryIntent(
                    raw_query=query,
                    normalized_query=normalized,
                    topic_phrase=topic,
                    field_phrase=field,
                )
    return None


def _normalized_alias_map() -> Dict[str, Tuple[str, ...]]:
    return {
        canonicalize_field_phrase(key): tuple(
            sorted({canonicalize_field_phrase(alias) for alias in aliases if canonicalize_field_phrase(alias)})
        )
        for key, aliases in FIELD_ALIASES.items()
    }


def _infer_canonical_label(label: str) -> Optional[str]:
    label_norm = canonicalize_field_phrase(label)
    alias_map = _normalized_alias_map()
    for canonical, aliases in alias_map.items():
        if label_norm == canonical or label_norm in aliases:
            return canonical
    return None


def _build_label_candidates(rows: Sequence[Tuple[str, str]]) -> List[_LabelCandidate]:
    alias_map = _normalized_alias_map()
    candidates: List[_LabelCandidate] = []
    for label, value in rows:
        phrases = {canonicalize_field_phrase(label), normalize_query_text(label)}
        canonical = _infer_canonical_label(label)
        if canonical is not None:
            phrases.add(canonical)
            phrases.update(alias_map.get(canonical, ()))
        cleaned_phrases = tuple(sorted(p for p in phrases if p))
        candidates.append(_LabelCandidate(label=label, value=value, phrases=cleaned_phrases))
    return candidates


def _best_fuzzy_candidate(field_phrase: str, candidates: Sequence[_LabelCandidate]) -> Optional[FieldMatch]:
    fuzz, process = _import_rapidfuzz()
    if fuzz is None or process is None:  # pragma: no cover - dependency missing in LSP env
        return None

    choices: List[str] = []
    phrase_to_candidate: Dict[str, _LabelCandidate] = {}
    for candidate in candidates:
        for phrase in candidate.phrases:
            if phrase not in phrase_to_candidate:
                choices.append(phrase)
                phrase_to_candidate[phrase] = candidate

    if not choices:
        return None

    result = process.extractOne(field_phrase, choices, scorer=fuzz.WRatio)
    if not result:
        return None
    phrase, score, _ = result
    if score < 82:
        return None
    candidate = phrase_to_candidate[phrase]
    return FieldMatch(
        label=candidate.label,
        value=candidate.value,
        strategy="rapidfuzz",
        score=float(score),
        candidate_phrase=phrase,
    )


@lru_cache(maxsize=1)
def _get_sentence_model():
    SentenceTransformer = _import_sentence_transformer_class()
    if SentenceTransformer is None:  # pragma: no cover - dependency missing in LSP env
        return None
    try:  # pragma: no cover - model download/env dependent
        return SentenceTransformer(_SEMANTIC_MODEL_NAME)
    except Exception:
        return None


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_best_candidate(field_phrase: str, candidates: Sequence[_LabelCandidate]) -> Optional[FieldMatch]:
    model = _get_sentence_model()
    if model is None:  # pragma: no cover - model availability env-dependent
        return None

    choices: List[str] = []
    phrase_to_candidate: Dict[str, _LabelCandidate] = {}
    for candidate in candidates:
        for phrase in candidate.phrases:
            if phrase not in phrase_to_candidate:
                choices.append(phrase)
                phrase_to_candidate[phrase] = candidate
    if not choices:
        return None

    try:  # pragma: no cover - model execution env-dependent
        embeddings = model.encode([field_phrase] + choices, normalize_embeddings=True)
    except Exception:
        return None

    query_embedding = embeddings[0]
    best_phrase = ""
    best_score = -1.0
    for phrase, emb in zip(choices, embeddings[1:]):
        score = _cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_phrase = phrase

    if best_score < 0.56:
        return None
    candidate = phrase_to_candidate[best_phrase]
    return FieldMatch(
        label=candidate.label,
        value=candidate.value,
        strategy="semantic",
        score=best_score,
        candidate_phrase=best_phrase,
    )


def match_infobox_field(field_phrase: str, rows: Sequence[Tuple[str, str]]) -> Optional[FieldMatch]:
    """Match a user's requested field phrase to one infobox row.

    Match order:
      1. exact alias / canonical phrase match
      2. RapidFuzz fuzzy label match
      3. sentence-transformers semantic fallback
    """
    request = canonicalize_field_phrase(field_phrase)
    if not request:
        return None

    candidates = _build_label_candidates(rows)

    for candidate in candidates:
        if request in candidate.phrases:
            return FieldMatch(
                label=candidate.label,
                value=candidate.value,
                strategy="alias",
                score=100.0,
                candidate_phrase=request,
            )

    fuzzy_match = _best_fuzzy_candidate(request, candidates)
    if fuzzy_match is not None:
        return fuzzy_match

    return _semantic_best_candidate(request, candidates)


def query_scaffold_vocab_hints() -> List[str]:
    """Closed-class / scaffold words used by the generic NLP engine.

    This is intentionally *not* every field alias token. Field/content words
    should be left to the generic NLP matcher; command autocorrect should only
    fix the structural glue words around them (`what`, `does`, `use`, `how`,
    `many`, ...). That's what prevents bad snaps like `tall` -> `tell` while
    still allowing `dose` -> `does`.
    """
    return sorted(
        {
            "what",
            "who",
            "where",
            "when",
            "how",
            "many",
            "much",
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "of",
            "the",
            "a",
            "an",
            "have",
            "has",
            "make",
            "earn",
            "bring",
            "use",
            "based",
            "located",
            "founded",
            "born",
            "died",
            "married",
            "to",
            "in",
        }
    )


def field_content_vocab_hints() -> List[str]:
    """Content words that belong to the field-matching layer.

    Command autocorrect should not rewrite these tokens. If a word is already a
    known field/content token (`area`, `population`, `currency`, `tall`, ...),
    it should be passed through untouched so the generic NLP engine can do its
    job. This is what prevents regressions like `area` -> `are`.
    """
    tokens = set()
    for field_aliases in FIELD_ALIASES.values():
        for phrase in field_aliases:
            for token in re.findall(r"[a-zA-Z0-9']+", phrase.lower()):
                tokens.add(token)
    return sorted(tokens)
