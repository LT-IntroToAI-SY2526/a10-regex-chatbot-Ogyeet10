"""
WikiBot - a Wikipedia-powered natural language chatbot.

Backend: scrapes Wikipedia infoboxes via the MediaWiki API + BeautifulSoup, then
extracts structured facts using *real* regular expressions (re module).

Frontend: pattern-action dispatcher (`pa_list`) using a tiny `%`/`_` wildcard
matcher, wrapped in a rich/prompt_toolkit TUI ported from the NotSteam project
(arrow-key history, autosuggest, word completer, branded bottom toolbar,
interactive disambiguation, help panel, etc.).

i need to be stopped. I (had claude) write intagration tests for this 😭

LLMs these days are so insanely powerful that with just a couple hours of time at home we intagrated an entire NLP pipeline into this project WITHOUT any LLM api.

in full tranparencey i wrote almost none of this code but tbh most modern day engineers dont ether 

Built for Assignment 10 (Wikipedia Chatbot) - a regex-focused exercise.
"""

import difflib
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import requests
from bs4 import BeautifulSoup, Tag

from match import match
from nlp_engine import (
    FieldMatch,
    extract_field_query_intent,
    field_content_vocab_hints,
    match_infobox_field,
    query_scaffold_vocab_hints,
)

# Rich for a polished terminal UI
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# prompt_toolkit for history, autosuggest, completer, bottom toolbar
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

try:
    from prompt_toolkit.application.current import get_app_or_none  # type: ignore
except Exception:  # pragma: no cover - older prompt_toolkit
    get_app_or_none = None  # type: ignore


console = Console()
VERSION = "1.0.0"

# Track the last topic the user asked about - enables "...and when did he die?"-style
# context follow-ups (mirrors NotSteam's _last_selected_game pattern).
_last_topic: Optional[str] = None


# ---------------------------------------------------------------------------
# Wikipedia backend
#
# The starter code calls `.text` on the entire <table class=infobox>,
# squashing the structured two-column markup into a single soup of characters.
# Then it tries to ride to the rescue with non-greedy regex like `\D{0,80}?` to
# bridge the gap between a label and its value - which breaks the moment the
# value contains a digit (e.g. Microsoft's "Founded April 4, 1975 ..." - the
# `4` in "April 4" trips `\D` and the match dies).
#
# We do it properly: parse the infobox as the two-column table it is, build
# a `{label: value}` dict, and run a small targeted regex on each individual
# value. Each extractor becomes ~3 lines and is robust.
# ---------------------------------------------------------------------------


# Wikipedia tags we never want bleeding into our text. References [1] etc. are
# fine to keep visible-but-we-strip-them, edit links / styles / nav navbars
# are pure noise.
_BOX_NOISE_SELECTORS = (
    "sup.reference",
    "sup.noprint",
    ".reference",
    ".mw-editsection",
    ".navbar",
    ".plainlinks",
    "style",
    "script",
)

# Reference-marker pattern for any survivors that slipped past the CSS selectors
# (e.g. inline "[1]" / "[a]" / "[note 3]").
_REF_MARKER_RE = re.compile(r"\[\s*(?:\d+|[a-z]|note\s*\d+|citation needed)\s*\]", re.IGNORECASE)


def get_page_html(title: str) -> str:
    """Fetch the rendered HTML of a Wikipedia page via the MediaWiki API.

    Retries on HTTP 429 (rate limit) up to 5 times, honoring Retry-After.
    Raises LookupError if the API returns an explicit error (no such page,
    etc.) and ConnectionError on persistent network failure.
    """
    for _attempt in range(5):
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "redirects": True,
            },
            headers={"User-Agent": "wikibot-class-project/1.0"},
        )
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 5))
            console.print(
                f"[yellow]Rate limited - waiting {wait}s before retrying '{title}'...[/yellow]"
            )
            time.sleep(wait)
            continue
        if response.status_code == 200 and response.text.strip():
            data = response.json()
            if "error" not in data:
                time.sleep(0.5)  # be polite, but not glacial
                return data["parse"]["text"]["*"]
            raise LookupError(
                data["error"].get("info") or f"Wikipedia error for '{title}'"
            )
    raise ConnectionError(
        f"Could not retrieve Wikipedia page for '{title}' after 5 attempts"
    )


def _search_page_titles(query: str, limit: int = 8) -> List[str]:
    """Ask Wikipedia's typo-tolerant search for likely page titles.

    This is the API-side complement to our command-word autocorrect. If the
    user gets the *topic* wrong (e.g. `micorsoft`), `parse&page=...` fails
    hard because there is no page by that exact title. Wikipedia's search API
    already knows how to recover from many misspellings, so we query it here
    and use the returned titles as fallback candidates.

    Returns an ordered list of candidate titles. On network/API failure we
    return an empty list rather than surfacing a second error to the user.
    """
    try:
        for _attempt in range(5):
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "namespace": 0,
                    "format": "json",
                    "redirects": "resolve",
                },
                headers={"User-Agent": "wikibot-class-project/1.0"},
                timeout=15,
            )
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 3))
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            titles = data[1] if isinstance(data, list) and len(data) > 1 else []
            return [str(title).strip() for title in titles if str(title).strip()]
    except Exception:
        return []
    return []


def _normalize_topic_for_similarity(title: str) -> str:
    """Normalize a page title/query for fuzzy title comparisons.

    We strip parenthetical qualifiers (`Amazon (company)` -> `amazon`) so the
    similarity check rewards the semantic title rather than punctuation noise.
    """
    lowered = title.lower()
    lowered = re.sub(r"\s*\([^)]*\)", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _topic_similarity(query: str, candidate: str) -> float:
    """Similarity score used to choose the best Wikipedia search fallback."""
    return difflib.SequenceMatcher(
        None,
        _normalize_topic_for_similarity(query),
        _normalize_topic_for_similarity(candidate),
    ).ratio()


def _normalize_label(s: str) -> str:
    """Normalize an infobox label for lookup: lowercase, strip parenthesized
    qualifiers like '(s)', collapse whitespace, drop trailing colon."""
    s = re.sub(r"\([^)]*\)", "", s)        # strip "(s)", "(formal)", etc.
    s = re.sub(r"[\u00a0\s]+", " ", s)     # nbsp + whitespace -> single space
    return s.strip().rstrip(":").lower()


def _clean_cell_text(s: str) -> str:
    """Clean a single cell's text: drop ref markers, normalize nbsp / spaces,
    collapse runs of newlines, trim each line."""
    s = _REF_MARKER_RE.sub("", s)
    s = s.replace("\u00a0", " ")
    # Collapse runs of inline whitespace per line, then trim each line and
    # squash multiple blank lines.
    cleaned_lines = []
    for line in s.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# Bullet markers Wikipedia uses to indent sub-rows under a section header
_BULLET_PREFIX_RE = re.compile(r"^[\u2022\u00b7\u25aa\u25e6\u25fe]+\s*")


def _infobox_dom(html: str) -> Tag:
    """Return the first <table class=infobox> with noise removed and block
    elements pre-marked with explicit newlines. Shared prep for both the
    lookup-dict and ordered-row views of an infobox."""
    soup = BeautifulSoup(html, "html.parser")
    box: Optional[Tag] = soup.find(class_="infobox")  # type: ignore[assignment]
    if box is None:
        raise LookupError("Page has no infobox")

    # Drop nodes that contribute only noise (reference superscripts, edit
    # links, nav navbars, style/script blocks, etc.).
    for sel in _BOX_NOISE_SELECTORS:
        for tag in box.select(sel):
            tag.decompose()

    # Force structural HTML to explicit newlines. Inline elements (<span>,
    # <a>, <sup>, ...) stay as-is so adjacent text nodes concatenate cleanly
    # without fragmenting numbers like "6,356.752".
    for br in box.find_all("br"):
        br.replace_with("\n")
    for tag in box.find_all(["li", "div", "p", "dd"]):
        tag.insert_before("\n")

    return box


def _iter_rows(box: Tag):
    """Yield (raw_label_text, raw_value_text) for infobox rows.

    Primary shape: one direct `<th>` plus one or more direct `<td>` cells.
    Secondary shape: some infobox variants (notably chembox-style scientific
    pages like Water) use `<td>`/`<td>` pairs with the label in the first td and
    the value in the second. Supporting both shapes keeps the parser general
    without special-casing individual pages.

    NOTE: strip=False is critical. With strip=True, BS4 strips each text
    node individually, which would wipe the `\\n` text nodes we injected
    before <br>/<li>/<div>.
    """
    for row in box.find_all("tr"):
        ths = row.find_all("th", recursive=False)
        tds = row.find_all("td", recursive=False)

        # Section/header-only row (used heavily by country/city infoboxes for
        # things like Population / Government / Establishment). Preserve it so
        # the higher-level parser can promote following bullet sub-rows.
        if len(ths) == 1 and not tds:
            yield ths[0].get_text(""), ""
            continue

        if len(ths) == 1 and len(tds) >= 1:
            yield ths[0].get_text(""), tds[0].get_text("")
            continue

        # Chembox-style fallback: two sibling td cells, first acts as label.
        if not ths and len(tds) >= 2:
            yield tds[0].get_text(""), tds[1].get_text("")
            continue

        yield "", ""


def parse_infobox_rows(html: str) -> List[Tuple[str, str]]:
    """All primary infobox rows as `(display_label, value)` pairs in
    document order.

    `display_label` preserves Wikipedia's original casing (e.g. "Founded",
    "Operating income") and has the bullet marker stripped, so it's
    suitable for showing to a human. Use this for full-infobox renders
    like `about_topic`.

    Header-only rows (section dividers) are skipped - they're useful for
    lookup convenience but not for display.
    """
    box = _infobox_dom(html)
    rows: List[Tuple[str, str]] = []
    for th_text, td_text in _iter_rows(box):
        if not th_text or not td_text:
            continue
        # Strip bullet, collapse whitespace, drop trailing colon - but keep
        # original casing for display.
        label = _BULLET_PREFIX_RE.sub("", th_text).strip()
        label = re.sub(r"[\u00a0\s]+", " ", label).rstrip(":").strip()
        value = _clean_cell_text(td_text)
        if not label or not value:
            continue
        rows.append((label, value))
    return rows


def parse_infobox(html: str) -> Dict[str, str]:
    """Parse a Wikipedia page's first infobox into a `{label: value}` lookup
    dict.

    Labels are normalized: lowercase, parenthesized qualifiers stripped,
    trailing colon dropped. First occurrence wins on duplicates.

    Two non-obvious things this handles that the naive `.text` approach
    cannot:

    1. **Inline tag fragmentation.** Wikipedia wraps grouped numbers in
       `<span class="nowrap">6,</span><span class="nowrap">356.752</span>`
       so they don't word-wrap. Naive `get_text("\\n")` would split them.
       We use an empty separator + explicit newlines on block elements.

    2. **Section sub-rows.** Country/city infoboxes use a header-only row
       like `<tr><th>Population</th></tr>` followed by bulleted sub-rows
       like `<tr><th>\u2022 2020 census</th><td>126,146,099</td></tr>`.
       To make these findable by `_lookup(box, "Population")` we:
         (a) Set the section name to the *first* sub-row's value
             (e.g. `box["population"] = "1,210,854,977"`)
         (b) ALSO store each sub-row under a composite key like
             `"population 2011 census"`, so `get_population` can iterate
             all population-related keys to find the most useful number.

    For *ordered* iteration over primary rows (skipping the synthetic
    section keys) use `parse_infobox_rows()`.
    """
    box = _infobox_dom(html)
    result: Dict[str, str] = {}
    current_section: Optional[str] = None

    for th_text, td_text in _iter_rows(box):
        # Header-only row: treat as a section divider and remember it. The
        # next bullet-prefixed sub-rows will be hoisted to this name.
        if th_text and not td_text:
            current_section = _normalize_label(th_text)
            continue

        if not th_text or not td_text:
            continue

        raw_label = _normalize_label(th_text)
        bare_label = _BULLET_PREFIX_RE.sub("", raw_label).strip()
        value = _clean_cell_text(td_text)

        if not bare_label or not value:
            continue

        was_subrow = raw_label != bare_label  # had a bullet prefix

        if bare_label not in result:
            result[bare_label] = value

        # Section linkage: hoist bullet sub-rows so they're findable via
        # the parent section name.
        if was_subrow and current_section and current_section != bare_label:
            # (a) Composite key: "<section> <subrow>" -> value. Lets
            # downstream code iterate all section-related entries by
            # filtering on a common prefix.
            composite = f"{current_section} {bare_label}".strip()
            if composite not in result:
                result[composite] = value
            # (b) Bare section name maps to the *first* sub-row value, so
            # `_lookup(box, "Population")` returns something useful for
            # Japan and India.
            if current_section not in result:
                result[current_section] = value

        # Non-bullet rows can also act as section headers for what follows
        # (some infoboxes use a plain th row to mark a section that itself
        # has a primary value).
        if not was_subrow:
            current_section = bare_label

    return result


def _lookup(box: Dict[str, str], *aliases: str) -> Optional[str]:
    """Return the first non-empty value for any matching alias label.

    Aliases are matched against the same normalization used during parsing,
    so `_lookup(box, "Founder", "Founders", "Founded by")` quietly handles
    pluralization and phrasing variants.
    """
    for alias in aliases:
        v = box.get(_normalize_label(alias))
        if v:
            return v
    return None


def _first_line(value: str) -> str:
    """First non-empty line of a multi-line cell value."""
    for line in value.split("\n"):
        line = line.strip()
        if line:
            return line
    return value.strip()


# --- Per-fact extractors -----------------------------------------------------
#
# Each one: parse infobox -> look up by label alias(es) -> apply one tiny
# regex (or take first line) to pull the answer out of the value cell.
# Compare with the old approach's 80-char non-greedy gymnastics on flat text.


# A radius cell starts directly with the number, e.g. "6,356.752 km" or
# "3376.2±0.1 km (0.531 Earths)". We anchor with re.match so we don't get
# fooled into picking up the `0.1` uncertainty after the `\u00b1` sign.
_LEADING_NUMBER_RE = re.compile(r"\s*(\d[\d,]*(?:\.\d+)?)")


def get_polar_radius(planet_name: str) -> str:
    """Polar / mean radius in km.

    Planet infoboxes vary: some have a dedicated `<th>Polar radius</th>` row,
    others bundle all the radii into a single `Dimensions` cell. We try the
    direct dict lookup first, then fall back to searching for an inline
    "Polar radius: NUM km" pattern in any cell.
    """
    box = parse_infobox(get_page_html(planet_name))
    value = _lookup(box, "Polar radius", "Mean radius", "Equatorial radius", "Radius")
    if value:
        m = _LEADING_NUMBER_RE.match(value)
        if m:
            return m.group(1)
    # Fallback: scan every cell for an inline "polar radius ... km" phrase
    for v in box.values():
        m = re.search(
            r"(?:polar|mean)\s*radius[^0-9]{0,40}(\d[\d,]*(?:\.\d+)?)\s*km",
            v,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
    raise AttributeError("Page infobox has no parseable radius information")


def get_birth_date(name: str) -> str:
    """Birth date in YYYY-MM-DD form (Wikipedia includes a hidden ISO date
    inside Born cells via <span class=bday> which our text extraction picks up)."""
    box = parse_infobox(get_page_html(name))
    value = _lookup(box, "Born")
    if not value:
        raise AttributeError("Page infobox has no birth information")
    m = re.search(r"\b(?P<birth>\d{4}-\d{2}-\d{2})\b", value)
    if not m:
        raise AttributeError(
            "Page infobox has no birth date in xxxx-xx-xx format"
        )
    return m.group("birth")


def get_death_date(name: str) -> str:
    """Death date in YYYY-MM-DD form."""
    box = parse_infobox(get_page_html(name))
    value = _lookup(box, "Died")
    if not value:
        raise AttributeError("Page infobox has no death info (or person isn't deceased)")
    m = re.search(r"\b(?P<death>\d{4}-\d{2}-\d{2})\b", value)
    if not m:
        raise AttributeError("Page infobox has no death date in xxxx-xx-xx format")
    return m.group("death")


def get_capital(country_name: str) -> str:
    """Capital city of a country / region."""
    box = parse_infobox(get_page_html(country_name))
    value = _lookup(
        box,
        "Capital",
        "Capital and largest city",
        "Capital city",
        "Capital and largest metropolitan area",
    )
    if not value:
        raise AttributeError("Page infobox has no capital listed")
    # First line is the capital; strip any coordinate annotations.
    capital = _first_line(value)
    capital = re.sub(r"\s*\(.*$", "", capital).strip()  # drop "(de facto)" etc.
    return capital


_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b")

# Magnitude words used in informal population figures (e.g. India's infobox
# value "1.48 billion"). Includes Indian-system words (crore = 10M,
# lakh = 100k) since those show up on en.wikipedia.org for South Asian pages.
_MAGNITUDE: Dict[str, int] = {
    "thousand": 1_000,
    "lakh":     100_000,
    "lakhs":    100_000,
    "million":  1_000_000,
    "crore":    10_000_000,
    "crores":   10_000_000,
    "billion":  1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_MAGNITUDE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion|lakhs?|crores?)\b",
    re.IGNORECASE,
)


def _parse_magnitude_number(text: str) -> Optional[str]:
    """Parse a magnitude expression like "1.48 billion" into a comma-grouped
    integer string like "1,480,000,000". Returns None if no match."""
    m = _MAGNITUDE_RE.search(text)
    if not m:
        return None
    n = float(m.group(1)) * _MAGNITUDE[m.group(2).lower()]
    return f"{int(n):,}"


def get_population(place_name: str) -> str:
    """Most-prominent population figure (city / country / region).

    Wikipedia is wildly inconsistent here:
      - Country pages sometimes use a top-level `Population` row.
      - Other countries (Japan, India) bury population in bullet sub-rows
        under a `Population` section header. The parser hoists these into
        composite keys like `"population 2011 census"` so we can find them.
      - Some pages show only a colloquial figure like "1.48 billion".

    Strategy: collect every population-related value, scan for the largest
    comma-grouped number (skipping density rows that have "/km" markers),
    and fall back to parsing magnitude words like "billion" if no precise
    figure is available.
    """
    html = get_page_html(place_name)
    box = parse_infobox(html)
    rows = parse_infobox_rows(html)

    # Collect candidate values from every population-related label.
    candidates: List[str] = []
    direct = _lookup(box, "Population")
    if direct:
        candidates.append(direct)
    for label, value in box.items():
        if "population" in label and value not in candidates:
            candidates.append(value)

    # City pages often store the useful numbers in row labels like
    # `Urban (2011)` / `Metro (2025)` rather than under a plain population key.
    # The lookup dict can lose those because earlier duplicate labels such as
    # `Urban` area rows win. Scan the ordered primary rows directly too.
    for label, value in rows:
        label_low = label.lower()
        if any(
            key in label_low
            for key in ("population", "urban", "metro", "city proper")
        ) and value not in candidates:
            candidates.append(value)

    if not candidates:
        raise AttributeError("Page infobox has no population information")

    # Pass 1: find the largest comma-grouped or 4+digit number anywhere
    # across the candidates. Skip per-km² density lines.
    best_int = -1
    best_str = ""
    for value in candidates:
        for line in value.split("\n"):
            if re.search(r"/\s*km|per\s*km|density", line, re.IGNORECASE):
                continue
            for n in _NUMBER_RE.findall(line):
                n_int = int(n.replace(",", ""))
                if n_int > best_int:
                    best_int = n_int
                    best_str = n

    if best_str:
        return best_str

    # Pass 2: no exact number found. Try parsing a magnitude expression like
    # "1.48 billion" - common on country pages with informal estimates.
    for value in candidates:
        parsed = _parse_magnitude_number(value)
        if parsed:
            return parsed

    raise AttributeError("Population cell present but no parseable number")


def get_founded_year(entity_name: str) -> str:
    """Founding year for a company / university / organization."""
    box = parse_infobox(get_page_html(entity_name))
    value = _lookup(
        box,
        "Founded",
        "Established",
        "Formation",
        "Founded in",
        "Date founded",
        "Inception",
    )
    if not value:
        raise AttributeError("Page infobox has no founding date")
    # Prefer an ISO-format date if present (most accurate), else first 4-digit year
    m = re.search(r"\b(\d{4})-\d{2}-\d{2}\b", value)
    if m:
        return m.group(1)
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", value)
    if not m:
        raise AttributeError("Founded cell present but no 4-digit year found")
    return m.group(1)


def get_headquarters(company_name: str) -> str:
    """Headquarters location.

    Cells come in two shapes:
      "Redmond, Washington, U.S."                 (single-line, comma-separated)
      "Apple Park,\nCupertino, California,\nU.S." (multi-line, each line may
                                                   have trailing comma)
    We split into lines, strip lone-comma / pure-whitespace lines (those were
    the source of the double-comma artifact), then re-join with commas.
    """
    box = parse_infobox(get_page_html(company_name))
    value = _lookup(box, "Headquarters", "Head office", "Headquartered in")
    if not value:
        raise AttributeError("Page infobox has no headquarters listed")

    parts: List[str] = []
    for line in value.split("\n"):
        # Strip whitespace AND trailing/leading commas. If nothing meaningful
        # remains, drop the line entirely.
        cleaned = line.strip().strip(",").strip()
        if cleaned and not re.fullmatch(r"\d+", cleaned):
            parts.append(cleaned)

    if not parts:
        return _first_line(value)
    return ", ".join(parts)


def get_founder(entity_name: str) -> str:
    """Founder(s)."""
    box = parse_infobox(get_page_html(entity_name))
    value = _lookup(box, "Founder", "Founders", "Founded by")
    if not value:
        raise AttributeError("Page infobox has no founder listed")
    # Cell often has one founder per line - join with " and " for natural reading.
    names = [n.strip() for n in value.split("\n") if n.strip()]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def get_spouse(name: str) -> str:
    """Spouse(s) of a person."""
    box = parse_infobox(get_page_html(name))
    value = _lookup(box, "Spouse", "Spouses", "Partner", "Partners")
    if not value:
        raise AttributeError("Page infobox has no spouse listed")
    # Spouses often listed with date ranges in parens; strip those for the name.
    names = []
    for line in value.split("\n"):
        line = line.strip()
        if not line:
            continue
        bare = re.sub(r"\s*\(.*$", "", line).strip()
        if bare:
            names.append(bare)
    if not names:
        raise AttributeError("Spouse cell present but no parseable name")
    if len(names) == 1:
        return names[0]
    return ", ".join(names)


# ---------------------------------------------------------------------------
# UI helpers (ported / adapted from NotSteam main.py)
# ---------------------------------------------------------------------------


def _bottom_toolbar() -> HTML:
    """Branded bottom toolbar (left brand / center hint / right hint).

    Adapts to terminal width when prompt_toolkit can give us the column count;
    falls back to a compact one-liner on any error.
    """
    try:
        cols = 80
        if get_app_or_none is not None:
            app = get_app_or_none()
            if app is not None:
                try:
                    size = app.output.get_size()
                    cols = getattr(size, "columns", cols) or cols
                except Exception:
                    pass

        brand_plain = f"WikiBot v{VERSION}"
        center_plain = "Enter send"
        right_plain = "Ctrl-C exit"

        start_center = max((cols - len(center_plain)) // 2, len(brand_plain) + 1)
        RIGHT_MARGIN = 1
        start_right = max(
            cols - len(right_plain) - RIGHT_MARGIN,
            start_center + len(center_plain) + 1,
        )
        pad_center = max(start_center - len(brand_plain), 1)
        pad_right = max(start_right - (start_center + len(center_plain)), 1)

        left = f'<b><style bg="#FFFFFF">WikiBot</style></b> v{VERSION}'
        center = '<b><style bg="#FFFFFF">Enter</style></b> send'
        right = '<b><style bg="#FFFFFF">Ctrl^C</style></b> exit'
        line = " " + left + (" " * pad_center) + center + (" " * pad_right) + right
        return HTML(f'<style bg="#606060" fg="#242424">{line}</style>')
    except Exception:
        return HTML(
            f'<style bg="#606060" fg="#242424">'
            f'<b><style bg="#0b5fff">WikiBot</style></b> v{VERSION}'
            f' | Enter: send | Ctrl-C: exit</style>'
        )


# Single shared prompt session so history persists across queries.
_history = InMemoryHistory()
_session: PromptSession = PromptSession(
    history=_history, auto_suggest=AutoSuggestFromHistory()
)

# Vocabulary the autocompleter will suggest. Includes question stems plus
# common topic words to make typing fast and forgiving.
_completer = WordCompleter(
    [
        # Question stems
        "when was",
        "when did",
        "what is",
        "what is the",
        "where is",
        "who is",
        "who founded",
        "who is the founder of",
        "tell me about",
        # Patterns we support
        "when was * born",
        "when did * die",
        "what is the polar radius of",
        "what is the capital of",
        "what is the population of",
        "when was * founded",
        "where is * headquartered",
        "who founded *",
        "who is * married to",
        # Meta
        "help",
        "commands",
        "how do i use this",
        "?",
        "bye",
        "exit",
        "quit",
    ],
    ignore_case=True,
)


def _render_pick_list(candidates: List[str], title: str) -> None:
    """Show a numbered Rich table for disambiguation."""
    table = Table(title=title, box=box.ROUNDED, show_header=True)
    table.add_column("#", style="magenta", no_wrap=True)
    table.add_column("Page", style="cyan")
    for idx, c in enumerate(candidates, start=1):
        table.add_row(str(idx), c)
    console.print(table)


def _prompt_pick_index(max_index: int) -> int:
    """Prompt for a 1..max_index choice (0 cancels)."""
    while True:
        try:
            raw = _session.prompt(
                f"Pick 1-{max_index} (or 0 to cancel): ",
                bottom_toolbar=_bottom_toolbar,
            )
        except Exception:
            raw = console.input(f"Pick 1-{max_index} (or 0 to cancel): ")
        raw = (raw or "").strip()
        if raw.isdigit():
            i = int(raw)
            if 0 <= i <= max_index:
                return i
        console.print("[yellow]Invalid selection[/yellow]")


def _try_extract_disambiguation_options(html: str, limit: int = 10) -> List[str]:
    """When the API returns a disambiguation page, scrape the candidate links."""
    soup = BeautifulSoup(html, "html.parser")
    options: List[str] = []
    # Disambiguation pages are mostly bullet lists of internal links
    for li in soup.find_all("li"):
        a = li.find("a")
        if not a:
            continue
        title = (a.get("title") or a.text or "").strip()
        # Skip junk links (edit sections, references, etc.)
        if not title or title.startswith(("Help:", "Wikipedia:", "Category:", "Edit ")):
            continue
        if title not in options:
            options.append(title)
        if len(options) >= limit:
            break
    return options


def resolve_topic(name: str) -> Optional[str]:
    """Resolve a user-typed topic to a Wikipedia page title.

    Resolution pipeline:
      1. Try the exact page title.
      2. If Wikipedia says the page doesn't exist, ask Wikipedia's typo-tolerant
         search API for suggestions and show them in the same numbered picker UI
         we use for disambiguation.
      3. If the chosen result is itself a disambiguation page, show the normal
         disambiguation UI.

    Returns the final chosen page title, or None if the user cancels / nothing
    matches.
    """
    global _last_topic

    def _topic_variants(original: str) -> List[str]:
        variants = [original]
        article_match = re.match(r"^(the|a|an)\s+(.+)$", original, re.IGNORECASE)
        if article_match:
            variants.append(article_match.group(2).strip())
        if original and original[:1].islower():
            variants.append(original[:1].upper() + original[1:])
        if article_match:
            stripped = article_match.group(2).strip()
            if stripped and stripped[:1].islower():
                variants.append(stripped[:1].upper() + stripped[1:])
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for variant in variants:
            key = variant.lower()
            if variant and key not in seen:
                seen.add(key)
                ordered.append(variant)
        return ordered

    try:
        html = get_page_html(name)
    except LookupError as e:
        html = ""
        for variant in _topic_variants(name)[1:]:
            try:
                html = get_page_html(variant)
                name = variant
                break
            except Exception:
                continue

        if not html:
            candidates = _search_page_titles(name)
            if not candidates:
                console.print(f"[red]Couldn't load Wikipedia page for '{name}': {e}[/red]")
                return None

            ranked = sorted(
                candidates,
                key=lambda title: _topic_similarity(name, title),
                reverse=True,
            )
            _render_pick_list(ranked, title=f"Search results for '{name}'")
            pick = _prompt_pick_index(len(ranked))
            if pick == 0:
                return None

            chosen = ranked[pick - 1]
            console.print(f"[dim](using search result: {chosen})[/dim]")
            try:
                html = get_page_html(chosen)
            except Exception as chosen_error:
                console.print(
                    f"[red]Couldn't load Wikipedia page for '{chosen}': {chosen_error}[/red]"
                )
                return None
            name = chosen
    except Exception as e:
        console.print(f"[red]Couldn't load Wikipedia page for '{name}': {e}[/red]")
        return None

    # Detect disambiguation pages: they have the .mw-disambig class on body or
    # a clear "may refer to:" lead.
    soup = BeautifulSoup(html, "html.parser")
    is_disambig = bool(soup.find(id="disambigbox")) or bool(
        soup.find("table", class_="dmbox-disambig")
    )
    if not is_disambig:
        # Fall back to the "may refer to:" textual signal
        first_p = soup.find("p")
        if first_p and "may refer to" in first_p.text.lower():
            is_disambig = True

    if is_disambig:
        options = _try_extract_disambiguation_options(html)
        if not options:
            console.print(
                f"[yellow]'{name}' is a disambiguation page but I couldn't parse the options.[/yellow]"
            )
            return None
        _render_pick_list(options, title=f"'{name}' could mean...")
        pick = _prompt_pick_index(len(options))
        if pick == 0:
            return None
        chosen = options[pick - 1]
        _last_topic = chosen
        return chosen

    _last_topic = name
    return name


# ---------------------------------------------------------------------------
# Action functions: each one takes the captured matches and returns a list of
# answer lines. Returning [] / None lets the caller fall back to "No answers".
# ---------------------------------------------------------------------------


def _safe_call(fn: Callable[[str], str], topic: str, label: str) -> List[str]:
    """Resolve `topic`, call the extractor, and translate exceptions into
    user-facing messages. Keeps action functions small & boring."""
    resolved = resolve_topic(topic)
    if not resolved:
        return ["No answers"]
    try:
        value = fn(resolved)
    except LookupError as e:
        return [f"[yellow]{e}[/yellow]"]
    except AttributeError as e:
        return [f"[yellow]{e}[/yellow]"]
    except ConnectionError as e:
        return [f"[red]{e}[/red]"]
    if not value:
        return ["No answers"]
    return [f"[bold cyan]{label}:[/bold cyan] {value}"]


def _format_multiline_value(value: str) -> str:
    """Format an infobox value for inline answer display.

    Single-line answers stay compact. Multi-line cells become bullets so a
    generic field query like `what are Microsoft's products` reads cleanly.
    """
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    if not lines:
        return value.strip()
    if len(lines) == 1:
        return lines[0]
    return "\n".join(f"- {line}" for line in lines)


def _format_generic_answer(label: str, value: str) -> str:
    """Format a generic infobox field answer for one-line-ish CLI display.

    Keep the full infobox cell value. Multi-line answers render as bullets so
    they stay readable, but we do not discard any trailing lines or metadata.
    """
    label_norm = label.lower().strip()
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    if not lines:
        return value.strip()

    bullet_fields = {
        "products",
        "services",
        "brands",
        "founders",
        "official languages",
        "languages",
        "colors",
        "colours",
    }
    if label_norm in bullet_fields:
        return _format_multiline_value("\n".join(lines))
    if len(lines) == 1:
        return lines[0]
    return _format_multiline_value("\n".join(lines))


def generic_field_query(query_words: Sequence[str]) -> Optional[List[str]]:
    """Generic natural-language infobox query engine.

    This is the hybrid Tier 1-3 stack:
      - rule-based query normalization / intent extraction
      - spaCy tokenization + RapidFuzz fuzzy label matching
      - sentence-transformers semantic fallback for label selection

    It intentionally runs *after* the hand-written specific handlers but
    *before* the broad `what is %` / `who is %` catchalls.
    """
    intent = extract_field_query_intent(" ".join(query_words))
    if intent is None:
        return None

    def _display_label_from_lookup_key(key: str) -> str:
        words = key.split()
        if not words:
            return key
        return " ".join([words[0].capitalize(), *[w.lower() for w in words[1:]]])

    def _extract_colors_from_text(text: str) -> List[str]:
        color_names = [
            "red",
            "blue",
            "green",
            "yellow",
            "white",
            "black",
            "orange",
            "purple",
            "brown",
            "gold",
            "silver",
            "scarlet",
            "azure",
        ]
        found: List[str] = []
        lowered = text.lower()
        for color in color_names:
            if re.search(rf"\b{re.escape(color)}\b", lowered) and color not in found:
                found.append(color)
        return found

    def _candidate_topic_phrases() -> List[str]:
        candidates = [intent.topic_phrase]
        normalized_field = intent.field_phrase.strip().lower()
        if "flag" in intent.normalized_query and not intent.topic_phrase.lower().startswith("flag of "):
            candidates.append(f"flag of {intent.topic_phrase}")
        if intent.topic_phrase.lower().startswith("the "):
            candidates.append(intent.topic_phrase[4:].strip())
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for candidate in candidates:
            low = candidate.lower()
            if low not in seen and candidate.strip():
                seen.add(low)
                ordered.append(candidate)
        return ordered

    last_error: Optional[List[str]] = ["No answers"]

    for topic_candidate in _candidate_topic_phrases():
        resolved = resolve_topic(topic_candidate)
        if not resolved:
            last_error = ["No answers"]
            continue

        try:
            html = get_page_html(resolved)
            rows = parse_infobox_rows(html)
            lookup = parse_infobox(html)
        except (LookupError, ConnectionError) as e:
            last_error = [f"[yellow]{e}[/yellow]"]
            continue

        # First try visible primary rows. This preserves the original Wikipedia
        # label when a direct row exists (e.g. `Number of employees`).
        field_match = match_infobox_field(intent.field_phrase, rows)

        # Then fall back to promoted lookup keys: they contain synthetic section
        # names like `area`, `population`, or `establishment` that are often a
        # better fit for generic questions than the visible leaf-row labels.
        synthetic_rows = [
            (_display_label_from_lookup_key(label), value)
            for label, value in lookup.items()
        ]
        if field_match is None:
            field_match = match_infobox_field(intent.field_phrase, synthetic_rows)

        # Generic flag-color fallback: many flag pages do not have a dedicated
        # `Colors` row, but the `Design` row explicitly names them. This is a
        # structural rule that applies across many vexillology pages.
        if field_match is None and intent.field_phrase.strip().lower() in {"colors", "colours"}:
            design_match = match_infobox_field("design", rows)
            if design_match is not None:
                colors = _extract_colors_from_text(design_match.value)
                if colors:
                    field_match = FieldMatch(
                        label="Colors",
                        value="\n".join(color.capitalize() for color in colors),
                        strategy="derived",
                        score=100.0,
                        candidate_phrase="colors from design",
                    )

        if field_match is None:
            last_error = ["No answers"]
            continue

        return [
            f"[bold cyan]{field_match.label}:[/bold cyan] {_format_generic_answer(field_match.label, field_match.value)}"
        ]

    return last_error


def birth_date(matches: List[str]) -> List[str]:
    return _safe_call(get_birth_date, " ".join(matches), "Birth date")


def death_date(matches: List[str]) -> List[str]:
    return _safe_call(get_death_date, " ".join(matches), "Died")


def polar_radius(matches: List[str]) -> List[str]:
    topic = " ".join(matches)
    resolved = resolve_topic(topic)
    if not resolved:
        return ["No answers"]
    try:
        value = get_polar_radius(resolved)
    except (LookupError, AttributeError) as e:
        return [f"[yellow]{e}[/yellow]"]
    return [f"[bold cyan]Polar radius:[/bold cyan] {value} km"]


def capital_of(matches: List[str]) -> List[str]:
    return _safe_call(get_capital, " ".join(matches), "Capital")


def population_of(matches: List[str]) -> List[str]:
    return _safe_call(get_population, " ".join(matches), "Population")


def founded_year(matches: List[str]) -> List[str]:
    return _safe_call(get_founded_year, " ".join(matches), "Founded")


def headquarters_of(matches: List[str]) -> List[str]:
    return _safe_call(get_headquarters, " ".join(matches), "Headquarters")


def founder_of(matches: List[str]) -> List[str]:
    return _safe_call(get_founder, " ".join(matches), "Founder")


def spouse_of(matches: List[str]) -> List[str]:
    return _safe_call(get_spouse, " ".join(matches), "Spouse")


def about_topic(matches: List[str]) -> Optional[List[str]]:
    """Render the topic's full infobox in source order.

    Earlier versions only surfaced a small curated subset of "interesting"
    fields (Founded / Founder / Headquarters / ...). That left a ton of useful
    infobox information on the floor and made `tell me about X` feel weaker
    than the data we had already scraped.

    We now dump every primary infobox row returned by `parse_infobox_rows()`.
    Multi-line values stay multi-line so lists like founders / products /
    offices remain readable instead of being flattened into ugly ` / ` chains.
    """
    topic = " ".join(matches)
    resolved = resolve_topic(topic)
    if not resolved:
        return ["No answers"]
    try:
        rows = parse_infobox_rows(get_page_html(resolved))
    except (LookupError, ConnectionError) as e:
        return [f"[yellow]{e}[/yellow]"]

    def _display_value(value: str) -> str:
        """Format one infobox cell for Rich table display.

        Keep multi-line structure intact. For list-like values we add bullets so
        a field like "Founders" renders as a readable stack instead of a dense
        paragraph. Single-line values are returned unchanged.
        """
        lines = [line.strip() for line in value.split("\n") if line.strip()]
        if not lines:
            return value.strip()
        if len(lines) == 1:
            return lines[0]
        return "\n".join(f"- {line}" for line in lines)

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold magenta", no_wrap=True)
    table.add_column("Value", style="cyan", overflow="fold")

    for label, value in rows:
        table.add_row(label, _display_value(value))

    url_slug = resolved.replace(" ", "_")
    console.print(
        Panel(
            (
                f"[dim]Source: en.wikipedia.org/wiki/{url_slug}[/dim]\n"
                f"[dim]{len(rows)} infobox fields[/dim]"
            ),
            title=f"[bold cyan]{resolved}[/bold cyan]",
            border_style="cyan",
        )
    )
    if rows:
        console.print(table)
    else:
        console.print("[yellow]No structured infobox facts found.[/yellow]")
    return None


def show_help(matches: List[str]) -> Optional[List[str]]:
    """Render the rich help panel."""
    intro = (
        "[bold cyan]WikiBot[/bold cyan] - ask questions about anything on Wikipedia.\n"
        "[dim]Patterns are matched left-to-right; arrow keys recall history.[/dim]"
    )
    examples = [
        (
            "People",
            [
                "when was grace hopper born",
                "when did alan turing die",
                "who is barack obama married to",
                "tell me about ada lovelace",
            ],
        ),
        (
            "Places",
            [
                "what is the capital of france",
                "what is the population of tokyo",
                "tell me about iceland",
            ],
        ),
        (
            "Companies / orgs",
            [
                "when was microsoft founded",
                "where is apple headquartered",
                "who founded spacex",
            ],
        ),
        (
            "Planets / science",
            [
                "what is the polar radius of earth",
                "what is the polar radius of mars",
            ],
        ),
        (
            "Meta",
            [
                "help",
                "bye",
            ],
        ),
    ]

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Topic", style="bold magenta", no_wrap=True)
    table.add_column("Examples", style="cyan")
    for topic, lines in examples:
        table.add_row(topic, "\n".join(f"- {l}" for l in lines))

    console.print(Panel(intro, border_style="cyan"))
    console.print(table)
    console.print(
        "[dim]Tip: if a name is ambiguous (e.g. 'Mercury') WikiBot will ask which one you meant.[/dim]"
    )
    return None


def bye_action(_: List[str]) -> None:
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Pattern-action list
# ---------------------------------------------------------------------------

Pattern = Sequence[str]
Action = Callable[[List[str]], Any]

# NOTE: ordering matters. More-specific patterns come first; the catch-all
# "tell me about %" comes near the end so it doesn't swallow more targeted
# phrasings.
pa_list: List[Tuple[Pattern, Action]] = [
    # Help / meta
    ("help".split(), show_help),
    ("commands".split(), show_help),
    ("how do i use this".split(), show_help),
    (["?"], show_help),
    # Birth date
    ("when was % born".split(), birth_date),
    ("what is % 's birth date".split(), birth_date),
    ("what is %s birth date".split(), birth_date),
    ("what is the birth date of %".split(), birth_date),
    # Death date
    ("when did % die".split(), death_date),
    ("what is % 's death date".split(), death_date),
    ("what is the death date of %".split(), death_date),
    # Polar radius (planet)
    ("what is the polar radius of %".split(), polar_radius),
    ("what is %s polar radius".split(), polar_radius),
    # Capital (country)
    ("what is the capital of %".split(), capital_of),
    ("what 's the capital of %".split(), capital_of),
    ("capital of %".split(), capital_of),
    # Population (city / country)
    ("what is the population of %".split(), population_of),
    ("what 's the population of %".split(), population_of),
    ("how many people live in %".split(), population_of),
    ("population of %".split(), population_of),
    # Founding year (companies / orgs / universities)
    ("when was % founded".split(), founded_year),
    ("when was % established".split(), founded_year),
    ("what year was % founded".split(), founded_year),
    # Headquarters (companies / orgs)
    ("where is % headquartered".split(), headquarters_of),
    ("where is the headquarters of %".split(), headquarters_of),
    ("what is the headquarters of %".split(), headquarters_of),
    ("where is %s headquarters".split(), headquarters_of),
    # Founders
    ("who founded %".split(), founder_of),
    ("who is the founder of %".split(), founder_of),
    ("who are the founders of %".split(), founder_of),
    # Spouse (people)
    ("who is % married to".split(), spouse_of),
    ("who is %s spouse".split(), spouse_of),
    ("who is the spouse of %".split(), spouse_of),
    # Generic "tell me about" - low-priority catch-all
    ("tell me about %".split(), about_topic),
    ("describe %".split(), about_topic),
    ("what is %".split(), about_topic),
    ("who is %".split(), about_topic),
    # Bye
    (["bye"], bye_action),
    (["exit"], bye_action),
    (["quit"], bye_action),
]

_ABOUT_TOPIC_CATCHALLS: Set[Tuple[str, ...]] = {
    tuple("tell me about %".split()),
    tuple("describe %".split()),
    tuple("what is %".split()),
    tuple("who is %".split()),
}


# ---------------------------------------------------------------------------
# Autocorrect for command words (progressive cutoff)
#
# Without this, a typo like "what is the *captial* of france" misses the
# specific pattern and falls through to the catch-all `what is %`, which
# tries to look up "the captial of france" as a Wikipedia page (it doesn't
# exist) and the user gets a useless error.
#
# Strategy: build a small vocabulary of every literal token used in pa_list
# (everything that isn't a `%` or `_` wildcard - so "what", "is", "the",
# "capital", "of", "born", "founded", ...). Snap each input word to the
# closest vocab entry whose similarity meets our cutoff.
#
# Cutoff is *progressive*: the first attempt is conservative (0.82) so we
# don't mangle proper names. If that attempt produces no usable answer, we
# loosen the cutoff and retry, picking up trickier typos (e.g. "borm" ->
# "born", ratio 0.75). We bottom out at 0.68 - any looser starts mangling
# real content words.
#
# Cost control: each retry only runs an action again if its corrected form
# differs from any previous attempt's. A no-typo query runs the action once.
# ---------------------------------------------------------------------------

# Built lazily on first use - pa_list must be fully defined first.
_COMMAND_VOCAB: Optional[Set[str]] = None

# Cutoffs tried in order. First wins; only retry if the strict pass produced
# no usable answer (no match, or a soft failure like "No answers" / error).
_AUTOCORRECT_CUTOFFS: Tuple[float, ...] = (0.82, 0.75, 0.68)


def _command_vocab() -> Set[str]:
    """Collect closed-class query scaffold tokens.

    Important design choice: this is *not* every word that might appear in a
    valid query. Content words (field names like `currency`, `height`,
    `revenue`, `headquarters`) are handled by the generic NLP engine and its
    fuzzy/semantic matching. The command autocorrect layer should only fix the
    structural glue words around them (`what`, `does`, `use`, `how`, ...).

    That separation is what prevents bad snaps like `tall` -> `tell` while
    still allowing scaffold typos like `dose` -> `does`.
    """
    global _COMMAND_VOCAB
    if _COMMAND_VOCAB is None:
        vocab: Set[str] = set()
        for pat, _ in pa_list:
            for tok in pat:
                if tok and tok not in ("%", "_"):
                    vocab.add(tok.lower())
        vocab.update(query_scaffold_vocab_hints())
        _COMMAND_VOCAB = vocab
    return _COMMAND_VOCAB


def _autocorrect_words(words: Sequence[str], cutoff: float = 0.82) -> List[str]:
    """Snap each word to its closest entry in the command vocabulary, IF
    the match meets `cutoff`. Words already in the vocab stay as-is;
    words with no close match stay as-is.

    Topic captures (proper nouns like "France", "Microsoft") are preserved
    because they don't have close matches in the small command vocab.
    """
    vocab = _command_vocab()
    content_vocab = set(field_content_vocab_hints())
    out: List[str] = []
    for w in words:
        wl = w.lower()
        # Field/content words belong to the NLP matcher, not the command-word
        # autocorrect layer. Protect them from scaffold rewrites such as
        # `area` -> `are`.
        if wl in content_vocab:
            out.append(w)
            continue
        if wl in vocab:
            out.append(w)
            continue
        matches = difflib.get_close_matches(wl, vocab, n=1, cutoff=cutoff)
        out.append(matches[0] if matches else w)
    return out


def _find_pa_match(
    words: Sequence[str],
    include_catchalls: bool = True,
) -> Optional[Tuple[Pattern, Action, List[str]]]:
    """Walk pa_list and return the first non-empty-capture match.

    Splitting matching from action-running lets the progressive autocorrect
    loop avoid running an action twice for the same corrected input."""
    for pat, act in pa_list:
        if not include_catchalls and tuple(pat) in _ABOUT_TOPIC_CATCHALLS:
            continue
        mat = match(pat, words)
        if mat is None:
            continue
        if any((m is None) or (str(m).strip() == "") for m in mat):
            continue
        return (pat, act, mat)
    return None


def _looks_failed(result: Optional[List[str]]) -> bool:
    """Heuristic: did this attempt produce a usable answer?

    A "failed" attempt is one we'd want to retry with looser autocorrect.
    Note that `None` (handler-rendered) counts as SUCCESS - those handlers
    only return None on a happy path; they return error lists on failure.
    """
    if result is None:
        return False
    if not result:
        return True
    if result in (["No answers"], ["I don't understand"]):
        return True
    first = result[0]
    if isinstance(first, str) and (
        first.startswith("[yellow]") or first.startswith("[red]")
    ):
        return True
    return False


def search_pa_list(src: Sequence[str]) -> Optional[List[str]]:
    """Run `src` through the pattern-action list with progressive autocorrect.

    Returns:
        - A list of answer strings to print, OR
        - None if a handler already rendered output directly via the console
          (e.g. about_topic / show_help draw their own panels).

    Falls back to ["I don't understand"] when no pattern matches at all.

    Applies autocorrect to known command words at progressively looser
    cutoffs so typos like "captial" or "borm" route to the right handler
    without sacrificing precision on the first pass.
    """
    joined = " ".join(w.lower() for w in src).strip()
    if not src or joined in {"show", "tell", "what"}:
        return ["I don't understand"]

    # We may try up to len(_AUTOCORRECT_CUTOFFS) attempts, but skip any
    # whose corrected form duplicates a previous attempt's (no point
    # running the same action twice).
    tried: Set[Tuple[str, ...]] = set()
    last_result: Optional[List[str]] = ["I don't understand"]
    last_corrected: List[str] = list(src)

    for cutoff in _AUTOCORRECT_CUTOFFS:
        corrected = _autocorrect_words(src, cutoff=cutoff)
        key = tuple(w.lower() for w in corrected)
        if key in tried:
            continue
        tried.add(key)

        # 1. Hand-written specific handlers first (birth date, capital, etc.)
        specific_match = _find_pa_match(corrected, include_catchalls=False)
        if specific_match is not None:
            _pat, act, mat = specific_match
            answer = act(mat)
            if not _looks_failed(answer):
                _maybe_announce(src, corrected)
                return answer if (answer is None or answer) else ["No answers"]
            last_result = answer
            last_corrected = corrected

        # 2. Generic NLP infobox query engine.
        generic_answer = generic_field_query(corrected)
        if generic_answer is not None:
            if not _looks_failed(generic_answer):
                _maybe_announce(src, corrected)
                return generic_answer if (generic_answer is None or generic_answer) else ["No answers"]
            last_result = generic_answer
            last_corrected = corrected
            continue

        # 3. Broad `tell me about` / `what is %` catchalls last.
        catchall_match = _find_pa_match(corrected, include_catchalls=True)
        if catchall_match is not None:
            _pat, act, mat = catchall_match
            answer = act(mat)
            if not _looks_failed(answer):
                _maybe_announce(src, corrected)
                return answer if (answer is None or answer) else ["No answers"]
            last_result = answer
            last_corrected = corrected
            continue

        last_result = ["I don't understand"]
        last_corrected = corrected

    # All attempts exhausted. Show the most-recent interpretation alongside
    # the failure result so the user can see what we tried.
    _maybe_announce(src, last_corrected)
    return last_result


def _maybe_announce(original: Sequence[str], corrected: Sequence[str]) -> None:
    """Print the dim `(interpreting as: ...)` line iff something changed."""
    if [w.lower() for w in corrected] != [w.lower() for w in original]:
        console.print(
            f"[dim](interpreting as: {' '.join(corrected)})[/dim]"
        )


def query_loop() -> None:
    """Main interactive loop. Mirrors NotSteam's polished entry experience."""
    intro = (
        f"[bold cyan]WikiBot[/bold cyan] [magenta]v{VERSION}[/magenta]\n"
        "[dim]Ask me anything that has a Wikipedia page![/dim]\n"
    )
    console.print(Panel.fit(intro, border_style="bright_blue"))
    console.print()
    console.print(
        "[dim]Tip: type [bold green]help[/bold green] to see example queries.[/dim]"
    )
    console.print()

    while True:
        try:
            console.print()
            try:
                query_text = _session.prompt(
                    "Your query? ",
                    completer=_completer,
                    bottom_toolbar=_bottom_toolbar,
                )
            except Exception:
                query_text = console.input("[bold green]Your query?[/bold green] ")

            sanitized = query_text.replace("?", "").strip()
            if not sanitized:
                continue
            query = sanitized.split()

            joined_lower = " ".join(w.lower() for w in query).strip()
            if joined_lower in {"bye", "exit", "quit", "q"}:
                console.print("\n[yellow]So long![/yellow]\n")
                break

            answers = search_pa_list(query)
            if answers is None:
                pass  # handler already rendered
            elif answers:
                for a in answers:
                    if a:
                        console.print(a)
            else:
                console.print("[yellow]No answers[/yellow]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]So long![/yellow]\n")
            break


def main() -> None:
    query_loop()


if __name__ == "__main__":
    main()
