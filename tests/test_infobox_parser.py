"""Unit tests for parse_infobox / _lookup / cell-cleaning helpers.

These run against hand-crafted HTML fragments so they don't need the network.
The goal is to lock down parser behavior independently of any specific
Wikipedia page's quirks.
"""

import pytest

# Import a10 without touching its prompt_toolkit/rich UI bits.
# This is the only place where we have to be careful about side-effects on
# import - everything else lives in functions.
from a10 import parse_infobox, _lookup, _normalize_label, _clean_cell_text, _first_line


# --- _normalize_label -------------------------------------------------------


def test_normalize_label_strips_parens_and_lowercases():
    assert _normalize_label("Spouse(s)") == "spouse"
    assert _normalize_label("Founder(s):") == "founder"
    assert _normalize_label("  FOUNDED  ") == "founded"


def test_normalize_label_collapses_nbsp_and_spaces():
    assert _normalize_label("Number\u00a0 of  employees") == "number of employees"


# --- _clean_cell_text -------------------------------------------------------


def test_clean_cell_strips_reference_markers():
    assert _clean_cell_text("Bill Gates[1] and Paul Allen[2]") == "Bill Gates and Paul Allen"


def test_clean_cell_normalizes_nbsp_to_space():
    assert _clean_cell_text("US$281.7\u00a0billion") == "US$281.7 billion"


def test_clean_cell_preserves_newlines_but_collapses_blank_lines():
    raw = "Bill Gates\n\n\nPaul Allen\n"
    assert _clean_cell_text(raw) == "Bill Gates\nPaul Allen"


# --- _first_line ------------------------------------------------------------


def test_first_line_returns_first_nonempty():
    assert _first_line("\n\nParis\nIle-de-France\n") == "Paris"


# --- parse_infobox ----------------------------------------------------------


SIMPLE_INFOBOX = """
<table class="infobox">
  <tr><th>Founded</th><td>April 4, 1975</td></tr>
  <tr><th>Founders</th><td>Bill Gates<br>Paul Allen</td></tr>
  <tr><th>Headquarters</th><td>Redmond, Washington</td></tr>
</table>
"""


def test_parse_basic_three_row_infobox():
    box = parse_infobox(SIMPLE_INFOBOX)
    assert box["founded"] == "April 4, 1975"
    assert box["founders"] == "Bill Gates\nPaul Allen"  # <br> -> newline
    assert box["headquarters"] == "Redmond, Washington"


def test_parse_no_infobox_raises():
    with pytest.raises(LookupError):
        parse_infobox("<html><body><p>nothing here</p></body></html>")


def test_parse_strips_reference_superscripts():
    html = """
    <table class="infobox">
      <tr><th>Founder</th><td>Bill Gates<sup class="reference">[1]</sup></td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box["founder"] == "Bill Gates"


def test_parse_strips_edit_section_links():
    html = """
    <table class="infobox">
      <tr><th>Industry</th><td>Tech<span class="mw-editsection">[edit]</span></td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert "edit" not in box["industry"].lower()


def test_parse_handles_list_items_as_newlines():
    html = """
    <table class="infobox">
      <tr><th>Founders</th><td><ul><li>Bill Gates</li><li>Paul Allen</li></ul></td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert "Bill Gates" in box["founders"]
    assert "Paul Allen" in box["founders"]
    # The two founders should be on separate lines, not jammed together
    assert "\n" in box["founders"]


def test_parse_ignores_rows_with_no_th():
    html = """
    <table class="infobox">
      <tr><td>this is the title cell</td></tr>
      <tr><th>Founded</th><td>1975</td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box == {"founded": "1975"}


def test_parse_keeps_first_occurrence_of_duplicate_label():
    html = """
    <table class="infobox">
      <tr><th>Founded</th><td>1975</td></tr>
      <tr><th>Founded</th><td>9999 (nested)</td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box["founded"] == "1975"


def test_parse_normalizes_parenthesized_label():
    html = """
    <table class="infobox">
      <tr><th>Spouse(s)</th><td>Jane Doe</td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box["spouse"] == "Jane Doe"


def test_parse_stitches_numbers_fragmented_across_inline_tags():
    """Wikipedia wraps grouped numbers in side-by-side <span>s. The old
    `get_text("\\n")` would split `6,356.752` into `6,` + `356.752`. We
    use an empty separator so adjacent inline text concatenates intact."""
    html = """
    <table class="infobox">
      <tr><th>Polar radius</th><td><span>6,</span><span>356.752</span> km</td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box["polar radius"] == "6,356.752 km"


def test_parse_section_tracks_subrows_under_header():
    """Country/city infoboxes use header-only rows like
    `<tr><th>Population</th></tr>` followed by bulleted sub-rows. The
    parser should make the section accessible by the parent name AND each
    sub-row reachable by its bare label."""
    html = """
    <table class="infobox">
      <tr><th>Population</th></tr>
      <tr><th>\u2022 2020 census</th><td>126,146,099</td></tr>
      <tr><th>\u2022 2024 estimate</th><td>123,500,000</td></tr>
    </table>
    """
    box = parse_infobox(html)
    # Sub-rows are still accessible by their (bullet-stripped) bare labels
    assert box["2020 census"] == "126,146,099"
    assert box["2024 estimate"] == "123,500,000"
    # AND the section name resolves to the first sub-row value
    assert box["population"] == "126,146,099"
    # Composite key also works
    assert box["population 2020 census"] == "126,146,099"


def test_parse_section_does_not_clobber_normal_label():
    """Section tracking shouldn't shadow a normal label that already has
    its own value."""
    html = """
    <table class="infobox">
      <tr><th>Founded</th><td>1975</td></tr>
      <tr><th>Founders</th></tr>
      <tr><th>\u2022 Lead</th><td>Bill Gates</td></tr>
    </table>
    """
    box = parse_infobox(html)
    assert box["founded"] == "1975"      # untouched
    assert box["founders"] == "Bill Gates"  # section gets primary value
    assert box["lead"] == "Bill Gates"   # sub-row also reachable


# --- _lookup ----------------------------------------------------------------


def test_lookup_returns_first_matching_alias():
    box = {"founded": "1975"}
    assert _lookup(box, "Founder", "Founded", "Established") == "1975"


def test_lookup_handles_alias_normalization():
    box = {"spouse": "Jane Doe"}
    # alias has parens which should be normalized away
    assert _lookup(box, "Spouse(s)") == "Jane Doe"


def test_lookup_returns_none_on_miss():
    box = {"founded": "1975"}
    assert _lookup(box, "Nonexistent", "Also missing") is None
