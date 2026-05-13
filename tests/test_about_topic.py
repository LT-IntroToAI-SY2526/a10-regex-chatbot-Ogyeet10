"""Tests for the `tell me about` / `about_topic` infobox dump.

This used to show only a tiny curated subset of fields. We now want the full
ordered infobox, with multi-line values kept readable.
"""

from rich.panel import Panel
from rich.table import Table

import a10


_MICROSOFT_INFOBOX = """
<html><body>
  <table class="infobox">
    <tr><th>Founded</th><td>April 4, 1975</td></tr>
    <tr><th>Founders</th><td>Bill Gates<br>Paul Allen</td></tr>
    <tr><th>Headquarters</th><td>Microsoft campus,<br>Redmond, Washington,<br>U.S.</td></tr>
    <tr><th>Industry</th><td>Information technology</td></tr>
    <tr><th>Products</th><td>Windows<br>Microsoft 365<br>Xbox</td></tr>
  </table>
</body></html>
"""


def test_about_topic_renders_full_ordered_infobox(monkeypatch):
    """`tell me about microsoft` should dump every infobox row in source
    order, not just a hand-picked summary."""
    printed = []

    monkeypatch.setattr(a10, "resolve_topic", lambda topic: "Microsoft")
    monkeypatch.setattr(a10, "get_page_html", lambda title: _MICROSOFT_INFOBOX)
    monkeypatch.setattr(a10.console, "print", lambda obj, *args, **kwargs: printed.append(obj))

    result = a10.about_topic(["microsoft"])

    assert result is None
    assert len(printed) == 2, "Expected a header panel and a table"
    assert isinstance(printed[0], Panel)
    assert isinstance(printed[1], Table)

    table = printed[1]
    field_cells = list(table.columns[0]._cells)
    value_cells = list(table.columns[1]._cells)

    # Every primary infobox row should be present, in source order.
    assert field_cells == [
        "Founded",
        "Founders",
        "Headquarters",
        "Industry",
        "Products",
    ]

    # Multi-line values should render as bullets rather than being flattened
    # into slash-separated soup.
    assert value_cells[1] == "- Bill Gates\n- Paul Allen"
    assert value_cells[2] == "- Microsoft campus,\n- Redmond, Washington,\n- U.S."
    assert value_cells[4] == "- Windows\n- Microsoft 365\n- Xbox"

    # Single-line values stay single-line.
    assert value_cells[0] == "April 4, 1975"
    assert value_cells[3] == "Information technology"
