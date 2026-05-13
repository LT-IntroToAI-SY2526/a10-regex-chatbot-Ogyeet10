"""End-to-end tests for generic infobox field queries."""

import a10


MICROSOFT_INFOBOX = """
<html><body>
  <table class="infobox">
    <tr><th>Revenue</th><td>US$281.7 billion</td></tr>
    <tr><th>Number of employees</th><td>228,000</td></tr>
    <tr><th>Headquarters</th><td>Microsoft campus,<br>Redmond, Washington,<br>U.S.</td></tr>
    <tr><th>Products</th><td>Windows<br>Microsoft 365<br>Xbox</td></tr>
    <tr><th>Industry</th><td>Information technology</td></tr>
  </table>
</body></html>
"""


def test_generic_field_query_possessive_revenue(monkeypatch):
    monkeypatch.setattr(a10, "resolve_topic", lambda topic: "Microsoft")
    monkeypatch.setattr(a10, "get_page_html", lambda title: MICROSOFT_INFOBOX)

    result = a10.generic_field_query("what is microsoft's revenue".split())
    assert result == ["[bold cyan]Revenue:[/bold cyan] US$281.7 billion"]


def test_generic_field_query_employee_count(monkeypatch):
    monkeypatch.setattr(a10, "resolve_topic", lambda topic: "Microsoft")
    monkeypatch.setattr(a10, "get_page_html", lambda title: MICROSOFT_INFOBOX)

    result = a10.generic_field_query("how many employees does microsoft have".split())
    assert result == ["[bold cyan]Number of employees:[/bold cyan] 228,000"]


def test_generic_field_query_multiline_products(monkeypatch):
    monkeypatch.setattr(a10, "resolve_topic", lambda topic: "Microsoft")
    monkeypatch.setattr(a10, "get_page_html", lambda title: MICROSOFT_INFOBOX)

    result = a10.generic_field_query("what products does microsoft make".split())
    assert result == [
        "[bold cyan]Products:[/bold cyan] - Windows\n- Microsoft 365\n- Xbox"
    ]


def test_search_pa_list_prefers_generic_field_engine_before_about_catchall(monkeypatch):
    calls = {"about": 0}

    monkeypatch.setattr(a10, "resolve_topic", lambda topic: "Microsoft")
    monkeypatch.setattr(a10, "get_page_html", lambda title: MICROSOFT_INFOBOX)

    def fake_about(matches):
        calls["about"] += 1
        return None

    new_pa_list = []
    for pat, act in a10.pa_list:
        if act.__name__ == "about_topic":
            new_pa_list.append((pat, fake_about))
        else:
            new_pa_list.append((pat, act))
    monkeypatch.setattr(a10, "pa_list", new_pa_list)
    monkeypatch.setattr(a10, "_COMMAND_VOCAB", None)

    result = a10.search_pa_list("what is microsoft's revenue".split())

    assert result == ["[bold cyan]Revenue:[/bold cyan] US$281.7 billion"]
    assert calls["about"] == 0


def test_search_pa_list_nonfield_still_falls_back_to_about_topic(monkeypatch):
    calls = {"about": 0}

    def fake_about(matches):
        calls["about"] += 1
        return None

    new_pa_list = []
    for pat, act in a10.pa_list:
        if act.__name__ == "about_topic":
            new_pa_list.append((pat, fake_about))
        else:
            new_pa_list.append((pat, act))
    monkeypatch.setattr(a10, "pa_list", new_pa_list)
    monkeypatch.setattr(a10, "_COMMAND_VOCAB", None)

    result = a10.search_pa_list("what is microsoft".split())

    assert result is None
    assert calls["about"] == 1
