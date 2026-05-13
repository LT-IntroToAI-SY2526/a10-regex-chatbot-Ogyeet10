"""Tests for topic-level Wikipedia typo tolerance.

This sits *below* the command-word autocorrect layer. The command layer fixes
`wen` -> `when` and `fowunded` -> `founded`; this file verifies that topic
resolution itself can recover from misspelled Wikipedia titles like
`micorsoft` by surfacing `opensearch` results in the same numbered picker UI
we use for disambiguation pages.
"""

import a10


_DUMMY_INFOBOX_HTML = """
<html><body>
  <table class="infobox">
    <tr><th>Founded</th><td>April 4, 1975</td></tr>
  </table>
</body></html>
"""


def test_resolve_topic_uses_search_results_ui_on_lookup_error(monkeypatch):
    """If the exact page lookup fails, resolve_topic should show Wikipedia
    search results in the same picker UI used for disambiguation."""
    rendered = []
    prompts = []
    printed = []

    def fake_get_page_html(title: str) -> str:
        if title == "micorsoft":
            raise LookupError("The page you specified doesn't exist.")
        if title == "Microsoft":
            return _DUMMY_INFOBOX_HTML
        raise AssertionError(f"unexpected title lookup: {title!r}")

    monkeypatch.setattr(a10, "get_page_html", fake_get_page_html)
    monkeypatch.setattr(
        a10,
        "_search_page_titles",
        lambda query, limit=8: ["Active Directory", "Microsoft"],
    )
    monkeypatch.setattr(
        a10,
        "_render_pick_list",
        lambda candidates, title: rendered.append((list(candidates), title)),
    )
    monkeypatch.setattr(
        a10,
        "_prompt_pick_index",
        lambda max_index: prompts.append(max_index) or 1,
    )
    monkeypatch.setattr(
        a10.console,
        "print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    resolved = a10.resolve_topic("micorsoft")

    assert resolved == "Microsoft"
    assert prompts == [2]
    assert rendered, "search-result picker UI was never shown"
    candidates, title = rendered[0]
    # Re-ranked by similarity, so Microsoft should be first even if the raw
    # search API returned it second.
    assert candidates[0] == "Microsoft"
    assert candidates[1] == "Active Directory"
    assert title == "Search results for 'micorsoft'"
    assert any(
        "using search result: Microsoft" in str(args[0])
        for args, _kwargs in printed
        if args
    )


def test_resolve_topic_can_cancel_search_results_ui(monkeypatch):
    """Canceling the picker should abort resolution cleanly."""

    monkeypatch.setattr(
        a10,
        "get_page_html",
        lambda title: (_ for _ in ()).throw(LookupError("missing"))
        if title == "micorsoft"
        else _DUMMY_INFOBOX_HTML,
    )
    monkeypatch.setattr(a10, "_search_page_titles", lambda query, limit=8: ["Microsoft"])
    monkeypatch.setattr(a10, "_render_pick_list", lambda candidates, title: None)
    monkeypatch.setattr(a10, "_prompt_pick_index", lambda max_index: 0)

    assert a10.resolve_topic("micorsoft") is None


def test_resolve_topic_with_no_search_hits_reports_failure(monkeypatch):
    """If Wikipedia search has no useful fallback titles, we should surface
    the original lookup failure and stop."""
    printed = []

    monkeypatch.setattr(
        a10,
        "get_page_html",
        lambda title: (_ for _ in ()).throw(LookupError("The page you specified doesn't exist.")),
    )
    monkeypatch.setattr(a10, "_search_page_titles", lambda query, limit=8: [])
    monkeypatch.setattr(
        a10.console,
        "print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    assert a10.resolve_topic("micorsoft") is None
    assert any(
        "Couldn't load Wikipedia page for 'micorsoft'" in str(args[0])
        for args, _kwargs in printed
        if args
    )


def test_resolve_topic_strips_leading_indefinite_article_before_search(monkeypatch):
    """Generic article stripping should handle natural phrases like
    `a bald eagle` without forcing the user into search results.

    This guards the crash-producing path from the conversation: the exact page
    doesn't exist, but the stripped title does.
    """
    looked_up = []

    def fake_get_page_html(title: str) -> str:
        looked_up.append(title)
        if title == "a bald eagle":
            raise LookupError("The page you specified doesn't exist.")
        if title == "bald eagle":
            return _DUMMY_INFOBOX_HTML
        raise AssertionError(f"unexpected title lookup: {title!r}")

    monkeypatch.setattr(a10, "get_page_html", fake_get_page_html)

    resolved = a10.resolve_topic("a bald eagle")

    assert resolved == "bald eagle"
    assert looked_up == ["a bald eagle", "bald eagle"]


def test_search_pa_list_user_case_flows_through_search_results(monkeypatch):
    """End-to-end for the exact user report:

      wen was micorsoft fowunded

    Command-word autocorrect should fix `wen`/`fowunded`, topic resolution
    should recover `micorsoft` via Wikipedia search results UI, and the final
    founded-year action should receive `Microsoft`.
    """
    rendered = []

    def fake_get_page_html(title: str) -> str:
        if title == "micorsoft":
            raise LookupError("The page you specified doesn't exist.")
        if title == "Microsoft":
            return _DUMMY_INFOBOX_HTML
        raise AssertionError(f"unexpected title lookup: {title!r}")

    monkeypatch.setattr(a10, "get_page_html", fake_get_page_html)
    monkeypatch.setattr(
        a10,
        "_search_page_titles",
        lambda query, limit=8: ["Active Directory", "Microsoft"],
    )
    monkeypatch.setattr(
        a10,
        "_render_pick_list",
        lambda candidates, title: rendered.append((list(candidates), title)),
    )
    monkeypatch.setattr(a10, "_prompt_pick_index", lambda max_index: 1)
    monkeypatch.setattr(
        a10,
        "get_founded_year",
        lambda title: "1975" if title == "Microsoft" else (_ for _ in ()).throw(AssertionError(title)),
    )

    result = a10.search_pa_list("wen was micorsoft fowunded".split())

    assert rendered, "search-results UI was not shown for the typo'd topic"
    assert result == ["[bold cyan]Founded:[/bold cyan] 1975"]
