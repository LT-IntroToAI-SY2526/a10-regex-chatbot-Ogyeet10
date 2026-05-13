"""
Shared pytest fixtures.

The `wiki_html` fixture is a cached HTTP fetcher: the first time a test asks
for a page, it hits the live MediaWiki API and writes the result to
`tests/fixtures/<title>.html`. Subsequent test runs read straight from disk -
fast, deterministic, and offline-friendly. The fixtures directory is gitignored
so the repo stays small.

If the cache is empty and we have no network, network-dependent tests are
skipped with a clear message instead of erroring out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import pytest
import requests

# Make the project root importable so `import a10` works from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)


def _safe_slug(title: str) -> str:
    """Filesystem-safe filename for a Wikipedia page title."""
    return title.replace("/", "_").replace(" ", "_").replace(":", "_") + ".html"


def _fetch_html(title: str) -> str:
    """Hit the live MediaWiki API. Retries on 429 with Retry-After backoff.

    Raises on persistent network/HTTP/API errors after up to 5 attempts.
    """
    import time

    for attempt in range(5):
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "redirects": True,
            },
            headers={"User-Agent": "wikibot-test-suite/1.0"},
            timeout=15,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 + attempt))
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error for {title!r}: {data['error']}")
        return data["parse"]["text"]["*"]
    raise RuntimeError(f"Wikipedia rate-limited 5x in a row for {title!r}")


@pytest.fixture(scope="session")
def wiki_html() -> Callable[[str], str]:
    """Return a `(title) -> html` fetcher backed by the on-disk fixture cache.

    Set REFRESH_FIXTURES=1 in the env to force re-download of every page on
    next access (useful when Wikipedia content changes break a test).
    """
    refresh = os.environ.get("REFRESH_FIXTURES") == "1"

    def get(title: str) -> str:
        path = FIXTURE_DIR / _safe_slug(title)
        if path.exists() and not refresh:
            return path.read_text(encoding="utf-8")
        try:
            html = _fetch_html(title)
        except (requests.RequestException, RuntimeError) as e:
            pytest.skip(f"No fixture for {title!r} and live fetch failed: {e}")
            raise  # unreachable, but appeases the type checker
        path.write_text(html, encoding="utf-8")
        return html

    return get
