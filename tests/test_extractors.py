"""Integration tests for the per-fact Wikipedia extractors.

These tests use real Wikipedia HTML, fetched once via the `wiki_html` fixture
and cached on disk under `tests/fixtures/` (gitignored). They're marked
`network` so they get skipped gracefully if the cache is empty AND there's
no internet.

Each extractor takes a *page title* and re-fetches the HTML. To avoid double
HTTP traffic we monkeypatch `get_page_html` to serve from the same fixture
cache the rest of the test suite uses.

When adding a new test:
  - Just reference a new page title in @pytest.mark.parametrize - the
    conftest fixture will pull it from Wikipedia on first run.
  - Run once with the network up to populate the cache, then commit nothing
    extra (fixtures/ is gitignored). Re-runs are offline-fast.
"""

import pytest

import a10


@pytest.fixture
def patched_page_html(monkeypatch, wiki_html):
    """Make a10.get_page_html serve from the fixture cache for all tests in
    this module."""
    monkeypatch.setattr(a10, "get_page_html", wiki_html)


pytestmark = [pytest.mark.network, pytest.mark.usefixtures("patched_page_html")]


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


class TestBirthDates:
    @pytest.mark.parametrize(
        "page,expected",
        [
            ("Grace Hopper",    "1906-12-09"),
            ("Ada Lovelace",    "1815-12-10"),
            ("Alan Turing",     "1912-06-23"),
            ("Albert Einstein", "1879-03-14"),
            ("Marie Curie",     "1867-11-07"),
            ("Steve Jobs",      "1955-02-24"),
            ("Isaac Newton",    "1643-01-04"),
            ("Charles Darwin",  "1809-02-12"),
            ("Tim Berners-Lee", "1955-06-08"),
            ("Linus Torvalds",  "1969-12-28"),
        ],
    )
    def test_birth_date(self, page, expected):
        assert a10.get_birth_date(page) == expected


class TestDeathDates:
    @pytest.mark.parametrize(
        "page,expected",
        [
            ("Grace Hopper",    "1992-01-01"),
            ("Alan Turing",     "1954-06-07"),
            ("Albert Einstein", "1955-04-18"),
            ("Marie Curie",     "1934-07-04"),
            ("Steve Jobs",      "2011-10-05"),
            ("Isaac Newton",    "1727-03-31"),
            ("Charles Darwin",  "1882-04-19"),
        ],
    )
    def test_death_date(self, page, expected):
        assert a10.get_death_date(page) == expected


class TestSpouses:
    @pytest.mark.parametrize(
        "page,must_contain",
        [
            ("Barack Obama",    "Michelle"),
            ("Bill Gates",      "Melinda"),
            ("Steve Jobs",      "Laurene"),
            ("Albert Einstein", "Mileva"),  # first listed spouse
        ],
    )
    def test_spouse_contains(self, page, must_contain):
        result = a10.get_spouse(page)
        assert must_contain.lower() in result.lower()


# ---------------------------------------------------------------------------
# Companies / organizations
# ---------------------------------------------------------------------------


class TestFoundedYear:
    @pytest.mark.parametrize(
        "page,expected",
        [
            # The headline bug case: "April 4, 1975" used to break the regex.
            ("Microsoft",          "1975"),
            ("Apple Inc.",         "1976"),
            ("Google",             "1998"),
            ("Amazon (company)",   "1994"),
            ("SpaceX",             "2002"),
            ("Tesla, Inc.",        "2003"),
            ("Meta Platforms",     "2004"),
            ("IBM",                "1911"),
            ("Stanford University", "1885"),
            ("Massachusetts Institute of Technology", "1861"),
            ("Harvard University", "1636"),
        ],
    )
    def test_founded_year(self, page, expected):
        assert a10.get_founded_year(page) == expected


class TestFounders:
    @pytest.mark.parametrize(
        "page,must_contain",
        [
            ("Microsoft",        ["Bill Gates", "Paul Allen"]),
            ("Apple Inc.",       ["Steve Jobs"]),
            ("SpaceX",           ["Elon Musk"]),
            ("Google",           ["Larry Page", "Sergey Brin"]),
            ("Amazon (company)", ["Jeff Bezos"]),
            ("Meta Platforms",   ["Mark Zuckerberg"]),
        ],
    )
    def test_founders_contain(self, page, must_contain):
        result = a10.get_founder(page)
        for needle in must_contain:
            assert needle in result, f"Expected {needle!r} in founders of {page!r}, got {result!r}"


class TestHeadquarters:
    @pytest.mark.parametrize(
        "page,city",
        [
            ("Microsoft",        "Redmond"),
            ("Apple Inc.",       "Cupertino"),
            ("Google",           "Mountain View"),
            ("Amazon (company)", "Seattle"),
            ("IBM",              "Armonk"),
            ("Tesla, Inc.",      "Austin"),
        ],
    )
    def test_headquarters_contains_city(self, page, city):
        hq = a10.get_headquarters(page)
        assert city in hq, f"Expected {city!r} in HQ of {page!r}, got {hq!r}"
        # Regression guard: no double-comma artifacts (the original bug)
        assert ", ," not in hq, f"Double-comma artifact in HQ of {page!r}: {hq!r}"
        assert ",," not in hq


# ---------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------


class TestCapitals:
    @pytest.mark.parametrize(
        "page,capital",
        [
            ("France",         "Paris"),
            ("Japan",          "Tokyo"),
            ("Germany",        "Berlin"),
            ("Italy",          "Rome"),
            ("Spain",          "Madrid"),
            ("Canada",         "Ottawa"),
            ("Brazil",         "Bras"),     # "Brasília" - just check first letters in case of unicode roundtrip
            ("India",          "New Delhi"),
            ("Egypt",          "Cairo"),
            ("United Kingdom", "London"),
        ],
    )
    def test_capital(self, page, capital):
        result = a10.get_capital(page)
        assert capital.lower() in result.lower(), (
            f"Expected {capital!r} in capital of {page!r}, got {result!r}"
        )


class TestPopulation:
    @pytest.mark.parametrize(
        "page,min_population",
        [
            ("Japan",          100_000_000),
            ("Germany",         70_000_000),
            ("France",          60_000_000),
            ("Canada",          30_000_000),
            ("Brazil",         200_000_000),
            ("India",        1_000_000_000),
            ("United Kingdom",  60_000_000),
            ("Egypt",           90_000_000),
        ],
    )
    def test_population_is_in_right_order_of_magnitude(self, page, min_population):
        pop = a10.get_population(page)
        # Must be a parseable integer (comma-grouped or bare digits)
        plain = pop.replace(",", "")
        assert plain.isdigit(), f"Population for {page!r} not numeric: {pop!r}"
        n = int(plain)
        assert n >= min_population, (
            f"Population for {page!r} = {n:,} is below the {min_population:,} floor"
        )
        # Sanity ceiling: nothing on Earth has > 2 billion people, so a number
        # bigger than that is almost certainly a GDP figure or area in m^2 we
        # accidentally captured.
        assert n < 2_000_000_000, (
            f"Population for {page!r} = {n:,} looks suspiciously large"
        )


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------


class TestPolarRadius:
    @pytest.mark.parametrize(
        # Approx polar radii in km from Wikipedia (used as sanity bounds)
        "page,low_km,high_km",
        [
            ("Earth",   6_300,    6_400),    # 6356.752
            ("Mars",    3_300,    3_400),    # 3376.2
            ("Venus",   6_000,    6_100),    # ~6051.8 (Venus has no separate polar value)
            ("Jupiter", 60_000,   70_000),   # 66,854
            ("Saturn",  50_000,   60_000),   # 54,364
            ("Mercury (planet)", 2_400, 2_500),  # 2439.7
        ],
    )
    def test_radius_is_in_sane_range(self, page, low_km, high_km):
        result = a10.get_polar_radius(page)
        # Regression guard: must not start with a period (original Earth bug)
        assert not result.startswith("."), (
            f"Radius for {page!r} starts with period: {result!r}"
        )
        # Strip commas/decimals and parse the integer portion
        intpart = int(result.replace(",", "").split(".")[0])
        assert low_km <= intpart <= high_km, (
            f"Radius for {page!r} = {intpart} km outside sane range "
            f"[{low_km}, {high_km}]"
        )


# ---------------------------------------------------------------------------
# Failure modes - we want the right exception, not silent garbage
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_living_person_has_no_death_date(self):
        """Living person infoboxes don't have a `Died` row - we should
        raise rather than silently return garbage."""
        with pytest.raises((AttributeError, LookupError)):
            a10.get_death_date("Barack Obama")

    def test_planet_has_no_founder(self):
        """Asking for the founder of a planet should fail cleanly."""
        with pytest.raises((AttributeError, LookupError)):
            a10.get_founder("Earth")

    def test_planet_has_no_spouse(self):
        with pytest.raises((AttributeError, LookupError)):
            a10.get_spouse("Mars")

    def test_person_has_no_polar_radius(self):
        with pytest.raises((AttributeError, LookupError)):
            a10.get_polar_radius("Albert Einstein")

    def test_person_has_no_capital(self):
        with pytest.raises((AttributeError, LookupError)):
            a10.get_capital("Alan Turing")

    def test_country_has_no_founder(self):
        """Countries don't have a Founder field (founders of *modern states*
        are buried in prose, not the infobox)."""
        with pytest.raises((AttributeError, LookupError)):
            a10.get_founder("France")


# ---------------------------------------------------------------------------
# parse_infobox sanity on real pages
# ---------------------------------------------------------------------------


class TestRealInfoboxes:
    """End-to-end checks that the parser produces sensible dicts for a
    variety of page types. Catches regressions where the parser silently
    drops a section, fragments a number, or mis-tracks bullet sub-rows."""

    def test_microsoft_infobox_has_core_fields(self, wiki_html):
        box = a10.parse_infobox(wiki_html("Microsoft"))
        assert "founded" in box
        assert "founders" in box
        assert "headquarters" in box
        assert "industry" in box

    def test_japan_infobox_has_subrow_population(self, wiki_html):
        """Japan's infobox keeps population in bullet sub-rows under a
        `Population` section header. Section tracking should hoist them."""
        box = a10.parse_infobox(wiki_html("Japan"))
        # Section name must resolve to one of the sub-row values
        assert "population" in box
        # And the bare sub-row labels are also present
        assert any("census" in k for k in box)

    def test_earth_radius_value_is_intact_number(self, wiki_html):
        """Regression: numbers split across <span class=nowrap> tags get
        reassembled instead of fragmented."""
        box = a10.parse_infobox(wiki_html("Earth"))
        polar = box.get("polar radius", "")
        # Must contain "6356" with optional comma (no fragmentation)
        flat = polar.replace(",", "").replace(" ", "")
        assert flat.startswith("6356"), f"polar radius corrupt: {polar!r}"
