"""Tests for the text cleaner module."""

from src.extraction.text_cleaner import (
    strip_html,
    strip_xbrl,
    decode_entities,
    remove_boilerplate,
    normalize_whitespace,
    extract_sections,
    clean,
)


class TestStripHtml:
    def test_removes_simple_tags(self):
        assert strip_html("<p>Hello</p>").strip() == "Hello"

    def test_removes_nested_tags(self):
        result = strip_html("<div><span>text</span></div>")
        assert "text" in result
        assert "<" not in result

    def test_removes_attributes(self):
        result = strip_html('<a href="http://example.com">link</a>')
        assert "link" in result
        assert "href" not in result

    def test_preserves_plain_text(self):
        assert strip_html("no tags here") == "no tags here"


class TestStripXbrl:
    def test_removes_ix_tags(self):
        result = strip_xbrl("<ix:nonFraction>42</ix:nonFraction>")
        assert "42" in result
        assert "ix:" not in result

    def test_removes_xmlns(self):
        result = strip_xbrl('xmlns:us-gaap="http://fasb.org"')
        assert "xmlns" not in result

    def test_case_insensitive(self):
        result = strip_xbrl("<IX:NONFRACTION>100</IX:NONFRACTION>")
        assert "100" in result
        assert "IX:" not in result


class TestDecodeEntities:
    def test_common_entities(self):
        assert "&" in decode_entities("&amp;")
        assert "<" in decode_entities("&lt;")
        assert ">" in decode_entities("&gt;")

    def test_quote_entities(self):
        assert '"' in decode_entities("&quot;")
        assert '"' in decode_entities("&ldquo;")
        assert '"' in decode_entities("&rdquo;")

    def test_dash_entities(self):
        assert "\u2014" in decode_entities("&mdash;")
        assert "\u2013" in decode_entities("&ndash;")

    def test_numeric_entities_stripped(self):
        result = decode_entities("&#9999;")
        assert "&#" not in result


class TestRemoveBoilerplate:
    def test_removes_sec_header(self):
        text = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION stuff"
        result = remove_boilerplate(text)
        assert "SECURITIES AND EXCHANGE COMMISSION" not in result

    def test_removes_form_type(self):
        result = remove_boilerplate("FORM 10-Q quarterly report")
        assert "FORM 10-Q" not in result

    def test_removes_table_of_contents(self):
        result = remove_boilerplate("Table of Contents next section")
        assert "Table of Contents" not in result

    def test_preserves_normal_text(self):
        text = "Revenue increased 15% year over year"
        assert remove_boilerplate(text) == text


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("too   many    spaces") == "too many spaces"

    def test_collapses_newlines(self):
        assert "\n\n\n" not in normalize_whitespace("a\n\n\n\nb")

    def test_strips_outer_whitespace(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_tabs_to_spaces(self):
        assert "\t" not in normalize_whitespace("tab\there")


class TestExtractSections:
    def test_finds_mda_section(self):
        text = "Intro text. Management's Discussion and Analysis of something. Revenue grew. Item 1A: Risk Factors here."
        sections = extract_sections(text)
        assert "mda" in sections
        assert "Revenue grew" in sections["mda"]

    def test_finds_risk_factors(self):
        text = "Start. Item 1A: Risk Factors We face competition. Item 3: Quantitative stuff."
        sections = extract_sections(text)
        assert "risk_factors" in sections

    def test_finds_liquidity(self):
        text = "Section. Liquidity and Capital Resources We maintain credit. End."
        sections = extract_sections(text)
        assert "liquidity" in sections

    def test_empty_text_returns_empty(self):
        assert extract_sections("") == {}

    def test_no_sections_returns_empty(self):
        assert extract_sections("just some random text without headers") == {}


class TestCleanPipeline:
    def test_full_pipeline(self):
        raw = """
        <html><body>
        <ix:nonFraction>42</ix:nonFraction>
        <p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
        <p>Revenue &amp; profit increased by 15%</p>
        </body></html>
        """
        result = clean(raw)
        assert "Revenue & profit increased by 15%" in result
        assert "<html>" not in result
        assert "ix:" not in result
        assert "SECURITIES AND EXCHANGE COMMISSION" not in result

    def test_preserves_financial_content(self):
        raw = "<p>Net income was $4.2 billion, up 23% from prior year.</p>"
        result = clean(raw)
        assert "$4.2 billion" in result
        assert "23%" in result
