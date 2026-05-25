"""
SEC filing text cleaner.
Strips HTML/XBRL tags, boilerplate, and noise from raw filing text.
Outputs clean plain text ready for LLM ingestion.
"""

import re
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


def strip_html(text: str) -> str:
    """Remove all HTML/XML tags."""
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def strip_xbrl(text: str) -> str:
    """Remove XBRL inline markup and namespace declarations."""
    text = re.sub(r"<ix:[^>]+>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</ix:[^>]+>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"xmlns[^=]*=\"[^\"]*\"", "", text)
    return text


def decode_entities(text: str) -> str:
    """Replace common HTML entities with readable characters."""
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
        "&quot;": '"', "&#160;": " ", "&#8212;": "—", "&#8211;": "–",
        "&#8217;": "'", "&#8220;": '"', "&#8221;": '"', "&ldquo;": '"',
        "&rdquo;": '"', "&lsquo;": "'", "&rsquo;": "'", "&mdash;": "—",
        "&ndash;": "–",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    # Catch remaining numeric entities
    text = re.sub(r"&#\d+;", " ", text)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common SEC filing boilerplate that adds noise for LLMs."""
    patterns = [
        r"Table of Contents",
        r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
        r"Washington,? D\.?C\.? \d{5}",
        r"FORM 10-[QK]",
        r"\(Mark One\)",
        r"Commission file number.{0,120}",   # bounded — avoid greedy match on long XBRL lines
        r"Check the appropriate box.{0,120}",
        r"Indicate by check mark.{0,200}",
        r"Securities registered pursuant to.{0,200}",
        r"Exact name of registrant.{0,200}",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace and blank lines."""
    text = re.sub(r"[ \t]+", " ", text)           # multiple spaces/tabs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3+ newlines → double newline
    text = re.sub(r" \n", "\n", text)               # trailing spaces before newline
    return text.strip()


def extract_sections(text: str) -> dict[str, str]:
    """
    Extract key 10-Q sections by header pattern matching.
    Returns a dict with section name → text.
    """
    section_patterns = {
        "mda": r"(?:item\s*2[\.\:]?\s*)?management['\u2019]?s?\s+discussion\s+and\s+analysis",
        "risk_factors": r"item\s*1a[\.\:]?\s*risk\s+factors",
        "quantitative_disclosures": r"item\s*3[\.\:]?\s*quantitative\s+and\s+qualitative",
        "financial_statements": r"item\s*1[\.\:]?\s*financial\s+statements",
        "results_of_operations": r"results\s+of\s+operations",
        "liquidity": r"liquidity\s+and\s+capital\s+resources",
        "forward_looking": r"forward[- ]looking\s+statements",
    }

    sections = {}
    text_lower = text.lower()

    for name, pattern in section_patterns.items():
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            start = matches[0].start()
            # Find next section start as the end boundary (rough cut)
            end = len(text)
            for other_name, other_pattern in section_patterns.items():
                if other_name == name:
                    continue
                other_matches = list(re.finditer(other_pattern, text_lower))
                for m in other_matches:
                    if start < m.start() < end:
                        end = m.start()
            sections[name] = text[start:end].strip()

    return sections


def clean(raw_text: str) -> str:
    """Full cleaning pipeline: strip → decode → remove boilerplate → normalize."""
    text = strip_xbrl(raw_text)
    text = strip_html(text)
    text = decode_entities(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text


def clean_file(filepath: Path) -> tuple[str, dict[str, str]]:
    """
    Clean a single raw filing file.
    Returns (full_clean_text, sections_dict).
    """
    raw = filepath.read_text(encoding="utf-8", errors="ignore")
    full_text = clean(raw)
    sections = extract_sections(full_text)
    return full_text, sections


def clean_and_save(ticker: str) -> list[Path]:
    """
    Clean all raw filings for a ticker and save to data/processed/<ticker>/.
    Returns list of saved paths.
    """
    raw_dir = DATA_RAW / ticker.upper()
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw filings found for {ticker}. Run edgar_fetcher first.")

    out_dir = DATA_PROCESSED / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for raw_file in sorted(raw_dir.glob("*.txt")):
        print(f"[{ticker}] Cleaning {raw_file.name}...")
        full_text, sections = clean_file(raw_file)

        # Save full cleaned text
        clean_path = out_dir / raw_file.name.replace(".txt", "_clean.txt")
        clean_path.write_text(full_text, encoding="utf-8")

        # Save extracted sections as separate files for targeted LLM prompting
        for section_name, section_text in sections.items():
            if len(section_text) > 200:  # skip empty/noise sections
                section_path = out_dir / raw_file.name.replace(".txt", f"_{section_name}.txt")
                section_path.write_text(section_text, encoding="utf-8")

        print(f"[{ticker}] Saved {clean_path.name} | sections found: {list(sections.keys())}")
        saved.append(clean_path)

    return saved


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "GS", "BAC"]
    for ticker in tickers:
        try:
            clean_and_save(ticker)
        except FileNotFoundError as e:
            print(f"Skipping {ticker}: {e}")
