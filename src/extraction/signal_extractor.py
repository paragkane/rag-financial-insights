"""
LLM signal extractor using Claude Sonnet 4.6.
Takes cleaned MD&A / filing text and returns structured alpha signals as JSON.

Signals extracted:
  - sentiment_score:      float -1.0 (very negative) to +1.0 (very positive)
  - guidance_direction:   "raised" | "lowered" | "maintained" | "none"
  - guidance_magnitude:   float % change implied (0.0 if none)
  - guidance_confidence:  float 0.0 to 1.0
  - risk_flags:           list of active risk categories
  - earnings_framing:     "beat" | "miss" | "in-line" | "not_mentioned"
  - tone:                 "optimistic" | "cautious" | "neutral" | "defensive"
  - key_themes:           list of up to 5 dominant themes in management language
  - reasoning:            brief explanation of the scores (for eval/audit)
"""

import json
import os
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a quantitative financial analyst specializing in extracting
structured alpha signals from SEC filings and earnings call transcripts.

Your job is to read the provided text and return a precise JSON signal object.
Be conservative and data-driven. Do not infer what is not stated.
If information is absent, use the default null/neutral values specified."""

EXTRACTION_PROMPT = """Analyze the following text from a SEC 10-Q filing (Management's Discussion
and Analysis section) and extract structured trading signals.

Return ONLY valid JSON matching this exact schema — no prose, no markdown:

{{
  "sentiment_score": <float from -1.0 to 1.0, where -1=very negative, 0=neutral, 1=very positive>,
  "guidance_direction": <"raised" | "lowered" | "maintained" | "none">,
  "guidance_magnitude": <float, estimated % change in guidance, 0.0 if none>,
  "guidance_confidence": <float 0.0 to 1.0, how explicitly stated the guidance is>,
  "risk_flags": <list from: "liquidity", "competition", "macro", "regulatory", "execution", "demand", "margin">,
  "earnings_framing": <"beat" | "miss" | "in-line" | "not_mentioned">,
  "tone": <"optimistic" | "cautious" | "neutral" | "defensive">,
  "key_themes": <list of up to 5 short theme strings, e.g. ["strong iPhone demand", "services growth"]>,
  "reasoning": <1-2 sentence explanation of your sentiment_score and guidance_direction>
}}

Filing text:
---
{text}
---"""


class FilingSignal(BaseModel):
    """Pydantic model for validated signal output."""
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    guidance_direction: str = Field(pattern="^(raised|lowered|maintained|none)$")
    guidance_magnitude: float = Field(ge=0.0)
    guidance_confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str]
    earnings_framing: str = Field(pattern="^(beat|miss|in-line|not_mentioned)$")
    tone: str = Field(pattern="^(optimistic|cautious|neutral|defensive)$")
    key_themes: list[str]
    reasoning: str


def extract_signals(text: str, max_chars: int = 12000) -> FilingSignal:
    """
    Extract structured signals from filing text using Claude Sonnet 4.6.

    Args:
        text:       Cleaned filing text (MD&A section preferred)
        max_chars:  Truncate input to this length to control cost

    Returns:
        FilingSignal with validated fields
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Truncate to keep costs predictable — MD&A is the most signal-dense section
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n[... truncated for length ...]"

    prompt = EXTRACTION_PROMPT.format(text=truncated)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,  # deterministic — critical for reproducible signals
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return FilingSignal(**data)


def extract_from_file(filepath: Path) -> FilingSignal:
    """Extract signals from a cleaned filing text file."""
    text = filepath.read_text(encoding="utf-8")
    return extract_signals(text)


SIGNALS_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "_signal_cache"


def _cache_path(ticker: str, date: str) -> Path:
    SIGNALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SIGNALS_CACHE_DIR / f"{ticker.upper()}_{date}.json"


def _load_cached(ticker: str, date: str) -> dict | None:
    p = _cache_path(ticker, date)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(ticker: str, date: str, signal: dict) -> None:
    _cache_path(ticker, date).write_text(json.dumps(signal, indent=2))


def extract_ticker(ticker: str, section: str = "mda") -> list[dict]:
    """
    Extract signals for all processed filings of a ticker.
    Uses a file cache — skips Claude API call if signal already extracted.

    Args:
        ticker:  Ticker symbol e.g. "AAPL"
        section: Which section file to use ("mda", "clean", etc.)

    Returns:
        List of dicts with date + signal fields
    """
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed" / ticker.upper()
    if not processed_dir.exists():
        raise FileNotFoundError(f"No processed data for {ticker}. Run edgar_fetcher + text_cleaner first.")

    pattern = f"*_{section}.txt" if section != "clean" else "*_clean.txt"
    files = sorted(processed_dir.glob(pattern))

    if not files:
        # Fall back to full clean file
        files = sorted(processed_dir.glob("*_clean.txt"))

    results = []
    for f in files:
        date_str = f.name.split("_")[0]

        # Check cache first — skip Claude API call if already extracted
        cached = _load_cached(ticker, date_str)
        if cached:
            results.append(cached)
            print(f"[{ticker}] {date_str} → cached ✓  sentiment={cached['sentiment_score']:+.2f}")
            continue

        print(f"[{ticker}] Extracting signals from {f.name}...")
        try:
            signal = extract_from_file(f)
            row = {"ticker": ticker, "date": date_str, **signal.model_dump()}
            _save_cache(ticker, date_str, row)
            results.append(row)
            print(f"[{ticker}] {date_str} → sentiment={signal.sentiment_score:+.2f}, "
                  f"guidance={signal.guidance_direction}, tone={signal.tone}")
        except Exception as e:
            print(f"[{ticker}] Error on {f.name}: {e}")

    return results


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    signals = extract_ticker(ticker)
    print(json.dumps(signals, indent=2))
