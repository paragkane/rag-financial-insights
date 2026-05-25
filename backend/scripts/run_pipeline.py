"""
Full pipeline runner.
Fetch filings → clean → extract signals → align with prices → backtest.

Usage:
    python scripts/run_pipeline.py
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.extraction.edgar_fetcher import fetch_tickers
from src.extraction.price_fetcher import fetch_and_save
from src.extraction.text_cleaner import clean_and_save
from src.extraction.signal_extractor import extract_ticker
from src.backtesting.signal_aligner import align_and_save
from src.backtesting.engine import run_full_backtest, print_summary

TICKERS = [
    # Tech (8)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL",
    # Finance (8)
    "JPM", "GS", "BAC", "MS", "WFC", "BLK", "C", "USB",
    # Healthcare (6)
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "CVS",
    # Consumer (5)
    "WMT", "HD", "NKE", "TGT", "MCD",
    # Energy (3)
    "XOM", "CVX", "COP",
]
FILINGS_PER_TICKER = 16  # 4 years of quarterly filings = ~480 filings total
FORM_TYPE = "10-Q"

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def run():
    print("=" * 60)
    print("PIPELINE START")
    print(f"Tickers: {TICKERS}")
    print(f"Filings per ticker: {FILINGS_PER_TICKER}")
    print("=" * 60)

    # ── Step 1: Fetch raw filings from EDGAR ──────────────────────
    print("\n[Step 1/5] Fetching SEC filings from EDGAR...")
    fetch_tickers(TICKERS, form_type=FORM_TYPE, limit=FILINGS_PER_TICKER)

    # ── Step 2: Fetch historical prices ───────────────────────────
    print("\n[Step 2/5] Fetching historical prices...")
    fetch_and_save(TICKERS)

    # ── Step 3: Clean filings ──────────────────────────────────────
    print("\n[Step 3/5] Cleaning filings...")
    for ticker in TICKERS:
        try:
            clean_and_save(ticker)
        except FileNotFoundError as e:
            print(f"  Skipping {ticker}: {e}")

    # ── Step 4: Extract signals with Claude ───────────────────────
    print("\n[Step 4/5] Extracting signals with Claude Sonnet 4.6...")
    all_signals = {}
    for ticker in TICKERS:
        print(f"\n  [{ticker}]")
        try:
            signals = extract_ticker(ticker, section="mda")
            if not signals:
                signals = extract_ticker(ticker, section="clean")
            all_signals[ticker] = signals
            time.sleep(0.5)  # be polite to the API
        except Exception as e:
            print(f"  [{ticker}] Signal extraction failed: {e}")
            all_signals[ticker] = []

    # Save raw signals for reference
    signals_path = DATA_PROCESSED / "raw_signals.json"
    signals_path.write_text(json.dumps(all_signals, indent=2))
    print(f"\n  Raw signals saved → {signals_path.name}")

    # ── Step 5: Align signals with prices ─────────────────────────
    print("\n[Step 5/5] Aligning signals with price data...")
    for ticker, signals in all_signals.items():
        if signals:
            align_and_save(signals, ticker)
        else:
            print(f"  [{ticker}] No signals to align.")

    # ── Backtest ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    tickers_with_data = [
        t for t in TICKERS
        if (DATA_PROCESSED / f"{t}_signals_aligned.parquet").exists()
    ]

    for horizon in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"]:
        results = run_full_backtest(tickers_with_data, horizon)
        print_summary(results)

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running.")
    run()
