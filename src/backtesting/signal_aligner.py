"""
Signal aligner — joins extracted LLM signals with forward price returns.

For each filing signal, finds the next trading day's open (T+1) as the
entry point, then looks up 1d, 5d, and 21d forward returns from that date.

This gives us the core dataset for backtesting:
  signal date → next trading day → forward returns
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def next_trading_day(date: str, price_df: pd.DataFrame) -> str | None:
    """Return the first trading day on or after the given date string (YYYY-MM-DD)."""
    idx = pd.Timestamp(date)
    future = price_df.index[price_df.index >= idx]
    return str(future[0].date()) if len(future) > 0 else None


def align(signals: list[dict], ticker: str) -> pd.DataFrame:
    """
    Join a list of signal dicts with forward returns for a ticker.

    Args:
        signals:  Output from signal_extractor.extract_ticker()
        ticker:   Ticker symbol to load price data for

    Returns:
        DataFrame with one row per filing, columns:
          date, entry_date, close, fwd_return_1d/5d/21d, + all signal fields
    """
    from src.extraction.price_fetcher import load_prices

    prices = load_prices(ticker)
    rows = []

    for sig in signals:
        filing_date = sig["date"]
        entry_date = next_trading_day(filing_date, prices)

        if entry_date is None:
            print(f"[{ticker}] No trading day found after {filing_date}, skipping.")
            continue

        if entry_date not in prices.index.astype(str).tolist():
            print(f"[{ticker}] Entry date {entry_date} not in price index, skipping.")
            continue

        price_row = prices.loc[entry_date]
        row = {
            "ticker": ticker,
            "filing_date": filing_date,
            "entry_date": entry_date,
            "entry_price": round(float(price_row["close"]), 4),
            "fwd_return_1d": round(float(price_row["fwd_return_1d"]), 6) if pd.notna(price_row["fwd_return_1d"]) else None,
            "fwd_return_5d": round(float(price_row["fwd_return_5d"]), 6) if pd.notna(price_row["fwd_return_5d"]) else None,
            "fwd_return_21d": round(float(price_row["fwd_return_21d"]), 6) if pd.notna(price_row["fwd_return_21d"]) else None,
            **{k: v for k, v in sig.items() if k not in ("ticker", "date")},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_aligned(df: pd.DataFrame, ticker: str) -> Path:
    """Save aligned dataset to data/processed/<ticker>_signals_aligned.parquet"""
    path = DATA_DIR / f"{ticker.upper()}_signals_aligned.parquet"
    df.to_parquet(path, index=False)
    return path


def load_aligned(ticker: str) -> pd.DataFrame:
    """Load the aligned signals+returns dataset for a ticker."""
    path = DATA_DIR / f"{ticker.upper()}_signals_aligned.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No aligned data for {ticker}. Run align() first.")
    return pd.read_parquet(path)


def align_and_save(signals: list[dict], ticker: str) -> pd.DataFrame:
    """Align signals with prices and save. Returns the aligned DataFrame."""
    df = align(signals, ticker)
    if df.empty:
        print(f"[{ticker}] No rows aligned.")
        return df
    path = save_aligned(df, ticker)
    print(f"[{ticker}] Aligned {len(df)} filings → {path.name}")
    print(df[["filing_date", "entry_date", "entry_price",
              "fwd_return_1d", "fwd_return_5d", "fwd_return_21d",
              "sentiment_score", "guidance_direction", "tone"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    from src.extraction.signal_extractor import extract_ticker
    from src.extraction.price_fetcher import fetch_and_save

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    fetch_and_save([ticker])
    signals = extract_ticker(ticker)
    align_and_save(signals, ticker)
