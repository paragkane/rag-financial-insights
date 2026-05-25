"""
Historical price fetcher using yfinance.
Pulls daily OHLCV data and computes forward returns at 1, 5, and 21 day horizons.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def fetch_prices(ticker: str, start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """Download daily adjusted close prices for a ticker."""
    from datetime import date
    if end is None:
        end = str(date.today())
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")
    df = df[["Close"]].copy()
    df.columns = ["close"]
    df.index.name = "date"
    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward return columns at 1, 5, and 21 trading day horizons."""
    df = df.copy()
    for days in [1, 5, 21]:
        df[f"fwd_return_{days}d"] = df["close"].shift(-days) / df["close"] - 1
    return df


def save_prices(ticker: str, df: pd.DataFrame) -> Path:
    """Save price DataFrame to data/processed/<ticker>_prices.parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{ticker.upper()}_prices.parquet"
    df.to_parquet(path)
    return path


def load_prices(ticker: str) -> pd.DataFrame:
    """Load saved price data for a ticker."""
    path = DATA_DIR / f"{ticker.upper()}_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No price data found for {ticker}. Run fetch first.")
    return pd.read_parquet(path)


def fetch_and_save(tickers: list[str], start: str = "2019-01-01", end: str | None = None) -> dict:
    """
    Main entry point. Fetch prices + forward returns for a list of tickers.

    Returns:
        Dict mapping ticker -> saved file path
    """
    results = {}
    for ticker in tickers:
        print(f"[{ticker}] Fetching prices {start} → {end}...")
        try:
            df = fetch_prices(ticker, start=start, end=end)
            df = compute_forward_returns(df)
            path = save_prices(ticker, df)
            print(f"[{ticker}] {len(df)} rows saved → {path.name}")
            results[ticker] = str(path)
        except Exception as e:
            print(f"[{ticker}] Error: {e}")
    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "GS", "BAC"]
    fetch_and_save(tickers)
