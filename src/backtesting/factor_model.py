"""
Factor model for return neutralization.

Removes common risk factors from raw returns so we isolate
the alpha signal from broad market and sector moves.

Two levels of neutralization:
  1. Market-adjusted:  return - beta * SPY_return
  2. Sector-adjusted:  return - sector_ETF_return (same horizon)

Without this, a signal that says "optimistic" during a bull market
looks good simply because everything went up — not because the signal works.

Factors used:
  Market:     SPY  (S&P 500)
  Technology: XLK
  Finance:    XLF
  Healthcare: XLV
  Consumer:   XLP
  Energy:     XLE
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

SECTOR_MAP = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK",
    "NVDA": "XLK", "META": "XLK", "CRM":   "XLK", "ORCL": "XLK",
    # Finance
    "JPM":  "XLF", "GS":   "XLF", "BAC":   "XLF", "MS":   "XLF",
    "WFC":  "XLF", "BLK":  "XLF", "C":     "XLF", "USB":  "XLF",
    # Healthcare
    "JNJ":  "XLV", "UNH":  "XLV", "PFE":   "XLV",
    "MRK":  "XLV", "ABBV": "XLV", "CVS":   "XLV",
    # Consumer
    "WMT":  "XLP", "HD":   "XLP", "NKE":   "XLP", "TGT": "XLP", "MCD": "XLP",
    # Energy
    "XOM":  "XLE", "CVX":  "XLE", "COP":   "XLE",
}

FACTOR_TICKERS = ["SPY", "XLK", "XLF", "XLV", "XLP", "XLE"]


def fetch_factor_prices(start: str = "2019-01-01") -> pd.DataFrame:
    """Download daily adjusted close prices for all factor ETFs."""
    print("Fetching factor ETF prices (SPY, XLK, XLF, XLV, XLP, XLE)...")
    raw = yf.download(FACTOR_TICKERS, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    prices.index.name = "date"
    path = DATA_DIR / "factor_prices.parquet"
    prices.to_parquet(path)
    print(f"Factor prices saved → {path.name} ({len(prices)} rows)")
    return prices


def load_factor_prices() -> pd.DataFrame:
    path = DATA_DIR / "factor_prices.parquet"
    if not path.exists():
        return fetch_factor_prices()
    return pd.read_parquet(path)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series,
                 window: int = 252) -> pd.Series:
    """
    Rolling beta of stock vs market using 252-day window.
    beta = cov(stock, market) / var(market)
    """
    cov = stock_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def get_factor_return(factor_prices: pd.DataFrame, factor: str,
                      date: str, horizon_days: int) -> float | None:
    """
    Get the actual forward return of a factor ETF over horizon_days
    starting from date. Matches exactly what we measure for the stock.
    """
    idx = factor_prices.index.astype(str).tolist()
    if date not in idx:
        return None
    pos = idx.index(date)
    end_pos = pos + horizon_days
    if end_pos >= len(factor_prices):
        return None
    start_price = factor_prices[factor].iloc[pos]
    end_price   = factor_prices[factor].iloc[end_pos]
    if start_price == 0:
        return None
    return float(end_price / start_price - 1)


def neutralize(df: pd.DataFrame, horizon: str) -> pd.Series:
    """
    Compute factor-neutralized returns for a column of raw forward returns.

    For each row:
      market_adj_return = raw_return - beta * SPY_forward_return
      sector_adj_return = raw_return - sector_ETF_forward_return

    We use sector-adjusted as the primary neutralized return since it
    controls for both market and sector effects simultaneously.

    Args:
        df:      Aligned signals DataFrame with ticker, entry_date, fwd_return_*
        horizon: Column name e.g. "fwd_return_5d"

    Returns:
        Series of neutralized returns, same index as df
    """
    factor_prices = load_factor_prices()
    horizon_days = {"fwd_return_1d": 1, "fwd_return_5d": 5, "fwd_return_21d": 21}[horizon]

    neutralized = []
    for _, row in df.iterrows():
        raw = row[horizon]
        if pd.isna(raw):
            neutralized.append(np.nan)
            continue

        ticker  = row["ticker"]
        entry   = row["entry_date"]
        sector  = SECTOR_MAP.get(ticker, "SPY")

        sector_ret = get_factor_return(factor_prices, sector, entry, horizon_days)
        if sector_ret is None:
            # Fall back to raw if factor data unavailable
            neutralized.append(raw)
            continue

        neutralized.append(raw - sector_ret)

    return pd.Series(neutralized, index=df.index, name=f"{horizon}_neutralized")


def add_neutralized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add market and sector neutralized return columns to aligned DataFrame."""
    df = df.copy()
    for horizon in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"]:
        if horizon in df.columns:
            df[f"{horizon}_neutralized"] = neutralize(df, horizon)
    return df


def load_aligned_with_factors(tickers: list[str]) -> pd.DataFrame:
    """Load all aligned data and add neutralized return columns."""
    frames = []
    for t in tickers:
        p = DATA_DIR / f"{t}_signals_aligned.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))

    if not frames:
        raise ValueError("No aligned data found.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Adding factor neutralization to {len(df)} filings...")
    df = add_neutralized_columns(df)
    return df


if __name__ == "__main__":
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "JPM", "GS", "BAC", "MS", "WFC", "BLK",
        "JNJ", "UNH", "PFE", "WMT", "HD", "NKE", "XOM", "CVX",
    ]
    fetch_factor_prices()
    df = load_aligned_with_factors(TICKERS)
    print(df[["ticker", "entry_date", "fwd_return_5d",
              "fwd_return_5d_neutralized"]].to_string())
