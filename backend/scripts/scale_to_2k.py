"""Scale the multi-agent pipeline to 2,000+ filings.

Resumable: skips filings whose signals are already in _signal_cache.
Sharded:   `--shard 0/4` picks tickers where i % 4 == 0 (run on 4 boxes).
Publishes: `--publish` pushes the resulting signals to Cloudflare KV.

Examples
--------
    # Pilot — 2 tickers, 4 filings each (~8 Anthropic calls):
    python scripts/scale_to_2k.py --tickers AAPL,MSFT --limit 4

    # Full run with KV publish:
    python scripts/scale_to_2k.py --universe sp100 --limit 8 --publish

    # Shard 0 of 4:
    python scripts/scale_to_2k.py --universe sp100 --shard 0/4 --publish
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.agents.reconciler_agent import run_pipeline
from src.extraction.edgar_fetcher import fetch_tickers
from src.extraction.signal_extractor import _cache_path, _save_cache
from src.extraction.text_cleaner import clean_and_save

UNIVERSES: dict[str, list[str]] = {
    "smoke": ["AAPL", "MSFT"],
    "tech10": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "CRM", "ORCL", "ADBE", "AVGO",
    ],
    "sp100": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ", "V", "PG",
        "UNH", "HD", "MA", "BAC", "PFE", "CVX", "ABBV", "KO", "PEP", "WMT",
        "DIS", "MRK", "TMO", "CSCO", "ABT", "ACN", "MCD", "WFC", "DHR", "VZ",
        "ADBE", "CRM", "TXN", "NFLX", "NEE", "QCOM", "HON", "LIN", "PM", "UPS",
        "AMGN", "IBM", "ORCL", "LOW", "AVGO", "COST", "INTU", "SBUX", "GS", "MS",
        "BLK", "C", "USB", "BRK-B", "XOM", "COP", "T", "BMY", "MDLZ", "ISRG",
        "AXP", "TGT", "NKE", "BA", "CAT", "DE", "GE", "MMM", "RTX", "LMT",
        "GILD", "REGN", "VRTX", "ZTS", "CL", "EL", "KHC", "SYK", "BDX", "ADP",
        "BKNG", "MO", "PYPL", "SCHW", "SPGI", "ICE", "CB", "PNC", "TFC", "COF",
        "DUK", "SO", "AEP", "EXC", "F", "GM", "EBAY", "FDX", "EMR", "NSC",
    ],
}


def resolve_universe(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.universe:
        if args.universe not in UNIVERSES:
            sys.exit(f"unknown universe: {args.universe}. options: {list(UNIVERSES)}")
        return UNIVERSES[args.universe]
    sys.exit("must pass --tickers or --universe")


def apply_shard(tickers: list[str], shard_spec: str | None) -> list[str]:
    if not shard_spec:
        return tickers
    try:
        idx, total = (int(x) for x in shard_spec.split("/"))
    except Exception:
        sys.exit(f"--shard must look like '0/4', got: {shard_spec}")
    return [t for i, t in enumerate(tickers) if i % total == idx]


def process_ticker(ticker: str, limit: int) -> dict:
    print(f"\n=== {ticker} ===")
    stats = {"ticker": ticker, "fetched": 0, "cached": 0, "extracted": 0, "errors": 0}

    fetch_tickers([ticker], form_type="10-Q", limit=limit)
    cleaned_paths = clean_and_save(ticker)
    stats["fetched"] = len(cleaned_paths)

    for p in cleaned_paths:
        date = p.name.split("_")[0]
        if _cache_path(ticker, date).exists():
            stats["cached"] += 1
            print(f"  {ticker} {date}  cached, skip")
            continue
        try:
            t0 = time.time()
            ctx = run_pipeline(
                filing_id=f"{ticker}_{date}",
                ticker=ticker,
                filing_text=p.read_text(encoding="utf-8"),
            )
            signal = ctx.final_signal or {}
            row = {"ticker": ticker, "date": date, **signal}
            _save_cache(ticker, date, row)
            stats["extracted"] += 1
            dt = time.time() - t0
            print(
                f"  {ticker} {date}  sent={signal.get('sentiment_score', 0):+.2f}"
                f"  rev={ctx.revisions}  ({dt:.1f}s)"
            )
        except Exception as e:
            stats["errors"] += 1
            print(f"  {ticker} {date}  ERROR: {e}")

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", help="comma-separated, e.g. AAPL,MSFT")
    p.add_argument("--universe", help=f"one of: {','.join(UNIVERSES)}")
    p.add_argument("--limit", type=int, default=8, help="filings per ticker")
    p.add_argument("--shard", help="N/total — e.g. 0/4 picks tickers where i%%4==0")
    p.add_argument("--publish", action="store_true", help="push cached signals to KV at end")
    args = p.parse_args()

    tickers = apply_shard(resolve_universe(args), args.shard)
    print(f"Processing {len(tickers)} tickers, limit={args.limit}/each")

    totals = {"fetched": 0, "cached": 0, "extracted": 0, "errors": 0}
    for ticker in tickers:
        s = process_ticker(ticker, args.limit)
        for k in totals:
            totals[k] += s[k]

    print("\n=== SUMMARY ===")
    print(json.dumps(totals, indent=2))

    if args.publish:
        from src.publishing.kv_publisher import publish_from_cache
        publish_from_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
