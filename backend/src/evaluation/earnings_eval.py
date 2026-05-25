"""
Evaluation module — measures LLM-extracted signal accuracy against actual
earnings surprises and forward price returns.

For each ticker, loads aligned signals (sentiment, tone, guidance, earnings
framing) and compares them to real EPS surprise data from yfinance.

Metrics produced:
  - Sentiment vs 5d price direction accuracy
  - Tone vs return direction accuracy
  - Guidance calibration (raised → positive returns?)
  - Earnings framing accuracy (beat/miss vs actual EPS surprise)
  - Confusion matrix for tone predictions vs price direction
"""

import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL",
    "JPM", "GS", "BAC", "MS", "WFC", "BLK", "C", "USB",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "CVS",
    "WMT", "HD", "NKE", "TGT", "MCD",
    "XOM", "CVX", "COP",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aligned_signals(ticker: str) -> pd.DataFrame | None:
    """Load aligned signals parquet for a ticker. Returns None if missing."""
    path = DATA_DIR / f"{ticker.upper()}_signals_aligned.parquet"
    if not path.exists():
        print(f"[{ticker}] No aligned data found at {path.name}, skipping.")
        return None
    df = pd.read_parquet(path)
    # Ensure date columns are proper datetime
    for col in ("filing_date", "entry_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_earnings_data(ticker: str) -> pd.DataFrame | None:
    """
    Fetch actual EPS surprise data from yfinance.

    Returns a DataFrame with columns:
        report_date, eps_actual, eps_estimate, eps_surprise, surprise_pct
    Returns None if data is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required. Install with: "
            "venv/bin/pip install yfinance"
        )

    try:
        tk = yf.Ticker(ticker)

        # Try earnings_dates first (more complete, has surprise info)
        earnings_dates = None
        try:
            earnings_dates = tk.earnings_dates
        except Exception:
            pass

        if earnings_dates is not None and not earnings_dates.empty:
            df = earnings_dates.copy()
            df = df.reset_index()

            # Normalise column names — yfinance uses varying capitalisation
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if "date" in cl or "earnings" in cl.lower():
                    if "date" in cl:
                        col_map[c] = "report_date"
                if "eps estimate" in cl:
                    col_map[c] = "eps_estimate"
                if "reported eps" in cl:
                    col_map[c] = "eps_actual"
                if "surprise" in cl and "%" in cl:
                    col_map[c] = "surprise_pct"
                elif "surprise" in cl:
                    col_map[c] = "eps_surprise"

            df = df.rename(columns=col_map)

            # The index from earnings_dates is usually the date itself
            if "report_date" not in df.columns:
                # First column is typically the date index
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "report_date"})

            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            df = df.dropna(subset=["report_date"])

            # Compute surprise if we have the raw numbers but not the delta
            if "eps_actual" in df.columns and "eps_estimate" in df.columns:
                if "eps_surprise" not in df.columns:
                    df["eps_surprise"] = pd.to_numeric(
                        df["eps_actual"], errors="coerce"
                    ) - pd.to_numeric(df["eps_estimate"], errors="coerce")
                if "surprise_pct" not in df.columns:
                    est = pd.to_numeric(df["eps_estimate"], errors="coerce")
                    df["surprise_pct"] = np.where(
                        est.abs() > 0.001,
                        (pd.to_numeric(df["eps_surprise"], errors="coerce") / est.abs()) * 100,
                        np.nan,
                    )

            # Keep only rows with actual reported data (future dates have NaN)
            if "eps_actual" in df.columns:
                df = df.dropna(subset=["eps_actual"])

            return df

        # Fallback: quarterly_earnings (less info, no surprise)
        try:
            qe = tk.quarterly_earnings
        except Exception:
            qe = None

        if qe is not None and not qe.empty:
            df = qe.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df = df.rename(columns={"date": "report_date"})
            elif df.columns[0] != "report_date":
                df = df.rename(columns={df.columns[0]: "report_date"})
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            return df

        print(f"[{ticker}] No earnings data available from yfinance.")
        return None

    except Exception as e:
        print(f"[{ticker}] Error fetching earnings data: {e}")
        return None


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def match_filing_to_earnings(
    filing_date: pd.Timestamp,
    earnings_df: pd.DataFrame,
    window_days: int = 30,
) -> pd.Series | None:
    """
    Find the nearest earnings report within `window_days` of a filing date.
    Returns the matching earnings row or None.
    """
    if earnings_df is None or earnings_df.empty:
        return None
    diffs = (earnings_df["report_date"] - filing_date).abs()
    closest_idx = diffs.idxmin()
    if diffs.loc[closest_idx] <= timedelta(days=window_days):
        return earnings_df.loc[closest_idx]
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_ticker(ticker: str) -> dict | None:
    """
    Run all evaluation metrics for a single ticker.
    Returns a dict of results or None if data is insufficient.
    """
    signals_df = load_aligned_signals(ticker)
    if signals_df is None or signals_df.empty:
        return None

    earnings_df = fetch_earnings_data(ticker)

    results: dict = {"ticker": ticker, "n_filings": len(signals_df)}

    # --- 1. Sentiment vs 5d price direction accuracy ---
    if "sentiment_score" in signals_df.columns and "fwd_return_5d" in signals_df.columns:
        sub = signals_df.dropna(subset=["sentiment_score", "fwd_return_5d"])
        if len(sub) > 0:
            predicted_pos = sub["sentiment_score"] > 0
            actual_pos = sub["fwd_return_5d"] > 0
            correct = (predicted_pos == actual_pos).sum()
            results["sentiment_direction_accuracy"] = round(correct / len(sub), 4)
            results["sentiment_direction_n"] = int(len(sub))
        else:
            results["sentiment_direction_accuracy"] = None
            results["sentiment_direction_n"] = 0
    else:
        results["sentiment_direction_accuracy"] = None
        results["sentiment_direction_n"] = 0

    # --- 2. Tone accuracy ---
    if "tone" in signals_df.columns and "fwd_return_5d" in signals_df.columns:
        sub = signals_df.dropna(subset=["tone", "fwd_return_5d"])
        tone_results = {}
        for tone_val in ("optimistic", "cautious", "neutral", "defensive"):
            mask = sub["tone"].str.lower() == tone_val
            tone_sub = sub[mask]
            if len(tone_sub) > 0:
                if tone_val in ("optimistic",):
                    correct = (tone_sub["fwd_return_5d"] > 0).sum()
                elif tone_val in ("cautious", "defensive"):
                    correct = (tone_sub["fwd_return_5d"] < 0).sum()
                else:  # neutral — count small absolute returns (<1%) as correct
                    correct = (tone_sub["fwd_return_5d"].abs() < 0.01).sum()
                tone_results[tone_val] = {
                    "accuracy": round(correct / len(tone_sub), 4),
                    "n": int(len(tone_sub)),
                }
            else:
                tone_results[tone_val] = {"accuracy": None, "n": 0}
        results["tone_accuracy"] = tone_results

        # Overall tone accuracy (optimistic→positive, cautious/defensive→negative)
        directional = sub[sub["tone"].str.lower().isin(["optimistic", "cautious", "defensive"])]
        if len(directional) > 0:
            is_opt = directional["tone"].str.lower() == "optimistic"
            correct = ((is_opt & (directional["fwd_return_5d"] > 0)) |
                       (~is_opt & (directional["fwd_return_5d"] < 0))).sum()
            results["tone_directional_accuracy"] = round(correct / len(directional), 4)
            results["tone_directional_n"] = int(len(directional))
        else:
            results["tone_directional_accuracy"] = None
            results["tone_directional_n"] = 0
    else:
        results["tone_accuracy"] = {}
        results["tone_directional_accuracy"] = None
        results["tone_directional_n"] = 0

    # --- 3. Guidance calibration ---
    if "guidance_direction" in signals_df.columns and "fwd_return_5d" in signals_df.columns:
        sub = signals_df.dropna(subset=["guidance_direction", "fwd_return_5d"])
        guidance_results = {}
        for gd in ("raised", "lowered", "maintained", "none"):
            mask = sub["guidance_direction"].str.lower() == gd
            g_sub = sub[mask]
            if len(g_sub) > 0:
                if gd == "raised":
                    correct = (g_sub["fwd_return_5d"] > 0).sum()
                elif gd == "lowered":
                    correct = (g_sub["fwd_return_5d"] < 0).sum()
                else:
                    correct = int(len(g_sub))  # no directional expectation
                guidance_results[gd] = {
                    "accuracy": round(correct / len(g_sub), 4),
                    "n": int(len(g_sub)),
                    "mean_5d_return": round(float(g_sub["fwd_return_5d"].mean()), 6),
                }
            else:
                guidance_results[gd] = {"accuracy": None, "n": 0, "mean_5d_return": None}
        results["guidance_calibration"] = guidance_results
    else:
        results["guidance_calibration"] = {}

    # --- 4. Earnings framing accuracy (vs actual EPS surprise) ---
    if "earnings_framing" in signals_df.columns and earnings_df is not None:
        match_results = {"correct": 0, "incorrect": 0, "unmatched": 0}
        for _, row in signals_df.iterrows():
            framing = row.get("earnings_framing")
            if pd.isna(framing) or str(framing).lower() == "not_mentioned":
                continue
            filing_dt = row.get("filing_date")
            if pd.isna(filing_dt):
                match_results["unmatched"] += 1
                continue
            earn_row = match_filing_to_earnings(filing_dt, earnings_df)
            if earn_row is None:
                match_results["unmatched"] += 1
                continue

            # Determine actual beat/miss from EPS surprise
            surprise = earn_row.get("eps_surprise")
            if surprise is None or pd.isna(surprise):
                surprise = earn_row.get("surprise_pct")

            if surprise is None or pd.isna(surprise):
                match_results["unmatched"] += 1
                continue

            surprise_val = float(surprise)
            framing_lower = str(framing).lower()
            if framing_lower == "beat" and surprise_val > 0:
                match_results["correct"] += 1
            elif framing_lower == "miss" and surprise_val < 0:
                match_results["correct"] += 1
            elif framing_lower == "in-line" and abs(surprise_val) < 0.02:
                match_results["correct"] += 1
            else:
                match_results["incorrect"] += 1

        total_judged = match_results["correct"] + match_results["incorrect"]
        results["earnings_framing"] = {
            "correct": match_results["correct"],
            "incorrect": match_results["incorrect"],
            "unmatched": match_results["unmatched"],
            "accuracy": (
                round(match_results["correct"] / total_judged, 4)
                if total_judged > 0
                else None
            ),
            "n_judged": total_judged,
        }
    else:
        results["earnings_framing"] = {
            "correct": 0, "incorrect": 0, "unmatched": 0,
            "accuracy": None, "n_judged": 0,
        }

    # --- 5. Confusion matrix: tone vs actual 5d price direction ---
    if "tone" in signals_df.columns and "fwd_return_5d" in signals_df.columns:
        sub = signals_df.dropna(subset=["tone", "fwd_return_5d"])
        if len(sub) > 0:
            # Map tone to predicted direction
            def tone_to_direction(t: str) -> str:
                t = str(t).lower()
                if t == "optimistic":
                    return "positive"
                elif t in ("cautious", "defensive"):
                    return "negative"
                return "neutral"

            sub = sub.copy()
            sub["predicted_dir"] = sub["tone"].apply(tone_to_direction)
            sub["actual_dir"] = sub["fwd_return_5d"].apply(
                lambda r: "positive" if r > 0 else ("negative" if r < 0 else "neutral")
            )

            # Build confusion matrix as nested dict
            labels = ["positive", "neutral", "negative"]
            cm: dict[str, dict[str, int]] = {}
            for pred in labels:
                cm[pred] = {}
                for act in labels:
                    cm[pred][act] = int(
                        ((sub["predicted_dir"] == pred) & (sub["actual_dir"] == act)).sum()
                    )
            results["confusion_matrix"] = cm
        else:
            results["confusion_matrix"] = {}
    else:
        results["confusion_matrix"] = {}

    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def run_full_evaluation(tickers: list[str] | None = None) -> dict:
    """
    Evaluate all tickers and produce aggregate metrics.
    Returns a dict with per-ticker and aggregate results.
    """
    if tickers is None:
        tickers = TICKERS

    all_results: list[dict] = []
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  Evaluating {ticker}")
        print(f"{'='*60}")
        res = evaluate_ticker(ticker)
        if res is not None:
            all_results.append(res)

    # Aggregate metrics
    aggregate: dict = {}

    # Sentiment direction accuracy (weighted by number of observations)
    sent_accs = [
        (r["sentiment_direction_accuracy"], r["sentiment_direction_n"])
        for r in all_results
        if r.get("sentiment_direction_accuracy") is not None
    ]
    if sent_accs:
        total_n = sum(n for _, n in sent_accs)
        if total_n > 0:
            aggregate["sentiment_direction_accuracy"] = round(
                sum(a * n for a, n in sent_accs) / total_n, 4
            )
            aggregate["sentiment_direction_n"] = total_n

    # Tone directional accuracy
    tone_accs = [
        (r["tone_directional_accuracy"], r["tone_directional_n"])
        for r in all_results
        if r.get("tone_directional_accuracy") is not None
    ]
    if tone_accs:
        total_n = sum(n for _, n in tone_accs)
        if total_n > 0:
            aggregate["tone_directional_accuracy"] = round(
                sum(a * n for a, n in tone_accs) / total_n, 4
            )
            aggregate["tone_directional_n"] = total_n

    # Earnings framing accuracy
    ef_correct = sum(r["earnings_framing"]["correct"] for r in all_results)
    ef_incorrect = sum(r["earnings_framing"]["incorrect"] for r in all_results)
    ef_total = ef_correct + ef_incorrect
    aggregate["earnings_framing_accuracy"] = (
        round(ef_correct / ef_total, 4) if ef_total > 0 else None
    )
    aggregate["earnings_framing_n"] = ef_total

    # Guidance: mean 5d return by direction across all tickers
    guidance_agg: dict[str, list[float]] = {
        "raised": [], "lowered": [], "maintained": [], "none": [],
    }
    for r in all_results:
        for gd, vals in r.get("guidance_calibration", {}).items():
            if vals.get("mean_5d_return") is not None:
                guidance_agg.setdefault(gd, []).append(vals["mean_5d_return"])
    aggregate["guidance_mean_5d_returns"] = {
        gd: round(float(np.mean(vs)), 6) if vs else None
        for gd, vs in guidance_agg.items()
    }

    # Aggregate confusion matrix
    agg_cm: dict[str, dict[str, int]] = {
        d: {"positive": 0, "neutral": 0, "negative": 0}
        for d in ("positive", "neutral", "negative")
    }
    for r in all_results:
        cm = r.get("confusion_matrix", {})
        for pred, actuals in cm.items():
            for act, count in actuals.items():
                agg_cm.setdefault(pred, {}).setdefault(act, 0)
                agg_cm[pred][act] += count
    aggregate["confusion_matrix"] = agg_cm

    output = {
        "per_ticker": all_results,
        "aggregate": aggregate,
        "n_tickers_evaluated": len(all_results),
        "n_tickers_total": len(tickers),
    }
    return output


def save_results(results: dict, path: Path | None = None) -> Path:
    """Save evaluation results to JSON."""
    if path is None:
        path = DATA_DIR / "evaluation_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")
    return path


def print_summary(results: dict) -> None:
    """Print a clean summary table of evaluation results."""
    agg = results.get("aggregate", {})

    print("\n" + "=" * 72)
    print("  LLM SIGNAL EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"  Tickers evaluated: {results['n_tickers_evaluated']}"
        f" / {results['n_tickers_total']}"
    )
    print("-" * 72)

    # Sentiment
    sa = agg.get("sentiment_direction_accuracy")
    sn = agg.get("sentiment_direction_n", 0)
    print(f"  Sentiment vs 5d return direction : "
          f"{sa:.1%} ({sn} obs)" if sa is not None else
          "  Sentiment vs 5d return direction : N/A")

    # Tone
    ta = agg.get("tone_directional_accuracy")
    tn = agg.get("tone_directional_n", 0)
    print(f"  Tone directional accuracy        : "
          f"{ta:.1%} ({tn} obs)" if ta is not None else
          "  Tone directional accuracy        : N/A")

    # Earnings framing
    ea = agg.get("earnings_framing_accuracy")
    en = agg.get("earnings_framing_n", 0)
    print(f"  Earnings framing accuracy        : "
          f"{ea:.1%} ({en} obs)" if ea is not None else
          "  Earnings framing accuracy        : N/A")

    # Guidance calibration
    print("-" * 72)
    print("  Guidance calibration (mean 5d return):")
    gm = agg.get("guidance_mean_5d_returns", {})
    for gd in ("raised", "lowered", "maintained", "none"):
        val = gm.get(gd)
        if val is not None:
            print(f"    {gd:>12s} : {val:+.4%}")
        else:
            print(f"    {gd:>12s} : N/A")

    # Confusion matrix
    cm = agg.get("confusion_matrix", {})
    if any(sum(row.values()) > 0 for row in cm.values()):
        print("-" * 72)
        print("  Confusion matrix (predicted \u2193 \\ actual \u2192):")
        labels = ["positive", "neutral", "negative"]
        header = f"  {'':>12s}  " + "  ".join(f"{lab:>10s}" for lab in labels)
        print(header)
        for pred in labels:
            vals = "  ".join(f"{cm.get(pred, {}).get(act, 0):>10d}" for act in labels)
            print(f"  {pred:>12s}  {vals}")

    # Per-ticker summary table
    per = results.get("per_ticker", [])
    if per:
        print("-" * 72)
        print("  Per-ticker summary:")
        print(f"  {'Ticker':>6s}  {'Files':>5s}  {'Sent%':>6s}  {'Tone%':>6s}  {'EF%':>6s}")
        for r in sorted(per, key=lambda x: x["ticker"]):
            sa_t = r.get("sentiment_direction_accuracy")
            ta_t = r.get("tone_directional_accuracy")
            ef = r.get("earnings_framing", {})
            ef_a = ef.get("accuracy")
            sent_s = f"{sa_t:>5.0%}" if sa_t is not None else "  N/A"
            tone_s = f"{ta_t:>5.0%}" if ta_t is not None else "  N/A"
            ef_s = f"{ef_a:>5.0%}" if ef_a is not None else "  N/A"
            print(f"  {r['ticker']:>6s}  {r['n_filings']:>5d}  {sent_s}  {tone_s}  {ef_s}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Accept optional ticker list from command line
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        tickers = TICKERS

    # Ensure yfinance is available
    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("yfinance not found. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "yfinance"],
            stdout=subprocess.DEVNULL,
        )

    results = run_full_evaluation(tickers)
    save_results(results)
    print_summary(results)
