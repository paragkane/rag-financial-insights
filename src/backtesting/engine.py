"""
Backtesting engine.

Takes the aligned signals+returns dataset and answers one question:
"Do the LLM-extracted signals have statistically significant predictive
power over future stock returns?"

Metrics computed per signal:
  - hit_rate:     % of trades where signal direction matched price direction
  - mean_return:  average forward return when signal was active
  - sharpe_ratio: mean_return / std_return (annualized)
  - t_stat:       statistical significance of mean_return vs zero
  - p_value:      probability this result is random noise (want < 0.05)
  - n:            number of observations
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "backtest_results"

ANNUALIZATION = {
    "fwd_return_1d": 252,
    "fwd_return_5d": 52,
    "fwd_return_21d": 12,
}


def sharpe(returns: pd.Series, horizon: str) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free rate)."""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    periods_per_year = ANNUALIZATION.get(horizon, 252)
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def hit_rate(returns: pd.Series, signal_direction: pd.Series) -> float:
    """
    % of times the signal correctly predicted price direction.
    signal_direction: +1 (bullish) or -1 (bearish)
    """
    correct = ((returns > 0) & (signal_direction > 0)) | \
              ((returns < 0) & (signal_direction < 0))
    return float(correct.mean())


def ttest(returns: pd.Series) -> tuple[float, float]:
    """One-sample t-test: is mean return significantly different from zero?"""
    if len(returns) < 3:
        return 0.0, 1.0
    t, p = stats.ttest_1samp(returns.dropna(), 0)
    return float(t), float(p)


def run_sentiment_backtest(df: pd.DataFrame, horizon: str = "fwd_return_5d") -> dict:
    """
    Test: does sentiment score correlate with forward returns?

    Uses two approaches:
    1. Continuous: Pearson correlation + linear regression (more statistically powerful)
    2. Bucketed: high (>0.1) vs low (<=0.0) sentiment comparison
    """
    df = df.dropna(subset=[horizon, "sentiment_score"]).copy()
    if len(df) < 3:
        return {}

    returns = df[horizon]
    scores = df["sentiment_score"]

    # ── Continuous correlation ─────────────────────────────────────
    corr, corr_p = stats.pearsonr(scores, returns)

    # Linear regression: return = alpha + beta * sentiment
    slope, intercept, r_value, reg_p, std_err = stats.linregress(scores, returns)

    # ── Bucketed: above-median vs below-median sentiment ──────────
    median_score = scores.median()
    high = df[scores > median_score][horizon]
    low  = df[scores <= median_score][horizon]

    signal_dir = (scores > median_score).astype(int) * 2 - 1  # +1 or -1
    t, p = ttest(returns)

    # Hit rate using median split
    hr = hit_rate(returns, signal_dir)

    return {
        "signal": "sentiment_score",
        "horizon": horizon,
        "n_total": len(df),
        "pearson_correlation": round(float(corr), 4),
        "correlation_p_value": round(float(corr_p), 4),
        "regression_slope": round(float(slope), 6),
        "regression_p_value": round(float(reg_p), 4),
        "high_sentiment_mean_return": round(float(high.mean()), 6) if len(high) > 0 else None,
        "low_sentiment_mean_return":  round(float(low.mean()), 6) if len(low) > 0 else None,
        "overall_sharpe": round(sharpe(returns, horizon), 4),
        "hit_rate": round(hr, 4),
        "t_stat": round(t, 4),
        "p_value": round(p, 4),
        "significant": bool(corr_p < 0.05 or reg_p < 0.05 or p < 0.05),
    }


def run_tone_backtest(df: pd.DataFrame, horizon: str = "fwd_return_5d") -> dict:
    """
    Test: does management tone predict returns?
    optimistic → long, defensive/cautious → short
    """
    df = df.dropna(subset=[horizon]).copy()
    if df.empty:
        return {}

    tone_map = {"optimistic": 1, "neutral": 0, "cautious": -1, "defensive": -1}
    df["tone_signal"] = df["tone"].map(tone_map).fillna(0)

    directional = df[df["tone_signal"] != 0]
    if directional.empty:
        return {}

    returns = directional[horizon]
    signal_dir = directional["tone_signal"]
    t, p = ttest(returns)

    return {
        "signal": "tone",
        "horizon": horizon,
        "n_total": len(df),
        "tone_counts": df["tone"].value_counts().to_dict(),
        "optimistic_mean_return": round(float(df[df["tone"] == "optimistic"][horizon].mean()), 6) if len(df[df["tone"] == "optimistic"]) > 0 else None,
        "cautious_mean_return": round(float(df[df["tone"].isin(["cautious", "defensive"])][horizon].mean()), 6) if len(df[df["tone"].isin(["cautious", "defensive"])]) > 0 else None,
        "overall_sharpe": round(sharpe(returns, horizon), 4),
        "hit_rate": round(hit_rate(returns, signal_dir), 4),
        "t_stat": round(t, 4),
        "p_value": round(p, 4),
        "significant": p < 0.05,
    }


def run_guidance_backtest(df: pd.DataFrame, horizon: str = "fwd_return_5d") -> dict:
    """
    Test: does guidance revision direction predict returns?
    raised → long, lowered → short
    """
    df = df.dropna(subset=[horizon]).copy()
    if df.empty:
        return {}

    guidance_map = {"raised": 1, "maintained": 0, "lowered": -1, "none": 0}
    df["guidance_signal"] = df["guidance_direction"].map(guidance_map).fillna(0)

    directional = df[df["guidance_signal"] != 0]
    if directional.empty:
        return {"signal": "guidance_direction", "horizon": horizon, "note": "no raised/lowered guidance in dataset"}

    returns = directional[horizon]
    signal_dir = directional["guidance_signal"]
    t, p = ttest(returns)

    return {
        "signal": "guidance_direction",
        "horizon": horizon,
        "n_total": len(df),
        "guidance_counts": df["guidance_direction"].value_counts().to_dict(),
        "raised_mean_return": round(float(df[df["guidance_direction"] == "raised"][horizon].mean()), 6) if len(df[df["guidance_direction"] == "raised"]) > 0 else None,
        "lowered_mean_return": round(float(df[df["guidance_direction"] == "lowered"][horizon].mean()), 6) if len(df[df["guidance_direction"] == "lowered"]) > 0 else None,
        "overall_sharpe": round(sharpe(returns, horizon), 4),
        "hit_rate": round(hit_rate(returns, signal_dir), 4),
        "t_stat": round(t, 4),
        "p_value": round(p, 4),
        "significant": p < 0.05,
    }


def run_full_backtest(tickers: list[str], horizon: str = "fwd_return_5d") -> dict:
    """
    Run all signal backtests across a list of tickers.
    Pools all aligned data together for statistical power.
    """
    frames = []
    for ticker in tickers:
        path = DATA_DIR / f"{ticker.upper()}_signals_aligned.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            print(f"[{ticker}] No aligned data found, skipping.")

    if not frames:
        raise ValueError("No aligned data found for any ticker.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nPooled dataset: {len(df)} filings across {df['ticker'].nunique()} tickers\n")

    # Add factor-neutralized returns if available
    neutralized_horizon = f"{horizon}_neutralized"
    if neutralized_horizon not in df.columns:
        try:
            from src.backtesting.factor_model import add_neutralized_columns
            df = add_neutralized_columns(df)
            print("Factor neutralization applied.\n")
        except Exception as e:
            print(f"Factor neutralization skipped: {e}\n")

    # Run on raw returns
    raw_results = {
        "sentiment": run_sentiment_backtest(df, horizon),
        "tone":      run_tone_backtest(df, horizon),
        "guidance":  run_guidance_backtest(df, horizon),
    }

    # Run on neutralized returns if available
    neutralized_results = {}
    if neutralized_horizon in df.columns:
        neutralized_results = {
            "sentiment": run_sentiment_backtest(df, neutralized_horizon),
            "tone":      run_tone_backtest(df, neutralized_horizon),
            "guidance":  run_guidance_backtest(df, neutralized_horizon),
        }

    results = {
        "n_filings": len(df),
        "n_tickers": df["ticker"].nunique(),
        "tickers": df["ticker"].unique().tolist(),
        "horizon": horizon,
        **raw_results,
        "neutralized": neutralized_results,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"backtest_{horizon}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out_path}")

    return results


def print_summary(results: dict) -> None:
    """Print a clean readable summary of backtest results."""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS — horizon: {results['horizon']}")
    print(f"Dataset: {results['n_filings']} filings, {results['n_tickers']} tickers")
    print(f"{'='*60}\n")

    def _print_signal(r: dict, label: str = "") -> None:
        if not r:
            return
        sig = "✅ SIGNIFICANT" if r.get("significant") else "❌ not significant"
        prefix = f"  [{label}] " if label else "  "
        print(f"Signal: {r.get('signal')}  {label}")
        print(f"{prefix}Hit rate:         {r.get('hit_rate', 'n/a')}")
        print(f"{prefix}Sharpe:           {r.get('overall_sharpe', 'n/a')}")
        print(f"{prefix}Pearson corr:     {r.get('pearson_correlation', 'n/a')}  (p={r.get('correlation_p_value', 'n/a')})")
        print(f"{prefix}Regression slope: {r.get('regression_slope', 'n/a')}  (p={r.get('regression_p_value', 'n/a')})")
        print(f"{prefix}t-stat:           {r.get('t_stat', 'n/a')}  p={r.get('p_value', 'n/a')}  {sig}")

    for key in ["sentiment", "tone", "guidance"]:
        raw = results.get(key, {})
        neu = results.get("neutralized", {}).get(key, {})
        if not raw:
            continue
        _print_signal(raw, "raw")
        if neu:
            _print_signal(neu, "sector-neutralized")
        print()


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    for horizon in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"]:
        results = run_full_backtest(tickers, horizon)
        print_summary(results)
