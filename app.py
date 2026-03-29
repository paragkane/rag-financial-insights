"""
LLM Financial Signal Extraction & Backtesting Dashboard
========================================================
Interactive Streamlit dashboard for exploring LLM-extracted signals
from SEC 10-Q filings and their backtesting results.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Financial Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
BACKTEST_DIR = DATA_DIR / "backtest_results"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL",
    "JPM", "GS", "BAC", "MS", "WFC", "BLK", "C", "USB",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "CVS",
    "WMT", "HD", "NKE", "TGT", "MCD",
    "XOM", "CVX", "COP",
]

SECTOR_MAP = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL"],
    "Finance": ["JPM", "GS", "BAC", "MS", "WFC", "BLK", "C", "USB"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "CVS"],
    "Consumer": ["WMT", "HD", "NKE", "TGT", "MCD"],
    "Energy": ["XOM", "CVX", "COP"],
}

TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

HORIZONS = ["1d", "5d", "21d"]

# Consistent color palette
SECTOR_COLORS = {
    "Tech": "#636EFA",
    "Finance": "#EF553B",
    "Healthcare": "#00CC96",
    "Consumer": "#AB63FA",
    "Energy": "#FFA15A",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
    }
    h1 {
        color: #1f2937;
    }
    h2, h3 {
        color: #374151;
    }
    .stMetric label {
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading signal data...")
def load_all_signals() -> pd.DataFrame:
    """Load and concatenate all ticker signal parquet files."""
    frames = []
    for ticker in TICKERS:
        fpath = DATA_DIR / f"{ticker}_signals_aligned.parquet"
        try:
            df = pd.read_parquet(fpath)
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Ensure filing_date is datetime
    for col in ["filing_date", "entry_date"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    # Add sector column
    combined["sector"] = combined["ticker"].map(TICKER_TO_SECTOR)
    return combined


@st.cache_data(show_spinner="Loading backtest results...")
def load_backtest(horizon: str) -> dict:
    """Load a single backtest JSON."""
    fpath = BACKTEST_DIR / f"backtest_fwd_return_{horizon}.json"
    try:
        with open(fpath) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner="Loading all backtests...")
def load_all_backtests() -> dict[str, dict]:
    results = {}
    for h in HORIZONS:
        results[h] = load_backtest(h)
    return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def fmt_pct(v, decimals=2):
    if pd.isna(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_float(v, decimals=4):
    if pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def significance_badge(sig: bool) -> str:
    return "Yes" if sig else "No"


def build_backtest_comparison(backtests: dict[str, dict]) -> pd.DataFrame:
    """Build a summary comparison table across horizons."""
    rows = []
    for h, bt in backtests.items():
        if not bt:
            continue
        label = f"{h} horizon"
        # Raw sentiment
        s = bt.get("sentiment", {})
        rows.append({
            "Horizon": label,
            "Type": "Raw Sentiment",
            "Pearson r": s.get("pearson_correlation"),
            "r p-value": s.get("correlation_p_value"),
            "Slope": s.get("regression_slope"),
            "High Sent. Return": s.get("high_sentiment_mean_return"),
            "Low Sent. Return": s.get("low_sentiment_mean_return"),
            "Sharpe": s.get("overall_sharpe"),
            "Hit Rate": s.get("hit_rate"),
            "Significant": s.get("significant"),
        })
        # Neutralized sentiment
        ns = bt.get("neutralized", {}).get("sentiment", {})
        if ns:
            rows.append({
                "Horizon": label,
                "Type": "Neutralized Sentiment",
                "Pearson r": ns.get("pearson_correlation"),
                "r p-value": ns.get("correlation_p_value"),
                "Slope": ns.get("regression_slope"),
                "High Sent. Return": ns.get("high_sentiment_mean_return"),
                "Low Sent. Return": ns.get("low_sentiment_mean_return"),
                "Sharpe": ns.get("overall_sharpe"),
                "Hit Rate": ns.get("hit_rate"),
                "Significant": ns.get("significant"),
            })
        # Tone
        t = bt.get("tone", {})
        rows.append({
            "Horizon": label,
            "Type": "Raw Tone",
            "Pearson r": None,
            "r p-value": None,
            "Slope": None,
            "High Sent. Return": t.get("optimistic_mean_return"),
            "Low Sent. Return": t.get("cautious_mean_return"),
            "Sharpe": t.get("overall_sharpe"),
            "Hit Rate": t.get("hit_rate"),
            "Significant": t.get("significant"),
        })
        # Neutralized tone
        nt = bt.get("neutralized", {}).get("tone", {})
        if nt:
            rows.append({
                "Horizon": label,
                "Type": "Neutralized Tone",
                "Pearson r": None,
                "r p-value": None,
                "Slope": None,
                "High Sent. Return": nt.get("optimistic_mean_return"),
                "Low Sent. Return": nt.get("cautious_mean_return"),
                "Sharpe": nt.get("overall_sharpe"),
                "Hit Rate": nt.get("hit_rate"),
                "Significant": nt.get("significant"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("LLM Financial Signal Extraction & Backtesting")
st.markdown(
    "_Extracting forward-looking signals from SEC 10-Q filings using Claude, "
    "then backtesting their predictive power across 30 major US equities._"
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df_all = load_all_signals()
backtests = load_all_backtests()

if df_all.empty:
    st.error(
        "No signal data found. Ensure parquet files exist in "
        f"`{DATA_DIR}` with the naming pattern `{{TICKER}}_signals_aligned.parquet`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Signal Explorer", "Backtest Results", "Sector Analysis"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption(
    f"Data: {len(df_all)} filings across {df_all['ticker'].nunique()} tickers"
)
if "filing_date" in df_all.columns:
    min_d = df_all["filing_date"].min()
    max_d = df_all["filing_date"].max()
    if pd.notna(min_d) and pd.notna(max_d):
        st.sidebar.caption(
            f"Date range: {min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"
        )

# ===================================================================
# SECTION 1 -- Overview
# ===================================================================

if section == "Overview":
    st.header("Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Filings", f"{len(df_all):,}")
    col2.metric("Tickers", df_all["ticker"].nunique())
    if "filing_date" in df_all.columns:
        col3.metric("Earliest Filing", df_all["filing_date"].min().strftime("%Y-%m-%d"))
        col4.metric("Latest Filing", df_all["filing_date"].max().strftime("%Y-%m-%d"))

    st.subheader("Filings per Ticker")
    counts = df_all["ticker"].value_counts().reindex(TICKERS).fillna(0).astype(int)
    fig_counts = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Ticker", "y": "Number of Filings"},
        color=[TICKER_TO_SECTOR.get(t, "Other") for t in counts.index],
        color_discrete_map=SECTOR_COLORS,
    )
    fig_counts.update_layout(
        showlegend=True,
        legend_title_text="Sector",
        height=380,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    # Key backtest results summary
    st.subheader("Key Backtest Results")

    comp_df = build_backtest_comparison(backtests)
    if not comp_df.empty:
        # Style the dataframe
        display_df = comp_df.copy()
        for col in ["Pearson r", "Slope", "Sharpe"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: fmt_float(v) if pd.notna(v) else "-"
                )
        for col in ["r p-value", "High Sent. Return", "Low Sent. Return", "Hit Rate"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: fmt_float(v) if pd.notna(v) else "-"
                )
        if "Significant" in display_df.columns:
            display_df["Significant"] = display_df["Significant"].apply(
                lambda v: significance_badge(v) if pd.notna(v) else "-"
            )

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No backtest results available.")

    # Sentiment distribution
    if "sentiment_score" in df_all.columns:
        st.subheader("Sentiment Score Distribution (All Filings)")
        fig_hist = px.histogram(
            df_all,
            x="sentiment_score",
            nbins=30,
            color="sector",
            color_discrete_map=SECTOR_COLORS,
            labels={"sentiment_score": "Sentiment Score", "sector": "Sector"},
            opacity=0.8,
            barmode="overlay",
        )
        fig_hist.update_layout(height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)


# ===================================================================
# SECTION 2 -- Signal Explorer
# ===================================================================

elif section == "Signal Explorer":
    st.header("Signal Explorer")

    available_tickers = sorted(df_all["ticker"].unique())
    selected_ticker = st.selectbox("Select Ticker", available_tickers, index=0)
    df_ticker = df_all[df_all["ticker"] == selected_ticker].copy()

    if df_ticker.empty:
        st.warning(f"No data available for {selected_ticker}.")
        st.stop()

    df_ticker = df_ticker.sort_values("filing_date")
    sector = TICKER_TO_SECTOR.get(selected_ticker, "Unknown")
    st.caption(f"Sector: **{sector}** | Filings: **{len(df_ticker)}**")

    # -- Sentiment over time --
    if "sentiment_score" in df_ticker.columns and "filing_date" in df_ticker.columns:
        st.subheader(f"Sentiment Score Over Time -- {selected_ticker}")
        fig_sent = go.Figure()
        fig_sent.add_trace(
            go.Scatter(
                x=df_ticker["filing_date"],
                y=df_ticker["sentiment_score"],
                mode="lines+markers",
                name="Sentiment Score",
                line=dict(color=SECTOR_COLORS.get(sector, "#636EFA"), width=2.5),
                marker=dict(size=8),
            )
        )
        fig_sent.update_layout(
            xaxis_title="Filing Date",
            yaxis_title="Sentiment Score",
            height=380,
            margin=dict(t=20, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # -- Tone distribution --
    if "tone" in df_ticker.columns:
        st.subheader(f"Tone Distribution -- {selected_ticker}")
        tone_counts = df_ticker["tone"].value_counts()
        fig_tone = px.bar(
            x=tone_counts.index,
            y=tone_counts.values,
            labels={"x": "Tone", "y": "Count"},
            color=tone_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_tone.update_layout(
            showlegend=False,
            height=320,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_tone, use_container_width=True)

    # -- Forward returns comparison --
    fwd_cols = [c for c in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"] if c in df_ticker.columns]
    if fwd_cols and "filing_date" in df_ticker.columns:
        st.subheader(f"Forward Returns by Horizon -- {selected_ticker}")
        fig_fwd = go.Figure()
        color_map = {"fwd_return_1d": "#636EFA", "fwd_return_5d": "#EF553B", "fwd_return_21d": "#00CC96"}
        for col in fwd_cols:
            label = col.replace("fwd_return_", "").upper()
            fig_fwd.add_trace(
                go.Bar(
                    x=df_ticker["filing_date"],
                    y=df_ticker[col],
                    name=label,
                    marker_color=color_map.get(col, "#636EFA"),
                    opacity=0.75,
                )
            )
        fig_fwd.update_layout(
            barmode="group",
            xaxis_title="Filing Date",
            yaxis_title="Forward Return",
            yaxis_tickformat=".2%",
            height=380,
            margin=dict(t=20, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fwd, use_container_width=True)

    # -- Full filings table --
    st.subheader(f"All Filings -- {selected_ticker}")
    display_cols = [
        c
        for c in [
            "filing_date",
            "entry_date",
            "entry_price",
            "sentiment_score",
            "guidance_direction",
            "guidance_magnitude",
            "guidance_confidence",
            "risk_flags",
            "earnings_framing",
            "tone",
            "key_themes",
            "reasoning",
            "fwd_return_1d",
            "fwd_return_5d",
            "fwd_return_21d",
        ]
        if c in df_ticker.columns
    ]
    st.dataframe(
        df_ticker[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


# ===================================================================
# SECTION 3 -- Backtest Results
# ===================================================================

elif section == "Backtest Results":
    st.header("Backtest Results")

    if all(not bt for bt in backtests.values()):
        st.warning("No backtest result files found.")
        st.stop()

    # -- Comparison table --
    st.subheader("Results Comparison Across Horizons")
    comp_df = build_backtest_comparison(backtests)
    if not comp_df.empty:
        styled = comp_df.copy()
        for col in ["Pearson r", "Slope", "Sharpe"]:
            if col in styled.columns:
                styled[col] = styled[col].apply(lambda v: fmt_float(v) if pd.notna(v) else "-")
        for col in ["r p-value", "High Sent. Return", "Low Sent. Return", "Hit Rate"]:
            if col in styled.columns:
                styled[col] = styled[col].apply(lambda v: fmt_float(v) if pd.notna(v) else "-")
        if "Significant" in styled.columns:
            styled["Significant"] = styled["Significant"].apply(
                lambda v: significance_badge(v) if pd.notna(v) else "-"
            )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # -- Scatter: sentiment vs forward return --
    st.subheader("Sentiment Score vs Forward Return")

    horizon_tabs = st.tabs([f"{h} Horizon" for h in HORIZONS])
    for i, h in enumerate(HORIZONS):
        with horizon_tabs[i]:
            fwd_col = f"fwd_return_{h}"
            if fwd_col not in df_all.columns or "sentiment_score" not in df_all.columns:
                st.info(f"Missing data for {h} horizon scatter plot.")
                continue

            plot_df = df_all.dropna(subset=["sentiment_score", fwd_col]).copy()
            if plot_df.empty:
                st.info(f"No valid data for {h} horizon.")
                continue

            fig_scatter = px.scatter(
                plot_df,
                x="sentiment_score",
                y=fwd_col,
                color="sector",
                color_discrete_map=SECTOR_COLORS,
                hover_data=["ticker", "filing_date"],
                labels={
                    "sentiment_score": "Sentiment Score",
                    fwd_col: f"Forward Return ({h})",
                    "sector": "Sector",
                },
                opacity=0.7,
            )

            # Add OLS trendline manually via numpy
            mask = plot_df["sentiment_score"].notna() & plot_df[fwd_col].notna()
            x_vals = plot_df.loc[mask, "sentiment_score"].values
            y_vals = plot_df.loc[mask, fwd_col].values
            if len(x_vals) > 2:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = np.polyval(coeffs, x_line)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"OLS (slope={coeffs[0]:.5f})",
                        line=dict(color="rgba(0,0,0,0.6)", width=2, dash="dash"),
                    )
                )

            # Highlight significance
            bt = backtests.get(h, {})
            sent_results = bt.get("sentiment", {})
            sig = sent_results.get("significant", False)
            corr = sent_results.get("pearson_correlation", None)
            pval = sent_results.get("correlation_p_value", None)
            annotation_text = f"r = {corr:.4f}, p = {pval:.4f}" if corr is not None else ""
            if sig:
                annotation_text += " (SIGNIFICANT)"

            fig_scatter.update_layout(
                height=480,
                margin=dict(t=40, b=40),
                yaxis_tickformat=".2%",
                annotations=[
                    dict(
                        text=annotation_text,
                        xref="paper",
                        yref="paper",
                        x=0.01,
                        y=0.99,
                        showarrow=False,
                        font=dict(size=13, color="red" if sig else "#555"),
                    )
                ],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # -- Sharpe ratio comparison --
    st.subheader("Sharpe Ratio Comparison")
    sharpe_rows = []
    for h in HORIZONS:
        bt = backtests.get(h, {})
        for label, key_path in [
            ("Raw Sentiment", ("sentiment", "overall_sharpe")),
            ("Raw Tone", ("tone", "overall_sharpe")),
            ("Neutralized Sentiment", ("neutralized", "sentiment", "overall_sharpe")),
            ("Neutralized Tone", ("neutralized", "tone", "overall_sharpe")),
        ]:
            obj = bt
            for k in key_path:
                if isinstance(obj, dict):
                    obj = obj.get(k)
                else:
                    obj = None
                    break
            if obj is not None:
                sharpe_rows.append({"Horizon": h, "Signal": label, "Sharpe": obj})

    if sharpe_rows:
        sharpe_df = pd.DataFrame(sharpe_rows)
        fig_sharpe = px.bar(
            sharpe_df,
            x="Horizon",
            y="Sharpe",
            color="Signal",
            barmode="group",
            labels={"Sharpe": "Sharpe Ratio", "Horizon": "Forward Return Horizon"},
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sharpe.update_layout(height=400, margin=dict(t=20, b=40))
        st.plotly_chart(fig_sharpe, use_container_width=True)


# ===================================================================
# SECTION 4 -- Sector Analysis
# ===================================================================

elif section == "Sector Analysis":
    st.header("Sector Analysis")

    # -- Average sentiment by sector --
    if "sentiment_score" in df_all.columns:
        st.subheader("Average Sentiment Score by Sector")
        sector_avg = (
            df_all.groupby("sector")["sentiment_score"]
            .agg(["mean", "std", "count"])
            .reindex(SECTOR_MAP.keys())
        )
        sector_avg.columns = ["Mean Sentiment", "Std Dev", "Filings"]

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig_bar = px.bar(
                x=sector_avg.index,
                y=sector_avg["Mean Sentiment"],
                error_y=sector_avg["Std Dev"],
                labels={"x": "Sector", "y": "Mean Sentiment Score"},
                color=sector_avg.index,
                color_discrete_map=SECTOR_COLORS,
            )
            fig_bar.update_layout(showlegend=False, height=380, margin=dict(t=20, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_table:
            display_sector = sector_avg.copy()
            display_sector["Mean Sentiment"] = display_sector["Mean Sentiment"].apply(
                lambda v: f"{v:.4f}"
            )
            display_sector["Std Dev"] = display_sector["Std Dev"].apply(
                lambda v: f"{v:.4f}"
            )
            display_sector["Filings"] = display_sector["Filings"].astype(int)
            st.dataframe(display_sector, use_container_width=True)

    # -- Sentiment over time by sector --
    if "sentiment_score" in df_all.columns and "filing_date" in df_all.columns:
        st.subheader("Average Sentiment Over Time by Sector")
        time_sector = (
            df_all.assign(
                quarter=df_all["filing_date"].dt.to_period("Q").astype(str)
            )
            .groupby(["quarter", "sector"])["sentiment_score"]
            .mean()
            .reset_index()
        )
        if not time_sector.empty:
            fig_ts = px.line(
                time_sector,
                x="quarter",
                y="sentiment_score",
                color="sector",
                color_discrete_map=SECTOR_COLORS,
                markers=True,
                labels={
                    "quarter": "Quarter",
                    "sentiment_score": "Avg Sentiment Score",
                    "sector": "Sector",
                },
            )
            fig_ts.update_layout(height=420, margin=dict(t=20, b=40), hovermode="x unified")
            st.plotly_chart(fig_ts, use_container_width=True)

    # -- Tone breakdown by sector --
    if "tone" in df_all.columns:
        st.subheader("Tone Distribution by Sector")
        tone_sector = (
            df_all.groupby(["sector", "tone"]).size().reset_index(name="count")
        )
        fig_tone_s = px.bar(
            tone_sector,
            x="sector",
            y="count",
            color="tone",
            barmode="stack",
            labels={"sector": "Sector", "count": "Number of Filings", "tone": "Tone"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_tone_s.update_layout(height=380, margin=dict(t=20, b=40))
        st.plotly_chart(fig_tone_s, use_container_width=True)

    # -- Sector-level forward returns --
    st.subheader("Sector-Level Mean Forward Returns")
    fwd_cols = [c for c in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"] if c in df_all.columns]
    if fwd_cols:
        sector_fwd = df_all.groupby("sector")[fwd_cols].mean().reindex(SECTOR_MAP.keys())
        sector_fwd.columns = [c.replace("fwd_return_", "").upper() for c in sector_fwd.columns]

        fig_fwd_s = go.Figure()
        colors = {"1D": "#636EFA", "5D": "#EF553B", "21D": "#00CC96"}
        for col in sector_fwd.columns:
            fig_fwd_s.add_trace(
                go.Bar(
                    x=sector_fwd.index,
                    y=sector_fwd[col],
                    name=col,
                    marker_color=colors.get(col, "#636EFA"),
                )
            )
        fig_fwd_s.update_layout(
            barmode="group",
            yaxis_tickformat=".2%",
            xaxis_title="Sector",
            yaxis_title="Mean Forward Return",
            height=400,
            margin=dict(t=20, b=40),
        )
        fig_fwd_s.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_fwd_s, use_container_width=True)

    # -- Sector-level backtest performance --
    st.subheader("Sector-Level Backtest Performance")
    st.caption("Mean forward return by sentiment tercile, grouped by sector.")

    if "sentiment_score" in df_all.columns and fwd_cols:
        sector_bt_rows = []
        for sector_name in SECTOR_MAP:
            sector_df = df_all[df_all["sector"] == sector_name]
            if sector_df.empty or "sentiment_score" not in sector_df.columns:
                continue
            try:
                sector_df = sector_df.copy()
                sector_df["sent_tercile"] = pd.qcut(
                    sector_df["sentiment_score"], q=3, labels=["Low", "Mid", "High"],
                    duplicates="drop",
                )
            except ValueError:
                continue
            for fwd in fwd_cols:
                if fwd not in sector_df.columns:
                    continue
                tercile_means = sector_df.groupby("sent_tercile", observed=True)[fwd].mean()
                h_label = fwd.replace("fwd_return_", "").upper()
                for tercile in ["Low", "Mid", "High"]:
                    val = tercile_means.get(tercile, np.nan)
                    sector_bt_rows.append({
                        "Sector": sector_name,
                        "Horizon": h_label,
                        "Tercile": tercile,
                        "Mean Return": val,
                    })

        if sector_bt_rows:
            sbt_df = pd.DataFrame(sector_bt_rows)
            horizon_select = st.selectbox(
                "Select Horizon",
                ["1D", "5D", "21D"],
                index=2,
                key="sector_bt_horizon",
            )
            subset = sbt_df[sbt_df["Horizon"] == horizon_select]
            if not subset.empty:
                fig_sbt = px.bar(
                    subset,
                    x="Sector",
                    y="Mean Return",
                    color="Tercile",
                    barmode="group",
                    color_discrete_map={
                        "Low": "#EF553B",
                        "Mid": "#FFA15A",
                        "High": "#00CC96",
                    },
                    labels={"Mean Return": f"Mean {horizon_select} Return"},
                )
                fig_sbt.update_layout(
                    yaxis_tickformat=".2%",
                    height=400,
                    margin=dict(t=20, b=40),
                )
                fig_sbt.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_sbt, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built with Streamlit and Plotly. Data sourced from SEC EDGAR 10-Q filings, "
    "signals extracted via Claude (Anthropic)."
)
