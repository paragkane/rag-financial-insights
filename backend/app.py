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

SECTOR_COLORS = {
    "Tech": "#6366f1",
    "Finance": "#f43f5e",
    "Healthcare": "#10b981",
    "Consumer": "#a855f7",
    "Energy": "#f59e0b",
}

TONE_COLORS = {
    "optimistic": "#10b981",
    "neutral": "#6b7280",
    "cautious": "#f59e0b",
    "defensive": "#ef4444",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0;
    }

    /* Headers */
    h1 { color: #f1f5f9 !important; font-weight: 700 !important; }
    h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
    h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
    p, li, span { color: #94a3b8; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 24px;
    }
    .hero-title {
        color: #f1f5f9;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 24px;
        line-height: 1.6;
    }
    .hero-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 24px;
    }
    .hero-stat {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .hero-stat-value {
        color: #818cf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .hero-stat-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Pipeline steps */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .pipeline-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        position: relative;
    }
    .pipeline-step::after {
        content: "→";
        position: absolute;
        right: -14px;
        top: 50%;
        transform: translateY(-50%);
        color: #6366f1;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .pipeline-step:last-child::after { content: ""; }
    .pipeline-num {
        color: #6366f1;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .pipeline-label {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .pipeline-detail {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 4px;
    }

    /* Key finding cards */
    .finding-card {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .finding-card.significant {
        border-left-color: #10b981;
    }
    .finding-card.not-significant {
        border-left-color: #f59e0b;
    }
    .finding-title {
        color: #f1f5f9;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .finding-detail {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Dividers */
    hr { border-color: #334155 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Selectbox */
    div[data-baseweb="select"] { background-color: #1e293b; }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 24px 0;
        border-top: 1px solid #1e293b;
        margin-top: 40px;
    }
    .footer a { color: #6366f1; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading signal data...")
def load_all_signals() -> pd.DataFrame:
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
    for col in ["filing_date", "entry_date"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    combined["sector"] = combined["ticker"].map(TICKER_TO_SECTOR)
    return combined


@st.cache_data(show_spinner="Loading backtest results...")
def load_backtest(horizon: str) -> dict:
    fpath = BACKTEST_DIR / f"backtest_fwd_return_{horizon}.json"
    try:
        with open(fpath) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner="Loading all backtests...")
def load_all_backtests() -> dict[str, dict]:
    return {h: load_backtest(h) for h in HORIZONS}


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def dark_layout(fig, height=450, **kwargs):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        margin=dict(t=40, b=50, l=60, r=20),
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#6366f1",
            font=dict(color="#f1f5f9", size=13),
        ),
        **kwargs,
    )
    fig.update_xaxes(
        gridcolor="rgba(51,65,85,0.5)",
        zerolinecolor="#334155",
        title_font=dict(size=12, color="#94a3b8"),
        tickfont=dict(size=11, color="#94a3b8"),
    )
    fig.update_yaxes(
        gridcolor="rgba(51,65,85,0.5)",
        zerolinecolor="#334155",
        title_font=dict(size=12, color="#94a3b8"),
        tickfont=dict(size=11, color="#94a3b8"),
    )
    return fig


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df_all = load_all_signals()
backtests = load_all_backtests()

if df_all.empty:
    st.error("No signal data found. Run the pipeline first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Navigation")
    section = st.radio(
        "Go to",
        ["Home", "Signal Explorer", "Backtest Results", "Sector Analysis"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"**{len(df_all):,}** filings")
    st.markdown(f"**{df_all['ticker'].nunique()}** tickers")
    if "filing_date" in df_all.columns:
        min_d = df_all["filing_date"].min()
        max_d = df_all["filing_date"].max()
        if pd.notna(min_d) and pd.notna(max_d):
            st.markdown(f"**{min_d.strftime('%b %Y')}** – **{max_d.strftime('%b %Y')}**")
    st.divider()
    st.markdown(
        '<p style="color: #475569; font-size: 0.75rem;">'
        'Built by Parag Kane<br>'
        '<a href="https://github.com/paragkane/llm-financial-signals" '
        'style="color: #6366f1;">GitHub Repo</a></p>',
        unsafe_allow_html=True,
    )


# ===================================================================
# HOME PAGE
# ===================================================================

if section == "Home":
    # Hero
    n_filings = len(df_all)
    n_tickers = df_all["ticker"].nunique()
    n_sectors = len(SECTOR_MAP)
    date_range = ""
    if "filing_date" in df_all.columns:
        min_d = df_all["filing_date"].min()
        max_d = df_all["filing_date"].max()
        if pd.notna(min_d) and pd.notna(max_d):
            date_range = f"{min_d.strftime('%Y')}–{max_d.strftime('%Y')}"

    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">LLM Financial Signal Extraction<br>& Backtesting Pipeline</div>
        <div class="hero-subtitle">
            Extracting quantitative alpha signals from SEC 10-Q filings using
            <strong style="color: #818cf8;">Claude Sonnet 4.6</strong>,
            then backtesting their predictive power with full statistical rigor
            including sector factor neutralization.
        </div>
        <div class="hero-stat-grid">
            <div class="hero-stat">
                <div class="hero-stat-value">{n_filings}</div>
                <div class="hero-stat-label">SEC Filings Analyzed</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{n_tickers}</div>
                <div class="hero-stat-label">Tickers</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{n_sectors}</div>
                <div class="hero-stat-label">Sectors</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{date_range}</div>
                <div class="hero-stat-label">Date Range</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline visualization
    st.markdown("### How It Works")
    st.markdown("""
    <div class="pipeline-grid">
        <div class="pipeline-step">
            <div class="pipeline-num">01</div>
            <div class="pipeline-label">EDGAR Fetch</div>
            <div class="pipeline-detail">Download 10-Q filings from SEC</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">02</div>
            <div class="pipeline-label">Text Cleaning</div>
            <div class="pipeline-detail">Strip HTML/XBRL, extract MD&A</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">03</div>
            <div class="pipeline-label">Signal Extraction</div>
            <div class="pipeline-detail">Claude Sonnet 4.6 at temp=0</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">04</div>
            <div class="pipeline-label">Price Alignment</div>
            <div class="pipeline-detail">Match to T+1 entry + returns</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-num">05</div>
            <div class="pipeline-label">Backtesting</div>
            <div class="pipeline-detail">Sharpe, t-test, factor neutralization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key findings
    st.markdown("### Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="finding-card significant">
            <div class="finding-title">Raw Sentiment — Statistically Significant</div>
            <div class="finding-detail">
                5-day horizon: Sharpe 0.92, p=0.015<br>
                21-day horizon: Sharpe 0.75, p&lt;0.001
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="finding-card not-significant">
            <div class="finding-title">After Factor Neutralization — Signal Disappears</div>
            <div class="finding-detail">
                Sector-adjusted returns show no significance — raw signal
                captures sector momentum, not firm-level alpha.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    # Sentiment distribution + tone pie side by side
    col_hist, col_pie = st.columns([3, 2])
    with col_hist:
        st.markdown("### Sentiment Score Distribution")
        fig_hist = px.histogram(
            df_all, x="sentiment_score", nbins=35, color="sector",
            color_discrete_map=SECTOR_COLORS,
            labels={"sentiment_score": "Sentiment Score", "sector": "Sector"},
            opacity=0.85, barmode="overlay",
        )
        dark_layout(fig_hist, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_pie:
        st.markdown("### Tone Breakdown")
        tone_counts = df_all["tone"].value_counts().reset_index()
        tone_counts.columns = ["tone", "count"]
        fig_pie = px.pie(
            tone_counts, values="count", names="tone",
            color="tone", color_discrete_map=TONE_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont=dict(size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
        dark_layout(fig_pie, height=400)
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Filings per ticker bar
    st.markdown("### Filings by Ticker")
    counts = df_all["ticker"].value_counts().reindex(TICKERS).fillna(0).astype(int)
    fig_counts = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Ticker", "y": "Filings"},
        color=[TICKER_TO_SECTOR.get(t, "Other") for t in counts.index],
        color_discrete_map=SECTOR_COLORS,
    )
    fig_counts.update_traces(
        hovertemplate="<b>%{x}</b><br>Filings: %{y}<extra></extra>",
    )
    dark_layout(fig_counts, height=380, showlegend=True, legend_title_text="Sector")
    st.plotly_chart(fig_counts, use_container_width=True)


# ===================================================================
# SIGNAL EXPLORER
# ===================================================================

elif section == "Signal Explorer":
    st.markdown("## Signal Explorer")
    st.markdown('<p style="color: #94a3b8;">Drill into individual ticker signals extracted by Claude.</p>', unsafe_allow_html=True)

    available_tickers = sorted(df_all["ticker"].unique())
    selected_ticker = st.selectbox("Select Ticker", available_tickers, index=0)
    df_ticker = df_all[df_all["ticker"] == selected_ticker].copy().sort_values("filing_date")

    if df_ticker.empty:
        st.warning(f"No data for {selected_ticker}.")
        st.stop()

    sector = TICKER_TO_SECTOR.get(selected_ticker, "Unknown")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sector", sector)
    col2.metric("Filings", len(df_ticker))
    col3.metric("Avg Sentiment", f"{df_ticker['sentiment_score'].mean():.2f}")
    most_common_tone = df_ticker["tone"].mode().iloc[0] if not df_ticker["tone"].mode().empty else "N/A"
    col4.metric("Dominant Tone", most_common_tone.title())
    avg_5d = df_ticker["fwd_return_5d"].mean() if "fwd_return_5d" in df_ticker.columns else 0
    col5.metric("Avg 5d Return", f"{avg_5d:.2%}")

    st.markdown("###")

    # Sentiment over time + forward returns
    col_sent, col_fwd = st.columns(2)

    with col_sent:
        st.markdown("### Sentiment Over Time")
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=df_ticker["filing_date"], y=df_ticker["sentiment_score"],
            mode="lines+markers",
            line=dict(color=SECTOR_COLORS.get(sector, "#6366f1"), width=3),
            marker=dict(size=10, line=dict(width=2, color="#0f172a")),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sentiment: %{y:.2f}<extra></extra>",
            name="Sentiment",
        ))
        fig_sent.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
        dark_layout(fig_sent, height=400, hovermode="x unified",
                    xaxis_title="Filing Date", yaxis_title="Sentiment Score")
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_fwd:
        st.markdown("### Forward Returns by Filing")
        fig_fwd = go.Figure()
        fwd_colors = {"fwd_return_1d": "#6366f1", "fwd_return_5d": "#f43f5e", "fwd_return_21d": "#10b981"}
        for col_name, color in fwd_colors.items():
            if col_name in df_ticker.columns:
                label = col_name.replace("fwd_return_", "").upper()
                fig_fwd.add_trace(go.Bar(
                    x=df_ticker["filing_date"], y=df_ticker[col_name],
                    name=label, marker_color=color, opacity=0.8,
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: %{{y:.2%}}<extra></extra>",
                ))
        dark_layout(fig_fwd, height=400, barmode="group",
                    xaxis_title="Filing Date", yaxis_title="Return",
                    yaxis_tickformat=".1%", hovermode="x unified")
        st.plotly_chart(fig_fwd, use_container_width=True)

    # Tone over time
    col_tone, col_risk = st.columns(2)
    with col_tone:
        st.markdown("### Tone Distribution")
        tone_cts = df_ticker["tone"].value_counts()
        fig_tone = px.bar(
            x=tone_cts.index, y=tone_cts.values,
            labels={"x": "Tone", "y": "Count"},
            color=tone_cts.index, color_discrete_map=TONE_COLORS,
        )
        fig_tone.update_traces(
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        )
        dark_layout(fig_tone, height=350, showlegend=False)
        st.plotly_chart(fig_tone, use_container_width=True)

    with col_risk:
        st.markdown("### Risk Flags Frequency")
        risk_flags = []
        for flags in df_ticker["risk_flags"].dropna():
            if isinstance(flags, list):
                risk_flags.extend(flags)
        if risk_flags:
            from collections import Counter
            risk_counts = Counter(risk_flags).most_common(10)
            risk_df = pd.DataFrame(risk_counts, columns=["Flag", "Count"])
            fig_risk = px.bar(
                risk_df, x="Count", y="Flag", orientation="h",
                color="Count", color_continuous_scale=["#334155", "#6366f1", "#818cf8"],
            )
            fig_risk.update_traces(
                hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
            )
            dark_layout(fig_risk, height=350, showlegend=False)
            fig_risk.update_layout(coloraxis_showscale=False)
            fig_risk.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("No risk flags found.")

    # Full table
    st.markdown(f"### All Filings — {selected_ticker}")
    display_cols = [c for c in [
        "filing_date", "entry_date", "entry_price", "sentiment_score",
        "guidance_direction", "tone", "earnings_framing", "key_themes",
        "fwd_return_1d", "fwd_return_5d", "fwd_return_21d",
    ] if c in df_ticker.columns]
    st.dataframe(df_ticker[display_cols].reset_index(drop=True),
                 use_container_width=True, height=400)


# ===================================================================
# BACKTEST RESULTS
# ===================================================================

elif section == "Backtest Results":
    st.markdown("## Backtest Results")
    st.markdown('<p style="color: #94a3b8;">Statistical analysis of signal predictive power across time horizons.</p>', unsafe_allow_html=True)

    if all(not bt for bt in backtests.values()):
        st.warning("No backtest results found.")
        st.stop()

    # Results summary cards
    st.markdown("### Summary")
    cols = st.columns(3)
    for i, h in enumerate(HORIZONS):
        bt = backtests.get(h, {})
        sent = bt.get("sentiment", {})
        sharpe = sent.get("overall_sharpe", "N/A")
        pval = sent.get("p_value", 1.0)
        sig = sent.get("significant", False)
        with cols[i]:
            label = f"{h.upper()} Horizon"
            sig_text = "Significant" if sig else "Not Significant"
            sig_color = "#10b981" if sig else "#f59e0b"
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
                <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                <div style="color: #f1f5f9; font-size: 1.6rem; font-weight: 700; margin: 8px 0;">Sharpe {sharpe}</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">p-value: {pval}</div>
                <div style="color: {sig_color}; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">{sig_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("###")

    # Scatter plots
    st.markdown("### Sentiment vs Forward Returns")
    horizon_tabs = st.tabs([f"{h.upper()} Horizon" for h in HORIZONS])

    for i, h in enumerate(HORIZONS):
        with horizon_tabs[i]:
            fwd_col = f"fwd_return_{h}"
            if fwd_col not in df_all.columns:
                st.info(f"Missing data for {h}.")
                continue

            plot_df = df_all.dropna(subset=["sentiment_score", fwd_col]).copy()
            if plot_df.empty:
                continue

            fig_scatter = px.scatter(
                plot_df, x="sentiment_score", y=fwd_col,
                color="sector", color_discrete_map=SECTOR_COLORS,
                hover_data={"ticker": True, "filing_date": True, "tone": True,
                            "sentiment_score": ":.2f", fwd_col: ":.2%"},
                labels={"sentiment_score": "Sentiment Score",
                        fwd_col: f"Forward Return ({h.upper()})"},
                opacity=0.7,
            )
            fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color="#0f172a")))

            # Trendline
            x_vals = plot_df["sentiment_score"].values
            y_vals = plot_df[fwd_col].values
            if len(x_vals) > 2:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                fig_scatter.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(coeffs, x_line),
                    mode="lines", name=f"OLS (slope={coeffs[0]:.5f})",
                    line=dict(color="#f43f5e", width=2.5, dash="dash"),
                ))

            bt = backtests.get(h, {}).get("sentiment", {})
            sig = bt.get("significant", False)
            corr = bt.get("pearson_correlation")
            pval = bt.get("correlation_p_value")
            ann = f"r = {corr:.4f}, p = {pval:.4f}" if corr is not None else ""
            if sig:
                ann += "  SIGNIFICANT"

            dark_layout(fig_scatter, height=500, yaxis_tickformat=".1%")
            fig_scatter.add_annotation(
                text=ann, xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, font=dict(size=14, color="#10b981" if sig else "#f59e0b"),
                bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155", borderwidth=1, borderpad=8,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Sharpe comparison
    st.markdown("### Sharpe Ratio Comparison")
    sharpe_rows = []
    for h in HORIZONS:
        bt = backtests.get(h, {})
        for label, keys in [
            ("Raw Sentiment", ["sentiment", "overall_sharpe"]),
            ("Raw Tone", ["tone", "overall_sharpe"]),
            ("Neutralized Sentiment", ["neutralized", "sentiment", "overall_sharpe"]),
            ("Neutralized Tone", ["neutralized", "tone", "overall_sharpe"]),
        ]:
            obj = bt
            for k in keys:
                obj = obj.get(k) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if obj is not None:
                sharpe_rows.append({"Horizon": h.upper(), "Signal": label, "Sharpe": obj})

    if sharpe_rows:
        sharpe_df = pd.DataFrame(sharpe_rows)
        color_map = {
            "Raw Sentiment": "#6366f1", "Raw Tone": "#10b981",
            "Neutralized Sentiment": "#818cf8", "Neutralized Tone": "#6ee7b7",
        }
        fig_sharpe = px.bar(
            sharpe_df, x="Horizon", y="Sharpe", color="Signal",
            barmode="group", color_discrete_map=color_map,
            labels={"Sharpe": "Sharpe Ratio"},
        )
        fig_sharpe.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Horizon: %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
        )
        fig_sharpe.add_hline(y=0, line_dash="dash", line_color="#475569")
        dark_layout(fig_sharpe, height=420)
        st.plotly_chart(fig_sharpe, use_container_width=True)


# ===================================================================
# SECTOR ANALYSIS
# ===================================================================

elif section == "Sector Analysis":
    st.markdown("## Sector Analysis")
    st.markdown('<p style="color: #94a3b8;">Cross-sector comparison of sentiment, tone, and returns.</p>', unsafe_allow_html=True)

    # Average sentiment by sector
    col_bar, col_ts = st.columns(2)

    with col_bar:
        st.markdown("### Avg Sentiment by Sector")
        sector_avg = df_all.groupby("sector")["sentiment_score"].agg(["mean", "std"]).reindex(SECTOR_MAP.keys())
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=sector_avg.index, y=sector_avg["mean"],
            error_y=dict(type="data", array=sector_avg["std"], color="#94a3b8", thickness=1.5),
            marker_color=[SECTOR_COLORS[s] for s in sector_avg.index],
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>",
        ))
        dark_layout(fig_bar, height=420, showlegend=False,
                    xaxis_title="Sector", yaxis_title="Mean Sentiment")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_ts:
        st.markdown("### Sentiment Over Time")
        time_sector = (
            df_all.assign(quarter=df_all["filing_date"].dt.to_period("Q").astype(str))
            .groupby(["quarter", "sector"])["sentiment_score"].mean().reset_index()
        )
        fig_ts = px.line(
            time_sector, x="quarter", y="sentiment_score", color="sector",
            color_discrete_map=SECTOR_COLORS, markers=True,
            labels={"quarter": "Quarter", "sentiment_score": "Avg Sentiment", "sector": "Sector"},
        )
        fig_ts.update_traces(
            line=dict(width=2.5), marker=dict(size=7),
            hovertemplate="<b>%{fullData.name}</b><br>Quarter: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        )
        dark_layout(fig_ts, height=420, hovermode="x unified", xaxis_tickangle=-45)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Tone by sector + returns by sector
    col_tone, col_ret = st.columns(2)

    with col_tone:
        st.markdown("### Tone Distribution by Sector")
        tone_sector = df_all.groupby(["sector", "tone"]).size().reset_index(name="count")
        fig_tone = px.bar(
            tone_sector, x="sector", y="count", color="tone",
            barmode="stack", color_discrete_map=TONE_COLORS,
            labels={"sector": "Sector", "count": "Filings", "tone": "Tone"},
        )
        fig_tone.update_traces(
            hovertemplate="<b>%{x}</b> — %{fullData.name}<br>Count: %{y}<extra></extra>",
        )
        dark_layout(fig_tone, height=420)
        st.plotly_chart(fig_tone, use_container_width=True)

    with col_ret:
        st.markdown("### Mean Forward Returns by Sector")
        fwd_cols = [c for c in ["fwd_return_1d", "fwd_return_5d", "fwd_return_21d"] if c in df_all.columns]
        sector_fwd = df_all.groupby("sector")[fwd_cols].mean().reindex(SECTOR_MAP.keys())
        fig_ret = go.Figure()
        ret_colors = {"fwd_return_1d": "#6366f1", "fwd_return_5d": "#f43f5e", "fwd_return_21d": "#10b981"}
        for col_name, color in ret_colors.items():
            if col_name in sector_fwd.columns:
                label = col_name.replace("fwd_return_", "").upper()
                fig_ret.add_trace(go.Bar(
                    x=sector_fwd.index, y=sector_fwd[col_name],
                    name=label, marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2%}}<extra></extra>",
                ))
        fig_ret.add_hline(y=0, line_dash="dash", line_color="#475569")
        dark_layout(fig_ret, height=420, barmode="group",
                    yaxis_tickformat=".1%", xaxis_title="Sector", yaxis_title="Mean Return")
        st.plotly_chart(fig_ret, use_container_width=True)

    # Tercile analysis
    st.markdown("### Sentiment Tercile Performance")
    st.markdown('<p style="color: #94a3b8;">Do high-sentiment filings outperform low-sentiment filings within each sector?</p>', unsafe_allow_html=True)

    horizon_select = st.selectbox("Horizon", ["1D", "5D", "21D"], index=2)
    fwd_col = f"fwd_return_{horizon_select.lower()}"

    if fwd_col in df_all.columns:
        tercile_rows = []
        for sector_name in SECTOR_MAP:
            sdf = df_all[df_all["sector"] == sector_name].copy()
            if len(sdf) < 6:
                continue
            try:
                sdf["tercile"] = pd.qcut(sdf["sentiment_score"], q=3,
                                         labels=["Low", "Mid", "High"], duplicates="drop")
            except ValueError:
                continue
            for terc in ["Low", "Mid", "High"]:
                val = sdf[sdf["tercile"] == terc][fwd_col].mean()
                tercile_rows.append({"Sector": sector_name, "Tercile": terc, "Return": val})

        if tercile_rows:
            tdf = pd.DataFrame(tercile_rows)
            fig_terc = px.bar(
                tdf, x="Sector", y="Return", color="Tercile", barmode="group",
                color_discrete_map={"Low": "#ef4444", "Mid": "#f59e0b", "High": "#10b981"},
                labels={"Return": f"Mean {horizon_select} Return"},
            )
            fig_terc.update_traces(
                hovertemplate="<b>%{x}</b> — %{fullData.name}<br>Return: %{y:.2%}<extra></extra>",
            )
            fig_terc.add_hline(y=0, line_dash="dash", line_color="#475569")
            dark_layout(fig_terc, height=420, yaxis_tickformat=".1%")
            st.plotly_chart(fig_terc, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="footer">
    LLM Financial Signal Extraction & Backtesting Pipeline<br>
    <a href="https://github.com/paragkane/llm-financial-signals">github.com/paragkane/llm-financial-signals</a>
    &nbsp;|&nbsp; Built by Parag Kane
</div>
""", unsafe_allow_html=True)
