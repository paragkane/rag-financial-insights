# LLM Financial Signal Extraction & Backtesting Pipeline

An end-to-end pipeline that extracts quantitative alpha signals from SEC 10-Q filings using **Claude Sonnet 4.6**, then backtests those signals against historical price data with full statistical rigor — including sector factor neutralization to distinguish true alpha from sector momentum.

**363 filings · 30 tickers · 5 sectors · 2021–2026**

---

## Results Dashboard

![Backtest Results Dashboard](charts/dashboard_screenshot.png)

## Results (363 filings · 30 tickers)

### Raw Signal Performance

| Signal | Horizon | Sharpe | Hit Rate | t-stat | p-value |
|---|---|---|---|---|---|
| Sentiment Score | 1-Day | -0.66 | 50.4% | -0.79 | 0.43 |
| Sentiment Score | **5-Day** | **0.92** | 49.0% | **2.43** | **0.015 ✅** |
| Sentiment Score | **21-Day** | **0.75** | 50.0% | **4.14** | **0.000 ✅** |
| Tone | 5-Day | 1.82 | 40.9% | 1.19 | 0.25 |
| Tone | 21-Day | 0.94 | 50.0% | 1.28 | 0.21 |

### After Sector Factor Neutralization

| Signal | Horizon | Sharpe | Hit Rate | t-stat | p-value |
|---|---|---|---|---|---|
| Sentiment Score | 5-Day | -0.30 | 45.5% | -0.36 | 0.72 |
| Sentiment Score | 21-Day | -0.37 | 49.5% | -0.44 | 0.66 |
| Tone | 21-Day | 1.95 | 50.0% | 0.58 | 0.57 |

**Key findings:**
- Raw sentiment score is **statistically significant** at 5-day (p=0.015) and 21-day (p=0.000) horizons with positive Sharpe ratios
- After sector factor neutralization (SPY + sector ETFs), significance disappears — suggesting the raw signal partially captures **sector momentum** rather than pure filing-level alpha
- This is a common finding in NLP-based alpha research: text signals often proxy for sector-level sentiment rather than firm-specific information
- Signal is strongest at longer horizons — management language takes days to weeks to be fully priced in
- Tested across 30 tickers in tech, finance, healthcare, consumer, and energy for generalizability

---

## What It Does

1. **Fetches** SEC 10-Q filings via the free EDGAR API (no key required)
2. **Cleans** raw iXBRL/HTML filings — strips tags, extracts MD&A, risk factors, liquidity sections
3. **Extracts** structured alpha signals using Claude Sonnet 4.6 at temperature=0:
   - `sentiment_score` — float -1.0 to +1.0
   - `guidance_direction` — raised / lowered / maintained / none
   - `tone` — optimistic / cautious / neutral / defensive
   - `risk_flags` — liquidity, macro, competition, regulatory, execution
   - `earnings_framing` — beat / miss / in-line
   - `key_themes` — up to 5 dominant themes in management language
4. **Aligns** signals to next trading day entry prices with 1d/5d/21d forward returns
5. **Backtests** using Pearson correlation, linear regression, Sharpe ratio, and t-test significance

---

## Why Claude over GPT-4o

Claude Sonnet 4.6 was chosen over GPT-4o for this task because:
- Superior long-context comprehension on dense financial prose (MD&A sections run 10K–30K chars)
- More conservative and precise on domain-specific language — less hallucination of financial metrics
- Temperature=0 with strict JSON schema produces highly consistent, reproducible signal extraction

---

## Project Structure

```
llm-financial-signals/
├── src/
│   ├── extraction/
│   │   ├── edgar_fetcher.py      # SEC EDGAR 10-Q/10-K downloader
│   │   ├── price_fetcher.py      # Historical prices + forward returns (yfinance)
│   │   ├── text_cleaner.py       # HTML/XBRL stripper, section extractor
│   │   └── signal_extractor.py   # Claude Sonnet 4.6 signal extraction + file cache
│   ├── backtesting/
│   │   ├── signal_aligner.py     # Filing date → entry price → forward returns
│   │   ├── engine.py             # Sharpe, hit rate, t-stat, Pearson correlation
│   │   ├── factor_model.py       # Sector factor neutralization (SPY + sector ETFs)
│   │   └── visualizer.py         # Plotly charts + dashboard
│   └── evaluation/
│       └── earnings_eval.py      # LLM signal accuracy vs actual earnings surprises
├── scripts/
│   └── run_pipeline.py           # End-to-end runner (one command)
├── app.py                        # Streamlit interactive dashboard
├── notebooks/
│   └── walkthrough.ipynb         # Step-by-step pipeline walkthrough
├── tests/                        # Unit + integration tests (pytest)
├── data/
│   ├── raw/                      # Raw SEC filings (gitignored)
│   └── processed/                # Cleaned text, parquet files (gitignored)
└── charts/                       # Plotly HTML charts (gitignored)
```

---

## Setup

```bash
git clone https://github.com/paragkane/llm-financial-signals
cd llm-financial-signals

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Run

```bash
# Full pipeline: fetch → clean → extract → align → backtest
python scripts/run_pipeline.py

# Interactive dashboard (requires processed data)
streamlit run app.py

# Run tests
pytest tests/ -v

# Run evaluation (LLM signals vs earnings surprises)
python -m src.evaluation.earnings_eval
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Data | SEC EDGAR API (free), yfinance |
| Signal validation | Pydantic |
| Backtesting | Custom Pandas engine + SciPy |
| Statistics | Pearson correlation, OLS regression, t-test |
| Visualization | Plotly |
| Caching | File-based signal cache (skip redundant API calls) |
| CI/CD | GitHub Actions (lint + test) |

---

## Tickers Tested (30)

| Sector | Tickers |
|---|---|
| Technology (8) | AAPL, MSFT, GOOGL, AMZN, NVDA, META, CRM, ORCL |
| Finance (8) | JPM, GS, BAC, MS, WFC, BLK, C, USB |
| Healthcare (6) | JNJ, UNH, PFE, MRK, ABBV, CVS |
| Consumer (5) | WMT, HD, NKE, TGT, MCD |
| Energy (3) | XOM, CVX, COP |
