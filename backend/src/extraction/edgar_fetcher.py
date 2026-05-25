"""
SEC EDGAR 10-Q/10-K filing fetcher.
Uses the free EDGAR full-text search API — no API key required.
"""

import time
import requests
from pathlib import Path

EDGAR_HEADERS = {
    "User-Agent": "paragkane prg.kane98@gmail.com",  # EDGAR requires identity
    "Accept-Encoding": "gzip, deflate",
}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def get_cik(ticker: str) -> str | None:
    """Resolve ticker symbol to SEC CIK number."""
    # Use the company tickers JSON — most reliable
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_filings(cik: str, form_type: str = "10-Q", limit: int = 8) -> list[dict]:
    """Fetch list of filings for a given CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])

    results = []
    for form, accession, date, doc in zip(forms, accession_numbers, filing_dates, primary_docs):
        if form == form_type:
            results.append({
                "form": form,
                "accession": accession.replace("-", ""),
                "accession_dashed": accession,
                "date": date,
                "primary_doc": doc,
                "cik": cik,
            })
        if len(results) >= limit:
            break

    return results


def get_readable_doc(cik: str, accession: str, primary_doc: str) -> str:
    """
    Return the best readable HTM document for a filing.

    EDGAR's primaryDocument field reliably points to the main filing HTM.
    We trust it directly if it ends in .htm — only fall back to index
    parsing if primaryDocument is a non-HTML file (e.g. .xml, .xsd).
    """
    # Primary doc is an HTM — trust it directly (covers AAPL, MSFT, JPM, etc.)
    if primary_doc.lower().endswith(".htm"):
        return primary_doc

    # Primary doc is XML/XSD — parse the filing index to find the HTM
    acc_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{acc_dashed}-index.htm"
    resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)

    if resp.status_code != 200:
        return primary_doc

    import re
    links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', resp.text, re.IGNORECASE)

    def is_main_doc(path: str) -> bool:
        name = path.split("/")[-1].lower()
        excludes = ["ex-", "-ex", "exhibit", "filingsummary",
                    "r1.", "r2.", "r3.", "r4.", "r5.", "r6.", "r7.", "r8.", "r9."]
        return not any(ex in name for ex in excludes)

    candidates = [link.split("/")[-1] for link in links if is_main_doc(link)]
    return candidates[0] if candidates else primary_doc


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str | None:
    """Download the raw HTML text of a filing, using the best readable document."""
    doc = get_readable_doc(cik, accession, primary_doc)
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=60)
    if resp.status_code == 200:
        return resp.text
    return None


def save_filing(ticker: str, date: str, form: str, text: str) -> Path:
    """Save filing text to data/raw/<ticker>/."""
    out_dir = DATA_DIR / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}_{form.replace('-', '')}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def fetch_tickers(tickers: list[str], form_type: str = "10-Q", limit: int = 4) -> dict:
    """
    Main entry point. Fetch filings for a list of tickers.

    Args:
        tickers:   List of ticker symbols e.g. ["AAPL", "MSFT", "JPM"]
        form_type: "10-Q" or "10-K"
        limit:     Max filings per ticker

    Returns:
        Dict mapping ticker -> list of saved file paths
    """
    results = {}

    for ticker in tickers:
        print(f"[{ticker}] Resolving CIK...")
        cik = get_cik(ticker)
        if not cik:
            print(f"[{ticker}] CIK not found, skipping.")
            continue

        print(f"[{ticker}] CIK={cik}. Fetching {form_type} filings...")
        filings = get_filings(cik, form_type=form_type, limit=limit)
        print(f"[{ticker}] Found {len(filings)} filings.")

        saved = []
        for filing in filings:
            text = fetch_filing_text(filing["cik"], filing["accession"], filing["primary_doc"])
            if text:
                path = save_filing(ticker, filing["date"], filing["form"], text)
                saved.append(str(path))
                print(f"[{ticker}] Saved {filing['date']} {filing['form']} -> {path.name}")
            else:
                print(f"[{ticker}] Failed to fetch {filing['date']} {filing['form']}")
            time.sleep(0.5)  # EDGAR rate limit: be polite

        results[ticker] = saved
        time.sleep(1)

    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "GS", "BAC"]
    fetch_tickers(tickers, form_type="10-Q", limit=4)
