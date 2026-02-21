import re
import csv
import argparse
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional


WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"

AIQ_PAGE = "https://www.globalxetfs.com/funds/aiq/"
BOTZ_PAGE = "https://www.globalxetfs.com/funds/botz/"

REMX_HOLDINGS = "https://stockanalysis.com/etf/remx/holdings/"
DRNZ_HOLDINGS = "https://www.quiverquant.com/etf/REX%20Drone%20ETF"
XLE_PAGE = "https://www.ssga.com/us/en/intermediary/etfs/state-street-energy-select-sector-spdr-etf-xle"


def _clean_ticker(t: str) -> str:
    if not t:
        return ""
    t = t.strip().upper()
    # keep only first token before space (e.g., "NVDA US" -> "NVDA")
    if " " in t:
        t = t.split(" ")[0]
    t = t.replace("$", "")
    # reject digit-only or too long
    if not re.match(r"^[A-Z]{1,6}$", t):
        return ""
    return t


def _write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "source", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fetch_html(url: str) -> str:
    r = requests.get(url, timeout=20, headers={"User-Agent": "news-sync/0.1"})
    r.raise_for_status()
    return r.text


def _read_wiki_table(url: str, column: str) -> List[str]:
    html = _fetch_html(url)
    tables = pd.read_html(StringIO(html))
    # pick first table that contains column
    for df in tables:
        if column in df.columns:
            return [str(x) for x in df[column].tolist()]
    return []


def _read_csv_tickers(url: str, col_candidates: List[str]) -> List[str]:
    r = requests.get(url, timeout=20, headers={"User-Agent": "news-sync/0.1"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    for c in col_candidates:
        if c in df.columns:
            return [str(x) for x in df[c].tolist()]
    return []


def _read_html_table_tickers(url: str, col_candidates: List[str]) -> List[str]:
    html = _fetch_html(url)
    tables = pd.read_html(StringIO(html))
    for df in tables:
        for c in col_candidates:
            if c in df.columns:
                return [str(x) for x in df[c].tolist()]
    return []


def _extract_xle_holdings(url: str) -> List[str]:
    # Try to find a CSV link in the HTML
    html = _fetch_html(url)
    m = re.search(r"https?://[^\"']+\\.csv", html, re.I)
    if m:
        try:
            return _read_csv_tickers(m.group(0), ["Ticker", "Symbol"])
        except Exception:
            pass
    # Fallback: parse tables from page (top holdings)
    return _read_html_table_tickers(url, ["Ticker", "Symbol"])


def _find_globalx_csv(page_url: str) -> str:
    html = _fetch_html(page_url)
    # look for holdings CSV
    m = re.search(r"https?://assets\\.globalxetfs\\.com/holdings/[^\"']+\\.csv", html, re.I)
    return m.group(0) if m else ""


def _filter_by_market_cap(tickers: List[str], api_key: str, min_cap: int) -> List[str]:
    if not api_key:
        return tickers
    filtered = []
    # FMP profile endpoint supports comma-separated tickers
    batch = 100
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(chunk)}?apikey={api_key}"
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "news-sync/0.1"})
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue
        for item in data if isinstance(data, list) else []:
            t = item.get("symbol")
            cap = item.get("mktCap") or item.get("marketCap")
            if t and cap and cap >= min_cap:
                filtered.append(t)
    return sorted({t for t in filtered})


def build_watchlists(
    out_dir: str,
    fmp_key: str = "",
    min_cap: int = 200000000,
) -> Dict[str, List[str]]:
    out = {}

    sp500 = [_clean_ticker(t) for t in _read_wiki_table(WIKI_SP500, "Symbol")]
    sp500 = sorted({t for t in sp500 if t})
    out["sp500"] = sp500

    ndx = [_clean_ticker(t) for t in _read_wiki_table(WIKI_NDX, "Ticker")]
    ndx = sorted({t for t in ndx if t})
    out["nasdaq100"] = ndx

    ai = []
    aiq_csv = _find_globalx_csv(AIQ_PAGE)
    botz_csv = _find_globalx_csv(BOTZ_PAGE)
    if aiq_csv:
        try:
            ai += _read_csv_tickers(aiq_csv, ["Ticker"])
        except Exception:
            pass
    if botz_csv:
        try:
            ai += _read_csv_tickers(botz_csv, ["Ticker"])
        except Exception:
            pass
    ai = sorted({t for t in (_clean_ticker(x) for x in ai) if t})
    out["ai"] = ai

    rare = _read_html_table_tickers(REMX_HOLDINGS, ["Symbol", "Ticker"])
    rare = sorted({t for t in (_clean_ticker(x) for x in rare) if t})
    out["rare_earth"] = rare

    drones = _read_html_table_tickers(DRNZ_HOLDINGS, ["Ticker"])
    drones = sorted({t for t in (_clean_ticker(x) for x in drones) if t})
    out["drones"] = drones

    energy = _extract_xle_holdings(XLE_PAGE)
    energy = sorted({t for t in (_clean_ticker(x) for x in energy) if t})
    out["energy"] = energy

    # write
    base = Path(out_dir)
    rows = []
    for k, tickers in out.items():
        for t in tickers:
            rows.append({"ticker": t, "source": k, "note": ""})
        _write_csv(base / f"{k}.csv", [{"ticker": t, "source": k, "note": ""} for t in tickers])

    # union
    universe = sorted({r["ticker"] for r in rows})
    _write_csv(base / "universe_all.csv", [{"ticker": t, "source": "union", "note": ""} for t in universe])

    # optional market-cap filter (Financial Modeling Prep)
    if fmp_key:
        filtered = _filter_by_market_cap(universe, fmp_key, min_cap)
        _write_csv(base / "universe_filtered.csv", [{"ticker": t, "source": "union", "note": f">={min_cap}"} for t in filtered])

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build stock watchlists for local research use.")
    ap.add_argument("--out-dir", default="./data/watchlists")
    ap.add_argument("--fmp-key", default="")
    ap.add_argument("--min-market-cap", type=int, default=200000000)
    args = ap.parse_args()
    build_watchlists(
        out_dir=args.out_dir,
        fmp_key=args.fmp_key,
        min_cap=args.min_market_cap,
    )
