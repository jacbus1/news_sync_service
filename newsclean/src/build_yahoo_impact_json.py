from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from ticker_resolver import resolve_tickers


_FED_RE = re.compile(r"\b(fed|federal reserve|fomc|powell|interest rate|rate cut|rate hike|treasury yield|cpi|ppi|jobs|payroll)\b", re.I)
_EARN_RE = re.compile(r"\b(earnings|eps|guidance|outlook|revenue|profit|results|beat|miss)\b", re.I)
_REPORT_RE = re.compile(r"\b(sec|10-?k|10-?q|8-?k|13f|s-?1|analyst|rating|price target|upgrade|downgrade|report)\b", re.I)
_NON_MARKET_RE = re.compile(r"\b(lottery|powerball|pick 3|jackpot|401\(k\)|social security|retirement rule)\b", re.I)

_POS_RE = re.compile(r"\b(beat|beats|surge|jumps?|rise|rall(y|ies)|record high|strong|upgrade)\b", re.I)
_NEG_RE = re.compile(r"\b(miss|misses|drop|drops|fell|falls|plunge|selloff|downgrade|warning|cuts?)\b", re.I)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,6}(?:\.[A-Z]{1,2})?)\b")
_EXCHANGE_RE = re.compile(
    r"\b(?:NASDAQ|NYSE|AMEX|OTC|TSX|TSXV)\s*:\s*([A-Z]{1,6}(?:\.[A-Z]{1,2})?)\b",
    re.I,
)
_CRYPTO_RE = re.compile(r"\b([A-Z]{2,10}-USD)\b")
_PAREN_RE = re.compile(r"\(([A-Z]{1,5})\)")
_PAREN_STOP = {"Q1", "Q2", "Q3", "Q4", "FY", "EPS", "CEO", "CFO", "ETF", "USD", "CAD", "EU", "UK", "PE", "PV"}
_BAD_TICKERS = {
    "FED",
    "FOMC",
    "ECB",
    "SPX",
    "DOWI",
    "IUXX",
    "GDP",
    "CPI",
    "PPI",
    "EPS",
    "ETF",
}


def _extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    out.extend(_CASHTAG_RE.findall(text.upper()))
    out.extend([m.upper() for m in _EXCHANGE_RE.findall(text)])
    out.extend(_CRYPTO_RE.findall(text.upper()))
    for m in _PAREN_RE.findall(text.upper()):
        if m in _PAREN_STOP:
            continue
        out.append(m)
    noise = {"USD", "USDT", "USDC"}
    return [t for t in sorted(set(out)) if t not in noise]


@dataclass
class PriceSnapshot:
    ticker: str
    price: Optional[float]
    change_pct: Optional[float]
    anomaly: str


def _norm_title_key(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[a-z0-9 ._-]{2,24}:\s+", "", t)
    t = re.sub(r"\s+-\s+.*$", "", t)
    return t


def _as_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _fetch_price_snapshot(ticker: str) -> PriceSnapshot:
    t = (ticker or "").strip().upper()
    if not t:
        return PriceSnapshot(ticker="", price=None, change_pct=None, anomaly="")
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None) or {}
        last_price = fi.get("last_price") or fi.get("lastPrice")
        prev_close = fi.get("previous_close") or fi.get("previousClose")
        last_vol = fi.get("last_volume") or fi.get("lastVolume")
        avg_vol = fi.get("ten_day_average_volume") or fi.get("three_month_average_volume") or fi.get("threeMonthAverageVolume")
        year_high = fi.get("year_high") or fi.get("yearHigh")
        year_low = fi.get("year_low") or fi.get("yearLow")

        price = float(last_price) if last_price is not None else None
        chg = None
        if price is not None and prev_close:
            pc = float(prev_close)
            if pc != 0:
                chg = (price / pc - 1.0) * 100.0

        anomalies: List[str] = []
        if chg is not None and abs(chg) >= 5:
            if chg > 0:
                anomalies.append(f"Strong daily surge {chg:.2f}%")
            else:
                anomalies.append(f"Sharp daily drop {abs(chg):.2f}%")
        elif chg is not None and abs(chg) >= 3:
            if chg > 0:
                anomalies.append(f"Notable gain {chg:.2f}%")
            else:
                anomalies.append(f"Notable loss {abs(chg):.2f}%")

        try:
            if last_vol and avg_vol and float(avg_vol) > 0:
                vr = float(last_vol) / float(avg_vol)
                if vr >= 2.0:
                    anomalies.append(f"Unusual volume ({int(float(last_vol)):,}, {vr:.2f}x avg)")
        except Exception:
            pass

        try:
            if price is not None and year_high:
                yh = float(year_high)
                if yh > 0 and price >= yh * 0.995:
                    anomalies.append("52-week high alert")
        except Exception:
            pass
        try:
            if price is not None and year_low:
                yl = float(year_low)
                if yl > 0 and price <= yl * 1.005:
                    anomalies.append("52-week low alert")
        except Exception:
            pass

        return PriceSnapshot(
            ticker=t,
            price=price,
            change_pct=chg,
            anomaly=" + ".join(anomalies),
        )
    except Exception:
        return PriceSnapshot(ticker=t, price=None, change_pct=None, anomaly="")


def _is_likely_ticker(t: str) -> bool:
    s = (t or "").strip().upper()
    if not s:
        return False
    if s in _BAD_TICKERS:
        return False
    if not _TICKER_RE.fullmatch(s):
        return False
    # Pure one-letter symbols are often noisy in headline parsing.
    if len(s) == 1:
        return False
    return True


def _weak_category(title: str, content: str, tickers: List[str]) -> str:
    text = f"{title} {content}"
    if _FED_RE.search(text) and not tickers:
        return "macro"
    if _EARN_RE.search(text):
        return "earnings"
    if _REPORT_RE.search(text):
        return "report"
    return "other"


def _weak_sentiment(title: str, content: str) -> str:
    text = f"{title} {content}"
    pos = len(_POS_RE.findall(text))
    neg = len(_NEG_RE.findall(text))
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def _sector_from_ticker(ticker: str) -> str:
    t = (ticker or "").upper()
    tech = {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "QCOM"}
    if t in tech:
        return "Technology"
    return "US Equities"


def _reason_for(cat: str, sentiment: str, title: str, content: str, ticker: str) -> str:
    if cat == "macro":
        return "Fed/rates headline indicates a likely broad market impact through yields, liquidity, and risk appetite."
    if cat == "earnings":
        return f"{ticker} earnings/guidance headline is likely to move price expectations and sector positioning."
    if cat == "report":
        return "Report/regulatory headline can shift valuation assumptions and near-term positioning."
    return "Stock-related headline with potential impact on positioning and intraday volatility."


def _summary_one_to_two_sentences(title: str, content: str) -> str:
    c = (content or "").strip()
    t = (title or "").strip()
    if not c or c.lower() == t.lower():
        return t
    s = re.sub(r"<[^>]+>", " ", c)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 280:
        s = s[:277].rstrip() + "..."
    return s


def _load_rows(db_path: str, since_hours: int) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        since = _as_iso_z(datetime.now(timezone.utc) - timedelta(hours=since_hours))
        q = """
        SELECT id, source, title, url, published_at, ingested_at, content
        FROM events_raw
        WHERE (lower(source) LIKE '%yahoo%' OR lower(url) LIKE '%yahoo.com%')
          AND coalesce(published_at, ingested_at, '') >= ?
        ORDER BY id DESC
        """
        return conn.execute(q, (since,)).fetchall()
    finally:
        conn.close()


def build_json(db_path: str, out_path: str, since_hours: int = 72, max_items: int = 600) -> Dict:
    rows = _load_rows(db_path, since_hours=since_hours)
    seen = set()
    out_stock = []
    out_fed = []
    all_tickers = set()

    price_cache: Dict[str, PriceSnapshot] = {}

    for _id, source, title, url, published_at, ingested_at, content in rows[:max_items]:
        title = (title or "").strip()
        content = (content or "").strip()
        if not title:
            continue
        if _NON_MARKET_RE.search(f"{title} {content}"):
            continue
        key = _norm_title_key(title)
        if not key or key in seen:
            continue
        seen.add(key)

        tickers = _extract_tickers(f"{title} {content}") or resolve_tickers(title, content)
        tickers = sorted({t for t in (tickers or []) if _is_likely_ticker(t)})
        for t in tickers:
            all_tickers.add(t)

        cat = _weak_category(title, content, tickers)
        sentiment = _weak_sentiment(title, content)
        summary = _summary_one_to_two_sentences(title, content)

        if cat == "macro":
            if _FED_RE.search(f"{title} {content}"):
                out_fed.append(
                    {
                        "category": "FED News",
                        "sentiment_to_market": sentiment,
                        "ai_analysis": sentiment,
                        "reason": _reason_for(cat, sentiment, title, content, ""),
                        "english_summary": summary,
                        "source": source or "Yahoo Finance",
                        "hyperlink": url,
                        "published_at": published_at or ingested_at,
                        "weak_label": {"category": cat, "sentiment": sentiment},
                    }
                )
            continue

        # Stock flow: require ticker
        if not tickers:
            continue
        tk = tickers[0]
        if tk in price_cache:
            snap = price_cache[tk]
        else:
            snap = _fetch_price_snapshot(tk)
            price_cache[tk] = snap
        sector = _sector_from_ticker(tk)
        price_line = ""
        if snap.price is not None and snap.change_pct is not None:
            price_line = f"{snap.price:.2f} USD ({snap.change_pct:+.2f}%)"
        elif snap.price is not None:
            price_line = f"{snap.price:.2f} USD"
        out_stock.append(
            {
                "category": sector,
                "ticker": f"${tk}",
                "price": price_line,
                "anomalies": snap.anomaly,
                "ai_analysis": sentiment,
                "reason": _reason_for(cat, sentiment, title, content, tk),
                "english_summary": summary,
                "source": source or "Yahoo Finance",
                "hyperlink": url,
                "published_at": published_at or ingested_at,
                "weak_label": {"category": cat, "sentiment": sentiment},
            }
        )

    payload = {
        "generated_at_utc": _as_iso_z(datetime.now(timezone.utc)),
        "window_hours": since_hours,
        "yahoo_tickers_found": sorted(all_tickers),
        "counts": {
            "yahoo_ticker_count": len(all_tickers),
            "stock_items": len(out_stock),
            "fed_items": len(out_fed),
        },
        "stock_items": out_stock,
        "fed_items": out_fed,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./data/news_sync.db")
    ap.add_argument("--out", default="./data/yahoo_impact.json")
    ap.add_argument("--since-hours", type=int, default=120)
    ap.add_argument("--max-items", type=int, default=600)
    args = ap.parse_args()

    data = build_json(args.db, args.out, since_hours=args.since_hours, max_items=args.max_items)
    print(f"wrote: {args.out}")
    print(json.dumps(data.get("counts", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
