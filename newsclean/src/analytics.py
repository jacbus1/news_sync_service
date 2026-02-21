import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple


# Capture tickers either as cashtags ($AAPL) or bare uppercase tokens (AAPL).
# Use finditer to avoid `re.findall` group-shape gotchas.
_TICKER_RE = re.compile(r"(?:\$(?P<cash>[A-Z]{1,6})|(?P<bare>\b[A-Z]{2,6}\b))")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_STOPWORDS = {
    "the","and","for","with","from","will","this","that","news","stock","market","usd",
    "into","over","under","after","before","about","their","there","while","would","could",
    "also","more","less","than","then","when","what","which","your","have","has","had",
    "but","not","are","was","were","you","they","them","our","out","who","why","how",
    "says","said","new","now","today","week","month","year","price","prices","stocks",
    "crypto","bitcoin","ethereum","shares","share","company","companies","inc","corp",
}
_POS_WORDS = {
    "beat","beats","surge","soar","soars","rally","rallies","rise","rises","gain","gains",
    "upgrade","upgraded","record","strong","growth","bull","bullish","profit","profits",
    "approval","approved","buyback","dividend","win","wins","positive","optimistic",
}
_NEG_WORDS = {
    "miss","misses","drop","drops","plunge","plunges","fall","falls","lawsuit","probe","sec",
    "downgrade","downgraded","loss","losses","cut","cuts","warning","warns","negative",
    "bear","bearish","risk","recession","fraud","halt","halts","hack","bankrupt","bankruptcy",
}


@dataclass
class FearGreedResult:
    value: int
    label: str
    sentiment_score: float
    volume_score: float
    count_24h: int
    avg_7d: float


def _extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    s = text.upper()
    out: List[str] = []
    for m in _TICKER_RE.finditer(s):
        t = (m.group("cash") or m.group("bare") or "").strip().upper()
        if 2 <= len(t) <= 6:
            out.append(t)
    noise = {"THE", "AND", "FOR", "WITH", "FROM", "WILL", "THIS", "THAT", "NEWS", "STOCK", "MARKET", "USD"}
    return [t for t in out if t and t not in noise]


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return [w for w in words if w not in _STOPWORDS and len(w) >= 3]


def _label_from_value(v: int) -> str:
    if v <= 24:
        return "Extreme Fear"
    if v <= 44:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


def _get_events_since(conn: sqlite3.Connection, since_dt: datetime) -> List[Dict]:
    rows = conn.execute(
        "SELECT source, title, url, published_at, ingested_at, content FROM events_raw "
        "WHERE ingested_at >= ?",
        (since_dt.isoformat(),)
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "source": r[0], "title": r[1], "url": r[2],
            "published_at": r[3], "ingested_at": r[4], "content": r[5]
        })
    return out


def compute_heat(sqlite_path: str, minutes: int = 15, top_tickers: int = 10, top_words: int = 12):
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(minutes=minutes)
    conn = sqlite3.connect(sqlite_path)
    try:
        events = _get_events_since(conn, since_dt)
    finally:
        conn.close()

    tickers = compute_top_tickers(events, top_n=top_tickers)
    words = compute_top_words(events, top_n=top_words)

    src_ctr = Counter()
    for e in events:
        src = e.get("source") or "unknown"
        src_ctr[src] += 1

    return {
        "window_minutes": minutes,
        "events": events,
        "top_tickers": tickers,
        "top_words": words,
        "top_sources": src_ctr.most_common(8),
    }


def compute_fear_greed(sqlite_path: str) -> FearGreedResult:
    now = datetime.now(timezone.utc)
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)

    conn = sqlite3.connect(sqlite_path)
    try:
        events_24h = _get_events_since(conn, last_24h)
        events_7d = _get_events_since(conn, last_7d)
    finally:
        conn.close()

    count_24h = len(events_24h)
    avg_7d = (len(events_7d) / 7.0) if events_7d else 0.0

    pos = 0
    neg = 0
    for e in events_24h:
        text = f"{e.get('title','')} {e.get('content','')}".lower()
        for w in _POS_WORDS:
            if w in text:
                pos += 1
        for w in _NEG_WORDS:
            if w in text:
                neg += 1

    # sentiment score in [-1, 1]
    sent = 0.0
    if pos + neg > 0:
        sent = (pos - neg) / float(pos + neg)
    sent_score = int(round((sent + 1.0) * 50))

    ratio = count_24h / (avg_7d + 1.0)
    volume_score = int(max(0, min(100, round(50 * ratio))))

    value = int(round(0.6 * sent_score + 0.4 * volume_score))
    return FearGreedResult(
        value=value,
        label=_label_from_value(value),
        sentiment_score=sent_score,
        volume_score=volume_score,
        count_24h=count_24h,
        avg_7d=avg_7d,
    )


def compute_top_tickers(events: List[Dict], top_n: int = 20) -> List[Tuple[str, int]]:
    ctr = Counter()
    for e in events:
        text = f"{e.get('title','')} {e.get('content','')}"
        for t in _extract_tickers(text):
            ctr[t] += 1
    return ctr.most_common(top_n)


def compute_top_words(events: List[Dict], top_n: int = 30) -> List[Tuple[str, int]]:
    ctr = Counter()
    for e in events:
        text = f"{e.get('title','')} {e.get('content','')}"
        for w in _tokens(text):
            ctr[w] += 1
    return ctr.most_common(top_n)


def render_daily_report(date: datetime, fgi: FearGreedResult, top_tickers, top_words) -> str:
    lines = []
    lines.append(f"# Daily Heatmap & Fear/Greed ({date.date().isoformat()} UTC)")
    lines.append("")
    lines.append(f"Fear & Greed Index: **{fgi.value} ({fgi.label})**")
    lines.append(f"- Sentiment score: {fgi.sentiment_score}")
    lines.append(f"- Volume score: {fgi.volume_score}")
    lines.append(f"- Items (24h): {fgi.count_24h}")
    lines.append(f"- Avg items (7d): {fgi.avg_7d:.2f}")
    lines.append("")
    lines.append("## Most Discussed Tickers")
    for t, c in top_tickers:
        lines.append(f"- ${t}: {c}")
    lines.append("")
    lines.append("## Most Used Words")
    for w, c in top_words:
        lines.append(f"- {w}: {c}")
    return "\n".join(lines) + "\n"


def write_daily_report(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
