import re
import os
from functools import lru_cache

from http_client import build_session

try:
    import financedatabase as fd
except Exception:
    fd = None

_ALIASES = {
    "palantir": "PLTR",
    "tyson foods": "TSN",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "tesla": "TSLA",
}


_STOP = {
    "the","a","an","and","or","for","with","from","to","in","on","of","at","by","as","is","are","was","were",
    "today","week","month","year","q1","q2","q3","q4","earnings","results","beats","misses","guidance","data",
    "u.s.","us","stock","stocks","market","markets","price","prices","index","report","reports",
}

# Explicit symbol patterns (higher confidence)
_CASHTAG_RE = re.compile(r"\\$([A-Z]{1,6}(?:\\.[A-Z]{1,2})?)\\b")
_EXCHANGE_RE = re.compile(r"\\b(?:NASDAQ|NYSE|AMEX|OTC|TSX|TSXV)\\s*:\\s*([A-Z]{1,6}(?:\\.[A-Z]{1,2})?)\\b", re.I)
_CRYPTO_RE = re.compile(r"\\b([A-Z]{2,10}-USD)\\b")
_PAREN_RE = re.compile(r"\\(([A-Z]{1,5})\\)")
_PAREN_STOP = {"Q1","Q2","Q3","Q4","FY","EPS","CEO","CFO","ETF","USD","CAD","EU","UK","PE","PV"}


def _load_equities():
    if fd is None:
        return None
    try:
        return fd.Equities()
    except Exception:
        return None


_EQUITIES = _load_equities()
_YH_SESSION = build_session(total_retries=2, backoff_factor=0.4)
_USE_FD = os.getenv("USE_FINANCEDB", "0") == "1"


def _extract_phrases(text: str):
    words = re.findall(r"[A-Za-z0-9&'.-]+", text)
    words = [w for w in words if w.lower() not in _STOP and len(w) > 2]
    phrases = set()
    if words:
        phrases.add(" ".join(words[:6]))
    for i in range(len(words) - 1):
        phrases.add(f"{words[i]} {words[i+1]}")
    for w in words[:8]:
        phrases.add(w)
    return [p for p in phrases if p]


def _extract_symbols(result):
    symbols = []
    if result is None:
        return symbols
    try:
        import pandas as pd
        if isinstance(result, pd.DataFrame):
            if "symbol" in result.columns:
                symbols = result["symbol"].tolist()
            elif "Symbol" in result.columns:
                symbols = result["Symbol"].tolist()
        elif isinstance(result, dict):
            # often returns dict keyed by symbol
            symbols = list(result.keys())
    except Exception:
        pass
    return [s for s in symbols if isinstance(s, str)]


def _yahoo_candidates(title: str) -> list:
    """
    Generate a small set of candidate queries for Yahoo Finance search.
    We deliberately keep this cheap (no heavy NLP deps).
    """
    t = (title or "").strip()
    if not t:
        return []

    # Prefer segments after common question prefixes: "Why X ...", "Is X ...", "How X ...", etc.
    lowered = t.lower()
    for pref in ("is ", "why ", "how ", "what ", "where ", "when "):
        if lowered.startswith(pref):
            t2 = t[len(pref):].strip()
            if t2:
                t = t2
            break

    # Extract consecutive Capitalized Words (e.g., "Tyson Foods", "Randy Smallwood", "GE Vernova")
    # Also allow leading uppercase acronyms like "GE".
    caps = re.findall(r"\\b(?:[A-Z]{2,5}\\s+)?(?:[A-Z][a-z0-9&'.-]+)(?:\\s+[A-Z][a-z0-9&'.-]+){0,3}\\b", t)
    out = []
    for c in caps[:6]:
        if c.lower() not in _STOP and len(c) >= 3:
            out.append(c)

    # Add a truncated title fallback
    words = re.findall(r"[A-Za-z0-9&'.-]+", t)
    words = [w for w in words if w.lower() not in _STOP]
    if words:
        out.append(" ".join(words[:8]))

    # Dedup while keeping order
    seen = set()
    deduped = []
    for q in out:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(q)
    return deduped[:6]


@lru_cache(maxsize=512)
def _yahoo_search(query: str) -> list:
    """
    Yahoo Finance search endpoint.
    Returns a list of symbols ordered by Yahoo relevance.
    """
    q = (query or "").strip()
    if not q:
        return []
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "quotesCount": 8,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "true",
    }
    try:
        r = _YH_SESSION.get(url, params=params, timeout=8, headers={"User-Agent": "mlstockbot/0.1"})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    # Build tokens from query for name matching (avoid returning random tickers).
    q_words = re.findall(r"[A-Za-z0-9&'.-]+", q)
    q_words = [w.lower() for w in q_words if w.lower() not in _STOP and len(w) >= 4]
    # Avoid matching generic finance words (these cause false positives like "Tech", "Market", etc.)
    generic = {
        "stock","stocks","market","markets","shares","equity","equities","tech","financial","finance","dollar",
        "prices","price","rally","drops","falls","gains","losses","rotation","sector","index","indices",
        "weekly","monthly","daily","chart","futures","commodities","precious","metals","canadian","global",
        "dividend","dividends","income","reliable","today","green","glows","soared","beats","misses",
        "etf","fund","portfolio","options","rates","yield","yields",
        "energy",  # too generic; causes false positives (e.g., matches many companies)
    }
    q_words = [w for w in q_words if w not in generic]

    # Require stronger matching:
    # - If we have 2+ meaningful tokens, require at least 2 tokens to appear in the security name.
    # - If we have 1 meaningful token, require it to appear.
    required_matches = 2 if len(q_words) >= 2 else 1

    out = []
    for qt in (data.get("quotes") or []):
        sym = (qt.get("symbol") or "").upper().strip()
        qtype = (qt.get("quoteType") or "").upper().strip()
        region = (qt.get("region") or "").upper().strip()
        exch = (qt.get("exchange") or "").upper().strip()
        name = (qt.get("shortname") or qt.get("longname") or "").lower()

        if not sym:
            continue

        # Allow crypto pairs like BTC-USD (but these are usually handled elsewhere).
        if _CRYPTO_RE.fullmatch(sym):
            out.append(sym)
            continue

        # For equities: be conservative.
        if qtype != "EQUITY":
            continue
        if region and region != "US":
            continue
        if exch and exch not in {"NMS", "NYQ", "ASE", "NGM", "NCM"}:
            continue
        if not re.fullmatch(r"[A-Z]{1,5}(?:\\.[A-Z]{1,2})?", sym):
            continue

        # Name match requirement (reduces false positives):
        # accept only when enough meaningful tokens appear in the security name.
        if not (q_words and name):
            continue
        match_count = sum(1 for w in q_words if w in name)
        if match_count < required_matches:
            continue

        out.append(sym)

    # Preserve order, dedup
    seen = set()
    deduped = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped[:5]


@lru_cache(maxsize=256)
def _fd_search(term: str):
    if _EQUITIES is None or not term:
        return []
    try:
        res = _EQUITIES.search(term, case_sensitive=False)
    except Exception:
        return []
    return _extract_symbols(res)[:5]


def resolve_tickers(title: str, content: str) -> list:
    raw = f"{title or ''} {content or ''}"
    text_l = raw.lower()

    out = []

    # 1) Explicit symbols present in text
    out.extend([m.upper() for m in _CASHTAG_RE.findall(raw.upper())])
    out.extend([m.upper() for m in _EXCHANGE_RE.findall(raw)])
    out.extend(_CRYPTO_RE.findall(raw.upper()))
    for m in _PAREN_RE.findall(raw.upper()):
        if m in _PAREN_STOP:
            continue
        out.append(m)

    # 2) Manual aliases (cheap wins)
    for k, v in _ALIASES.items():
        # Use word/phrase boundary matching to avoid substring bugs (e.g., "meta" in "metals").
        if re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", text_l):
            out.append(v)

    # 3) Yahoo Finance search for broad ticker coverage
    # Only attempt Yahoo lookup when the headline contains at least one company-ish token.
    # This prevents false positives on generic market/commodity headlines.
    for q in _yahoo_candidates(title or ""):
        # Require at least one non-stop, non-generic word of length >= 4.
        q_words = re.findall(r"[A-Za-z0-9&'.-]+", q)
        q_words = [w.lower() for w in q_words if w.lower() not in _STOP and len(w) >= 4]
        if not q_words:
            continue
        hits = _yahoo_search(q)
        if hits:
            out.extend(hits)
            break

    # 4) FinanceDatabase search (optional fallback)
    # Disabled by default because fuzzy matches can introduce false positives.
    if _USE_FD:
        for q in _yahoo_candidates(title or ""):
            out.extend(_fd_search(q))
            if len(set(out)) >= 5:
                break

    # Final: dedup + sort for stable output
    return sorted(set([s for s in out if s]))
