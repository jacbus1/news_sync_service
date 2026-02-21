from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from urllib.parse import urlparse
import html
import re
import json
import html as _html_mod
import unicodedata

from classifier import CATEGORIES, classify_item
from market_data import get_quote, format_price_line
from translator import translate_to_zh
from llm_client import chat as llm_chat
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None
from company_names_zh import load_company_names_zh
from sector_classifier import classify_sector


@dataclass
class DigestConfig:
    max_total_items: int = 10
    max_per_category: int = 3
    include_price: bool = True
    include_details: bool = True
    details_max_chars: int = 240
    summary_min_sentences: int = 3
    summary_max_sentences: int = 6
    translate_zh: bool = False
    translate_base_url: str = ""
    translate_model: str = ""
    translate_fallback_model: str = ""
    translate_timeout_seconds: int = 25
    price_max_tickers_total: int = 6
    price_max_per_item: int = 1
    impact_filter_enabled: bool = True


def _item_details(it: dict, max_chars: int) -> str:
    """
    Prefer a snippet if present in content_enricher output.
    """
    content = (it.get("content") or "").strip()
    title = (it.get("title") or "").strip()
    if not content:
        return ""
    # If content is basically the title only, nothing to add.
    if title and content.lower() == title.lower():
        return ""
    # Keep a short, single-paragraph detail.
    s = html.unescape(content.replace("\n", " ").strip())
    # Strip any embedded HTML tags and raw URLs that make summaries messy (common in Google News RSS).
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    # Normalize spaced domains like "marketwatch. com" -> "marketwatch.com"
    s = re.sub(r"([A-Za-z0-9])\s*\.\s*([A-Za-z0-9])", r"\1.\2", s)
    # Strip CSS / style artifacts that sometimes leak from pages (e.g., transcripts).
    s = re.sub(r"@media\s*\([^)]*\)\s*\{[^{}]{0,2000}\}", " ", s, flags=re.I)
    s = re.sub(r"\.[a-z0-9_-]+\s*\{[^{}]{0,2000}\}", " ", s, flags=re.I)
    for _ in range(3):
        ns = re.sub(r"\{[^{}]{0,1200}\}", " ", s)
        if ns == s:
            break
        s = ns
    s = s.replace("{", " ").replace("}", " ")
    # Remove common boilerplate
    s = re.sub(r"\bSource\s*:\s*[A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)+\.?\s*", " ", s, flags=re.I)
    s = re.sub(r"\bSee\s+link\s+for\s+full\s+details\b\.?\s*", " ", s, flags=re.I)
    s = re.sub(r"\blink\s+for\s+full\s+details\b\.?\s*", " ", s, flags=re.I)
    s = re.sub(r"\bPublished\s*:\s*[^.]+\.(\s+)?", " ", s, flags=re.I)
    # Strip common wire prefixes like "(RTTNews) - " / "RTTNews - "
    s = re.sub(r"^\s*\(?\s*RTTNews\s*\)?\s*-\s*", " ", s, flags=re.I)
    s = re.sub(r"\(\s*RTTNews\s*\)", " ", s, flags=re.I)
    s = s.replace("來源：marketwatch.com。欲知更多詳情，請點擊連結。", " ")
    s = s.replace("欲知更多詳情，請點擊連結。", " ")
    # Drop "Key Points"/Chinese equivalent from details.
    s = re.sub(r"^\s*Key Points\s*[:\-]?\s*", " ", s, flags=re.I)
    s = re.sub(r"^\s*關鍵點\s*[:：]?\s*", " ", s)
    # Drop leading ticker-only prefix like "(ILMN) ..." that adds noise.
    s = re.sub(r"^\s*\([A-Z]{1,6}\)\s*", " ", s)
    # Also remove stray "Key Points"/"關鍵點" labels inside the snippet.
    s = re.sub(r"\bKey Points\b\s*[:\-]?\s*", " ", s, flags=re.I)
    s = re.sub(r"\b關鍵點\b\s*[:：]?\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Remove assistant-like / conversational filler (must not appear in public posts).
    assistantish = re.compile(
        r"\b(i couldn't find|i cannot find|if you would like|please note|alternatively,|"
        r"i can try to|provide a translation|once it becomes available|confidential|proprietary|"
        r"do you have a specific link|need a quote)\b",
        re.I,
    )
    if assistantish.search(s):
        kept = []
        for sent in _split_sentences(s):
            if assistantish.search(sent):
                continue
            kept.append(sent.strip())
        s = " ".join([x for x in kept if x]).strip()

    # Remove transcript speaker blocks (names/roles repeated, high noise).
    if ("chief executive officer" in s.lower() or "executive chairman" in s.lower()) and (s.count("—") >= 1 or s.count("-") >= 6):
        kept = []
        for sent in _split_sentences(s):
            sl = sent.lower()
            if "chief executive officer" in sl or "executive chairman" in sl or "founder" in sl:
                continue
            kept.append(sent.strip())
        s = " ".join([x for x in kept if x]).strip()
    if title and s.lower().startswith(title.lower()):
        s = s[len(title) :].strip(" -:\n\t")
    if not s:
        return ""
    if len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."
    return s


def _split_sentences(text: str) -> List[str]:
    # Very lightweight sentence splitter; good enough for headlines/snippets.
    s = (text or "").replace("\n", " ").strip()
    if not s:
        return []
    out = []
    cur = []
    for ch in s:
        cur.append(ch)
        if ch in ".!?":
            sent = "".join(cur).strip()
            cur = []
            if sent:
                out.append(sent)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _llm_summary_fallback(it: dict, cfg: DigestConfig) -> str:
    """
    If rule-based summary extraction is empty, ask local LLM to produce
    a short factual summary. Try primary model first, then fallbacks.
    """
    base_url = (cfg.translate_base_url or "").strip()
    primary = (cfg.translate_model or "").strip()
    extra = [m.strip() for m in (cfg.translate_fallback_model or "").split(",") if m.strip()]
    models = [m for m in [primary] + extra if m]
    if not models:
        return ""

    title = (it.get("title") or "").strip()
    content = (it.get("content") or "").strip()
    if not (title or content):
        return ""

    # Keep prompt payload bounded.
    raw = f"Title: {title}\n\nContent:\n{content[:1800]}".strip()
    prompt = (
        "Summarize this market news in 1-2 factual English sentences. "
        "No advice, no opinions, no filler, no boilerplate. "
        "Do not mention inability to find data. "
        "Output plain text only.\n\n" + raw
    )
    messages = [
        {"role": "system", "content": "You are a neutral financial news summarizer."},
        {"role": "user", "content": prompt},
    ]

    assistantish = re.compile(
        r"\b(i couldn't find|i cannot find|if you would like|please note|alternatively,|"
        r"i can try to|provide a translation|once it becomes available|confidential|proprietary|"
        r"do you have a specific link|need a quote|unable to|cannot access)\b",
        re.I,
    )
    for m in models:
        try:
            out = (llm_chat(base_url, m, messages, timeout=int(cfg.translate_timeout_seconds)) or "").strip()
        except Exception:
            out = ""
        if not out:
            continue
        out = html.unescape(out)
        out = re.sub(r"<[^>]+>", " ", out)
        out = re.sub(r"https?://\S+", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        if not out:
            continue
        if assistantish.search(out):
            continue
        # keep at most first 2 sentences / 250 chars
        sents = [s.strip() for s in _split_sentences(out) if s.strip()]
        if sents:
            out = " ".join(sents[:2]).strip()
        if len(out) > 250:
            out = out[:247].rstrip() + "..."
        if re.search(r"[A-Za-z0-9]{8,}", out):
            return out
    return ""


_ALLOWED_GICS = {
    "Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
}


def _extract_json_obj(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _sector_zh_to_en(sector_zh: str) -> str:
    m = {
        "科技": "Technology",
        "醫療": "Health Care",
        "金融": "Financials",
        "消費": "Consumer Discretionary",
        "能源": "Energy",
        "公用事業": "Utilities",
        "房地產": "Real Estate",
        "原材料": "Materials",
        "通訊服務": "Communication Services",
        "工業": "Industrials",
        "必需消費": "Consumer Staples",
        "經濟": "Macro",
        "美股": "US Equities",
        "加密": "Crypto",
        "國防": "Industrials",
        "體育": "Consumer Discretionary",
    }
    return m.get((sector_zh or "").strip(), "")


def _heuristic_impact(it: dict, category_key: str) -> Dict[str, Any]:
    txt = f"{it.get('title') or ''} {it.get('content') or ''}".lower()
    tickers = [t for t in (it.get("tickers") or []) if t]
    strong = any(
        k in txt
        for k in [
            "earnings",
            "guidance",
            "fomc",
            "federal reserve",
            "interest rate",
            "cpi",
            "ppi",
            "merger",
            "acquisition",
            "regulation",
            "sanction",
            "war",
            "tariff",
            "supply chain",
        ]
    )
    has_impact = bool(strong or category_key in {"earnings", "report"} or tickers)
    sec_en = _sector_zh_to_en(classify_sector(it, category_key))
    if sec_en in {"Macro", "US Equities", "Crypto", ""}:
        sec_en = "Technology" if any(t in {"NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"} for t in tickers) else "Financials"
    # Strict FED policy scope (avoid false positives like "Fed Cattle Exchange").
    fed_false_positive = any(
        k in txt
        for k in [
            "fed cattle",
            "fed cattle exchange",
            "cattle exchange",
            "federation",
            "powell industries",
        ]
    )
    fed_news = (not fed_false_positive) and any(
        k in txt
        for k in [
            "federal reserve",
            "fomc",
            "powell",
            "dot plot",
            "fed minutes",
            "fomc minutes",
            "fed funds",
            "fed chair",
            "rate cut",
            "rate cuts",
            "rate hike",
            "rate hikes",
            "treasury yield",
            "10-year yield",
            "30-year yield",
            "u.s. rates",
            "us rates",
        ]
    )

    sentiment = "neutral"
    if any(x in txt for x in ["beats", "surge", "jumps", "rise", "strong", "record"]):
        sentiment = "positive"
    if any(x in txt for x in ["misses", "plunge", "falls", "drop", "weak", "cut"]):
        sentiment = "negative"

    # FED-specific multi-factor heuristic:
    # Surprise/tone + long-end yields + risk-asset transmission.
    fed_neg = 0
    fed_pos = 0
    if fed_news:
        if any(x in txt for x in ["hawkish", "higher for longer", "sticky inflation", "inflation remains elevated", "upside inflation risk"]):
            fed_neg += 2
        if any(x in txt for x in ["dovish", "disinflation", "soft landing", "cooling inflation"]):
            fed_pos += 2
        if any(x in txt for x in ["fewer cuts", "pause cuts", "hold rates", "rate cuts delayed", "keep rates unchanged"]):
            fed_neg += 1
        if any(x in txt for x in ["more cuts", "faster cuts", "front-loaded cuts", "cuts this year"]):
            fed_pos += 1
        if any(x in txt for x in ["10-year yield rises", "10-year yield rose", "30-year yield rises", "long-end yields rise", "yield curve steepens"]):
            fed_neg += 2
        if any(x in txt for x in ["10-year yield falls", "10-year yield dropped", "30-year yield falls", "long-end yields fall", "yield curve bull steepening"]):
            fed_pos += 2
        if any(x in txt for x in ["risk-off", "stocks fall", "equities slide", "nasdaq drops", "s&p 500 falls", "vix jumps"]):
            fed_neg += 1
        if any(x in txt for x in ["risk-on", "stocks rise", "equities rally", "nasdaq gains", "s&p 500 rises", "vix falls"]):
            fed_pos += 1
        if fed_neg > fed_pos:
            sentiment = "negative"
        elif fed_pos > fed_neg:
            sentiment = "positive"
        else:
            sentiment = "neutral"
    impact_type = "other"
    if category_key == "earnings":
        impact_type = "earnings"
    elif fed_news or any(k in txt for k in ["interest rate", "cpi", "ppi", "jobs", "payroll"]):
        impact_type = "macro"
    elif any(k in txt for k in ["war", "sanction", "geopolitical", "tariff"]):
        impact_type = "geopolitical"
    elif any(k in txt for k in ["merger", "acquisition", "takeover"]):
        impact_type = "merger"
    elif any(k in txt for k in ["sec", "regulation", "probe", "investigation"]):
        impact_type = "regulation"
    reason = ""
    if fed_news:
        if sentiment == "negative":
            reason = "Fed tone/yield setup is restrictive: higher long-end yields and delayed cuts pressure equity valuation, especially rate-sensitive sectors."
        elif sentiment == "positive":
            reason = "Fed tone/yield setup is supportive: easing yield pressure and a more dovish path improve risk-asset valuation."
        else:
            reason = "Fed signal is largely in-line with expectations; market impact depends on follow-through in long-end yields and liquidity conditions."

    return {
        "has_market_impact": has_impact,
        "sectors": [sec_en] if sec_en else [],
        "tickers": tickers,
        "impact_type": impact_type,
        "sentiment": sentiment,
        "reason": reason,
    }


def _analyze_market_impact(it: dict, category_key: str, cfg: DigestConfig) -> Dict[str, Any]:
    base_url = (cfg.translate_base_url or "").strip()
    model = (cfg.translate_model or "").strip()
    fallbacks = [m.strip() for m in (cfg.translate_fallback_model or "").split(",") if m.strip()]
    models = [m for m in [model] + fallbacks if m]
    if not models:
        return _heuristic_impact(it, category_key)

    title = (it.get("title") or "").strip()
    content = (it.get("content") or "").strip()
    tickers = [t for t in (it.get("tickers") or []) if t]
    prompt = (
        "You are a market impact classifier.\n"
        "Output strict JSON only with fields:\n"
        '{"has_market_impact": true/false, "sectors": [], "tickers": [], '
        '"impact_type":"earnings|macro|geopolitical|merger|regulation|other", '
        '"sentiment":"positive|negative|neutral", "reason":"one short sentence"}\n'
        "Use only GICS sectors:\n"
        "Technology, Health Care, Financials, Consumer Discretionary, Communication Services, "
        "Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials.\n"
        "News:\n"
        f"Title: {title}\n"
        f"Tickers: {', '.join(tickers)}\n"
        f"Content: {content[:2200]}"
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON generator."},
        {"role": "user", "content": prompt},
    ]
    for m in models:
        try:
            raw = llm_chat(base_url, m, messages, timeout=int(cfg.translate_timeout_seconds))
        except Exception:
            raw = ""
        obj = _extract_json_obj(raw or "")
        if not obj:
            continue
        # Normalize and validate
        hs = bool(obj.get("has_market_impact"))
        secs = obj.get("sectors") or []
        if not isinstance(secs, list):
            secs = []
        secs = [str(x).strip() for x in secs if str(x).strip() in _ALLOWED_GICS]
        tks = obj.get("tickers") or []
        if not isinstance(tks, list):
            tks = []
        tks = [str(x).strip().upper() for x in tks if str(x).strip()]
        impact_type = str(obj.get("impact_type") or "other").strip().lower()
        if impact_type not in {"earnings", "macro", "geopolitical", "merger", "regulation", "other"}:
            impact_type = "other"
        sentiment = str(obj.get("sentiment") or "neutral").strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        reason = str(obj.get("reason") or "").strip()
        return {
            "has_market_impact": hs,
            "sectors": secs,
            "tickers": tks or tickers,
            "impact_type": impact_type,
            "sentiment": sentiment,
            "reason": reason,
        }
    return _heuristic_impact(it, category_key)


def _anomaly_line(it: dict, en_summary: str) -> str:
    tickers = [t for t in (it.get("tickers") or []) if t]
    if not tickers:
        return ""
    q = get_quote(tickers[0], cache_ttl_seconds=60)
    if q is None:
        return ""
    pct = float(q.day_change_pct)
    if abs(pct) >= 5:
        if pct > 0:
            return f"Anomaly: Strong daily surge {pct:.2f}%"
        return f"Anomaly: Sharp daily drop {abs(pct):.2f}%"
    if abs(pct) >= 3:
        if pct > 0:
            return f"Anomaly: Notable gain {pct:.2f}%"
        return f"Anomaly: Notable loss {abs(pct):.2f}%"
    if re.search(r"\b(52-week high|52 week high|record high)\b", en_summary or "", re.I):
        return "Anomaly: 52-week high alert"
    if re.search(r"\b(52-week low|52 week low)\b", en_summary or "", re.I):
        return "Anomaly: 52-week low alert"
    return ""


def _get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return ""


_TICKER_HINT_RE = re.compile(r"\(([A-Z]{1,6})\)")
_STOP_TICKER_HINTS = {
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "FY",
    "EPS",
    "ETF",
    "CEO",
    "CFO",
    "USA",
    "USD",
    "PE",
    "PV",
}


def _ticker_hint_from_text(text: str) -> str:
    """
    Best-effort: extract a ticker from patterns like "(ILMN)" in title/content.
    Used only for display/translation templates (doesn't need to be perfect).
    """
    s = (text or "").strip()
    if not s:
        return ""
    for m in _TICKER_HINT_RE.finditer(s):
        t = (m.group(1) or "").strip().upper()
        if not t or t in _STOP_TICKER_HINTS:
            continue
        # Avoid grabbing years like (2026) and other numeric junk.
        if re.fullmatch(r"\d+", t):
            continue
        return t
    return ""


def _source_display(url: str, source: str) -> str:
    dom = _get_domain(url) or (source or "").strip()
    dom = dom.lower()
    if "finance.yahoo.com" in dom or dom == "yahoo.com":
        return "Yahoo Finance"
    if "nasdaq.com" in dom:
        return "Nasdaq"
    if "marketwatch.com" in dom:
        return "MarketWatch"
    if "investing.com" in dom:
        return "Investing.com"
    if "reuters.com" in dom:
        return "Reuters"
    if "news.google.com" in dom:
        return "Google News"
    return dom or "Source"


_TAG_FED = re.compile(r"\b(federal\s+reserve|fomc|powell|fed\s+funds|fed\s+minutes|fed\s+chair)\b", re.I)
_TAG_TRUMP = re.compile(r"\btrump\b", re.I)
_TAG_EARN = re.compile(r"\b(earnings|eps|guidance|results|transcript)\b", re.I)
_TAG_SEC = re.compile(r"\b(sec\b|10-?k|10-?q|8-?k|13f)\b", re.I)

# Index / benchmark tagging (SPX/DJI/NDX/VIX, S&P 500, Nasdaq 100, etc.)
_TAG_INDEX = re.compile(
    r"\b("
    r"s\s*&\s*p\s*500|s&p\s*500|s&p\s*500\s+index|"
    r"dow\s+jones|dow\s+industrials|dow\s+30|"
    r"nasdaq\s*100|nasdaq\s+100|nasdaq\s+composite|"
    r"\^gspc|\^dji|\^ndx|\^ixic|"
    r"spx|dji|ndx|vix"
    r")\b",
    re.I,
)

_PCT_SPACING_RE = re.compile(r"([+-]?\d+)\.\s+(\d+)%")


def _normalize_change_signs(text: str) -> str:
    """
    Cleanup for public posts:
    - Fix spacing artifacts like "-1. 23%" -> "-1.23%"
    - Avoid double-negative: "down -1.23%" -> "down 1.23%" and "下跌-1.23%" -> "下跌 1.23%"
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = _PCT_SPACING_RE.sub(r"\1.\2%", s)
    # English: down -X% / up +X%
    s = re.sub(r"\b(down|declined|lower|fell|dropped|slid|off)\s*[-+]\s*(\d+(?:\.\d+)?)%", r"\1 \2%", s, flags=re.I)
    s = re.sub(r"\b(up|rose|gained|higher|advanced|climbed|added)\s*[-+]\s*(\d+(?:\.\d+)?)%", r"\1 \2%", s, flags=re.I)
    # Chinese: 下跌-1.23% / 上漲+1.23%
    s = re.sub(r"(下跌|下滑|下降)\s*[-+]\s*(\d+(?:\.\d+)?)%", r"\1 \2%", s)
    s = re.sub(r"(上漲|上升)\s*[-+]\s*(\d+(?:\.\d+)?)%", r"\1 \2%", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sector_hashtag_en(sector_label: str) -> str:
    """Map sector label to English hashtag token (without leading #)."""
    sec = (sector_label or "").strip().lstrip("#")
    if not sec:
        return ""
    mapping = {
        "經濟": "Macro",
        "國防": "Defense",
        "科技": "Technology",
        "能源": "Energy",
        "金融": "Financials",
        "醫療": "Healthcare",
        "消費": "Consumer",
        "體育": "Sports",
        "加密": "Crypto",
        "原材料": "Materials",
        "房地產": "RealEstate",
        "美股": "USStocks",
    }
    if sec in mapping:
        return mapping[sec]
    # Fallback: keep ASCII letters/digits only for Telegram hashtag compatibility.
    return re.sub(r"[^A-Za-z0-9]+", "", sec)


def _tags_for_item(it: dict, category_key: str) -> List[str]:
    title = (it.get("title") or "")
    content = (it.get("content") or "")
    text = f"{title} {content}"

    tags: List[str] = []
    if category_key == "earnings":
        tags.append("Earning快訊")
    text_l = text.lower()
    fed_noise = any(x in text_l for x in ["fed cattle", "fed cattle exchange", "powell industries"])
    if _TAG_FED.search(text) and not fed_noise:
        tags.append("FED")
    if _TAG_TRUMP.search(text):
        tags.append("TRUMP")
    if _TAG_SEC.search(text):
        tags.append("SEC")
    if category_key != "earnings" and _TAG_EARN.search(text):
        tags.append("EARNINGS")

    seen = set()
    out = []
    for t in tags:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _ensure_period(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[-1] in ".!?":
        return s
    return s + "."


_KEYPOINTS_FIX_RE = re.compile(r"\b(Key Points)([A-Z])")


def _clean_topic(title: str) -> str:
    t = html.unescape((title or "").strip())
    t = re.sub(r"\s+", " ", t)
    # Drop "Key Points" prefixes (noise from some sources).
    t = re.sub(r"^\s*Key Points\s*[:\-]?\s*", "", t, flags=re.I)
    # Fix common concatenation artifact: "Key PointsAmazon" -> "Key Points Amazon"
    t = _KEYPOINTS_FIX_RE.sub(r"\1 \2", t)
    return t


def _build_summary_body(it: dict, cfg: DigestConfig) -> str:
    """
    Build a 3-6 sentence objective summary using snippet/content + basic metadata.
    (The headline goes to the Topic line.)
    """
    title = _clean_topic(it.get("title") or "(no title)")
    url = (it.get("url") or "").strip()
    # source/published are shown in the Source line; avoid repeating in summary.

    sentences: List[str] = []

    # Add 1-3 sentences from details/snippet when available.
    details = _item_details(it, cfg.details_max_chars)
    if details:
        details = details.replace("Snippet:", "").strip()
        # Avoid repeating metadata that we show elsewhere.
        details = re.sub(r"\bSource\s*:\s*[A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)+\.?\s*", "", details, flags=re.I)
        details = re.sub(r"\bSee\s+link\s+for\s+full\s+details\b\.?\s*", "", details, flags=re.I)
        details = re.sub(r"\blink\s+for\s+full\s+details\b\.?\s*", "", details, flags=re.I)
        details = re.sub(r"\bPublished\s*:\s*[^.]+\.?\s*", "", details, flags=re.I)
        # If a stray domain suffix remains (e.g., \"com.\"), drop it.
        details = re.sub(r"^(com|net|org)\s*\.\s*", "", details.strip(), flags=re.I)
        for s in _split_sentences(details)[:3]:
            if not s:
                continue
            if s.lower() in {title.lower(), _ensure_period(title).lower()}:
                continue
            # Remove "Key Points"/"關鍵點" labels if they leaked into sentence start.
            s = re.sub(r"^\s*Key Points\s*[:\-]?\s*", "", s, flags=re.I)
            s = re.sub(r"^\s*關鍵點\s*[:：]?\s*", "", s)
            # Skip junk fragments (punctuation-only / too short after filtering).
            if len(re.sub(r"[^A-Za-z0-9]+", "", s)) < 12:
                continue
            sentences.append(_ensure_period(s))

    # Cap to max sentences.
    sentences = sentences[: cfg.summary_max_sentences]

    # Final clamp
    sentences = sentences[: cfg.summary_max_sentences]
    out = " ".join([s.strip() for s in sentences if s.strip()]).strip()
    # If rule-based extraction failed, try LLM fallback chain.
    if not re.search(r"[A-Za-z0-9]", out):
        out = _llm_summary_fallback(it, cfg)
    # If still no alnum content, treat as empty.
    if not re.search(r"[A-Za-z0-9]", out):
        return ""
    return out


def _maybe_bilingual_summary(it: dict, cfg: DigestConfig) -> str:
    # Backwards-compatible helper; keep returning EN only.
    # Translation is emitted as a separate "中文總結：" line by the digest builders.
    return _build_summary_body(it, cfg)


def _maybe_translate_zh(en_text: str, cfg: DigestConfig) -> str:
    if not cfg.translate_zh:
        return ""
    zh = translate_to_zh(
        en_text,
        cfg.translate_base_url,
        cfg.translate_model,
        timeout=cfg.translate_timeout_seconds,
        fallback_model=cfg.translate_fallback_model,
    )
    return (zh or "").strip()


def _maybe_translate_title_zh(title_en: str, cfg: DigestConfig) -> str:
    if not cfg.translate_zh:
        return ""
    zh = translate_to_zh(
        title_en,
        cfg.translate_base_url,
        cfg.translate_model,
        timeout=cfg.translate_timeout_seconds,
        fallback_model=cfg.translate_fallback_model,
    )
    return (zh or "").strip()


def _build_item_block(it: dict, category_key: str, cfg: DigestConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Public-facing per-item block (Telegram HTML).
    """
    def _normalize_url(u: str) -> str:
        # RSS/feeds sometimes contain whitespace and trailing punctuation in URLs.
        s = re.sub(r"\s+", "", (u or "").strip())
        return s.rstrip(").,;]>\"'")

    def _compact_summary(it_: dict, max_chars: int = 250) -> str:
        """
        One paragraph, <= max_chars. English + optional Chinese (if translation works).
        """
        en_full = _build_summary_body(it_, cfg).strip()
        en_sents = _split_sentences(en_full)
        en = " ".join(en_sents[:2]).strip() if en_sents else en_full
        en = re.sub(r"\s+", " ", en).strip()
        if len(en) > 140:
            en = en[:137].rstrip() + "..."

        zh = _maybe_translate_zh(en, cfg)
        zh = re.sub(r"\s+", " ", (zh or "")).strip()
        if zh and len(zh) > 140:
            zh = zh[:137].rstrip() + "..."

        para = en if not zh else f"{en} 中文：{zh}"
        if len(para) > max_chars:
            # Trim Chinese first if present.
            if zh:
                zh_cap = max(0, max_chars - len(en) - len(" 中文：") - 3)
                if zh_cap > 0:
                    zh2 = zh[:zh_cap].rstrip() + "..."
                    para = f"{en} 中文：{zh2}"
                else:
                    para = en[: max_chars - 3].rstrip() + "..."
            else:
                para = para[: max_chars - 3].rstrip() + "..."
        return para.strip()

    _GOOGLE_TITLE_SRC_RE = re.compile(r"\s+-\s+([A-Za-z0-9&][A-Za-z0-9& .'/\\-]{1,30})\s*$")

    def _split_google_publisher(title: str, url_: str) -> Tuple[str, str]:
        """
        Google News often appends publisher like " - MSN" to titles.
        Return (clean_title, publisher) where publisher may be "".
        """
        t0 = _clean_topic(title)
        dom = _get_domain(url_)
        if "news.google.com" not in (dom or ""):
            return t0, ""
        m = _GOOGLE_TITLE_SRC_RE.search(t0)
        if not m:
            return t0, ""
        pub = (m.group(1) or "").strip()
        clean = t0[: m.start()].strip()
        return (clean or t0), pub

    def _summaries(it_: dict) -> Tuple[str, str]:
        """
        Summary length rules:
        - If we can output BOTH EN+ZH: each tries for 100-250 chars.
        - If only a single language (e.g., translation fails): 50-150 chars.
        """
        base = _build_summary_body(it_, cfg).strip()
        sents = [s.strip() for s in _split_sentences(base) if s.strip()]

        def rule_zh_from_en(en: str) -> str:
            """
            Lightweight offline zh-Hant fallback for common earnings/news phrases.
            (Used when LLM translation is unavailable in this runtime.)
            """
            e = re.sub(r"\s+", " ", (en or "")).strip()
            if not e:
                return ""
            # Remove wire prefixes / keypoints.
            e = re.sub(r"^\(?\s*RTTNews\s*\)?\s*-\s*", "", e, flags=re.I).strip()
            e = re.sub(r"^\s*Key Points\s*[:\-]?\s*", "", e, flags=re.I).strip()
            # Remove leading "(TICKER)" blocks.
            e = re.sub(r"^\\([A-Z]{1,6}\\)\\s*", "", e).strip()

            low = e.lower()

            # Strip leading company phrase if present: "Illumina Inc. (ILMN) ..." -> "..."
            e2 = re.sub(
                r"^[A-Z][A-Za-z0-9&.'\\-]*(?:\\s+[A-Z][A-Za-z0-9&.'\\-]*)*(?:\\s+(?:Inc\\.|Corp\\.|Corporation|Co\\.|Company|Ltd\\.|PLC|Group))?\\s*(?:\\([A-Z]{1,6}\\))?\\s*",
                "",
                e,
            ).strip()
            low2 = e2.lower()

            # Detect quarter
            q = ""
            if "first quarter" in low2:
                q = "第一季度"
            elif "second quarter" in low2:
                q = "第二季度"
            elif "third quarter" in low2:
                q = "第三季度"
            elif "fourth quarter" in low2:
                q = "第四季度"
            elif re.search(r"\bq1\b", low2):
                q = "第一季度"
            elif re.search(r"\bq2\b", low2):
                q = "第二季度"
            elif re.search(r"\bq3\b", low2):
                q = "第三季度"
            elif re.search(r"\bq4\b", low2):
                q = "第四季度"

            # Earnings-style patterns
            if ("released earnings" in low2) or re.search(r"\breported\b.*\bearnings\b", low2):
                parts = []
                head = f"這週公布{q}財報" if q else "這週公布財報"
                parts.append(head)
                if "above expectations" in low2 or "beat expectations" in low2:
                    parts.append("業績優於預期")
                if "revenue came up short" in low2 or "revenue fell short" in low2 or "revenue missed" in low2:
                    parts.append("但營收不及預期")
                if "from the same period last year" in low2 or "year-ago" in low2 or "last year" in low2:
                    if any(k in low2 for k in ["increase", "increases", "increased", "advance", "advances", "rise", "rises", "jump", "jumps", "up"]):
                        parts.append("較去年同期有所增加")
                # Join into one sentence with commas.
                z = "，".join(parts).strip("，") + "。"
            else:
                z = ""

            if not z:
                return ""
            if len(z) > 150:
                z = z[:147].rstrip() + "..."
            return z

        def build_en(min_len: int, max_len: int) -> str:
            if not sents:
                s = re.sub(r"\s+", " ", base).strip()
                if len(s) > max_len:
                    s = s[: max_len - 3].rstrip() + "..."
                return s
            parts: List[str] = []
            out = ""
            for s in sents:
                parts.append(s)
                out = re.sub(r"\s+", " ", " ".join(parts)).strip()
                if len(out) >= min_len:
                    break
            out = re.sub(r"\s+", " ", out).strip()
            if len(out) > max_len:
                out = out[: max_len - 3].rstrip() + "..."
            if not re.search(r"[A-Za-z0-9]", out):
                return ""
            return out

        # Try bilingual first
        en_long = build_en(100, 250)
        # Guardrails: remove boilerplate like "Source: ..." / "See link..." if any slipped in.
        en_long = re.sub(r"\bSource\s*:\s*[A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)+\.?\s*", " ", en_long, flags=re.I)
        en_long = re.sub(r"\bSee\s+link\s+for\s+full\s+details\b\.?\s*", " ", en_long, flags=re.I)
        en_long = re.sub(r"\s+", " ", en_long).strip()
        if len(en_long) > 250:
            en_long = en_long[:247].rstrip() + "..."

        zh = _maybe_translate_zh(en_long, cfg)
        zh = re.sub(r"\s+", " ", (zh or "")).strip()
        zh = zh.replace("來源：marketwatch.com。欲知更多詳情，請點擊連結。", "")
        zh = zh.replace("來源：marketwatch.com。欲知更多詳情請點擊連結。", "")
        zh = re.sub(r"\s+", " ", zh).strip()
        if zh:
            if len(zh) > 250:
                zh = zh[: 247].rstrip() + "..."
            return en_long, zh

        # Single-language fallback (shorter)
        en_short = build_en(50, 150)
        en_short = re.sub(r"\bSource\s*:\s*[A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)+\.?\s*", " ", en_short, flags=re.I)
        en_short = re.sub(r"\bSee\s+link\s+for\s+full\s+details\b\.?\s*", " ", en_short, flags=re.I)
        en_short = re.sub(r"\s+", " ", en_short).strip()
        if len(en_short) > 150:
            en_short = en_short[:147].rstrip() + "..."
        # Offline zh fallback for common phrases (best-effort).
        zh2 = rule_zh_from_en(en_short)
        return en_short, (zh2 or "")

    url = _normalize_url(it.get("url") or "")
    title_raw = it.get("title") or "(no title)"
    title_en, google_pub = _split_google_publisher(title_raw, url)
    source = _source_display(url, it.get("source") or "")
    # Display time should be "now" (posting time), not the article's original publish time.
    pub_et = ""
    if ZoneInfo is not None:
        try:
            pub_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M EST")
        except Exception:
            pub_et = ""

    # Impact analysis (LLM structured output + fallback heuristic).
    impact = _analyze_market_impact(it, category_key, cfg)
    impact_sector = ""
    if impact.get("sectors"):
        impact_sector = str((impact.get("sectors") or [""])[0]).strip()
    if not impact_sector:
        impact_sector = _sector_zh_to_en(classify_sector(it, category_key))
    impact_sent = (str(impact.get("sentiment") or "neutral").strip().lower() or "neutral").capitalize()

    tickers = [t for t in (it.get("tickers") or []) if t]
    if not tickers:
        tickers = [t for t in (impact.get("tickers") or []) if t]

    # Prefer known Chinese company names when we have a ticker; otherwise keep English name.
    names_zh = load_company_names_zh()
    tickers_for_name = [t for t in (it.get("tickers") or []) if t]
    if not tickers_for_name:
        hinted = _ticker_hint_from_text(f"{title_raw} {it.get('content') or ''}")
        if hinted:
            tickers_for_name = [hinted]
    zh_company = names_zh.get(tickers_for_name[0].upper()) if tickers_for_name else ""

    # Best-effort company name phrase from title (for fallback display when no zh mapping).
    company_en_prefix = ""
    m_q = re.match(r"^(.+?)\s+Q[1-4]\b", title_en, flags=re.I)
    if m_q:
        company_en_prefix = (m_q.group(1) or "").strip()
        # Trim trailing ticker parens: "Linde (LIN)" -> "Linde"
        company_en_prefix = re.sub(r"\s*\([A-Z]{1,6}\)\s*$", "", company_en_prefix).strip()
    # Best-effort company phrase from title (for removing English name leaks in Chinese lines).
    company_en_guess = ""
    m = re.match(r"^Why\s+(.+?)\s+(Plunged|Dropped|Fell)\s+Today\b", title_en, flags=re.I)
    if m:
        company_en_guess = (m.group(1) or "").strip()
        # also keep an ASCII-folded variant (e.g. "Estée" -> "Estee")
        try:
            company_en_guess_ascii = unicodedata.normalize("NFKD", company_en_guess).encode("ascii", "ignore").decode("ascii")
        except Exception:
            company_en_guess_ascii = company_en_guess
    else:
        company_en_guess_ascii = ""
    def _build_zh_title(title_en_: str) -> str:
        """
        Build a safer zh title.
        If we know company name mapping, prefer deterministic templates to avoid wrong company-name translations.
        """
        if not cfg.translate_zh:
            return ""
        t = (title_en_ or "").strip()
        if not t:
            return ""
        if not zh_company:
            return _maybe_translate_title_zh(t, cfg)

        tl = t.lower()
        # Pattern: "Why <Company> Plunged Today" -> "<Company> 今日股價大跌"
        if tl.startswith("why ") and (" plunged today" in tl or " dropped today" in tl or " fell today" in tl):
            return f"{zh_company} 今日股價大跌"

        # Earnings transcript/call
        if "earnings" in tl and "transcript" in tl:
            return f"{zh_company} 財報電話會議記錄"

        # Common earnings headline: "Q4 Profit Advances/Increases/Rises/Jumps"
        m_q = re.search(r"\bq([1-4])\b", tl)
        if m_q and "profit" in tl and any(k in tl for k in ["advance", "advances", "rise", "rises", "increase", "increases", "jump", "jumps"]):
            qn = m_q.group(1)
            qmap = {"1": "第一季度", "2": "第二季度", "3": "第三季度", "4": "第四季度"}
            return f"{zh_company} {qmap.get(qn, '季度')}利潤增長"

        # Simple phrase mapping for common financial headlines (avoid model hallucinations).
        def _rest_to_zh(rest_en: str) -> str:
            r = (rest_en or "").strip()
            if not r:
                return ""
            r_l = r.lower()
            # Quarter
            r = re.sub(r"\\bq1\\b", "第一季度", r, flags=re.I)
            r = re.sub(r"\\bq2\\b", "第二季度", r, flags=re.I)
            r = re.sub(r"\\bq3\\b", "第三季度", r, flags=re.I)
            r = re.sub(r"\\bq4\\b", "第四季度", r, flags=re.I)
            # Profit / earnings
            if "profit" in r_l and any(k in r_l for k in ["advance", "advances", "rise", "rises", "increases", "increase", "jumps", "jump"]):
                return "第四季度利潤增長" if "第四季度" in r else "利潤增長"
            if "earnings" in r_l and "call" in r_l and "transcript" in r_l:
                return "財報電話會議記錄"
            if "earnings" in r_l and "transcript" in r_l:
                return "財報記錄"
            # Fallback: model translate the rest only.
            return _maybe_translate_title_zh(r, cfg) or ""

        # Strip leading company name when the title begins with it, then translate the rest and prefix with zh company.
        # Only do this if it doesn't start with question words.
        if not re.match(r"^(why|how|what|when)\\b", tl):
            rest = re.sub(
                r"^[A-Z][A-Za-z&.'\\-]*(?:\\s+[A-Z][A-Za-z&.'\\-]*)*(?:\\s+(?:Inc\\.|Corp\\.|Corporation|Co\\.|Company|Ltd\\.|PLC|Group))?\\s*",
                "",
                t,
            ).strip()
            if rest and rest != t:
                rest_zh = _rest_to_zh(rest)
                rest_zh = re.sub(r"^\\s*關鍵點\\s*[:：-]?\\s*", "", rest_zh).strip()
                return (f"{zh_company} {rest_zh}".strip()) if rest_zh else zh_company

        # Fallback: translate full title, but if the model produced the wrong company name, we still show mapping.
        full_zh = _maybe_translate_title_zh(t, cfg) or ""
        full_zh = re.sub(r"^\\s*關鍵點\\s*[:：-]?\\s*", "", full_zh).strip()
        if not full_zh:
            return zh_company
        # If translated title doesn't contain company mapping, prefix it.
        if zh_company not in full_zh:
            return f"{zh_company} {full_zh}"
        return full_zh

    zh_title = _build_zh_title(title_en)
    # zh is already inside compact summary; keep title translation as separate optional line.

    en_sum, zh_sum = _summaries(it)
    en_sum = _normalize_change_signs(en_sum)
    zh_sum = _normalize_change_signs(zh_sum)
    anomaly = _anomaly_line({"tickers": tickers}, en_sum)

    zh_sum_clean = ""
    if zh_sum:
        # Remove noisy prefixes in Chinese and avoid wrong company-name parentheses.
        z = zh_sum
        z = re.sub(r"^\s*關鍵點\s*[:：]?\s*", "", z)
        z = re.sub(r"\s+", " ", z).strip()
        # Remove leading parentheses blocks (even if malformed/missing a closing bracket).
        z = re.sub(r"^[（(][A-Za-z0-9 .&'\-]{1,80}[）)]?\s*", "", z)
        z = re.sub(r"^[（(][^）)]{0,80}[）)]?\s*", "", z)
        # If we know the Chinese company name, prefix the sentence with it (no parentheses).
        if zh_company and not z.startswith(zh_company):
            z = f"{zh_company} " + z.lstrip()
        # If we don't know the Chinese name, fall back to the English company name (no parentheses).
        if (not zh_company) and company_en_prefix and not z.lower().startswith(company_en_prefix.lower()):
            z = f"{company_en_prefix} " + z.lstrip()
        if zh_company:
            # Remove: "<ZH_COMPANY> (Illumina Inc.) ..." or malformed "(Illumina Inc...."
            z = re.sub(
                rf"^{re.escape(zh_company)}\s+[（(][A-Za-z0-9 .&'\-]{{1,80}}[）)]?\s*",
                f"{zh_company} ",
                z,
            )
            # Also handle plain "(...)" blocks robustly.
            z = re.sub(rf"^{re.escape(zh_company)}\s*\([^)]{{1,120}}\)\s*", f"{zh_company} ", z)

            # Remove English company phrase if it leaks into the Chinese sentence.
            if company_en_guess:
                z = re.sub(rf"^{re.escape(zh_company)}\s+{re.escape(company_en_guess)}\s*", f"{zh_company} ", z, flags=re.I)
            if company_en_guess_ascii and company_en_guess_ascii != company_en_guess:
                z = re.sub(rf"^{re.escape(zh_company)}\s+{re.escape(company_en_guess_ascii)}\s*", f"{zh_company} ", z, flags=re.I)

            # Avoid: "<ZH_COMPANY> Estee Lauder ..." -> keep only the zh company name.
            z = re.sub(
                rf"^{re.escape(zh_company)}\s+[A-Za-z][A-Za-z .&'\-]{{2,80}}\s*",
                f"{zh_company} ",
                z,
            ).strip()
        zh_sum_clean = _normalize_change_signs(z).strip()

    # Hard fallback for Chinese when translation returned empty.
    if not zh_title:
        zh_title = (_maybe_translate_title_zh(title_en, cfg) or "").strip()
    if not zh_title:
        zh_title = title_en
    if not zh_sum_clean and en_sum:
        zh_sum_clean = (_maybe_translate_zh(en_sum, cfg) or "").strip()
    if url:
        is_google = "news.google.com" in (_get_domain(url) or "")
        if is_google:
            src = f"Source: {_html_mod.escape(source)}"
        else:
            src = f"Source: <a href=\"{_html_mod.escape(url)}\">{_html_mod.escape(source)}</a>"
    else:
        src = f"Source: {_html_mod.escape(source)}"
    if google_pub:
        src += f" (by {_html_mod.escape(google_pub)})"
    if pub_et:
        src += f" | {_html_mod.escape(pub_et)}"
    src_line = "Source：" + src.replace("Source: ", "")
    src_line_ascii = src_line.replace("：", ": ")

    # Category formatting
    fed_noise = any(x in (f"{title_en} {it.get('content') or ''}").lower() for x in ["fed cattle", "fed cattle exchange", "powell industries"])
    is_fed_news = bool(_TAG_FED.search(f"{title_en} {it.get('content') or ''}")) and not fed_noise
    category_en = "FED News" if is_fed_news else (impact_sector or "Stock News")
    category_zh = "FED News" if is_fed_news else ((classify_sector(it, category_key) or "").strip() or "其他")
    reason_en = str(impact.get("reason") or "").strip() or "Market impact is limited and mostly in-line with prevailing expectations."
    reason_zh = (_maybe_translate_zh(reason_en, cfg) or "").strip()

    # Combined text (backward compatibility)
    lines: List[str] = []
    lines.append(_html_mod.escape(f"Category: {category_en}"))
    if tickers:
        lines.append(_html_mod.escape("Ticker: " + ", ".join([f"${t}" for t in tickers])))
    if tickers:
        q0 = get_quote(tickers[0], cache_ttl_seconds=60)
        if q0 is not None:
            lines.append(_html_mod.escape(f"Price: {q0.price:.2f} USD ({q0.day_change_pct:+.2f}%)"))
    if anomaly:
        lines.append(_html_mod.escape(anomaly))
    lines.append(_html_mod.escape(f"AI Sentiment Analysis: {impact_sent}"))
    lines.append(_html_mod.escape(f"Reason: {reason_en}"))
    lines.append(_html_mod.escape(f"Topic: {title_en}"))
    if en_sum:
        lines.append(_html_mod.escape(en_sum))
    lines.append(src_line_ascii)
    text = "\n".join(lines).strip()

    # Split-language channel payloads
    en_lines: List[str] = []
    en_lines.append(_html_mod.escape(f"Category: {category_en}"))
    if tickers:
        en_lines.append(_html_mod.escape("Ticker: " + ", ".join([f"${t}" for t in tickers])))
    if tickers:
        q = get_quote(tickers[0], cache_ttl_seconds=60)
        if q is not None:
            en_lines.append(_html_mod.escape(f"Price: {q.price:.2f} USD ({q.day_change_pct:+.2f}%)"))
    if anomaly:
        en_lines.append(_html_mod.escape(anomaly))
    en_lines.append(_html_mod.escape(f"AI Sentiment Analysis: {impact_sent}"))
    en_lines.append(_html_mod.escape(f"Reason: {reason_en}"))
    en_lines.append(_html_mod.escape(f"Topic: {title_en}"))
    if en_sum:
        en_lines.append(_html_mod.escape(en_sum))
    en_lines.append(src_line_ascii)
    en_text = "\n".join(en_lines).strip()

    zh_lines: List[str] = []
    zh_lines.append(_html_mod.escape(f"分類: {category_zh}"))
    if tickers:
        zh_lines.append(_html_mod.escape("代號: " + ", ".join([f"${t}" for t in tickers])))
    if tickers:
        q2 = get_quote(tickers[0], cache_ttl_seconds=60)
        if q2 is not None:
            zh_lines.append(_html_mod.escape(f"股價: {q2.price:.2f} USD ({q2.day_change_pct:+.2f}%)"))
    if anomaly:
        zh_anomaly = anomaly.replace("Anomaly:", "異常:").replace("daily surge", "單日上升").replace("daily drop", "單日下跌")
        zh_lines.append(_html_mod.escape(zh_anomaly))
    zh_lines.append(_html_mod.escape(f"AI Sentiment Analysis: {impact_sent}"))
    zh_lines.append(_html_mod.escape(f"原因: {reason_zh or reason_en}"))
    zh_lines.append(_html_mod.escape(f"主題: {zh_title}"))
    if zh_sum_clean:
        zh_lines.append(_html_mod.escape(zh_sum_clean))
    elif en_sum:
        zh_lines.append(_html_mod.escape(en_sum))
    zh_lines.append(src_line_ascii.replace("Source", "來源"))
    zh_text = "\n".join(zh_lines).strip()

    meta: Dict[str, Any] = {
        "source": source,
        "url": url,
        "google_pub": google_pub,
        "translate_enabled": bool(cfg.translate_zh),
        "zh_title_ok": (not cfg.translate_zh) or bool(zh_title.strip()),
        "zh_summary_ok": (not cfg.translate_zh) or bool(zh_sum.strip()),
        "zh_title": (zh_title or "").strip(),
        "zh_summary": (zh_sum or "").strip(),
        "en_summary": (en_sum or "").strip(),
        "title_en": title_en,
        "has_market_impact": bool(impact.get("has_market_impact")),
        "impact_type": str(impact.get("impact_type") or "").strip().lower(),
        "impact_sector": impact_sector,
        "impact_sentiment": impact_sent.lower(),
        "impact_reason": str(impact.get("reason") or "").strip(),
        "en_text": en_text,
        "zh_text": zh_text,
    }
    return text, meta


def build_item_message_with_meta(it: dict, cfg: DigestConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Build a single public-facing message block for one news item.
    Category is determined by classifier.
    """
    cat = classify_item(it)
    return _build_item_block(it, cat, cfg)


def build_item_message(it: dict, cfg: DigestConfig) -> str:
    text, _meta = build_item_message_with_meta(it, cfg)
    return text


def _format_published_et(published_at: str) -> str:
    """
    Convert ISO8601 timestamp to Eastern Time display.
    Returns empty string if parsing/conversion fails.
    """
    s = (published_at or "").strip()
    if not s:
        return ""
    try:
        # Normalize Z to +00:00 for fromisoformat
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return ""
    if dt.tzinfo is None:
        return ""
    if ZoneInfo is None:
        return ""
    try:
        et = dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return ""
    return et.strftime("%Y-%m-%d %H:%M EST")


def build_digest(items: List[dict], cfg: DigestConfig) -> str:
    """
    Build a categorized Telegram message with stable, deterministic formatting.
    """
    # categorize
    buckets: Dict[str, List[dict]] = {c.key: [] for c in CATEGORIES}
    for it in items:
        buckets[classify_item(it)].append(it)

    # Build output
    out_lines: List[str] = []
    total = 0
    for cat in CATEGORIES:
        picked = buckets.get(cat.key) or []
        if not picked:
            continue
        picked = picked[: cfg.max_per_category]
        out_lines.append(cat.title)
        for it in picked:
            if total >= cfg.max_total_items:
                break
            tickers = it.get("tickers") or []
            tickers = [t for t in tickers if t]
            if tickers:
                out_lines.append("Stock Ticker: " + ", ".join([f"${t}" for t in tickers]))
                if cfg.include_price:
                    for t in tickers[:3]:
                        q = get_quote(t, cache_ttl_seconds=60)
                        if q is not None:
                            out_lines.append(format_price_line(q, include_ticker=len(tickers) > 1))
            out_lines.append(f"Topic: {_clean_topic(it.get('title') or '(no title)')}")
            en_sum = _build_summary_body(it, cfg)
            out_lines.append(f"Summary: {en_sum}")
            zh_sum = _maybe_translate_zh(en_sum, cfg)
            out_lines.append(f"中文總結：{zh_sum if zh_sum else '（翻譯暫不可用）'}")
            out_lines.append(f"Website: {it.get('url') or ''}")
            out_lines.append("")
            total += 1
        out_lines.append("")
        if total >= cfg.max_total_items:
            break

    text = "\n".join(out_lines).strip()
    return text or "(no items)"


def build_digests(items: List[dict], cfg: DigestConfig) -> Dict[str, str]:
    """
    Returns per-category digest texts: {category_key: message_text}
    Message text is a concatenation of per-item blocks (no category header).
    """
    buckets: Dict[str, List[dict]] = {c.key: [] for c in CATEGORIES}
    for it in items:
        buckets[classify_item(it)].append(it)

    out: Dict[str, str] = {}
    total = 0
    prices_left = max(0, int(cfg.price_max_tickers_total))
    for cat in CATEGORIES:
        picked = buckets.get(cat.key) or []
        if not picked:
            continue
        picked = picked[: cfg.max_per_category]

        lines: List[str] = []
        for it in picked:
            if total >= cfg.max_total_items:
                break
            block, _meta = _build_item_block(it, cat.key, cfg)
            lines.append(block)
            lines.append("")
            total += 1

        text = "\n".join(lines).strip()
        if text:
            out[cat.key] = text

        if total >= cfg.max_total_items:
            break

    return out


def build_canonical(digests: Dict[str, str]) -> str:
    """
    Canonical string for DB dedupe/throttle (not necessarily posted).
    """
    parts = []
    for cat in CATEGORIES:
        t = digests.get(cat.key)
        if not t:
            continue
        # Remove dynamic lines (prices/timestamps) so dedupe isn't broken by quotes changing.
        filtered = []
        for line in t.splitlines():
            if line.strip().lower().startswith("price ("):
                continue
            if line.strip().startswith("中文總結："):
                continue
            filtered.append(line)
        parts.append("\n".join(filtered).strip())
    return "\n\n---\n\n".join(parts).strip()
