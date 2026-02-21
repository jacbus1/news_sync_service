import json
import os
import re
from typing import Dict, Optional


_CACHE: Optional[Dict[str, str]] = None


def load_ticker_sectors() -> Dict[str, str]:
    """
    Return mapping: TICKER -> sector label (zh).
    Optional override file: data/ticker_sectors.json
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    base: Dict[str, str] = {
        # Seed: common / requested examples
        "AAPL": "科技",
        "MSFT": "科技",
        "NVDA": "科技",
        "AMZN": "消費",
        "GOOGL": "科技",
        "META": "科技",
        "TSLA": "消費",
        "EL": "消費",
        "ILMN": "醫療",
        "LIN": "原材料",
        "LADR": "房地產",
        "PLTR": "國防",
        "LMT": "國防",
        "NOC": "國防",
        "RTX": "國防",
        "XOM": "能源",
        "CVX": "能源",
        "JPM": "金融",
        "BAC": "金融",
        "GS": "金融",
        "BLK": "金融",
        "BTC": "加密",
        "ETH": "加密",
    }

    path = os.path.join(os.path.dirname(__file__), "..", "data", "ticker_sectors.json")
    path = os.path.abspath(path)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                extra = json.load(f) or {}
            for k, v in (extra or {}).items():
                kk = str(k or "").strip().upper()
                vv = str(v or "").strip()
                if kk and vv:
                    base[kk] = vv
    except Exception:
        pass

    _CACHE = base
    return base


_RE_FED = re.compile(r"\b(fed|fomc|powell|interest rate|rates?|inflation|cpi|ppi|jobs?|payrolls?|unemployment|yield|treasury)\b", re.I)
_RE_DEFENSE = re.compile(r"\b(pentagon|defense|defence|military|navy|army|air force|missile|drone|uav|lockheed|northrop|raytheon)\b", re.I)
_RE_TECH = re.compile(r"\b(ai|artificial intelligence|semiconductor|chip|gpu|cpu|cloud|cyber|software|saas|data center|robotics)\b", re.I)
_RE_CONSUMER = re.compile(r"\b(retail|consumer|restaurant|travel|hotel|airline|e-commerce|luxury|cosmetics|beauty|apparel|auto|vehicle)\b", re.I)
_RE_ENERGY = re.compile(r"\b(oil|gas|opec|crude|brent|wti|energy|solar|wind|renewable|power grid|utilities?)\b", re.I)
_RE_FIN = re.compile(r"\b(bank|banks|lender|mortgage|insurance|broker|asset management|hedge fund|etf|reits?)\b", re.I)
_RE_HEALTH = re.compile(r"\b(biotech|pharma|drug|clinical|fda|healthcare|medical|hospital)\b", re.I)
_RE_SPORTS = re.compile(r"\b(nfl|nba|mlb|nhl|soccer|football|basketball|baseball|hockey|olympics)\b", re.I)
_RE_CRYPTO = re.compile(r"\b(bitcoin|btc|ethereum|eth|crypto|blockchain|defi|stablecoin|solana|dogecoin)\b", re.I)


def classify_sector(it: dict, category_key: str) -> str:
    """
    Return a single sector bucket label (zh). Empty string means "no label".
    Priority:
    - Fed/macro => 經濟
    - Ticker map
    - Keyword rules
    """
    title = (it.get("title") or "")
    content = (it.get("content") or "")
    text = f"{title} {content}"

    if _RE_FED.search(text):
        return "經濟"

    tickers = [t for t in (it.get("tickers") or []) if t]
    if tickers:
        m = load_ticker_sectors()
        for t in tickers:
            lab = m.get(str(t).upper())
            if lab:
                return lab

    if _RE_CRYPTO.search(text):
        return "加密"
    if _RE_DEFENSE.search(text):
        return "國防"
    if _RE_TECH.search(text):
        return "科技"
    if _RE_ENERGY.search(text):
        return "能源"
    if _RE_FIN.search(text):
        return "金融"
    if _RE_HEALTH.search(text):
        return "醫療"
    if _RE_CONSUMER.search(text):
        return "消費"
    if _RE_SPORTS.search(text):
        return "體育"

    # Category fallback
    if category_key in {"earnings", "report", "stock"}:
        return "美股"
    return ""
