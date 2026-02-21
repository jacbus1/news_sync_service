import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    title: str


CATEGORIES = [
    Category("earnings", "Earning財報"),
    Category("report", "Report報告"),
    Category("stock", "Stock News股票新聞"),
    Category("other", "Others其他"),
]


_EARNINGS_STRONG_RE = re.compile(
    r"\b("
    r"earnings|eps|guidance|forecast|outlook|results|"
    r"q[1-4]\b|fy\\d{2,4}|quarter|full\\s*year"
    r")\b",
    re.I,
)
_EARNINGS_CONTEXT_RE = re.compile(
    r"\b(revenue|profit|loss|operating\\s*income|margin)\b",
    re.I,
)
_EARNINGS_BEAT_RE = re.compile(r"\b(beats|misses|estimate|estimates)\b", re.I)

_REPORT_RE = re.compile(
    r"\b("
    r"sec\b|"
    r"10-?k|10-?q|8-?k|13f|"
    r"s-?1|424b\\d*|prospectus|registration\\s+statement|"
    r"form\\s+(?:4|3|5)\\b|schedule\\s+13d|schedule\\s+13g"
    r"|report|reports|filing|filed|regulator|investigation|"
    r"analyst|rating|price\\s*target|"
    r"upgrad(?:e|ed|es|ing)|downgrad(?:e|ed|es|ing)|initiates|coverage|"
    r"cpi|pce|jobs\\s*report|durable\\s*goods|gdp|fed\\b|fomc|minutes"
    r")\b",
    re.I,
)


def classify_item(it: dict) -> str:
    """
    Priority:
      1) Earnings
      2) Report (SEC filing / analyst report / macro report)
      3) Stock news (has ticker)
      4) Other
    """
    title = (it.get("title") or "").strip()
    content = (it.get("content") or "").strip()
    text = f"{title} {content}".strip()
    tickers = it.get("tickers") or []

    # Earnings: require strong earnings terms, or beat/miss + financial context.
    if _EARNINGS_STRONG_RE.search(text) or (_EARNINGS_BEAT_RE.search(text) and _EARNINGS_CONTEXT_RE.search(text)):
        return "earnings"
    if _REPORT_RE.search(text):
        return "report"
    if tickers:
        return "stock"
    return "other"
