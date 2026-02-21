import os
from json import JSONDecodeError
from datetime import datetime, timezone
from http_client import build_session


def fetch_gdelt(max_records: int = 50, timeout: int = 20, retries: int = 3):
    session = build_session(total_retries=retries)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Default is stocks-first. Override via env: GDELT_QUERY
    # Note: GDELT query syntax supports boolean operators.
    q = os.getenv("GDELT_QUERY", "stocks OR shares OR earnings OR guidance OR NASDAQ OR NYSE OR S&P 500 OR Federal Reserve")
    params = {
        "query": q,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json"
    }
    r = session.get(url, params=params, timeout=timeout, headers={"User-Agent": "news-sync/0.1"})
    r.raise_for_status()
    try:
        data = r.json()
    except JSONDecodeError:
        return []
    articles = data.get("articles", [])
    out = []
    now = datetime.now(timezone.utc).isoformat()
    for a in articles:
        out.append({
            "source": "gdelt",
            "title": a.get("title") or a.get("name"),
            "url": a.get("url"),
            "published_at": a.get("seendate") or a.get("publishedAt"),
            "ingested_at": now,
            "content": a.get("title") or a.get("name") or ""
        })
    return out
