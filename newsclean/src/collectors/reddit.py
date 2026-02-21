from datetime import datetime, timezone
from http_client import build_session


def fetch_reddit(subreddit: str, limit: int = 50, timeout: int = 20, retries: int = 3):
    session = build_session(total_retries=retries)
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    params = {"limit": limit}
    r = session.get(url, params=params, timeout=timeout, headers={"User-Agent": "news-sync/0.1"})
    r.raise_for_status()
    data = r.json()
    out = []
    now = datetime.now(timezone.utc).isoformat()
    for c in data.get("data", {}).get("children", []):
        d = c.get("data", {})
        out.append({
            "source": f"reddit/{subreddit}",
            "title": d.get("title"),
            "url": d.get("url") or ("https://www.reddit.com" + (d.get("permalink") or "")),
            "published_at": datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat() if d.get("created_utc") else None,
            "ingested_at": now,
            "content": d.get("selftext") or d.get("title") or ""
        })
    return out
