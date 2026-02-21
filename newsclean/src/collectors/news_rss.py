from datetime import datetime, timezone
from urllib.parse import urlparse

import feedparser
import requests


def fetch_rss(urls, timeout: int = 20):
    if not urls:
        return []
    out = []
    now = datetime.now(timezone.utc).isoformat()
    for url in urls:
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "news-sync/0.1 (+rss)"},
            )
            resp.raise_for_status()
        except Exception:
            continue

        feed = feedparser.parse(resp.text)
        for e in feed.entries[:50]:
            published = None
            if getattr(e, "published_parsed", None):
                dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                published = dt.isoformat()
            link = getattr(e, "link", "") or ""
            domain = ""
            try:
                domain = (urlparse(link).netloc or urlparse(url).netloc or "").lower()
            except Exception:
                domain = ""
            out.append({
                "source": domain or "rss",
                "title": getattr(e, "title", ""),
                "url": link,
                "published_at": published,
                "ingested_at": now,
                "content": getattr(e, "summary", "") or getattr(e, "title", "")
            })
    return out
