import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from http_client import build_session

try:
    import lxml.html
except Exception:
    lxml = None


_WS_RE = re.compile(r"\s+")


@dataclass
class EnrichResult:
    url: str
    snippet: str


_CACHE: dict[str, EnrichResult] = {}


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = _WS_RE.sub(" ", s)
    return s


def fetch_meta_snippet(url: str, timeout: int = 10) -> Optional[str]:
    """
    Best-effort: fetch an article page and extract meta description (OG/Twitter/description).
    Keep it cheap (no full article extraction).
    """
    if not url:
        return None
    if url in _CACHE:
        return _CACHE[url].snippet
    if lxml is None:
        return None

    # Avoid long stalls: do not retry page fetches here. The main loop will retry next cycle.
    session = build_session(total_retries=0, backoff_factor=0.0)
    try:
        r = session.get(url, timeout=(3, timeout), headers={"User-Agent": "mlstockbot/0.1"})
    except Exception:
        return None
    if r.status_code != 200:
        return None
    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ctype:
        return None

    try:
        doc = lxml.html.fromstring(r.text)
    except Exception:
        return None

    metas = doc.xpath("//meta")
    candidates = []
    for m in metas:
        prop = (m.get("property") or "").strip().lower()
        name = (m.get("name") or "").strip().lower()
        content = _clean(m.get("content") or "")
        if not content:
            continue

        if prop in {"og:description", "twitter:description"}:
            candidates.append((3, content))
        elif name == "description":
            candidates.append((2, content))
        elif prop == "description":
            candidates.append((1, content))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    snippet = candidates[0][1]
    if len(snippet) > 340:
        snippet = snippet[:337].rstrip() + "..."
    _CACHE[url] = EnrichResult(url=url, snippet=snippet)
    return snippet


def _extract_main_text(html: str) -> Optional[str]:
    """
    Very lightweight article extraction:
    - prefer <article>, else the largest text block among <main>/<div>/<section>
    - collect paragraph-ish text
    """
    if lxml is None:
        return None
    try:
        doc = lxml.html.fromstring(html)
    except Exception:
        return None

    # Drop non-content nodes
    for bad in doc.xpath("//script|//style|//noscript|//header|//footer|//nav|//aside"):
        try:
            bad.drop_tree()
        except Exception:
            pass

    def text_len(node) -> int:
        try:
            return len(_clean(node.text_content() or ""))
        except Exception:
            return 0

    candidates = []
    for xp in ("//article", "//main", "//section", "//div"):
        for n in doc.xpath(xp):
            tl = text_len(n)
            if tl >= 400:
                candidates.append((tl, n))
        if candidates:
            break

    node = max(candidates, key=lambda x: x[0])[1] if candidates else doc

    parts = []
    for p in node.xpath(".//p|.//li"):
        t = _clean(p.text_content() or "")
        if len(t) < 40:
            continue
        parts.append(t)

    text = _clean(" ".join(parts))
    if len(text) < 300:
        # Fallback: all paragraphs in the doc
        parts = []
        for p in doc.xpath("//p"):
            t = _clean(p.text_content() or "")
            if len(t) < 40:
                continue
            parts.append(t)
        text = _clean(" ".join(parts))

    if len(text) < 200:
        return None
    return text


def fetch_article_text(url: str, timeout: int = 10) -> Optional[str]:
    if not url or lxml is None:
        return None

    # Prefer curl with hard timeouts; it avoids rare hangs in Python SSL reads on tiny VMs.
    curl = shutil.which("curl")
    if curl:
        try:
            cp = subprocess.run(
                [
                    curl,
                    "-L",
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    str(max(3, int(timeout))),
                    "-A",
                    "mlstockbot/0.1",
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(5, int(timeout) + 2),
                text=True,
            )
            if cp.returncode != 0 or not cp.stdout:
                return None
            return _extract_main_text(cp.stdout)
        except Exception:
            return None

    # Fallback: requests (best-effort)
    session = build_session(total_retries=0, backoff_factor=0.0)
    try:
        r = session.get(url, timeout=(3, timeout), headers={"User-Agent": "mlstockbot/0.1"})
    except Exception:
        return None
    if r.status_code != 200:
        return None
    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ctype:
        return None
    return _extract_main_text(r.text)


def enrich_items(items: list, timeout: int = 10, max_fetch: int = 8) -> None:
    """
    Mutates items in-place:
    - Best-effort fetch full article text (web crawler) for the top items.
    - Falls back to meta-description snippet.
    """
    n = 0
    for it in items:
        if n >= max_fetch:
            break
        url = it.get("url") or ""
        title = it.get("title") or ""
        content = (it.get("content") or "").strip()

        if not url:
            continue

        # Always attempt to read page content for published items.
        body = fetch_article_text(url, timeout=timeout)
        if body:
            # keep DB size sane
            if len(body) > 4000:
                body = body[:3997].rstrip() + "..."
            it["content"] = body
            it["snippet"] = body[:420]
            n += 1
            continue

        # Fallback: meta snippet if we likely only have the title / tiny snippet.
        if not url:
            continue
        if content and len(content) >= 120 and content.lower() != title.lower():
            continue

        snippet = fetch_meta_snippet(url, timeout=timeout)
        if not snippet:
            continue
        if snippet.lower() == title.lower():
            continue

        it["snippet"] = snippet
        it["content"] = snippet
        n += 1
