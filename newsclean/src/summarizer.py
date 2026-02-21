import json
import requests


def _strip_unknown_ticker_lines(text: str) -> str:
    if not text:
        return text
    out_lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("stock ticker:") and "unknown" in line.lower():
            continue
        out_lines.append(line)
    # collapse excessive blank lines
    cleaned = "\n".join(out_lines).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def summarize(items, api_key: str, base_url: str, model: str, timeout: int = 25) -> str:
    if not items:
        return "(no items)"

    # If no API key, do a simple heuristic digest
    if not api_key:
        lines = []
        for it in items[:10]:
            lines.append(f"- {it.get('title','(no title)')} ({it.get('source')})")
        return "\n".join(lines)

    url = base_url.rstrip('/') + '/chat/completions'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = (
        "You are preparing a real-time market alert for a public Telegram channel. "
        "Your tone must be objective, neutral, and factual. No opinions, no hype, no predictions, "
        "no investment advice, and no promotional language.\n\n"
        "Output format for each item (repeat per item, no bullets):\n"
        "Stock ticker: $TICKER (optional; omit this line if unknown)\n"
        "Summary: ...\n"
        "Website: ...\n"
        "\n"
        "Rules:\n"
        "- 4 to 6 items max\n"
        "- Summary should be 1-3 sentences, factual, and cite concrete information only\n"
        "- Website should be the original article URL\n"
        "- Use provided tickers field when available; if none, infer from title; if still unknown, omit ticker line\n"
        "- Prioritize breaking items and macro events\n"
        "- Exclude subjective phrases (e.g., 'bullish', 'bearish', 'good', 'bad') unless directly quoted in source\n"
        "- Keep total under ~2000 characters\n\n"
        + json.dumps(items, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial news summarizer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return _strip_unknown_ticker_lines(text)
    except Exception:
        # Fallback to heuristic digest on any LLM error/timeouts
        lines = []
        for it in items[:10]:
            tickers = it.get("tickers") or []
            parts = []
            if tickers:
                parts.append(f"Stock ticker: ${tickers[0]}")
            parts.append(f"Summary: {it.get('title','(no title)')}")
            parts.append(f"Website: {it.get('url','')}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)
