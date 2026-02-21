import os
import shutil
import subprocess
from typing import List, Dict

import requests


def chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 60,
) -> str:
    """
    OpenAI-compatible chat completion with Ollama CLI fallback.
    Fallback is enabled when `TRANSLATE_USE_OLLAMA_CLI=1`.
    """
    base_url = (base_url or "").strip()
    model = (model or "").strip()
    if not model:
        raise ValueError("missing model")

    # Try HTTP first (if base_url provided)
    if base_url:
        try:
            url = base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return (out or "").strip()
        except Exception:
            pass

    # CLI fallback
    if os.getenv("TRANSLATE_USE_OLLAMA_CLI", "0") != "1":
        return ""
    if shutil.which("ollama") is None:
        return ""

    # Flatten messages into a single prompt.
    chunks = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            chunks.append(content)
        elif role == "user":
            chunks.append(content)
        else:
            chunks.append(content)
    prompt = "\n\n".join(chunks).strip()
    if not prompt:
        return ""

    try:
        rr = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except Exception:
        return ""

    return (rr.stdout or "").strip()

