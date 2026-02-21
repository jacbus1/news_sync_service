import hashlib
import os
import shutil
import subprocess
from typing import Optional

import re
import requests


_CACHE: dict[str, str] = {}


def _find_ollama_bin() -> Optional[str]:
    p = shutil.which("ollama")
    if p:
        return p
    for c in ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/usr/bin/ollama"]:
        if os.path.exists(c):
            return c
    return None

def _ollama_cli_translate(model: str, text: str, timeout: int) -> Optional[str]:
    """
    Fallback path when HTTP access to Ollama is blocked.
    Requires `ollama` CLI available locally.
    """
    if not model or not text:
        return None
    ollama_bin = _find_ollama_bin()
    if not ollama_bin:
        return None
    prompt = (
        "Translate the following English text into Traditional Chinese (zh-Hant). "
        "Do not add or remove facts. Do not add opinions. Output only the translation.\n\n"
        + text.strip()
    )
    try:
        r = subprocess.run(
            [ollama_bin, "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except Exception:
        return None
    out = (r.stdout or "").strip()
    if not out:
        return None
    # Some models may echo the prompt; keep last non-empty paragraph.
    parts = [p.strip() for p in out.splitlines() if p.strip()]
    if not parts:
        return None
    return "\n".join(parts).strip()


def translate_to_zh(text: str, base_url: str, model: str, timeout: int = 30, fallback_model: str = "") -> Optional[str]:
    """
    Best-effort translation (Traditional Chinese). Returns None on failure.
    Uses an OpenAI-compatible /chat/completions endpoint (e.g., Ollama).
    """
    s = (text or "").strip()
    if not s:
        return None
    if not base_url or not model:
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    extra = []
    if fallback_model:
        # Allow comma-separated fallbacks, e.g. "gemma3:12b,qwen2.5-coder:7b"
        extra = [m.strip() for m in fallback_model.split(",") if m.strip()]
    models = [model] + [m for m in extra if m and m != model]

    for m in models:
        key = hashlib.sha1((m + "\n" + s).encode("utf-8")).hexdigest()
        if key in _CACHE:
            return _CACHE[key]
        def _try_cli() -> Optional[str]:
            cli_out = _ollama_cli_translate(m, s, timeout=timeout)
            if cli_out:
                _CACHE[key] = cli_out
                return cli_out
            return None

        payload = {
            "model": m,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate the given English text into Traditional Chinese (zh-Hant). "
                        "Do not add or remove facts. Do not add opinions. Output only the translation."
                    ),
                },
                {"role": "user", "content": s},
            ],
            "temperature": 0.1,
        }
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            out = (out or "").strip()
            # Suppress common non-translation outputs.
            low = out.lower()
            # Must look like Traditional Chinese output; otherwise treat as failure.
            # (Prevents assistant-style English responses leaking into public posts.)
            if not re.search(r"[\u4e00-\u9fff]", out):
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if out in {"（翻譯暫不可用）", "翻譯暫不可用", "無法翻譯。", "無法翻譯"}:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if "unable to translate" in low or "cannot translate" in low:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if "無法翻譯" in out or "抱歉" in out:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if "無法找到您要翻譯的內容" in out or "請提供更多資訊" in out or "請提供更多信息" in out:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if "請提供更多資訊或完整的英文文本" in out or "請提供更多信息或完整的英文文本" in out:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            # Assistant-like / conversational filler (must not appear as "translation").
            if re.search(
                r"\b(i couldn't find|i cannot find|if you would like|please note|alternatively,|"
                r"i can try to|provide a translation|once it becomes available|confidential|proprietary|"
                r"do you have a specific link|need a quote)\b",
                low,
            ):
                cli = _try_cli()
                if cli:
                    return cli
                continue
            # Reject HTML/CSS artifacts.
            if "<a href" in low or "@media" in low or "{" in out or "}" in out:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            if not out:
                cli = _try_cli()
                if cli:
                    return cli
                continue
            _CACHE[key] = out
            return out
        except Exception:
            # Always try CLI fallback when HTTP path fails.
            cli_out = _try_cli()
            if cli_out:
                return cli_out
            continue

        # HTTP succeeded but response was unusable; still try CLI fallback.
        cli_out = _try_cli()
        if cli_out:
            return cli_out

    return None
