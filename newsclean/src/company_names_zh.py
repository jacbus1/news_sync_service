import json
from pathlib import Path
from typing import Dict


# Minimal seed mapping (Traditional Chinese). Users can extend via data/company_names_zh.json.
_SEED: Dict[str, str] = {
    "AAPL": "蘋果",
    "MSFT": "微軟",
    "GOOGL": "谷歌",
    "GOOG": "谷歌",
    "AMZN": "亞馬遜",
    "META": "Meta",
    "TSLA": "特斯拉",
    "NVDA": "輝達",
    "ILMN": "因美納",
    "LIN": "林德",
    "EL": "雅詩蘭黛",
}


def load_company_names_zh() -> Dict[str, str]:
    """
    Loads ticker->Traditional Chinese company name mapping.
    Optional user overrides: data/company_names_zh.json
    """
    out = dict(_SEED)
    p = Path(__file__).resolve().parent.parent / "data" / "company_names_zh.json"
    try:
        if p.exists():
            obj = json.loads(p.read_text())
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        continue
                    kk = k.strip().upper()
                    vv = v.strip()
                    if kk and vv:
                        out[kk] = vv
    except Exception:
        pass
    return out

