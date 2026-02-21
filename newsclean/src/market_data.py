import time
import io
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from typing import Optional


_ET = ZoneInfo("America/New_York")


@dataclass
class Quote:
    ticker: str
    price: float
    ts_utc: datetime
    day_change_pct: float


_CACHE: dict[str, tuple[float, Quote]] = {}
_FAIL_CACHE: dict[str, float] = {}
_FAIL_TTL_SECONDS = 900
_BAD_TICKERS = {"PE", "PV"}


def _to_utc_ts(idx_val) -> datetime:
    # yfinance index can be tz-aware or naive; normalize to UTC datetime
    if isinstance(idx_val, pd.Timestamp):
        dt = idx_val.to_pydatetime()
    else:
        dt = idx_val
    if not isinstance(dt, datetime):
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_quote(ticker: str, cache_ttl_seconds: int = 60) -> Optional[Quote]:
    t = (ticker or "").strip().upper()
    if not t:
        return None
    if t in _BAD_TICKERS:
        return None

    now = time.time()
    fail_ts = _FAIL_CACHE.get(t)
    if fail_ts and (now - fail_ts) <= _FAIL_TTL_SECONDS:
        return None

    cached = _CACHE.get(t)
    if cached and (now - cached[0]) <= cache_ttl_seconds:
        return cached[1]

    # Fast path: yfinance fast_info (usually much cheaper than downloading candles)
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None)
        if fi:
            last_price = fi.get("last_price") or fi.get("lastPrice")
            prev_close = fi.get("previous_close") or fi.get("previousClose")
            if last_price is not None and prev_close:
                last_close = float(last_price)
                prev_close = float(prev_close)
                day_change_pct = (last_close / prev_close - 1.0) * 100.0 if prev_close else 0.0
                q = Quote(
                    ticker=t,
                    price=last_close,
                    ts_utc=datetime.now(timezone.utc),
                    day_change_pct=day_change_pct,
                )
                _CACHE[t] = (now, q)
                return q
    except Exception:
        pass

    def _safe_download(*args, **kwargs):
        # Suppress noisy yfinance warnings for bad symbols.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return yf.download(*args, **kwargs)

    # Fallback: intraday "last" price + previous daily close
    intraday = _safe_download(t, period="1d", interval="5m", auto_adjust=False, progress=False, threads=False)
    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        _FAIL_CACHE[t] = now
        return None

    close_part = intraday["Close"]
    last_close = float(close_part.iloc[-1, 0]) if isinstance(close_part, pd.DataFrame) else float(close_part.iloc[-1])
    last_ts_utc = _to_utc_ts(intraday.index[-1])
    last_date_utc = last_ts_utc.date()

    daily = _safe_download(t, period="10d", interval="1d", auto_adjust=False, progress=False, threads=False)
    prev_close = None
    if daily is not None and not daily.empty and "Close" in daily.columns:
        tmp = daily.copy()
        try:
            tmp_idx = [(_to_utc_ts(x).date()) for x in tmp.index]
            tmp["__d"] = tmp_idx
            tmp = tmp[tmp["__d"] < last_date_utc]
        except Exception:
            pass
        if not tmp.empty:
            c = tmp["Close"]
            prev_close = float(c.iloc[-1, 0]) if isinstance(c, pd.DataFrame) else float(c.iloc[-1])

    if prev_close and prev_close != 0:
        day_change_pct = (last_close / prev_close - 1.0) * 100.0
    else:
        day_change_pct = 0.0

    q = Quote(
        ticker=t,
        price=last_close,
        ts_utc=last_ts_utc,
        day_change_pct=day_change_pct,
    )
    _CACHE[t] = (now, q)
    if t in _FAIL_CACHE:
        _FAIL_CACHE.pop(t, None)
    return q


def format_price_line(q: Quote, include_ticker: bool = False) -> str:
    # Display in ET for US stocks
    ts_et = q.ts_utc.astimezone(_ET)
    ts_s = ts_et.strftime("%H:%M ET")
    if include_ticker:
        return f"Price ({q.ticker}, {ts_s}): US ${q.price:.2f} ({q.day_change_pct:+.2f}% today)"
    return f"Price ({ts_s}): US ${q.price:.2f} ({q.day_change_pct:+.2f}% today)"
