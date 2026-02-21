import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import pywt
except Exception:
    pywt = None

try:
    from pykalman import KalmanFilter
except Exception:
    KalmanFilter = None

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


M7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]


@dataclass
class RegimeResult:
    ticker: str
    trend_score: float
    regime: str
    evidence: str


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    x = np.arange(window)
    def slope(y):
        if len(y) < window:
            return np.nan
        coef = np.polyfit(x, y, 1)
        return coef[0]
    return series.rolling(window).apply(slope, raw=False)


def denoise_ema(series: pd.Series, fast: int = 8, slow: int = 21) -> pd.Series:
    # Simple denoise: double EMA
    return _ema(_ema(series, fast), slow)


def denoise_fft(series: pd.Series, keep: int = 6) -> pd.Series:
    y = np.asarray(series).astype(float).reshape(-1)
    n = len(y)
    if n < 20:
        return series.copy()
    # FFT
    fft = np.fft.rfft(y)
    mags = np.abs(fft)
    # keep top-k frequencies (excluding DC)
    idx = np.argsort(mags)[::-1]
    keep_idx = set(idx[:keep])
    filtered = np.zeros_like(fft)
    for i in keep_idx:
        filtered[i] = fft[i]
    recon = np.fft.irfft(filtered, n=n)
    return pd.Series(recon, index=series.index)


def denoise_wavelet(series: pd.Series, wavelet: str = "db4", level: int = 3) -> pd.Series:
    if pywt is None:
        return series.copy()
    y = series.values.astype(float)
    if len(y) < 32:
        return series.copy()
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # soft-threshold detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2 * np.log(len(y))) if sigma > 0 else 0.0
    coeffs_denoised = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(c, thr, mode="soft"))
    recon = pywt.waverec(coeffs_denoised, wavelet)
    recon = recon[:len(y)]
    return pd.Series(recon, index=series.index)


def denoise_kalman(series: pd.Series) -> pd.Series:
    if KalmanFilter is None:
        return series.copy()
    y = np.asarray(series).astype(float).reshape(-1, 1)
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=y[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    state_means, _ = kf.filter(y)
    return pd.Series(state_means.flatten(), index=series.index)


def denoise_kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    close = series.values.astype(float)
    n = len(close)
    if n < period + 1:
        return series.copy()
    change = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    er = np.zeros(n)
    er[period:] = np.where(volatility[period:] == 0, 0, change / volatility[period:])
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros(n)
    kama[:period] = close[:period]
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return pd.Series(kama, index=series.index)


def denoise_alma(series: pd.Series, window: int = 20, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    y = series.values.astype(float)
    n = len(y)
    if n < window:
        return series.copy()
    m = offset * (window - 1)
    s = window / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(window)])
    w /= w.sum()
    out = np.full(n, np.nan)
    for i in range(window-1, n):
        out[i] = np.dot(w, y[i-window+1:i+1])
    # forward-fill initial
    for i in range(window-1):
        out[i] = y[i]
    return pd.Series(out, index=series.index)


def denoise_hma(series: pd.Series, period: int = 20) -> pd.Series:
    y = series.values.astype(float)
    if len(y) < period:
        return series.copy()
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma = lambda x, p: pd.Series(x).rolling(p).apply(lambda v: np.dot(np.arange(1, p+1), v) / np.sum(np.arange(1, p+1)), raw=True)
    wma_half = wma(y, half)
    wma_full = wma(y, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_p).fillna(method="bfill")
    return pd.Series(hma.values, index=series.index)


def denoise_hmm(series: pd.Series, n_states: int = 2) -> pd.Series:
    if GaussianHMM is None:
        return series.copy()
    y = np.asarray(series).astype(float)
    returns = np.diff(y) / y[:-1]
    returns = returns.reshape(-1, 1)
    if len(returns) < 50:
        return series.copy()
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(returns)
    states = model.predict(returns)
    # smooth: map state to mean return sign
    means = model.means_.flatten()
    state_signal = np.sign(means[states])
    # reconstruct a smoothed price path via cumulative sum of signed returns
    smooth = [y[0]]
    for i, s in enumerate(state_signal):
        smooth.append(smooth[-1] * (1 + s * abs(returns[i][0])))
    smooth = np.array(smooth)
    if len(smooth) < len(series):
        smooth = np.pad(smooth, (0, len(series)-len(smooth)), 'edge')
    return pd.Series(smooth[:len(series)], index=series.index)


def detect_regime(df: pd.DataFrame, denoise_mode: str = "ema") -> RegimeResult:
    close = df["close"].copy()
    if denoise_mode == "fft":
        dn = denoise_fft(close)
    elif denoise_mode == "wavelet":
        dn = denoise_wavelet(close)
    elif denoise_mode == "kalman":
        dn = denoise_kalman(close)
    elif denoise_mode == "kama":
        dn = denoise_kama(close)
    elif denoise_mode == "alma":
        dn = denoise_alma(close)
    elif denoise_mode == "hma":
        dn = denoise_hma(close)
    elif denoise_mode == "hmm":
        dn = denoise_hmm(close)
    else:
        dn = denoise_ema(close)
    slope = _rolling_slope(dn, 20)
    rsi = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)

    s_last = slope.iloc[-1]
    m_last = hist.iloc[-1]
    r_last = rsi.iloc[-1]
    if isinstance(s_last, pd.Series):
        s_last = s_last.iloc[0]
    if isinstance(m_last, pd.Series):
        m_last = m_last.iloc[0]
    if isinstance(r_last, pd.Series):
        r_last = r_last.iloc[0]
    slope_now = float(s_last) if not pd.isna(s_last) else 0.0
    macd_now = float(m_last) if not pd.isna(m_last) else 0.0
    rsi_now = float(r_last) if not pd.isna(r_last) else 50.0
    p_last = close.iloc[-1]
    e_last = _ema(close, 20).iloc[-1]
    if isinstance(p_last, pd.Series):
        p_last = p_last.iloc[0]
    if isinstance(e_last, pd.Series):
        e_last = e_last.iloc[0]
    price = float(p_last)
    ema20 = float(e_last)

    trend_score = 0.0
    if slope_now > 0:
        trend_score += 0.5
    if macd_now > 0:
        trend_score += 0.3
    if price > ema20:
        trend_score += 0.2

    # Simple regime decision
    if slope_now > 0 and macd_now > 0 and rsi_now > 50:
        regime = "Continuation (Uptrend)"
    elif slope_now < 0 and macd_now < 0 and rsi_now < 50:
        regime = "Continuation (Downtrend)"
    else:
        # potential turning / consolidation
        regime = "Potential Transition / Range"

    evidence = f"denoise={denoise_mode}, slope={slope_now:.4f}, macd_hist={macd_now:.4f}, rsi={rsi_now:.1f}, price_vs_ema20={'above' if price>ema20 else 'below'}"
    return RegimeResult(ticker="", trend_score=trend_score, regime=regime, evidence=evidence)


def fetch_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    data = df[["open", "high", "low", "close", "volume"]]
    # yfinance sometimes returns a DataFrame with columns as tuples; flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    if isinstance(data, pd.DataFrame) and isinstance(data.get("close"), pd.DataFrame):
        data["close"] = data["close"].iloc[:, 0]
    return data.dropna()


def run_m7(period: str = "1y", interval: str = "1d", denoise_mode: str = "ema") -> List[RegimeResult]:
    results = []
    for t in M7:
        df = fetch_prices(t, period=period, interval=interval)
        if df.empty or len(df) < 60:
            continue
        res = detect_regime(df, denoise_mode=denoise_mode)
        res.ticker = t
        results.append(res)
    return results


def backtest_regime(df: pd.DataFrame, denoise_mode: str = "ema") -> Dict[str, float]:
    close = df["close"].copy()
    if denoise_mode == "fft":
        dn = denoise_fft(close)
    elif denoise_mode == "wavelet":
        dn = denoise_wavelet(close)
    else:
        dn = denoise_ema(close)

    slope = _rolling_slope(dn, 20)
    rsi = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    ema20 = _ema(close, 20)

    signals = []
    for i in range(1, len(df) - 1):
        s_i = slope.iloc[i]
        m_i = hist.iloc[i]
        r_i = rsi.iloc[i]
        p_i = close.iloc[i]
        e_i = ema20.iloc[i]

        if isinstance(s_i, pd.Series):
            s_i = s_i.iloc[0]
        if isinstance(m_i, pd.Series):
            m_i = m_i.iloc[0]
        if isinstance(r_i, pd.Series):
            r_i = r_i.iloc[0]
        if isinstance(p_i, pd.Series):
            p_i = p_i.iloc[0]
        if isinstance(e_i, pd.Series):
            e_i = e_i.iloc[0]

        slope_now = float(s_i) if not pd.isna(s_i) else np.nan
        macd_now = float(m_i) if not pd.isna(m_i) else 0.0
        rsi_now = float(r_i) if not pd.isna(r_i) else 50.0
        price = float(p_i)
        ema = float(e_i)

        if np.isnan(slope_now) or np.isnan(ema):
            signals.append(0)
            continue

        if slope_now > 0 and macd_now > 0 and rsi_now > 50 and price > ema:
            signals.append(1)
        elif slope_now < 0 and macd_now < 0 and rsi_now < 50 and price < ema:
            signals.append(-1)
        else:
            signals.append(0)

    # next-day return sign
    next_ret = close.pct_change().shift(-1).iloc[1:1+len(signals)]
    preds = np.array(signals)
    actual = np.sign(next_ret.values)

    mask = preds != 0
    if mask.sum() == 0:
        return {"accuracy": 0.0, "coverage": 0.0}
    acc = (preds[mask] == actual[mask]).mean()
    coverage = mask.mean()
    return {"accuracy": float(acc), "coverage": float(coverage)}


def backtest_m7(period: str = "1y", interval: str = "1d", denoise_mode: str = "ema") -> Dict[str, Dict[str, float]]:
    out = {}
    for t in M7:
        df = fetch_prices(t, period=period, interval=interval)
        if df.empty or len(df) < 120:
            continue
        out[t] = backtest_regime(df, denoise_mode=denoise_mode)
    return out


if __name__ == "__main__":
    out = run_m7(denoise_mode="ema")
    for r in out:
        print(f"{r.ticker}: {r.regime} | score={r.trend_score:.2f} | {r.evidence}")
    print("\\nBacktest (EMA):")
    bt = backtest_m7(denoise_mode="ema")
    for k, v in bt.items():
        print(k, v)
