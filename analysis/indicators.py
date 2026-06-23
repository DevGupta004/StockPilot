"""Technical indicators.

Uses pandas-ta when available, otherwise falls back to hand-rolled numpy/pandas
implementations so the server works even if pandas-ta won't install (it can be
finicky on newer numpy). Every function takes an OHLCV DataFrame and returns plain
floats / bools so the result is JSON-serialisable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pandas-ta is optional; we hand-roll everything below as a fallback.
    import pandas_ta as pta  # noqa: F401
    _HAS_PTA = True
except Exception:  # noqa: BLE001 - import can fail on numpy mismatch
    _HAS_PTA = False


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _stoch(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult * std, mid, mid - mult * std


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute(df: pd.DataFrame) -> dict:
    """Compute the full indicator snapshot for the most recent bar.

    Returns a flat dict of latest values plus a few series-derived booleans
    (crossovers, volume spike, OBV trend). All values are JSON-safe scalars.
    """
    close = df["Close"]
    last = float(close.iloc[-1])

    ema9, ema21, ema50 = _ema(close, 9), _ema(close, 21), _ema(close, 50)
    rsi = _rsi(close)
    macd_line, macd_signal, macd_hist = _macd(close)
    atr = _atr(df)
    pct_k, pct_d = _stoch(df)
    bb_up, bb_mid, bb_low = _bollinger(close)
    obv = _obv(df)

    vol = df["Volume"]
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1])
    vol_last = float(vol.iloc[-1])

    def _f(series: pd.Series) -> float:
        val = series.iloc[-1]
        return float(val) if pd.notna(val) else float("nan")

    macd_cross_up = bool(
        macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0
    ) if len(macd_hist) > 1 else False
    ema_cross_up = bool(
        ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]
    ) if len(ema9) > 1 else False

    obv_trend_up = bool(obv.iloc[-1] > obv.iloc[-5]) if len(obv) > 5 else False

    return {
        "price": round(last, 2),
        "ema9": round(_f(ema9), 2),
        "ema21": round(_f(ema21), 2),
        "ema50": round(_f(ema50), 2),
        "rsi14": round(_f(rsi), 2),
        "macd": round(_f(macd_line), 4),
        "macd_signal": round(_f(macd_signal), 4),
        "macd_hist": round(_f(macd_hist), 4),
        "macd_cross_up": macd_cross_up,
        "ema_cross_up": ema_cross_up,
        "atr14": round(_f(atr), 2),
        "stoch_k": round(_f(pct_k), 2),
        "stoch_d": round(_f(pct_d), 2),
        "bb_upper": round(_f(bb_up), 2),
        "bb_mid": round(_f(bb_mid), 2),
        "bb_lower": round(_f(bb_low), 2),
        "obv_trend_up": obv_trend_up,
        "vol_last": round(vol_last, 0),
        "vol_avg20": round(vol_avg20, 0),
        "vol_spike_ratio": round(vol_last / vol_avg20, 2) if vol_avg20 else 0.0,
        "recent_high20": round(float(df["High"].rolling(20).max().iloc[-1]), 2),
        "recent_low20": round(float(df["Low"].rolling(20).min().iloc[-1]), 2),
        "engine": "pandas-ta" if _HAS_PTA else "builtin",
    }
