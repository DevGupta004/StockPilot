"""Technical indicators.

Uses pandas-ta when available, otherwise falls back to hand-rolled numpy/pandas
implementations so the server works even if pandas-ta won't install (it can be
finicky on newer numpy). Every function takes an OHLCV DataFrame and returns plain
floats / bools so the result is JSON-serialisable.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import CONFIG


def _clean(value: float, fallback: float) -> float:
    """Return ``value`` rounded to a finite float, or ``fallback`` if NaN/inf.

    Guards against non-finite indicator values (e.g. RSI when there are no down days,
    Stochastic on a flat range) reaching the JSON output — NaN/Infinity are not valid
    JSON and would corrupt the MCP response or propagate into entry/stop arithmetic.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return round(float(fallback), 4)
    return round(v if math.isfinite(v) else float(fallback), 4)

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


def volume_surge(df: pd.DataFrame, window: int | None = None,
                 base: int | None = None) -> dict | None:
    """Detect a volume increase over the last `window` bars vs the prior `base` bars.

    Windows default to the configured VOLUME_WINDOW / VOLUME_BASE (.env). Returns a
    JSON-safe dict (surge_ratio, max_day_spike, price_change_7d_pct, bias, raw averages)
    or None if the frame is too short (< window + base bars).
    bias: ACCUMULATION (vol up & price up) / DISTRIBUTION (vol up & price down) / MIXED.
    """
    window = CONFIG.thresholds.volume_window if window is None else window
    base = CONFIG.thresholds.volume_base if base is None else base
    if df is None or len(df) < window + base:
        return None
    vol = df["Volume"]
    recent = vol.iloc[-window:]
    prior = vol.iloc[-(window + base):-window]
    vol_recent_avg = float(recent.mean())
    vol_base_avg = float(prior.mean())
    if vol_base_avg <= 0:
        return None
    surge = vol_recent_avg / vol_base_avg
    max_spike = float(recent.max()) / vol_base_avg

    close = df["Close"]
    price_chg = (float(close.iloc[-1]) / float(close.iloc[-window]) - 1.0) * 100.0
    if surge >= 1.2 and price_chg > 1.0:
        bias = "ACCUMULATION"
    elif surge >= 1.2 and price_chg < -1.0:
        bias = "DISTRIBUTION"
    else:
        bias = "MIXED"

    return {
        "surge_ratio": round(surge, 2),
        "max_day_spike": round(max_spike, 2),
        "vol_7d_avg": round(vol_recent_avg, 0),
        "vol_base_avg": round(vol_base_avg, 0),
        "price_change_7d_pct": round(price_chg, 2),
        "bias": bias,
        "window": window,
        "base": base,
    }


def oversold(snapshot: dict, rsi_max: float | None = None,
             stoch_max: float | None = None) -> dict:
    """Classify an indicator snapshot as oversold and score how deep.

    Primary gate is RSI-14 at/below `rsi_max` (default OVERSOLD_RSI). Stochastic %K
    at/below `stoch_max` and a close beneath the lower Bollinger band deepen the score.
    `oversold_score` (0..1) ranks how stretched-to-the-downside a name is — higher = more
    oversold = a stronger mean-reversion (bounce) candidate. Oversold is NOT a buy on its
    own; a name can keep falling. Returns a JSON-safe dict.
    """
    rsi_max = CONFIG.thresholds.oversold_rsi if rsi_max is None else rsi_max
    stoch_max = CONFIG.thresholds.oversold_stoch if stoch_max is None else stoch_max

    rsi = float(snapshot.get("rsi14", 50.0))
    stoch_k = float(snapshot.get("stoch_k", 50.0))
    price = float(snapshot.get("price", 0.0))
    bb_low = float(snapshot.get("bb_lower", price))

    below_band = price > 0 and price < bb_low
    is_oversold = rsi <= rsi_max

    reasons: list[str] = []
    if is_oversold:
        reasons.append(f"RSI-14 {rsi:.1f} ≤ {rsi_max:.0f}")
    if stoch_k <= stoch_max:
        reasons.append(f"Stochastic %K {stoch_k:.1f} ≤ {stoch_max:.0f}")
    if below_band:
        reasons.append("close below lower Bollinger band")

    # Depth score: RSI is the backbone (how far below the threshold, scaled by the
    # threshold itself), with bonuses for a confirming Stochastic and a sub-band close.
    rsi_depth = max(0.0, (rsi_max - rsi) / rsi_max) if rsi_max > 0 else 0.0
    stoch_bonus = 0.15 if stoch_k <= stoch_max else 0.0
    band_bonus = 0.15 if below_band else 0.0
    score = min(1.0, 0.70 * rsi_depth + stoch_bonus + band_bonus)

    return {
        "is_oversold": bool(is_oversold),
        "rsi14": round(rsi, 2),
        "stoch_k": round(stoch_k, 2),
        "below_lower_band": bool(below_band),
        "oversold_score": round(score, 4),
        "rsi_max": rsi_max,
        "stoch_max": stoch_max,
        "reasons": reasons,
    }


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
    vbase = CONFIG.thresholds.volume_base
    vol_avg20 = float(vol.rolling(vbase).mean().iloc[-1])
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

    # Every numeric is cleaned to a finite float (NaN/inf -> a neutral fallback) so the
    # snapshot is always JSON-safe and downstream entry/stop math never sees NaN.
    return {
        "price": _clean(last, last),
        "ema9": _clean(_f(ema9), last),
        "ema21": _clean(_f(ema21), last),
        "ema50": _clean(_f(ema50), last),
        "rsi14": _clean(_f(rsi), 50.0),
        "macd": _clean(_f(macd_line), 0.0),
        "macd_signal": _clean(_f(macd_signal), 0.0),
        "macd_hist": _clean(_f(macd_hist), 0.0),
        "macd_cross_up": macd_cross_up,
        "ema_cross_up": ema_cross_up,
        "atr14": _clean(_f(atr), round(last * 0.02, 2)),
        "stoch_k": _clean(_f(pct_k), 50.0),
        "stoch_d": _clean(_f(pct_d), 50.0),
        "bb_upper": _clean(_f(bb_up), last),
        "bb_mid": _clean(_f(bb_mid), last),
        "bb_lower": _clean(_f(bb_low), last),
        "obv_trend_up": obv_trend_up,
        "vol_last": _clean(vol_last, 0.0),
        "vol_avg20": _clean(vol_avg20, 0.0),
        "vol_spike_ratio": _clean(vol_last / vol_avg20, 0.0) if vol_avg20 else 0.0,
        "recent_high20": _clean(float(df["High"].rolling(20).max().iloc[-1]), last),
        "recent_low20": _clean(float(df["Low"].rolling(20).min().iloc[-1]), last),
        "engine": "pandas-ta" if _HAS_PTA else "builtin",
    }
