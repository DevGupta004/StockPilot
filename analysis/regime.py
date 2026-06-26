"""Broad-market regime read for the Indian (NSE) market.

LONG-only delivery picks fight the tape in a falling market. ``assess`` reads the Nifty
50 index (``^NSEI``) once per scan and classifies the backdrop as RISK_ON / NEUTRAL /
RISK_OFF from the EMA stack and the 20-bar trend slope (reusing the same primitives the
per-stock engine uses). The scan keeps returning its top-3 either way — the regime only
shaves confidence and adds a banner (project decision), never suppresses.
"""

from __future__ import annotations

from analysis import indicators as ind
from analysis import patterns
from data.market import get_ohlcv
from utils.log import log

NIFTY = "^NSEI"

RISK_ON = "RISK_ON"
NEUTRAL = "NEUTRAL"
RISK_OFF = "RISK_OFF"

_UNKNOWN = {
    "regime": NEUTRAL,
    "nifty_trend": "unknown",
    "nifty_price": None,
    "note": "market regime unavailable (Nifty fetch failed) — treated as NEUTRAL",
}


def assess() -> dict:
    """Return {regime, nifty_trend, nifty_price, note}. Never raises."""
    try:
        df = get_ohlcv(NIFTY)
        if df is None or len(df) < 60:
            return dict(_UNKNOWN)
        t = ind.compute(df)
        trend, slope = patterns._trend(df["Close"])
        price = t["price"]
        above_stack = price > t["ema21"] > t["ema50"]
        below_stack = price < t["ema21"] < t["ema50"]

        if below_stack and trend == "down":
            regime, note = RISK_OFF, "Nifty below falling EMA21/EMA50 — RISK-OFF"
        elif above_stack and trend == "up":
            regime, note = RISK_ON, "Nifty above rising EMA21/EMA50 — RISK-ON"
        elif trend == "down" or below_stack:
            regime, note = RISK_OFF, "Nifty trend weak/below stack — RISK-OFF"
        else:
            regime, note = NEUTRAL, "Nifty mixed — NEUTRAL"

        return {
            "regime": regime,
            "nifty_trend": trend,
            "nifty_price": round(price, 2),
            "nifty_slope_pct": round(slope * 100, 2),
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001 - regime is best-effort context
        log(f"regime: assess failed: {exc}")
        return dict(_UNKNOWN)
