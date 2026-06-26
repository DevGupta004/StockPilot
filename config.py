"""Central configuration for the Short-Swing Stock Signal MCP server.

Everything tunable lives here: the stock universe, scoring weights, thresholds,
the hard holding horizon, and which data/news providers are active. Values can be
overridden via environment variables so the server can be reconfigured at launch
without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Hard rules — do not exceed the holding window.
# --------------------------------------------------------------------------- #
HORIZON_DAYS: int = 2  # NEVER hold a short-swing candidate longer than this.

DISCLAIMER: str = (
    "Educational research signal only — NOT financial advice, NOT a prediction. "
    "Short-term moves are largely noise; you can lose money. Do your own research."
)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Universe — resolved LIVE from NSE (no hardcoded stock lists). A name is an inline
# comma list ("RELIANCE.NS,TCS.NS"), a single ticker, or an index preset
# ("nifty50/100/200/500", "niftymidcap150", "niftysmallcap250", "niftynext50", "all")
# that data/universe.py fetches from the NSE archives / nsepython and caches per day.
# --------------------------------------------------------------------------- #
DEFAULT_PRESET: str = os.environ.get("STOCK_UNIVERSE", "nifty500")


def named_universe(name: str | None) -> list[str]:
    """Resolve a universe to NSE symbols. Presets are fetched live (cached per day).

    Order: explicit inline list / single ticker > given preset > STOCK_UNIVERSE env >
    "nifty500". Returns [] if a live preset fetch fails after retries, so callers can
    surface that instead of using stale hardcoded names.
    """
    # Lazy import avoids a circular import (universe.py imports CONFIG).
    from data.universe import fetch_universe

    name = name or DEFAULT_PRESET
    if "," in name:  # inline comma-separated list of symbols
        return [_nse(s) for s in name.split(",") if s.strip()]
    if "." in name and " " not in name:  # a single concrete ticker e.g. RELIANCE.NS
        return [_nse(name)]
    from data.universe import is_preset
    if is_preset(name):  # a known preset/index name -> fetch live
        return fetch_universe(name)
    if " " not in name:  # bare single symbol e.g. "RELIANCE" -> treat as NSE ticker
        return [_nse(name)]
    return fetch_universe(name)


def _nse(sym: str) -> str:
    """Normalize a symbol to a yfinance NSE ticker: append .NS if no exchange suffix."""
    s = sym.strip().upper()
    return s if s.endswith((".NS", ".BO")) else f"{s}.NS"


@dataclass(frozen=True)
class Weights:
    """Blend weights for the final score. technical + sentiment should sum to 1.0."""

    technical: float = _env_float("WEIGHT_TECHNICAL", 0.60)
    sentiment: float = _env_float("WEIGHT_SENTIMENT", 0.40)


@dataclass(frozen=True)
class Thresholds:
    # Default confidence bar separating ACTIONABLE from LOW CONFIDENCE.
    min_confidence: float = _env_float("MIN_CONFIDENCE", 0.55)
    # Volume spike multiple vs the baseline average to count as a confirming signal.
    volume_spike: float = _env_float("VOLUME_SPIKE_MULT", 1.5)
    # Default recent-vs-baseline volume surge multiple for scan_volume_spikes.
    min_surge: float = _env_float("MIN_SURGE", 2.0)
    # Volume lookback windows (trading days). `volume_base` = the "normal" baseline
    # (~1 month); `volume_window` = the recent burst compared against it. Match the
    # recent window to the holding horizon (short for 1-2 day swings).
    volume_base: int = _env_int("VOLUME_BASE", 20)
    volume_window: int = _env_int("VOLUME_WINDOW", 7)
    # Below this (last-bar volume / baseline avg) volume is "drying up" → conviction
    # penalty on the LONG technical score.
    dry_volume_ratio: float = _env_float("DRY_VOLUME_RATIO", 0.6)
    # ATR multiples used to derive target / stop.
    atr_target_mult: float = _env_float("ATR_TARGET_MULT", 1.5)
    atr_stop_mult: float = _env_float("ATR_STOP_MULT", 1.0)
    # Deepest pullback (in ATRs below spot) the entry is allowed to ask for. EMA21 is
    # often 5-8% below a trending stock — unreachable inside a 1-2 day hold — so the
    # buy level is clamped to at most this shallow, fillable dip, with a buy-at-open
    # fallback. Raise to wait for deeper dips, lower (→0) to enter nearer market.
    max_pullback_atr: float = _env_float("MAX_PULLBACK_ATR", 0.5)
    # In a RISK-OFF market (Nifty down-trend), LONG-only delivery picks face a headwind;
    # confidence is multiplied by this factor (picks still shown, just downgraded).
    risk_off_factor: float = _env_float("RISK_OFF_FACTOR", 0.85)
    # Oversold scan (scan_oversold). A name qualifies as oversold when RSI-14 is at/below
    # `oversold_rsi`; Stochastic %K at/below `oversold_stoch` and a close under the lower
    # Bollinger band add to the oversold depth score. `oversold_top_n` = how many to
    # return (the configurable "top N", default 5).
    oversold_rsi: float = _env_float("OVERSOLD_RSI", 30.0)
    oversold_stoch: float = _env_float("OVERSOLD_STOCH", 20.0)
    oversold_top_n: int = _env_int("OVERSOLD_TOP_N", 5)


@dataclass(frozen=True)
class Providers:
    # market: "yfinance" (default) | "yahoo_chart" (keyless raw Yahoo) | "twelvedata".
    # Whatever is set is tried first; the others are automatic fallbacks (yfinance ->
    # yahoo_chart -> twelvedata). yahoo_chart needs no key; twelvedata is skipped if
    # TWELVEDATA_API_KEY is unset.
    market: str = os.environ.get("MARKET_PROVIDER", "yfinance")
    # news: "marketaux" (default) or "rss"
    news: str = os.environ.get("NEWS_PROVIDER", "marketaux")
    # live spot price: "nsepython" (quasi-realtime NSE, default) or "yfinance" (delayed)
    live: str = os.environ.get("LIVE_PROVIDER", "nsepython")


@dataclass(frozen=True)
class Config:
    horizon_days: int = HORIZON_DAYS
    lookback_days: int = _env_int("LOOKBACK_DAYS", 270)  # ~9 months of dailies
    news_lookback_hours: int = _env_int("NEWS_LOOKBACK_HOURS", 72)
    # How far back to scan NSE corporate announcements for a fresh catalyst. Wide
    # enough to catch a marinating trigger (e.g. a takeover/stake disclosure that
    # moves the stock days later); recency-weighting keeps fresh filings dominant.
    catalyst_lookback_hours: int = _env_int("CATALYST_LOOKBACK_HOURS", 240)
    weights: Weights = field(default_factory=Weights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    providers: Providers = field(default_factory=Providers)
    cache_dir: str = os.environ.get(
        "CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache")
    )

    # Position sizing for delivery (CNC) trades.
    capital: float = _env_float("CAPITAL", 150000.0)   # default ₹1.5 lakh
    risk_per_trade: float = _env_float("RISK_PER_TRADE", 0.02)  # 2% of capital at risk
    # Delivery (CNC) mode: only LONG candidates — shorts can't be held as delivery on
    # NSE (intraday only). On by default for delivery traders.
    delivery_only: bool = os.environ.get("DELIVERY_ONLY", "true").lower() in (
        "1", "true", "yes", "on"
    )
    max_price: float | None = (
        _env_float("MAX_PRICE", 0.0) or None  # 0/unset = no price cap
    )

    # Provider API keys (never hardcode — read from env).
    marketaux_api_key: str | None = os.environ.get("MARKETAUX_API_KEY")
    twelvedata_api_key: str | None = os.environ.get("TWELVEDATA_API_KEY")


CONFIG = Config()
