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
        return [s.strip().upper() for s in name.split(",") if s.strip()]
    if "." in name and " " not in name:  # a single concrete ticker e.g. RELIANCE.NS
        return [name.strip().upper()]
    return fetch_universe(name)


@dataclass(frozen=True)
class Weights:
    """Blend weights for the final score. technical + sentiment should sum to 1.0."""

    technical: float = _env_float("WEIGHT_TECHNICAL", 0.60)
    sentiment: float = _env_float("WEIGHT_SENTIMENT", 0.40)


@dataclass(frozen=True)
class Thresholds:
    # Default confidence bar separating ACTIONABLE from LOW CONFIDENCE.
    min_confidence: float = _env_float("MIN_CONFIDENCE", 0.55)
    # Volume spike multiple vs 20-day average to count as a confirming signal.
    volume_spike: float = _env_float("VOLUME_SPIKE_MULT", 1.5)
    # Default 7-day vs baseline volume surge multiple for scan_volume_spikes.
    min_surge: float = _env_float("MIN_SURGE", 2.0)
    # ATR multiples used to derive target / stop.
    atr_target_mult: float = _env_float("ATR_TARGET_MULT", 1.5)
    atr_stop_mult: float = _env_float("ATR_STOP_MULT", 1.0)


@dataclass(frozen=True)
class Providers:
    # market: "yfinance" (default) or "twelvedata"
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
