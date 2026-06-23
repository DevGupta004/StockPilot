"""Market-data layer.

A small provider interface with two implementations — yfinance (free default) and
Twelve Data (free tier, used as a swappable fallback) — plus a per-day cache. The
public entry point is ``get_ohlcv`` which returns a pandas DataFrame indexed by date
with columns Open/High/Low/Close/Volume, or ``None`` if every provider failed.

Every network call is wrapped: rate-limits, timeouts, and missing data degrade
gracefully (try the fallback, then give up on the symbol) and never raise out.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd

from config import CONFIG
from utils.cache import DayCache
from utils.log import log
from utils.retry import retry_call

_CACHE = DayCache(CONFIG.cache_dir)
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


class MarketProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, ticker: str, lookback_days: int) -> pd.DataFrame | None:
        """Return a daily OHLCV DataFrame or None on failure."""


# --------------------------------------------------------------------------- #
# yfinance
# --------------------------------------------------------------------------- #
class YFinanceProvider(MarketProvider):
    name = "yfinance"

    def fetch(self, ticker: str, lookback_days: int) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            log("yfinance not installed")
            return None
        try:
            period = f"{max(lookback_days + 10, 60)}d"
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                log(f"yfinance: empty frame for {ticker}")
                return None
            # yfinance can return a MultiIndex column frame for single tickers.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[[c for c in _OHLCV_COLS if c in df.columns]].dropna()
            return df if not df.empty else None
        except Exception as exc:  # noqa: BLE001 - provider may raise anything
            log(f"yfinance: fetch failed for {ticker}: {exc}")
            return None


# --------------------------------------------------------------------------- #
# Twelve Data (fallback) — free tier ~800 calls/day, global incl. India.
# --------------------------------------------------------------------------- #
class TwelveDataProvider(MarketProvider):
    name = "twelvedata"
    BASE = "https://api.twelvedata.com/time_series"

    def fetch(self, ticker: str, lookback_days: int) -> pd.DataFrame | None:
        if not CONFIG.twelvedata_api_key:
            log("twelvedata: no API key configured")
            return None
        import httpx

        # Twelve Data wants EXCHANGE notation: RELIANCE.NS -> symbol=RELIANCE&exchange=NSE
        symbol, exchange = _split_nse(ticker)
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": max(lookback_days, 60),
            "apikey": CONFIG.twelvedata_api_key,
        }
        if exchange:
            params["exchange"] = exchange
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(self.BASE, params=params)
                resp.raise_for_status()
                payload = resp.json()
            if payload.get("status") == "error" or "values" not in payload:
                log(f"twelvedata: {payload.get('message', 'no values')} for {ticker}")
                return None
            rows = payload["values"]
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            rename = {"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "volume": "Volume"}
            df = df.rename(columns=rename)
            for col in _OHLCV_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[[c for c in _OHLCV_COLS if c in df.columns]].dropna()
            return df if not df.empty else None
        except Exception as exc:  # noqa: BLE001
            log(f"twelvedata: fetch failed for {ticker}: {exc}")
            return None


def _split_nse(ticker: str) -> tuple[str, str | None]:
    if ticker.upper().endswith(".NS"):
        return ticker[:-3], "NSE"
    if ticker.upper().endswith(".BO"):
        return ticker[:-3], "BSE"
    return ticker, None


def _make_provider(name: str) -> MarketProvider:
    return TwelveDataProvider() if name == "twelvedata" else YFinanceProvider()


def get_ohlcv(ticker: str, lookback_days: int | None = None) -> pd.DataFrame | None:
    """Fetch daily OHLCV with caching + automatic provider fallback.

    Returns a DataFrame indexed by date (Open/High/Low/Close/Volume) or None if no
    provider could supply usable data for ``ticker``.
    """
    lookback = lookback_days or CONFIG.lookback_days

    cached = _CACHE.get_data("ohlcv", ticker)
    if cached is not None:
        try:
            df = pd.read_json(io.StringIO(cached), orient="split")
            if not df.empty:
                return df
        except ValueError as exc:
            log(f"cache: bad ohlcv frame for {ticker}: {exc}")

    primary = CONFIG.providers.market
    order = [primary] + [p for p in ("yfinance", "twelvedata") if p != primary]

    for prov_name in order:
        provider = _make_provider(prov_name)

        def _fetch(p=provider):
            out = p.fetch(ticker, lookback)
            if out is None:  # raise so retry_call backs off + retries the provider
                raise RuntimeError("empty fetch")
            return out

        # Retry transient rate-limits/blips with backoff before falling to the next
        # provider. retry_call returns None after the last attempt (default=None).
        df = retry_call(_fetch, attempts=3, base=1.5,
                        label=f"ohlcv:{prov_name}:{ticker}", default=None)
        if df is not None and len(df) >= 30:  # need enough bars for indicators
            df.attrs["provider"] = provider.name
            df.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
            try:
                _CACHE.set("ohlcv", ticker, df.to_json(orient="split"))
            except (ValueError, TypeError) as exc:
                log(f"cache: could not serialise {ticker}: {exc}")
            return df
        log(f"market: {prov_name} insufficient data for {ticker}, trying fallback")

    log(f"market: ALL providers failed for {ticker}")
    return None


def _nse_realtime(ticker: str) -> dict | None:
    """Quasi-realtime NSE last-traded-price via nsepython (scrapes NSE; no API key).

    Returns {price, as_of, kind: "nse-realtime"} or None. nsepython is synchronous and
    can rate-limit / block / timeout — callers must tolerate None and fall back.
    """
    try:
        from nsepython import nse_quote_ltp
    except ImportError:
        log("nsepython not installed; skipping realtime quote")
        return None
    symbol = ticker.upper().removesuffix(".NS").removesuffix(".BO")
    try:
        price = float(nse_quote_ltp(symbol))
        if price and price > 0:
            return {
                "price": round(price, 2),
                "as_of": datetime.now(timezone.utc).isoformat(),
                "kind": "nse-realtime",
            }
    except Exception as exc:  # noqa: BLE001 - scraper can fail many ways
        log(f"nse realtime: failed for {ticker}: {exc}")
    return None


def prefilter_by_price(symbols: list[str], max_price: float,
                       chunk: int = 100) -> list[str]:
    """Cheaply narrow a large universe to symbols trading at/below ``max_price``.

    Batch-downloads the latest close for all symbols in a few yfinance calls (instead
    of one fetch per symbol), so a 500-name universe is filtered with ~5 requests
    before the heavy per-symbol pipeline runs. Retry-wrapped; on failure a chunk is
    kept (not dropped) so we never silently lose candidates. Cached per day.
    """
    if not symbols or max_price is None:
        return symbols

    cache_key = f"pricefilter_{int(max_price)}_{len(symbols)}"
    cached = _CACHE.get_data("prefilter", cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
    except ImportError:
        return symbols

    kept: list[str] = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]

        def _do(b=batch):
            return yf.download(b, period="5d", interval="1d", auto_adjust=False,
                               progress=False, threads=True, group_by="ticker")

        try:
            df = retry_call(_do, attempts=3, base=1.5,
                            label=f"prefilter[{i // chunk}]")
        except Exception as exc:  # noqa: BLE001
            log(f"prefilter: batch {i // chunk} failed ({exc}); keeping it unfiltered")
            kept.extend(batch)
            continue

        for sym in batch:
            try:
                close = df[sym]["Close"].dropna() if sym in df else df["Close"].dropna()
                if len(close) and float(close.iloc[-1]) <= max_price:
                    kept.append(sym)
            except Exception:  # noqa: BLE001 - keep on any parse ambiguity
                kept.append(sym)

    log(f"prefilter: {len(kept)}/{len(symbols)} symbols ≤ ₹{max_price:.0f}")
    _CACHE.set("prefilter", cache_key, kept)
    return kept


def get_spot_price(ticker: str) -> dict | None:
    """Fetch the most-current price available (near-live), with its timestamp.

    Provider order is set by LIVE_PROVIDER (default "nsepython"): tries nsepython's
    quasi-realtime NSE LTP first, then falls back to yfinance ``fast_info`` (delayed
    quote) and a 1-minute intraday bar. NOT cached. Returns {price, as_of (ISO ts),
    kind} or None. Even the realtime path is unofficial — never treat as a true tick.
    """
    if CONFIG.providers.live == "nsepython":
        nse = _nse_realtime(ticker)
        if nse:
            return nse
        # fall through to yfinance on any nsepython failure
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        tk = yf.Ticker(ticker)
        # fast_info is a delayed real-time-ish quote; cheapest path.
        try:
            fi = tk.fast_info
            price = float(fi.get("last_price") or fi.get("lastPrice"))
            if price and price > 0:
                return {
                    "price": round(price, 2),
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "kind": "delayed-quote",
                }
        except Exception:  # noqa: BLE001 - fall through to intraday
            pass

        intra = tk.history(period="1d", interval="1m")
        if intra is not None and not intra.empty:
            last = intra.iloc[-1]
            return {
                "price": round(float(last["Close"]), 2),
                "as_of": intra.index[-1].to_pydatetime().isoformat(),
                "kind": "intraday-1m",
            }
    except Exception as exc:  # noqa: BLE001
        log(f"spot: failed for {ticker}: {exc}")
    return None
