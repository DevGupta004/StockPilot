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
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from config import CONFIG
from utils.cache import DayCache
from utils.log import log
from utils.retry import retry_call

_CACHE = DayCache(CONFIG.cache_dir)
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

IST = timezone(timedelta(hours=5, minutes=30))
_MKT_CLOSE_H, _MKT_CLOSE_M = 15, 30  # NSE close 15:30 IST
# Re-attempt a stale-by-bar cache entry at most this often (seconds). Guards against
# refetching every call when the provider itself lags behind the expected session.
_STALE_REFRESH_SEC = 1800  # 30 min


def expected_last_bar_date() -> date:
    """The date of the newest *completed* daily bar we should have by now.

    Weekday holidays are not modelled (NSE holiday calendar varies); this is a
    freshness floor — if cached data is older than this, it is definitely stale.
    Rolls weekends back to Friday; before today's close, expects the prior session.
    """
    now = datetime.now(IST)
    d = now.date()
    # Roll a weekend back to Friday.
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    # If it's a weekday but today's session hasn't closed yet, the newest completed
    # daily bar is the previous trading day.
    if d == now.date():
        before_close = (now.hour, now.minute) < (_MKT_CLOSE_H, _MKT_CLOSE_M)
        if before_close:
            d -= timedelta(days=1)
            while d.weekday() >= 5:
                d -= timedelta(days=1)
    return d


def _frame_last_date(df: pd.DataFrame) -> date | None:
    try:
        return pd.Timestamp(df.index[-1]).date()
    except Exception:  # noqa: BLE001
        return None


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
# Yahoo chart API (keyless fallback) — same data backend as yfinance, but a
# completely independent code path: our own httpx session, User-Agent, and
# rate-limit bucket. This survives the most common yfinance-library failures
# (stale cookie/crumb, internal session rate-limit, lib version breakage), so it
# is a genuine second source with no API key required.
# --------------------------------------------------------------------------- #
class YahooChartProvider(MarketProvider):
    name = "yahoo_chart"
    HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
    _UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    def fetch(self, ticker: str, lookback_days: int) -> pd.DataFrame | None:
        import httpx

        # Pick a range bucket comfortably larger than the lookback.
        rng = "2y" if lookback_days > 250 else "1y" if lookback_days > 120 else "6mo"
        params = {"range": rng, "interval": "1d", "includeAdjustedClose": "true"}
        last_exc: Exception | None = None
        for host in self.HOSTS:  # try both Yahoo hosts before giving up
            url = f"{host}/v8/finance/chart/{ticker}"
            try:
                with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                    resp = client.get(url, params=params,
                                      headers={"User-Agent": self._UA})
                    resp.raise_for_status()
                    payload = resp.json()
                chart = (payload or {}).get("chart", {})
                if chart.get("error"):
                    log(f"yahoo_chart: {chart['error']} for {ticker}")
                    continue
                results = chart.get("result") or []
                if not results:
                    continue
                res = results[0]
                ts = res.get("timestamp") or []
                quote = (res.get("indicators", {}).get("quote") or [{}])[0]
                if not ts or not quote:
                    continue
                df = pd.DataFrame({
                    "Open": quote.get("open"),
                    "High": quote.get("high"),
                    "Low": quote.get("low"),
                    "Close": quote.get("close"),
                    "Volume": quote.get("volume"),
                }, index=pd.to_datetime(ts, unit="s"))
                df = df[_OHLCV_COLS].dropna()
                return df if not df.empty else None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if last_exc:
            log(f"yahoo_chart: fetch failed for {ticker}: {last_exc}")
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


_PROVIDERS: dict[str, type[MarketProvider]] = {
    "yfinance": YFinanceProvider,
    "yahoo_chart": YahooChartProvider,
    "twelvedata": TwelveDataProvider,
}

# Order in which sources are tried. yfinance first (richest), then the keyless raw
# Yahoo path (independent session — survives yfinance lib/session failures), then
# Twelve Data only if a key is configured.
_FALLBACK_ORDER = ["yfinance", "yahoo_chart", "twelvedata"]


def _make_provider(name: str) -> MarketProvider:
    return _PROVIDERS.get(name, YFinanceProvider)()


def get_ohlcv(ticker: str, lookback_days: int | None = None) -> pd.DataFrame | None:
    """Fetch daily OHLCV with caching + automatic provider fallback.

    Returns a DataFrame indexed by date (Open/High/Low/Close/Volume) or None if no
    provider could supply usable data for ``ticker``.
    """
    lookback = lookback_days or CONFIG.lookback_days

    # Load the per-day cache. Keep it only if its newest bar is recent enough; a frame
    # whose last bar predates the expected last completed session is STALE (provider
    # lag can write a day-old frame that the per-day cache would otherwise serve all
    # day). Stale frames are kept as a fallback in case the refetch fails.
    stale_df: pd.DataFrame | None = None
    cached = _CACHE.get_data("ohlcv", ticker)
    if cached is not None:
        try:
            df = pd.read_json(io.StringIO(cached), orient="split")
            if not df.empty:
                last = _frame_last_date(df)
                fresh = last is not None and last >= expected_last_bar_date()
                # A stale-by-bar entry is only worth refetching once per refresh window:
                # if the provider simply lags (holiday / not-yet-published session) the
                # cache will never reach `expected`, so accept it rather than hammer the
                # provider on every call (esp. the 500-symbol scan).
                age = _CACHE.age_seconds("ohlcv", ticker) or 0.0
                if fresh or age < _STALE_REFRESH_SEC:
                    df.attrs["provider"] = "cache" if fresh else "cache-stale"
                    return df
                df.attrs["provider"] = "cache-stale"
                stale_df = df  # genuinely old — try a refetch, fall back on failure
                log(f"cache: stale ohlcv for {ticker} (last bar {last} < expected "
                    f"{expected_last_bar_date()}, age {age:.0f}s); refetching")
        except ValueError as exc:
            log(f"cache: bad ohlcv frame for {ticker}: {exc}")

    primary = CONFIG.providers.market
    order = [primary] + [p for p in _FALLBACK_ORDER if p != primary]

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

    # Every provider failed. A stale cached frame beats nothing — return it flagged so
    # the freshness line can warn, rather than dropping the symbol entirely.
    if stale_df is not None:
        log(f"market: refetch failed for {ticker}, serving STALE cache")
        return stale_df

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
