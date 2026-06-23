"""Dynamic stock universe — fetched live from NSE, never hardcoded.

Presets resolve to NSE index constituents pulled from the public NSE archives CSVs
(no API key, one request each), or the full equity list via nsepython. Results are
cached per day and every network call is retry-wrapped with backoff, so a transient
rate-limit doesn't break a scan.

Returned symbols carry the yfinance ``.NS`` suffix (e.g. ``RELIANCE.NS``).
"""

from __future__ import annotations

import csv
import io

import httpx

from config import CONFIG
from utils.cache import DayCache
from utils.log import log
from utils.retry import retry_call

_CACHE = DayCache(CONFIG.cache_dir)
_HDR = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}

# Index name -> NSE archives constituent CSV. These are the live source of truth.
_INDEX_CSV = {
    "nifty50": "ind_nifty50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
    "niftymidcap150": "ind_niftymidcap150list.csv",
    "niftysmallcap250": "ind_niftysmallcap250list.csv",
    "niftynext50": "ind_niftynext50list.csv",
}
_BASE = "https://archives.nseindia.com/content/indices/"

# Friendly aliases the user / tools may pass.
_ALIASES = {
    "default": "nifty500",
    "nifty": "nifty50",
    "all": "all",
    "cheap": "nifty500",     # broad pool; the MAX_PRICE filter does the "sub-₹500" part
    "under500": "nifty500",
}


def _fetch_index_csv(index: str) -> list[str]:
    url = _BASE + _INDEX_CSV[index]

    def _do() -> list[str]:
        resp = httpx.get(url, headers=_HDR, timeout=20.0, follow_redirects=True)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        syms = [row["Symbol"].strip() for row in reader if row.get("Symbol", "").strip()]
        if not syms:
            raise ValueError(f"empty constituent list for {index}")
        return syms

    syms = retry_call(_do, attempts=3, base=1.5, label=f"universe:{index}")
    return [f"{s}.NS" for s in syms]


def _fetch_all_equities() -> list[str]:
    def _do() -> list[str]:
        from nsepython import nse_eq_symbols
        syms = nse_eq_symbols()
        if not syms:
            raise ValueError("nse_eq_symbols returned empty")
        return syms

    syms = retry_call(_do, attempts=3, base=2.0, label="universe:all")
    return [f"{s}.NS" for s in syms]


def fetch_universe(name: str) -> list[str]:
    """Resolve a preset/index name to a live list of NSE symbols (cached per day).

    Returns ``[]`` if the live source can't be reached after retries — callers should
    surface that rather than fall back to stale hardcoded names.
    """
    key = _ALIASES.get(name.lower(), name.lower())

    cached = _CACHE.get_data("universe", key)
    if cached:
        return cached

    try:
        if key == "all":
            syms = _fetch_all_equities()
        elif key in _INDEX_CSV:
            syms = _fetch_index_csv(key)
        else:
            log(f"universe: unknown preset '{name}', defaulting to nifty500")
            syms = _fetch_index_csv("nifty500")
    except Exception as exc:  # noqa: BLE001
        log(f"universe: live fetch failed for '{name}': {exc}")
        return []

    _CACHE.set("universe", key, syms)
    log(f"universe: loaded {len(syms)} symbols for '{key}' (live)")
    return syms


def is_preset(name: str) -> bool:
    key = _ALIASES.get(name.lower(), name.lower())
    return key == "all" or key in _INDEX_CSV
