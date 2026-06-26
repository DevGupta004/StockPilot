"""Corporate-catalyst layer (FREE NSE sources).

The technical+news stack is blind to the actual *trigger* behind a sharp move —
an order win, board outcome, takeover/stake disclosure, or block-deal accumulation.
Those land first on the NSE corporate-announcements feed and the bulk/block-deal
tape, BEFORE they ever reach an RSS market wrap. This module pulls both (no API key
required) and classifies them into high-impact event flags + a recency score.

Public entry: ``get_catalysts(ticker)`` -> normalised dict, cached per day, never
raises. On any network/parse failure it degrades to an empty-but-valid result.

Why free: the NSE site exposes the same JSON its own pages consume. We prime a
session cookie by hitting the homepage first (the site rejects cold API calls),
exactly the pattern nsepython already uses for live quotes elsewhere in this repo.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from config import CONFIG
from utils.cache import DayCache
from utils.log import log

_CACHE = DayCache(CONFIG.cache_dir)

_NSE_HOME = "https://www.nseindia.com"
_ANN_URL = "https://www.nseindia.com/api/corporate-announcements"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Announcement subject/description -> high-impact event flag. Order matters: an
# announcement is tagged with the FIRST pattern it matches, so put the strongest
# directional catalysts (orders, M&A) ahead of routine boilerplate.
_CATALYST_PATTERNS = {
    "order_win": re.compile(
        r"\b(order|contract|bags?|wins?|awarded|bagged|secures?|LOA|"
        r"letter of (award|intent)|deal worth|new client|go.?live)\b", re.I),
    "mna": re.compile(
        r"\b(acqui|acquisition|merger|amalgamat|takeover|stake|buyout|"
        r"slump sale|divest|joint venture|\bJV\b)\b", re.I),
    "results": re.compile(
        r"\b(financial results|board meeting|outcome of board|earnings|"
        r"quarterly result|q[1-4]\b|profit|revenue)\b", re.I),
    "fundraise": re.compile(
        r"\b(fund.?rais|preferential|QIP|rights issue|warrant|"
        r"allotment of (equity|shares)|debenture|NCD)\b", re.I),
    "rating_buyback": re.compile(
        r"\b(buy.?back|credit rating|rating (upgrade|revis)|bonus issue|"
        r"dividend|split|stock split)\b", re.I),
    "regulatory": re.compile(
        r"\b(sebi|takeover regulation|insider|penalt|fine|probe|"
        r"investigation|show cause|order under)\b", re.I),
}

# Routine filings that carry no directional signal — suppressed from the "material"
# count so trading-window / compliance noise doesn't masquerade as a catalyst.
_NOISE = re.compile(
    r"\b(trading window|newspaper|advertis|loss of share|duplicate share|"
    r"investor (meet|presentation) schedul|analyst.{0,15}call|"
    r"compliance certificate|reg\.?\s?74|depositor)\b", re.I)


def _ann_dt(raw: str) -> datetime | None:
    """Parse NSE's '17-Jun-2026 12:26:28' announcement timestamp (treated as UTC-ish)."""
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _classify(text: str) -> str | None:
    """Return the catalyst flag for an announcement, or None if non-material/noise."""
    if not text or _NOISE.search(text):
        return None
    for flag, pat in _CATALYST_PATTERNS.items():
        if pat.search(text):
            return flag
    return None


def _fetch_announcements(symbol: str, hours: int) -> list[dict] | None:
    """Recent NSE corporate announcements for a bare symbol, or None if unreachable."""
    import httpx

    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    headers = {
        "User-Agent": _UA,
        "Accept": "application/json",
        "Referer": f"{_NSE_HOME}/companies-listing/corporate-filings-announcements",
    }
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as c:
            c.get(_NSE_HOME)  # prime cookies — NSE rejects cold API hits with 401/403
            resp = c.get(_ANN_URL, params={"index": "equities", "symbol": symbol})
            resp.raise_for_status()
            rows = resp.json()
    except Exception as exc:  # noqa: BLE001 - scraped endpoint can fail many ways
        log(f"catalysts: announcements failed for {symbol}: {exc}")
        return None

    out: list[dict] = []
    for r in rows if isinstance(rows, list) else []:
        subject = r.get("desc") or ""
        detail = r.get("attchmntText") or ""
        text = f"{subject} {detail}"
        dt = _ann_dt(r.get("an_dt", ""))
        if dt and dt.timestamp() < cutoff:
            continue
        flag = _classify(text)
        out.append({
            "when": r.get("an_dt", ""),
            "subject": subject,
            "summary": detail[:200],
            "flag": flag,
            "material": flag is not None,
            "url": r.get("attchmntFile", ""),
        })
    return out


def _deal_rows(data) -> list[dict]:
    """Normalise nsepython's deal return (DataFrame OR dict OR list) to list-of-dicts."""
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return [] if data.empty else data.to_dict("records")
    if isinstance(data, dict):
        data = data.get("data", [])
    return list(data or [])


def _fetch_bulk_block(symbol: str) -> dict:
    """Today's bulk/block-deal footprint via nsepython (free). Empty dict on failure."""
    out: dict = {"bulk": [], "block": []}
    try:
        from nsepython import get_bulkdeals, get_blockdeals  # type: ignore
    except Exception:  # noqa: BLE001 - nsepython missing / different version
        return out
    target = symbol.upper()
    for key, fn in (("bulk", get_bulkdeals), ("block", get_blockdeals)):
        try:
            for row in _deal_rows(fn()):
                sym = str(row.get("symbol") or row.get("BD_SYMBOL") or "").upper()
                if sym == target:
                    out[key].append(row)
        except Exception as exc:  # noqa: BLE001
            log(f"catalysts: {key} deals failed for {symbol}: {exc}")
    return out


def get_catalysts(ticker: str, hours: int | None = None) -> dict:
    """Corporate catalysts for a ticker: announcements + bulk/block deals + flags.

    Returns: {ticker, provider, announcement_count, material_count, events{...},
    recency_hours (age of newest material announcement, or None), catalyst_score
    (0..1), announcements[...], deals{bulk,block}}. Always valid, never raises.
    """
    window = hours or CONFIG.catalyst_lookback_hours
    symbol = ticker.upper().removesuffix(".NS").removesuffix(".BO")

    cached = _CACHE.get_data("catalysts", ticker)
    if cached is not None:
        return cached

    anns = _fetch_announcements(symbol, window)
    contacted = anns is not None
    anns = anns or []
    deals = _fetch_bulk_block(symbol) if contacted else {"bulk": [], "block": []}

    events = {k: False for k in _CATALYST_PATTERNS}
    newest_material: datetime | None = None
    for a in anns:
        if a["flag"]:
            events[a["flag"]] = True
            dt = _ann_dt(a["when"])
            if dt and (newest_material is None or dt > newest_material):
                newest_material = dt

    has_deal = bool(deals["bulk"] or deals["block"])
    material_count = sum(1 for a in anns if a["material"])

    # Recency: a fresh material filing decays from 1.0 (now) toward 0 over the window.
    recency_hours = None
    recency = 0.0
    if newest_material:
        age_h = max(0.0, (datetime.now(timezone.utc) - newest_material)
                    .total_seconds() / 3600)
        recency_hours = round(age_h, 1)
        recency = max(0.0, 1.0 - age_h / max(window, 1))

    # Catalyst score: presence of a material filing + its recency + deal-tape activity.
    catalyst_score = min(1.0, 0.6 * recency + 0.25 * bool(material_count)
                         + 0.15 * has_deal)

    result = {
        "ticker": ticker,
        "provider": "nse" if contacted else "none (unreachable)",
        "announcement_count": len(anns),
        "material_count": material_count,
        "events": events,
        "has_deal_activity": has_deal,
        "recency_hours": recency_hours,
        "catalyst_score": round(catalyst_score, 4),
        "announcements": anns[:10],
        "deals": deals,
    }
    if contacted:  # don't cache a transient outage for the whole day
        _CACHE.set("catalysts", ticker, result)
    return result


def merge_into_news_events(news_events: dict, catalysts: dict) -> dict:
    """Fold catalyst flags into the existing news event dict (earnings/regulatory/mna).

    Keeps the news layer's schema intact for downstream scoring while letting a
    primary-source NSE filing light up the same high-impact flags that RSS/Marketaux
    headlines do — so a catalyst nobody has written an article about yet still counts.
    """
    ev = dict(news_events or {})
    cat = catalysts.get("events", {})
    ev["earnings"] = ev.get("earnings", False) or cat.get("results", False)
    ev["regulatory"] = ev.get("regulatory", False) or cat.get("regulatory", False)
    ev["mna"] = ev.get("mna", False) or cat.get("mna", False)
    # New first-class flags the news layer never had.
    ev["order_win"] = cat.get("order_win", False)
    ev["fundraise"] = cat.get("fundraise", False)
    return ev
