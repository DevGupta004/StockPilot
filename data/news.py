"""News + sentiment layer.

Provider interface with two implementations:
  * MarketauxProvider — default; free tier ~100 req/day, covers India, returns a
    per-article sentiment already mapped to the queried entity (-1..1).
  * RSSProvider — fallback; pulls Indian market RSS feeds (Economic Times,
    Moneycontrol, Mint, Business Standard) via feedparser and scores headlines
    locally with VADER.

``get_news_sentiment`` returns a normalised dict: a list of articles (title, url,
source, published, sentiment) plus an aggregate score in -1..1 and event flags
(earnings / regulatory / M&A). Network failures degrade to the fallback, then to an
empty-but-valid result — never an exception.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from config import CONFIG
from utils.cache import DayCache
from utils.log import log

_CACHE = DayCache(CONFIG.cache_dir)

# Map common NSE tickers to the company names / search terms news APIs understand.
_TICKER_TERMS = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever", "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel", "ITC.NS": "ITC", "KOTAKBANK.NS": "Kotak Mahindra",
    "LT.NS": "Larsen Toubro", "AXISBANK.NS": "Axis Bank", "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints", "MARUTI.NS": "Maruti Suzuki", "TITAN.NS": "Titan",
    "SUNPHARMA.NS": "Sun Pharma", "TATAMOTORS.NS": "Tata Motors", "WIPRO.NS": "Wipro",
    "ADANIENT.NS": "Adani Enterprises", "TATASTEEL.NS": "Tata Steel",
    "HCLTECH.NS": "HCL Technologies", "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid",
    "ONGC.NS": "ONGC",
}

_EVENT_PATTERNS = {
    "earnings": re.compile(r"\b(earnings|results|profit|q[1-4]|quarter|revenue)\b", re.I),
    "regulatory": re.compile(r"\b(sebi|rbi|regulat|probe|penalt|fine|ban|tax)\b", re.I),
    "mna": re.compile(r"\b(merger|acqui|acquisition|stake|buyout|takeover|deal)\b", re.I),
}


def term_for(ticker: str) -> str:
    return _TICKER_TERMS.get(ticker.upper(), ticker.split(".")[0])


class NewsProvider(ABC):
    name = "base"

    @abstractmethod
    def fetch(self, ticker: str, hours: int) -> list[dict] | None:
        """Articles (title/url/source/published/sentiment).

        Return a list (possibly empty = genuinely no news) on a SUCCESSFUL contact, or
        ``None`` if the provider could not be reached / is unconfigured — so the caller
        can avoid caching a transient failure for the whole day.
        """


# --------------------------------------------------------------------------- #
# Marketaux
# --------------------------------------------------------------------------- #
class MarketauxProvider(NewsProvider):
    name = "marketaux"
    BASE = "https://api.marketaux.com/v1/news/all"

    def fetch(self, ticker: str, hours: int) -> list[dict] | None:
        if not CONFIG.marketaux_api_key:
            log("marketaux: no API key configured")
            return None
        import httpx

        symbol = ticker.split(".")[0]  # Marketaux accepts bare NSE symbols
        published_after = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).strftime("%Y-%m-%dT%H:%M")
        params = {
            "symbols": f"{symbol}.NS,{symbol}",
            "filter_entities": "true",
            "language": "en",
            "published_after": published_after,
            "limit": 10,
            "api_token": CONFIG.marketaux_api_key,
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(self.BASE, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            log(f"marketaux: fetch failed for {ticker}: {exc}")
            return None

        out: list[dict] = []
        for art in payload.get("data", []):
            # Sentiment already mapped per-entity; average matching entities.
            ent_scores = [
                e.get("sentiment_score")
                for e in art.get("entities", [])
                if e.get("sentiment_score") is not None
            ]
            sentiment = sum(ent_scores) / len(ent_scores) if ent_scores else 0.0
            out.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("source", "marketaux"),
                "published": art.get("published_at", ""),
                "sentiment": float(sentiment),
            })
        return out


# --------------------------------------------------------------------------- #
# RSS fallback + local VADER sentiment
# --------------------------------------------------------------------------- #
class RSSProvider(NewsProvider):
    name = "rss"
    FEEDS = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.livemint.com/rss/markets",
        "https://www.business-standard.com/rss/markets-106.rss",
    ]

    def fetch(self, ticker: str, hours: int) -> list[dict] | None:
        try:
            import feedparser
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            log(f"rss: missing dependency: {exc}")
            return None

        term = term_for(ticker).lower()
        short = ticker.split(".")[0].lower()
        analyzer = SentimentIntensityAnalyzer()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out: list[dict] = []
        feeds_ok = 0

        for feed_url in self.FEEDS:
            try:
                parsed = feedparser.parse(feed_url)
                if getattr(parsed, "bozo", 0) and not parsed.entries:
                    raise ValueError(getattr(parsed, "bozo_exception", "parse error"))
            except Exception as exc:  # noqa: BLE001
                log(f"rss: feed failed {feed_url}: {exc}")
                continue
            feeds_ok += 1
            for entry in parsed.entries[:50]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                blob = f"{title} {summary}".lower()
                if term not in blob and short not in blob:
                    continue
                published = _entry_dt(entry)
                if published and published < cutoff:
                    continue
                score = analyzer.polarity_scores(f"{title}. {summary}")["compound"]
                out.append({
                    "title": title,
                    "url": entry.get("link", ""),
                    "source": parsed.feed.get("title", "rss"),
                    "published": published.isoformat() if published else "",
                    "sentiment": float(score),
                })
        # If every feed errored, signal failure (None) so the day isn't cached empty.
        return out if feeds_ok else None


def _entry_dt(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        tm = entry.get(key)
        if tm:
            try:
                return datetime(*tm[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def _make_provider(name: str) -> NewsProvider:
    return RSSProvider() if name == "rss" else MarketauxProvider()


def _flag_events(articles: list[dict]) -> dict[str, bool]:
    flags = {"earnings": False, "regulatory": False, "mna": False}
    for art in articles:
        text = art.get("title", "")
        for key, pat in _EVENT_PATTERNS.items():
            if pat.search(text):
                flags[key] = True
    return flags


def get_news_sentiment(ticker: str, hours: int | None = None) -> dict:
    """Aggregate recent news + sentiment for a ticker.

    Returns: {ticker, provider, article_count, aggregate_sentiment (-1..1),
    events {earnings, regulatory, mna}, articles[...]}. Always valid, never raises.
    """
    window = hours or CONFIG.news_lookback_hours

    cached = _CACHE.get_data("news", ticker)
    if cached is not None:
        return cached

    primary = CONFIG.providers.news
    order = [primary] + [p for p in ("marketaux", "rss") if p != primary]

    articles: list[dict] = []
    used = "none"
    contacted = False  # did ANY provider answer (even with no news)?
    for prov_name in order:
        provider = _make_provider(prov_name)
        fetched = provider.fetch(ticker, window)
        if fetched is None:  # provider unreachable/unconfigured — try the next
            log(f"news: {prov_name} unavailable for {ticker}, trying fallback")
            continue
        contacted = True
        used = provider.name
        articles = fetched
        if articles:  # got real news — stop; otherwise let a later provider try too
            break
        log(f"news: {prov_name} found no articles for {ticker}, trying fallback")

    scores = [a["sentiment"] for a in articles if a.get("sentiment") is not None]
    aggregate = sum(scores) / len(scores) if scores else 0.0

    result = {
        "ticker": ticker,
        "provider": used,
        "article_count": len(articles),
        "aggregate_sentiment": round(aggregate, 4),
        "events": _flag_events(articles),
        "articles": articles[:10],
    }
    # Only cache a real result. If every provider was unreachable, return a neutral
    # result but DON'T cache it, so a transient outage doesn't poison the whole day.
    if contacted:
        _CACHE.set("news", ticker, result)
    else:
        result["provider"] = "none (all sources unavailable)"
    return result
