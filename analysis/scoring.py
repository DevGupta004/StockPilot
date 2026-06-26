"""Scoring, entry/exit derivation, confidence, grading, and the 2-day rule.

Pipeline per symbol:
  technical signals -> normalised technical score (0..1, long-bias)
  news aggregate    -> normalised sentiment score (0..1)
  blended score     -> CONFIG.weights
  entry / target / stop / time-stop derived within HORIZON_DAYS
  confidence        -> from signal agreement + sentiment alignment + data quality
  grade             -> ACTIONABLE vs LOW CONFIDENCE vs min_confidence

The holding window is enforced structurally: ``time_stop_days`` is clamped to
CONFIG.horizon_days and the time-stop text is generated from it, so a candidate can
never carry a horizon longer than 2 trading days.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from config import CONFIG


def _pos(value: float, fallback: float) -> float:
    """Return a finite, positive float or ``fallback`` (guards entry/stop math)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) and v > 0 else fallback

ACTIONABLE = "ACTIONABLE"
LOW_CONF = "LOW CONFIDENCE - NOT RECOMMENDED"


@dataclass
class Candidate:
    rank: int
    ticker: str
    company: str
    label: str            # ACTIONABLE / LOW CONFIDENCE - NOT RECOMMENDED
    direction: str        # LONG (default) / SHORT
    confidence: float     # 0..1
    confidence_pct: int
    score: float          # blended 0..1, used purely for ranking
    technical_score: float
    sentiment_score: float
    price: float
    entry: dict           # {level, condition, when}
    target: float
    stop_loss: float
    time_stop_days: int
    time_stop: str        # human text, <= day+2
    rationale: str
    drivers: list[str]
    news_events: dict
    data_quality: float
    as_of: str = ""       # date of the price bar this is based on (data freshness)
    provider: str = ""    # which data provider supplied the bars
    live_price: float | None = None   # near-live (delayed) spot price, if available
    live_as_of: str = ""              # timestamp of the live price
    live_kind: str = ""               # "delayed-quote" / "intraday-1m"
    buy_by: str = ""                  # planned entry session (date + open)
    sell_by: str = ""                 # hard sell deadline (date + close, T+horizon)
    volume: dict | None = None        # 7-day volume-surge block (scan_volume_spikes)
    oversold: dict | None = None      # RSI/Stoch/BB oversold block (scan_oversold)
    catalysts: dict | None = None     # NSE corporate-catalyst block (filled post-rank)
    # Position sizing (filled after ranking; zeros until then).
    position: dict | None = None  # {shares, deploy, risk_amount, capital_pct, note}
    # Chart-pattern enrichment (filled post-rank for chart-enabled tools only).
    chart: dict | None = None         # {sparkline_30d, ascii, window_bars}
    patterns: dict | None = None      # detect() feature block + pattern_score
    final_score: float | None = None  # blended confidence + pattern_score (best-of-N)
    best: bool = False                # winner of the best-of-N chart+signal blend
    best_reason: str = ""             # why this one was picked as best

    def to_dict(self) -> dict:
        return asdict(self)


def size_position(entry_price: float, stop: float, direction: str,
                  capital: float, risk_pct: float) -> dict:
    """Risk-based delivery (CNC) position size.

    Shares = (capital * risk_pct) / per-share risk, then capped so the deployed
    amount never exceeds available capital. Returns the suggested share count, the
    rupees deployed, the rupees actually at risk to the stop, and the % of capital used.
    """
    per_share_risk = abs(entry_price - stop)
    if per_share_risk <= 0 or entry_price <= 0:
        return {"shares": 0, "deploy": 0.0, "risk_amount": 0.0,
                "capital_pct": 0.0, "note": "could not size (bad levels)"}

    risk_budget = capital * risk_pct
    shares = int(risk_budget // per_share_risk)
    note = f"risk-based ({risk_pct*100:.0f}% of ₹{capital:,.0f})"

    # Cap by available capital (can't deploy more than you have).
    max_affordable = int(capital // entry_price)
    if shares > max_affordable:
        shares = max_affordable
        note = "capped by available capital"
    if shares < 1 and max_affordable >= 1:
        shares = 1
        note = "min 1 share (stop wider than risk budget — size down or skip)"

    deploy = round(shares * entry_price, 2)
    risk_amount = round(shares * per_share_risk, 2)
    return {
        "shares": shares,
        "deploy": deploy,
        "risk_amount": risk_amount,
        "capital_pct": round(deploy / capital * 100, 1) if capital else 0.0,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Technical scoring
# --------------------------------------------------------------------------- #
def _technical_signals(t: dict) -> tuple[float, list[str], str]:
    """Return (score 0..1, list of contributing drivers, direction).

    Long-bias: we count bullish confirmations. A strongly bearish stack flips the
    direction to SHORT and scores the bearish confirmations instead (flagged for
    Indian intraday/short-sell constraints downstream).
    """
    bull: list[str] = []
    bear: list[str] = []
    price = t["price"]

    # Trend — EMA stack.
    if price > t["ema21"] > t["ema50"]:
        bull.append("price above rising EMA21/EMA50 stack")
    elif price < t["ema21"] < t["ema50"]:
        bear.append("price below falling EMA21/EMA50 stack")
    if t["ema_cross_up"]:
        bull.append("EMA9 crossed above EMA21")

    # MACD.
    if t["macd_hist"] > 0:
        bull.append("MACD histogram positive")
    else:
        bear.append("MACD histogram negative")
    if t["macd_cross_up"]:
        bull.append("MACD bullish crossover")

    # RSI momentum (avoid overbought blow-offs).
    rsi = t["rsi14"]
    if 50 <= rsi <= 68:
        bull.append(f"RSI constructive ({rsi:.0f})")
    elif rsi < 35:
        bear.append(f"RSI weak ({rsi:.0f})")
    elif rsi > 72:
        bear.append(f"RSI overbought ({rsi:.0f})")

    # Stochastic.
    if t["stoch_k"] > t["stoch_d"] and t["stoch_k"] < 80:
        bull.append("Stochastic turning up")
    elif t["stoch_k"] < t["stoch_d"] and t["stoch_k"] > 20:
        bear.append("Stochastic turning down")

    # Volume confirmation.
    if t["vol_spike_ratio"] >= CONFIG.thresholds.volume_spike:
        (bull if t["macd_hist"] > 0 else bear).append(
            f"volume spike {t['vol_spike_ratio']:.1f}x avg"
        )
    if t["obv_trend_up"]:
        bull.append("OBV rising")

    # Bollinger context — near lower band in an uptrend = pullback entry.
    if price <= t["bb_lower"] * 1.01 and price > t["ema50"]:
        bull.append("pullback to lower Bollinger band in uptrend")

    # Dry-volume penalty: a breakout/trend on volume well below its 20-day average has
    # no conviction behind it (the engine kept picking such names, e.g. KALYANKJIL at
    # 0.08x). Flag it and shave the technical score for the LONG case.
    vr = t.get("vol_spike_ratio", 1.0)
    dry_volume = vr < CONFIG.thresholds.dry_volume_ratio

    n_bull, n_bear = len(bull), len(bear)
    if n_bull >= n_bear:
        # Scale: ~8 possible bullish confirmations -> saturate near 1.
        score = min(1.0, n_bull / 8.0)
        if dry_volume:
            score *= 0.85
            bull = bull + [f"⚠ volume drying ({vr:.2f}x avg) — weak conviction"]
        return round(score, 4), bull, "LONG"
    score = min(1.0, n_bear / 8.0)
    return score, bear, "SHORT"


def _sentiment_score(news: dict) -> tuple[float, list[str]]:
    """Map aggregate sentiment (-1..1) to 0..1 and surface event drivers."""
    agg = news.get("aggregate_sentiment", 0.0)
    score = (agg + 1.0) / 2.0  # -1..1 -> 0..1
    drivers: list[str] = []
    count = news.get("article_count", 0)
    if count:
        tone = "positive" if agg > 0.1 else "negative" if agg < -0.1 else "neutral"
        drivers.append(f"{count} recent articles, {tone} tone ({agg:+.2f})")
    else:
        drivers.append("no recent news (sentiment neutral)")
    events = news.get("events", {})
    for key, label in (("earnings", "earnings event"),
                       ("regulatory", "regulatory news"),
                       ("mna", "M&A chatter")):
        if events.get(key):
            drivers.append(f"high-impact: {label}")
    return score, drivers


# --------------------------------------------------------------------------- #
# Entry / exit within the 2-day horizon
# --------------------------------------------------------------------------- #
def _entry_exit(t: dict, direction: str,
                horizon: int | None = None) -> tuple[dict, float, float, int, str]:
    price = _pos(t["price"], 0.0)
    atr = _pos(t.get("atr14"), price * 0.02)
    th = CONFIG.thresholds
    # Structurally clamp to <= 2 trading days no matter what the caller asks for.
    horizon = min(horizon or CONFIG.horizon_days, 2)

    if direction == "LONG":
        ema21 = t["ema21"]
        # Deepest dip we may ask for in the holding window. EMA21 is frequently
        # 5-8% below a trending name — unreachable in 1-2 days — so clamp the buy
        # level to a shallow, fillable dip (≤ max_pullback_atr × ATR below spot).
        dip_floor = price - th.max_pullback_atr * atr
        if price > ema21 and ema21 >= dip_floor:
            # EMA21 sits within a reachable dip — use it.
            level = round(ema21, 2)
            cond = (f"buy on pullback toward EMA21 ≈ ₹{ema21:.2f} "
                    f"(else buy at open on strength)")
            when = "next session, intraday"
        elif price > ema21:
            # EMA21 too far below — ask only for a shallow dip, fallback to open.
            level = round(dip_floor, 2)
            cond = (f"buy on dip to ₹{dip_floor:.2f} "
                    f"(~{th.max_pullback_atr:g}×ATR); else buy at open on strength")
            when = "next session, intraday"
        else:
            # Price at/below EMA21 already — buy at/near market on strength.
            level = round(price, 2)
            cond = f"buy at/near market ₹{price:.2f} on strength"
            when = "next session open"
        entry = {"level": level, "condition": cond, "when": when}
        base = level
        target = max(t["recent_high20"], base + th.atr_target_mult * atr)
        stop = base - th.atr_stop_mult * atr
    else:  # SHORT
        ema21 = t["ema21"]
        entry = {
            "level": round(price, 2),
            "condition": f"sell/short near market ₹{price:.2f} on weakness",
            "when": "next session open",
        }
        base = price
        target = min(t["recent_low20"], base - th.atr_target_mult * atr)
        stop = base + th.atr_stop_mult * atr

    time_stop = (
        f"exit by close of day+{horizon} (T+{horizon}) no matter what — "
        f"short-swing window is hard-capped at {horizon} trading days"
    )
    return entry, round(target, 2), round(stop, 2), horizon, time_stop


# --------------------------------------------------------------------------- #
# Confidence + grading
# --------------------------------------------------------------------------- #
def _confidence(tech_score: float, sent_score: float, news: dict,
                data_quality: float, direction: str) -> float:
    """Confidence from signal agreement, sentiment alignment, and data quality."""
    # Agreement: technical and sentiment pointing the same way boosts confidence.
    sent_centered = (sent_score - 0.5) * 2.0  # back to -1..1
    if direction == "SHORT":
        sent_centered = -sent_centered
    alignment = 0.5 + 0.5 * max(0.0, sent_centered)  # 0.5..1.0

    base = 0.6 * tech_score + 0.25 * alignment + 0.15 * data_quality

    # Penalise thin / missing news a little (less corroboration).
    if news.get("article_count", 0) == 0:
        base *= 0.92
    return max(0.0, min(1.0, base))


def grade(confidence: float, min_confidence: float) -> str:
    return ACTIONABLE if confidence >= min_confidence else LOW_CONF


# --------------------------------------------------------------------------- #
# Public: score one symbol
# --------------------------------------------------------------------------- #
def score_symbol(ticker: str, company: str, indicators: dict, news: dict,
                 data_quality: float, min_confidence: float,
                 horizon: int | None = None) -> Candidate:
    tech_score, tech_drivers, direction = _technical_signals(indicators)
    sent_score, sent_drivers = _sentiment_score(news)

    w = CONFIG.weights
    # For SHORT, technical score already reflects bearish strength; sentiment is
    # inverted inside _confidence. Blended score is used only for ranking.
    blended = w.technical * tech_score + w.sentiment * sent_score
    confidence = _confidence(tech_score, sent_score, news, data_quality, direction)

    entry, target, stop, horizon, time_stop = _entry_exit(
        indicators, direction, horizon)

    drivers = tech_drivers[:3] + sent_drivers[:2]
    key_tech = tech_drivers[0] if tech_drivers else "mixed technicals"
    key_news = sent_drivers[0] if sent_drivers else "no news"
    rationale = (
        f"{direction} bias: {key_tech}. News: {key_news}. "
        f"Target ≈ ₹{target}, stop ₹{stop}, exit by T+{horizon}."
    )

    return Candidate(
        rank=0,
        ticker=ticker,
        company=company,
        label=grade(confidence, min_confidence),
        direction=direction,
        confidence=round(confidence, 4),
        confidence_pct=round(confidence * 100),
        score=round(blended, 4),
        technical_score=round(tech_score, 4),
        sentiment_score=round(sent_score, 4),
        price=indicators["price"],
        entry=entry,
        target=target,
        stop_loss=stop,
        time_stop_days=horizon,
        time_stop=time_stop,
        rationale=rationale,
        drivers=drivers,
        news_events=news.get("events", {}),
        data_quality=round(data_quality, 3),
    )


def rank_and_grade(cands: list[Candidate], min_confidence: float,
                   top_n: int = 3) -> list[Candidate]:
    """Sort by score desc, take top_n, assign ranks, (re)grade against the bar."""
    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[:top_n]
    for i, c in enumerate(top, start=1):
        c.rank = i
        c.label = grade(c.confidence, min_confidence)
    return top
