"""Short-Swing Stock Signal — FastMCP server.

A daily research tool that scans a configurable NSE universe and ALWAYS returns the
top 3 short-swing candidates (graded honestly against a confidence bar), combining
technical analysis with recent news sentiment. Holding window is hard-capped at 2
trading days. Educational signal only — NOT financial advice.

Run (stdio):  PYTHONUNBUFFERED=1 python server.py
All logs go to stderr; stdout is reserved for the JSON-RPC protocol.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


# NSE session hours (IST). Holidays are not modelled — only weekends are skipped, so
# treat dates near known exchange holidays as approximate.
_MKT_OPEN = "09:15"
_MKT_CLOSE = "15:30"


def _next_trading_day(d: datetime) -> datetime:
    d += timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def _add_trading_days(d: datetime, n: int) -> datetime:
    for _ in range(n):
        d = _next_trading_day(d)
    return d


def _trade_schedule(as_of: str, horizon: int) -> tuple[str, str]:
    """Concrete buy/sell schedule from the data date within the T+horizon window.

    Buy = next trading session (entry on/after open). Sell-by = close of the
    horizon-th trading day after entry (the hard time stop). Weekends skipped;
    holidays NOT modelled, so this is approximate near them.
    """
    try:
        base = datetime.strptime(as_of, "%Y-%m-%d")
    except (ValueError, TypeError):
        return ("next session open", f"close of T+{horizon}")
    buy_day = _next_trading_day(base)
    sell_day = _add_trading_days(buy_day, horizon)
    buy = f"{buy_day:%a %d %b} {_MKT_OPEN}"
    sell = f"{sell_day:%a %d %b} {_MKT_CLOSE}"
    return buy, sell

from analysis import indicators as ind
from analysis import scoring
from config import CONFIG, DISCLAIMER, named_universe
from data.market import get_ohlcv, get_spot_price
from data.news import get_news_sentiment as fetch_news_sentiment
from data.news import term_for
from utils.log import log

mcp = FastMCP("stock-signals")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _company_name(ticker: str) -> str:
    return term_for(ticker)


def _data_quality(df) -> float:
    """0..1 score from bar count + liquidity (avg traded value proxy)."""
    bars = len(df)
    bar_score = min(1.0, bars / 200.0)
    try:
        avg_val = float((df["Close"] * df["Volume"]).tail(20).mean())
    except Exception:  # noqa: BLE001
        avg_val = 0.0
    # ₹50 crore/day avg turnover ~ very liquid large-cap -> saturate.
    liq_score = min(1.0, avg_val / 5.0e8)
    return round(0.5 * bar_score + 0.5 * liq_score, 3)


def _analyze_one(ticker: str, min_confidence: float) -> scoring.Candidate | None:
    df = get_ohlcv(ticker)
    if df is None:
        log(f"pipeline: skipping {ticker} (no data)")
        return None
    try:
        indicators = ind.compute(df)
    except Exception as exc:  # noqa: BLE001
        log(f"pipeline: indicator error for {ticker}: {exc}")
        return None
    news = fetch_news_sentiment(ticker)
    dq = _data_quality(df)
    cand = scoring.score_symbol(
        ticker, _company_name(ticker), indicators, news, dq, min_confidence
    )
    cand.as_of = str(df.index[-1].date())  # date of the last valid price bar
    cand.provider = df.attrs.get("provider", "yfinance/cache")
    spot = get_spot_price(ticker)  # near-live (delayed) quote, best-effort
    if spot:
        cand.live_price = spot["price"]
        cand.live_as_of = spot["as_of"]
        cand.live_kind = spot["kind"]
    cand.buy_by, cand.sell_by = _trade_schedule(cand.as_of, cand.time_stop_days)
    return cand


def _table(picks: list[scoring.Candidate]) -> str:
    """Compact, scannable view. Near-live price + last close + levels you act on."""
    rows = [
        "| # | Stock | Signal | Now | Buy by | Buy | Target | Stop | Sell by | Qty |",
        "|---|-------|--------|-----|--------|-----|--------|------|---------|-----|",
    ]
    for c in picks:
        flag = "✅" if c.label == scoring.ACTIONABLE else "⚠️"
        short = " 🚫CNC" if c.direction == "SHORT" else ""
        signal = f"{flag} {c.confidence_pct}%{short}"
        qty = (c.position or {}).get("shares", "-")
        now = f"₹{c.live_price}" if c.live_price is not None else "—"
        rows.append(
            f"| {c.rank} | {c.ticker.replace('.NS', '')} | {signal} "
            f"| {now} | {c.buy_by} | ₹{c.entry['level']} | ₹{c.target} "
            f"| ₹{c.stop_loss} | {c.sell_by} | {qty} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_daily_picks(universe: str | None = None,
                    min_confidence: float | None = None,
                    max_price: float | None = None,
                    capital: float | None = None,
                    risk_per_trade: float | None = None,
                    delivery_only: bool | None = None) -> dict:
    """Daily top-3 DELIVERY (CNC) trade candidates for the Indian (NSE) market.

    Run once a day. ALWAYS returns the top 3 picks (or fewer only if data fetches
    failed), ranked by a blended technical + news-sentiment score and graded honestly:
    each is labelled ACTIONABLE (confidence >= min_confidence) or
    "LOW CONFIDENCE - NOT RECOMMENDED". The header states how many cleared the bar.
    Holding window is hard-capped at 2 trading days, with concrete Buy-by / Sell-by
    dates per pick.

    In delivery mode (the default), only LONG candidates are returned — short selling
    cannot be held as delivery (CNC) on NSE, so shorts are excluded entirely rather
    than shown as un-actionable. This is a research/idea generator, NOT a predictor:
    it cannot tell you the "correct" stock or guarantee a move; it ranks the odds and
    grades them honestly, and sizes each position for your capital.

    Args:
        universe: Optional. A preset name ("nifty"/"default", or "under500"/"cheap"
            for liquid sub-₹500 names) or an inline comma-separated list of NSE symbols
            (e.g. "RELIANCE.NS,TCS.NS"). Defaults to the configured watchlist.
        min_confidence: Confidence threshold (0..1) separating ACTIONABLE picks from
            low-confidence ideas. Omit to use MIN_CONFIDENCE from .env (default 0.55).
        max_price: Optional. Only consider stocks whose latest price is at or below
            this rupee value (e.g. 500.0). Omit to use MAX_PRICE from .env.
        capital: Trading capital in rupees for position sizing. Omit to use CAPITAL
            from .env (default 150000 / ₹1.5L).
        risk_per_trade: Fraction of capital risked to the stop per trade. Omit to use
            RISK_PER_TRADE from .env (default 0.02 / 2%).
        delivery_only: If True (default), exclude SHORT candidates so every pick is a
            holdable LONG delivery trade. Set False to also surface (flagged) shorts.
            Omit to use DELIVERY_ONLY from .env (default True).

    When called with no arguments, the trading profile in .env (STOCK_UNIVERSE,
    MAX_PRICE, CAPITAL, RISK_PER_TRADE, MIN_CONFIDENCE, DELIVERY_ONLY) is applied.

    Returns:
        A dict with: header (summary line), actionable_count, picks (graded candidates
        with entry/target/stop/time-stop/Buy-by/Sell-by/rationale AND a position
        {shares, deploy, risk_amount, capital_pct}), table (markdown view), and a
        mandatory disclaimer.
    """
    # Fall back to env-backed CONFIG defaults when an argument is omitted, so the
    # trading profile in .env (capital, max_price, universe, bar) applies to a bare
    # "run my daily picks".
    min_confidence = (CONFIG.thresholds.min_confidence
                      if min_confidence is None else min_confidence)
    capital = CONFIG.capital if capital is None else capital
    risk_per_trade = (CONFIG.risk_per_trade
                      if risk_per_trade is None else risk_per_trade)
    cap = max_price if max_price is not None else CONFIG.max_price
    deliv = CONFIG.delivery_only if delivery_only is None else delivery_only

    symbols = named_universe(universe)
    log(f"get_daily_picks: scanning {len(symbols)} symbols, bar={min_confidence}, "
        f"max_price={cap}, capital={capital}, delivery_only={deliv}")

    cands: list[scoring.Candidate] = []
    failed: list[str] = []
    over_price: list[str] = []
    shorts_skipped: list[str] = []
    for sym in symbols:
        c = _analyze_one(sym, min_confidence)
        if c is None:
            failed.append(sym)
            continue
        if cap is not None and c.price > cap:
            over_price.append(sym)
            continue
        if deliv and c.direction == "SHORT":
            # Delivery (CNC) can't hold shorts — exclude from the pool entirely.
            shorts_skipped.append(sym)
            continue
        cands.append(c)

    if not cands:
        reason = "every data fetch failed" if not over_price else \
            f"no stock under ₹{cap:.0f} qualified"
        return {
            "header": f"0 candidates — {reason} today. "
                      f"Try a different universe / raise max_price and retry.",
            "actionable_count": 0,
            "picks": [],
            "failed_symbols": failed,
            "over_price_symbols": over_price,
            "disclaimer": DISCLAIMER,
        }

    picks = scoring.rank_and_grade(cands, min_confidence)

    # Attach delivery position sizing to each pick.
    for c in picks:
        c.position = scoring.size_position(
            c.entry["level"], c.stop_loss, c.direction, capital, risk_per_trade
        )
        if c.direction == "SHORT":
            # Short selling is intraday-only on NSE — cannot be held as delivery/CNC.
            c.position["delivery_note"] = (
                "SHORT not allowed as delivery (CNC) — NSE permits short selling "
                "intraday (MIS) only. Not executable as a 2-day delivery trade."
            )

    actionable = sum(1 for c in picks if c.label == scoring.ACTIONABLE)

    if actionable == 0:
        note = (f"0 of {len(picks)} cleared the bar — today is choppy, treat all "
                f"three as ideas only, NOT trades.")
    else:
        note = (f"{actionable} of {len(picks)} actionable today "
                f"(min_confidence {min_confidence:.2f}).")
    if deliv:
        note += " Delivery (CNC) mode: LONG-only."
    if cap is not None:
        note += f" Filtered to stocks ≤ ₹{cap:.0f}."
    if failed:
        note += f" {len(failed)} symbol(s) skipped (no data): {', '.join(failed[:5])}."
    if over_price:
        note += f" {len(over_price)} above price cap."
    if shorts_skipped:
        note += f" {len(shorts_skipped)} short setup(s) excluded (not delivery-able)."
    if len(picks) < 3:
        note += f" Only {len(picks)} valid candidate(s) available."

    # Data freshness: yfinance free daily bars can lag 1-2 sessions (the latest bar
    # may be NaN and is dropped). Surface the newest bar date so prices are never
    # mistaken for live/LTP — verify against your broker before acting.
    as_of_dates = sorted({c.as_of for c in picks if c.as_of})
    newest = as_of_dates[-1] if as_of_dates else "unknown"
    has_live = any(c.live_price is not None for c in picks)
    live_bit = (
        "'Now' = near-live DELAYED quote (yfinance, ~15 min lag). "
        if has_live else "Live quote unavailable; "
    )
    freshness = (
        f"Generated {_now_ist()}. {live_bit}"
        f"'Close' = last DAILY close (newest {newest}, source "
        f"{picks[0].provider or 'yfinance'}). Neither is an official tick — "
        f"confirm the live price in your broker before entry."
    )

    return {
        "header": note,
        "generated_at": _now_ist(),
        "data_freshness": freshness,
        "actionable_count": actionable,
        "horizon_days": CONFIG.horizon_days,
        "capital": capital,
        "risk_per_trade": risk_per_trade,
        "max_price": cap,
        "delivery_only": deliv,
        "picks": [c.to_dict() for c in picks],
        "table": _table(picks),
        "failed_symbols": failed,
        "over_price_symbols": over_price,
        "shorts_excluded": shorts_skipped,
        "disclaimer": DISCLAIMER,
    }


@mcp.tool()
def analyze_stock(ticker: str) -> dict:
    """Full technical + news breakdown and a graded short-swing verdict for one symbol.

    Args:
        ticker: NSE symbol with the .NS suffix (e.g. "RELIANCE.NS").

    Returns:
        A dict with the candidate verdict (direction, confidence, label, entry/target/
        stop/time-stop, rationale, drivers), the raw indicator snapshot, the news
        sentiment summary, and the disclaimer. Never raises — returns an error field
        if data could not be fetched.
    """
    ticker = ticker.upper().strip()
    df = get_ohlcv(ticker)
    if df is None:
        return {"ticker": ticker, "error": "no market data available",
                "disclaimer": DISCLAIMER}
    indicators = ind.compute(df)
    news = fetch_news_sentiment(ticker)
    dq = _data_quality(df)
    cand = scoring.score_symbol(
        ticker, _company_name(ticker), indicators, news, dq,
        CONFIG.thresholds.min_confidence
    )
    cand.rank = 1
    return {
        "ticker": ticker,
        "verdict": cand.to_dict(),
        "indicators": indicators,
        "news": news,
        "data_quality": dq,
        "disclaimer": DISCLAIMER,
    }


@mcp.tool()
def get_technicals(ticker: str, lookback_days: int = 180) -> dict:
    """Technical indicators + derived signals for one symbol.

    Args:
        ticker: NSE symbol with the .NS suffix (e.g. "TCS.NS").
        lookback_days: Calendar days of daily history to pull (default 180).

    Returns:
        A dict with the latest indicator snapshot (EMA 9/21/50, MACD, RSI-14,
        Stochastic, ATR, Bollinger Bands, volume/OBV) plus the bullish/bearish signal
        list and inferred direction. Returns an error field if data is unavailable.
    """
    ticker = ticker.upper().strip()
    df = get_ohlcv(ticker, lookback_days=lookback_days)
    if df is None:
        return {"ticker": ticker, "error": "no market data available",
                "disclaimer": DISCLAIMER}
    indicators = ind.compute(df)
    tech_score, drivers, direction = scoring._technical_signals(indicators)
    return {
        "ticker": ticker,
        "provider": df.attrs.get("provider", "unknown"),
        "indicators": indicators,
        "technical_score": round(tech_score, 4),
        "direction": direction,
        "signals": drivers,
        "disclaimer": DISCLAIMER,
    }


@mcp.tool()
def get_news_sentiment(ticker: str) -> dict:
    """Recent headlines with per-article sentiment and an aggregate score.

    Args:
        ticker: NSE symbol with the .NS suffix (e.g. "INFY.NS").

    Returns:
        A dict with the provider used, article_count, aggregate_sentiment (-1..1),
        event flags (earnings/regulatory/M&A), and up to 10 articles (title, url,
        source, published, sentiment). Always valid even if no news was found.
    """
    ticker = ticker.upper().strip()
    result = fetch_news_sentiment(ticker)
    result["disclaimer"] = DISCLAIMER
    return result


@mcp.tool()
def backtest(ticker: str, days: int = 90) -> dict:
    """Quick historical sanity-check of the entry/exit logic over recent history.

    Walks the last ``days`` of bars: at each bar where the long-bias technical stack
    fires, simulate entry at the next open and exit at the first of target / stop /
    T+2 close. Reports hit-rate and average return. This is a crude sanity check, NOT
    a validated strategy backtest.

    Args:
        ticker: NSE symbol with the .NS suffix.
        days: Number of recent trading days to test over (default 90).

    Returns:
        A dict with trades simulated, win_rate, avg_return_pct, and per-trade results.
    """
    ticker = ticker.upper().strip()
    df = get_ohlcv(ticker, lookback_days=max(days + 80, 180))
    if df is None or len(df) < days + 30:
        return {"ticker": ticker, "error": "insufficient data for backtest",
                "disclaimer": DISCLAIMER}

    horizon = min(CONFIG.horizon_days, 2)
    th = CONFIG.thresholds
    trades: list[dict] = []
    window = df.iloc[-(days + 30):]

    for i in range(30, len(window) - horizon - 1):
        sub = window.iloc[: i + 1]
        try:
            t = ind.compute(sub)
        except Exception:  # noqa: BLE001
            continue
        score, _, direction = scoring._technical_signals(t)
        if direction != "LONG" or score < 0.4:
            continue

        entry_price = float(window["Open"].iloc[i + 1])
        atr = t["atr14"] or entry_price * 0.02
        target = entry_price + th.atr_target_mult * atr
        stop = entry_price - th.atr_stop_mult * atr

        exit_price, reason = None, "time"
        for d in range(1, horizon + 1):
            bar = window.iloc[i + 1 + d] if (i + 1 + d) < len(window) else None
            if bar is None:
                break
            if float(bar["High"]) >= target:
                exit_price, reason = target, "target"
                break
            if float(bar["Low"]) <= stop:
                exit_price, reason = stop, "stop"
                break
            exit_price = float(bar["Close"])
        if exit_price is None:
            continue
        ret = (exit_price - entry_price) / entry_price * 100.0
        trades.append({
            "date": str(window.index[i + 1].date()),
            "entry": round(entry_price, 2),
            "exit": round(exit_price, 2),
            "reason": reason,
            "return_pct": round(ret, 2),
        })

    if not trades:
        return {"ticker": ticker, "trades": 0,
                "note": "no long signals fired in the window",
                "disclaimer": DISCLAIMER}

    wins = [t for t in trades if t["return_pct"] > 0]
    avg = sum(t["return_pct"] for t in trades) / len(trades)
    return {
        "ticker": ticker,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "avg_return_pct": round(avg, 3),
        "horizon_days": horizon,
        "results": trades[-20:],
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    log("starting stock-signals MCP server (stdio)")
    log(f"providers: market={CONFIG.providers.market} news={CONFIG.providers.news} "
        f"| horizon={CONFIG.horizon_days}d | universe size="
        f"{len(named_universe(None))}")
    try:
        mcp.run()
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: server crashed: {exc}")
        sys.exit(1)
