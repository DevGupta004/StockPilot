"""Short-Swing Stock Signal — FastMCP server.

A daily research tool that scans a configurable NSE universe and ALWAYS returns the
top 3 short-swing candidates (graded honestly against a confidence bar), combining
technical analysis with recent news sentiment. Holding window is hard-capped at 2
trading days. Educational signal only — NOT financial advice.

Run (stdio):  PYTHONUNBUFFERED=1 python server.py
All logs go to stderr; stdout is reserved for the JSON-RPC protocol.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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


def _market_clock() -> dict:
    """Current NSE session status from the live IST clock.

    session: "pre-open" (09:00-09:15), "open" (09:15-15:30), "closed" (other weekday
    times), "weekend" (Sat/Sun). NSE holidays are NOT modelled — weekends only — so
    status near an exchange holiday is approximate.
    """
    now = datetime.now(IST)
    weekend = now.weekday() >= 5
    hm = now.strftime("%H:%M")
    if weekend:
        session = "weekend"
    elif _MKT_OPEN <= hm <= _MKT_CLOSE:
        session = "open"
    elif "09:00" <= hm < _MKT_OPEN:
        session = "pre-open"
    else:
        session = "closed"
    is_trading_now = session in ("open", "pre-open")
    # Next tradable session: today if a weekday and we're still before the close,
    # otherwise the next weekday.
    if not weekend and hm <= _MKT_CLOSE:
        next_open = now
    else:
        next_open = _next_trading_day(now)
    return {
        "now_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "weekday": now.strftime("%A"),
        "session": session,
        "is_trading_now": is_trading_now,
        "next_session_open": next_open,
        "status_line": (
            f"Market {session.upper()} ({now:%a %d %b %H:%M} IST)"
            + ("" if is_trading_now else
               f" — next session {next_open:%a %d %b} {_MKT_OPEN}")
        ),
    }


def _trade_schedule(horizon: int, clock: dict,
                    entry_today: bool = False) -> tuple[str, str]:
    """Concrete buy/sell schedule anchored to the CURRENT clock (not the data bar).

    Buy = this session if we can still enter today (entry_today and market pre-open/
    open), else the next trading session. Sell-by = close of the horizon-th trading day
    after the buy day (the hard time stop). Weekends skipped; holidays NOT modelled.
    """
    if entry_today and clock.get("is_trading_now"):
        buy_day = datetime.strptime(clock["now_ist"][:10], "%Y-%m-%d")
        when = "today, this session"
    else:
        buy_day = clock["next_session_open"]
        when = "next session"
    sell_day = _add_trading_days(buy_day, horizon)
    buy = f"{buy_day:%a %d %b} {_MKT_OPEN} ({when})"
    sell = f"{sell_day:%a %d %b} {_MKT_CLOSE}"
    return buy, sell

from analysis import indicators as ind
from analysis import scoring
from config import CONFIG, DISCLAIMER, named_universe
from data import market
from data.market import get_ohlcv, get_spot_price, prefilter_by_price
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


def _analyze_one(ticker: str, min_confidence: float, horizon: int | None = None,
                 clock: dict | None = None,
                 entry_today: bool = False) -> scoring.Candidate | None:
    """Score one symbol. Does NOT fetch the live quote — that is done post-rank for
    the shortlist only (see _attach_live), to avoid hammering the live provider."""
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
        ticker, _company_name(ticker), indicators, news, dq, min_confidence, horizon
    )
    cand.as_of = str(df.index[-1].date())  # date of the last valid price bar
    cand.provider = df.attrs.get("provider", "yfinance/cache")
    clk = clock or _market_clock()
    cand.buy_by, cand.sell_by = _trade_schedule(
        cand.time_stop_days, clk, entry_today)
    return cand


def _attach_live(picks: list[scoring.Candidate]) -> None:
    """Fetch the near-live quote for the FINAL shortlist only (cheap, block-safe)."""
    for c in picks:
        spot = get_spot_price(c.ticker)  # best-effort; None on failure
        if spot:
            c.live_price = spot["price"]
            c.live_as_of = spot["as_of"]
            c.live_kind = spot["kind"]


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


def _render_markdown(result: dict, tool_name: str) -> str:
    """Build a self-contained markdown report from a scan result (ready to save)."""
    lines = [
        f"# {tool_name} — {result.get('generated_at', _now_ist())}",
        "",
        f"**{result.get('market_status', '')}**",
        "",
        result.get("header", ""),
        "",
        result.get("table", ""),
        "",
        "### Picks",
    ]
    for p in result.get("picks", []):
        pos = p.get("position") or {}
        lines.append(
            f"- **#{p['rank']} {p['ticker']}** ({p.get('company','')}) — "
            f"{p['direction']} {p['confidence_pct']}% [{p['label']}]"
        )
        vol = p.get("volume")
        if vol:
            lines.append(
                f"  - Volume: {vol['surge_ratio']}× surge, "
                f"{vol['price_change_7d_pct']:+.1f}% 7d, {vol['bias']}"
            )
        lines.append(
            f"  - Buy by {p['buy_by']} @ ₹{p['entry']['level']} → "
            f"target ₹{p['target']}, stop ₹{p['stop_loss']}, Sell by {p['sell_by']}"
        )
        if pos.get("shares"):
            lines.append(
                f"  - Size: {pos['shares']} sh, deploy ₹{pos.get('deploy')}, "
                f"risk ₹{pos.get('risk_amount')} ({pos.get('capital_pct')}% capital)"
            )
        lines.append(f"  - {p.get('rationale','')}")
    lines += [
        "",
        f"_{result.get('data_freshness','')}_",
        "",
        f"> {result.get('disclaimer','')}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def _freshness_line(picks: list[scoring.Candidate]) -> str:
    as_of_dates = sorted({c.as_of for c in picks if c.as_of})
    newest = as_of_dates[-1] if as_of_dates else "unknown"
    kinds = {c.live_kind for c in picks if c.live_price is not None}
    if "nse-realtime" in kinds:
        live_bit = "'Now' = NSE realtime quote (nsepython, quasi-live). "
    elif kinds:
        live_bit = "'Now' = near-live DELAYED quote (yfinance, ~15 min lag). "
    else:
        live_bit = "Live quote unavailable; "
    provider = picks[0].provider if picks else "yfinance"
    return (
        f"Generated {_now_ist()}. {live_bit}"
        f"Bars = last DAILY close (newest {newest}, source {provider}). "
        f"Not an official tick — confirm the live price in your broker before entry."
    )


def _run_scan(*, universe: str | None, min_confidence: float | None,
              max_price: float | None, capital: float | None,
              risk_per_trade: float | None, delivery_only: bool | None,
              horizon: int | None = None, entry_today: bool = False,
              tool_name: str = "daily_picks") -> dict:
    """Shared scan engine behind get_daily_picks and the predict_* tools.

    Resolves env-backed defaults, scans the universe, ranks the top 3, sizes each
    position, fetches the live quote for the shortlist only, and renders the result
    (header + market_status + table + markdown). Honest grading + disclaimer always.
    """
    min_confidence = (CONFIG.thresholds.min_confidence
                      if min_confidence is None else min_confidence)
    capital = CONFIG.capital if capital is None else capital
    risk_per_trade = (CONFIG.risk_per_trade
                      if risk_per_trade is None else risk_per_trade)
    cap = max_price if max_price is not None else CONFIG.max_price
    deliv = CONFIG.delivery_only if delivery_only is None else delivery_only
    horizon = min(horizon or CONFIG.horizon_days, 2)

    clock = _market_clock()
    symbols = named_universe(universe)
    if not symbols:
        return {
            "header": "0 candidates — could not load the stock universe (live NSE "
                      "fetch failed after retries). Check connectivity and retry.",
            "generated_at": _now_ist(),
            "market_status": clock["status_line"],
            "actionable_count": 0,
            "picks": [],
            "disclaimer": DISCLAIMER,
        }
    # Cheap batch pre-filter by price first, so the heavy per-symbol pipeline only runs
    # on affordable names (lets us scan large live universes like nifty500).
    scanned = len(symbols)
    if cap is not None and len(symbols) > 50:
        symbols = market.prefilter_by_price(symbols, cap)
    log(f"{tool_name}: scanning {len(symbols)}/{scanned} symbols, bar={min_confidence}, "
        f"max_price={cap}, capital={capital}, delivery_only={deliv}, "
        f"horizon={horizon}, session={clock['session']}")

    cands: list[scoring.Candidate] = []
    failed: list[str] = []
    over_price: list[str] = []
    shorts_skipped: list[str] = []
    for sym in symbols:
        c = _analyze_one(sym, min_confidence, horizon, clock, entry_today)
        if c is None:
            failed.append(sym)
            continue
        if cap is not None and c.price > cap:
            over_price.append(sym)
            continue
        if deliv and c.direction == "SHORT":
            shorts_skipped.append(sym)  # delivery can't hold shorts
            continue
        cands.append(c)

    if not cands:
        reason = "every data fetch failed" if not over_price else \
            f"no stock under ₹{cap:.0f} qualified"
        return {
            "header": f"0 candidates — {reason} today. "
                      f"Try a different universe / raise max_price and retry.",
            "generated_at": _now_ist(),
            "market_status": clock["status_line"],
            "actionable_count": 0,
            "picks": [],
            "failed_symbols": failed,
            "over_price_symbols": over_price,
            "disclaimer": DISCLAIMER,
        }

    picks = scoring.rank_and_grade(cands, min_confidence)
    _attach_live(picks)  # live quote for the shortlist only (block-safe)

    for c in picks:
        c.position = scoring.size_position(
            c.entry["level"], c.stop_loss, c.direction, capital, risk_per_trade
        )
        if c.direction == "SHORT":
            c.position["delivery_note"] = (
                "SHORT not allowed as delivery (CNC) — NSE permits short selling "
                "intraday (MIS) only. Not executable as a 2-day delivery trade."
            )

    actionable = sum(1 for c in picks if c.label == scoring.ACTIONABLE)
    if actionable == 0:
        note = (f"0 of {len(picks)} cleared the bar — today is choppy, treat all "
                f"three as ideas only, NOT trades.")
    else:
        note = (f"{actionable} of {len(picks)} actionable (T+{horizon}, "
                f"min_confidence {min_confidence:.2f}).")
    if deliv:
        note += " Delivery (CNC) mode: LONG-only."
    if cap is not None:
        note += f" Filtered to ≤ ₹{cap:.0f}."
    if failed:
        note += f" {len(failed)} skipped (no data): {', '.join(failed[:5])}."
    if over_price:
        note += f" {len(over_price)} above price cap."
    if shorts_skipped:
        note += f" {len(shorts_skipped)} short setup(s) excluded (not delivery-able)."
    if len(picks) < 3:
        note += f" Only {len(picks)} valid candidate(s) available."

    result = {
        "header": note,
        "generated_at": _now_ist(),
        "market_status": clock["status_line"],
        "data_freshness": _freshness_line(picks),
        "actionable_count": actionable,
        "horizon_days": horizon,
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
    result["markdown"] = _render_markdown(result, tool_name)
    return result


@mcp.tool()
def get_daily_picks(universe: str | None = None,
                    min_confidence: float | None = None,
                    max_price: float | None = None,
                    capital: float | None = None,
                    risk_per_trade: float | None = None,
                    delivery_only: bool | None = None,
                    horizon_days: int | None = None) -> dict:
    """Daily top-3 DELIVERY (CNC) trade candidates for the Indian (NSE) market.

    Run once a day. ALWAYS returns the top 3 picks (or fewer only if data fetches
    failed), ranked by a blended technical + news-sentiment score and graded honestly:
    each is labelled ACTIONABLE (confidence >= min_confidence) or
    "LOW CONFIDENCE - NOT RECOMMENDED". The header states how many cleared the bar.
    Holding window is hard-capped at 2 trading days, with concrete Buy-by / Sell-by
    dates anchored to the current NSE session (see market_status in the result).

    In delivery mode (the default), only LONG candidates are returned — short selling
    cannot be held as delivery (CNC) on NSE. This is a research/idea generator, NOT a
    predictor: it cannot tell you the "correct" stock or guarantee a move; it ranks the
    odds and grades them honestly, and sizes each position for your capital.

    Args:
        universe: Optional. A preset name ("nifty"/"default", or "under500"/"cheap"
            for liquid sub-₹500 names) or an inline comma-separated list of NSE symbols.
            Defaults to the configured watchlist.
        min_confidence: Confidence threshold (0..1). Omit to use MIN_CONFIDENCE (.env).
        max_price: Only consider stocks at/below this rupee price. Omit for MAX_PRICE.
        capital: Trading capital (₹) for sizing. Omit to use CAPITAL (.env, ₹1.5L).
        risk_per_trade: Fraction of capital risked to the stop. Omit for RISK_PER_TRADE.
        delivery_only: If True (default), exclude SHORT candidates. Omit for DELIVERY_ONLY.
        horizon_days: Holding window in trading days, 1 or 2 (clamped to 2). Default 2.

    When called with no arguments, the .env trading profile is applied.

    Returns:
        A dict with header, market_status, generated_at, actionable_count, picks
        (graded candidates with entry/target/stop/Buy-by/Sell-by/position), table,
        markdown (ready to save), and a mandatory disclaimer.
    """
    return _run_scan(
        universe=universe, min_confidence=min_confidence, max_price=max_price,
        capital=capital, risk_per_trade=risk_per_trade, delivery_only=delivery_only,
        horizon=horizon_days, tool_name="daily_picks",
    )


@mcp.tool()
def predict_delivery_2day(universe: str | None = None,
                          min_confidence: float | None = None,
                          max_price: float | None = None,
                          capital: float | None = None,
                          risk_per_trade: float | None = None) -> dict:
    """Predict the top 3 DELIVERY (CNC) trades to hold for up to 2 trading days (T+2).

    Same engine as get_daily_picks, fixed to a 2-trading-day delivery horizon and
    LONG-only. Each pick has Buy-by / Sell-by dates anchored to the current NSE session,
    a target, stop, and a position sized to your capital. Research signal, NOT a
    guaranteed prediction.

    Args mirror get_daily_picks (universe / min_confidence / max_price / capital /
    risk_per_trade); omit any to use the .env trading profile.
    """
    return _run_scan(
        universe=universe, min_confidence=min_confidence, max_price=max_price,
        capital=capital, risk_per_trade=risk_per_trade, delivery_only=True,
        horizon=2, tool_name="predict_delivery_2day",
    )


@mcp.tool()
def predict_buy_today_sell_tomorrow(universe: str | None = None,
                                    min_confidence: float | None = None,
                                    max_price: float | None = None,
                                    capital: float | None = None,
                                    risk_per_trade: float | None = None) -> dict:
    """Predict the top 3 DELIVERY trades to buy this session and exit by the next day (T+1).

    A shorter 1-trading-day delivery hold: if the market is open/pre-open now, Buy-by is
    "today, this session"; otherwise the next session. Sell-by is the close of the next
    trading day (one day earlier than the 2-day tool). LONG-only, position-sized.
    Research signal, NOT a guaranteed prediction.

    Args mirror get_daily_picks; omit any to use the .env trading profile.
    """
    return _run_scan(
        universe=universe, min_confidence=min_confidence, max_price=max_price,
        capital=capital, risk_per_trade=risk_per_trade, delivery_only=True,
        horizon=1, entry_today=True, tool_name="predict_buy_today_sell_tomorrow",
    )


def _volume_table(picks: list[scoring.Candidate]) -> str:
    """Volume-scan view: surge + bias columns alongside the usual trade levels."""
    rows = [
        "| # | Stock | Surge | Bias | Signal | Now | Buy by | Buy | Target | Stop "
        "| Sell by | Qty |",
        "|---|-------|-------|------|--------|-----|--------|-----|--------|------"
        "|---------|-----|",
    ]
    for c in picks:
        flag = "✅" if c.label == scoring.ACTIONABLE else "⚠️"
        short = " 🚫CNC" if c.direction == "SHORT" else ""
        v = c.volume or {}
        qty = (c.position or {}).get("shares", "-")
        now = f"₹{c.live_price}" if c.live_price is not None else "—"
        rows.append(
            f"| {c.rank} | {c.ticker.replace('.NS', '')} | {v.get('surge_ratio','-')}× "
            f"| {v.get('bias','-')} | {flag} {c.confidence_pct}%{short} | {now} "
            f"| {c.buy_by} | ₹{c.entry['level']} | ₹{c.target} | ₹{c.stop_loss} "
            f"| {c.sell_by} | {qty} |"
        )
    return "\n".join(rows)


@mcp.tool()
def scan_volume_spikes(universe: str | None = None, min_surge: float | None = None,
                       top_n: int = 10, max_price: float | None = None,
                       capital: float | None = None,
                       risk_per_trade: float | None = None,
                       delivery_only: bool | None = None) -> dict:
    """Find stocks with a big VOLUME increase over the last 7 days + a full trade plan.

    Scans the universe for names whose last-7-day average volume is at least `min_surge`
    times their prior ~20-day baseline (a sign of unusual interest), then runs the full
    delivery analysis on the survivors so each comes with entry/target/stop/Buy-by/
    Sell-by and a position sized to your capital. Ranked by surge strength.

    Honest note: a volume surge signals ATTENTION, not direction — it can be
    accumulation (price up) or distribution (price down). This is a research signal,
    NOT a prediction. Bias (ACCUMULATION/DISTRIBUTION/MIXED) is reported per stock.

    Args:
        universe: Preset name / inline list / default watchlist (as get_daily_picks).
        min_surge: Minimum 7-day vs baseline volume multiple to qualify. Omit to use
            MIN_SURGE (.env, default 2.0).
        top_n: Max number of movers to return (ranked by surge). Default 10.
        max_price / capital / risk_per_trade / delivery_only: as get_daily_picks; omit
            to use the .env trading profile.

    Returns:
        A dict with header, market_status, generated_at, picks (each with a `volume`
        block: surge_ratio, max_day_spike, price_change_7d_pct, bias), table, markdown,
        and a mandatory disclaimer.
    """
    min_confidence = CONFIG.thresholds.min_confidence
    min_surge = CONFIG.thresholds.min_surge if min_surge is None else min_surge
    capital = CONFIG.capital if capital is None else capital
    risk_per_trade = CONFIG.risk_per_trade if risk_per_trade is None else risk_per_trade
    cap = max_price if max_price is not None else CONFIG.max_price
    deliv = CONFIG.delivery_only if delivery_only is None else delivery_only
    clock = _market_clock()
    symbols = named_universe(universe)
    if not symbols:
        return {
            "header": "0 candidates — could not load the stock universe (live NSE "
                      "fetch failed after retries). Check connectivity and retry.",
            "generated_at": _now_ist(),
            "market_status": clock["status_line"],
            "picks": [],
            "disclaimer": DISCLAIMER,
        }
    if cap is not None and len(symbols) > 50:
        symbols = prefilter_by_price(symbols, cap)
    log(f"scan_volume_spikes: scanning {len(symbols)}, min_surge={min_surge}, "
        f"session={clock['session']}")

    surged: list[tuple[float, scoring.Candidate]] = []
    failed: list[str] = []
    for sym in symbols:
        df = get_ohlcv(sym)  # cached per-day
        if df is None:
            failed.append(sym)
            continue
        vs = ind.volume_surge(df)
        if vs is None or vs["surge_ratio"] < min_surge:
            continue
        if cap is not None and float(df["Close"].iloc[-1]) > cap:
            continue
        c = _analyze_one(sym, min_confidence, horizon=2, clock=clock)  # hits cache
        if c is None:
            failed.append(sym)
            continue
        if deliv and c.direction == "SHORT":
            continue
        c.volume = vs
        surged.append((vs["surge_ratio"], c))

    if not surged:
        return {
            "header": f"No stock showed a ≥{min_surge}× volume surge in the last 7 days "
                      f"in this universe today.",
            "generated_at": _now_ist(),
            "market_status": clock["status_line"],
            "picks": [],
            "failed_symbols": failed,
            "disclaimer": DISCLAIMER,
        }

    surged.sort(key=lambda x: x[0], reverse=True)
    picks = [c for _, c in surged[:top_n]]
    for i, c in enumerate(picks, start=1):
        c.rank = i
        c.label = scoring.grade(c.confidence, min_confidence)
    _attach_live(picks)
    for c in picks:
        c.position = scoring.size_position(
            c.entry["level"], c.stop_loss, c.direction, capital, risk_per_trade
        )

    actionable = sum(1 for c in picks if c.label == scoring.ACTIONABLE)
    note = (f"{len(picks)} stock(s) with ≥{min_surge}× volume surge in last 7d "
            f"({actionable} actionable). Surge = attention, not direction — check Bias.")
    if deliv:
        note += " Delivery (CNC) mode: LONG-only."

    result = {
        "header": note,
        "generated_at": _now_ist(),
        "market_status": clock["status_line"],
        "data_freshness": _freshness_line(picks),
        "actionable_count": actionable,
        "horizon_days": 2,
        "capital": capital,
        "min_surge": min_surge,
        "picks": [c.to_dict() for c in picks],
        "table": _volume_table(picks),
        "failed_symbols": failed,
        "disclaimer": DISCLAIMER,
    }
    result["markdown"] = _render_markdown(result, "scan_volume_spikes")
    return result


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


def _branch_name() -> str:
    """Current git branch for the report folder; 'default' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_PROJECT_DIR, capture_output=True, text=True, timeout=5,
        )
        name = out.stdout.strip()
        if name and name != "HEAD":  # HEAD = detached
            return "".join(c if c.isalnum() or c in "._-" else "-" for c in name)
    except Exception as exc:  # noqa: BLE001
        log(f"branch: could not resolve git branch: {exc}")
    return "default"


@mcp.tool()
def save_report(content: str, title: str = "") -> dict:
    """Save a tool's markdown report, organised by git branch and date.

    Writes to reports/<branch>/<YYYY-MM-DD>/<YYYY-MM-DD>.md (folders created if missing,
    branch = current git branch). Multiple runs the same day APPEND to that one daily
    file, each separated by a timestamped heading. The file is git-trackable (not
    ignored); this tool only writes it — it does not git add/commit.

    Typical flow: run a predict_/scan_ tool, then pass its `markdown` field here as
    `content` after the user confirms they want it saved.

    Args:
        content: The markdown to save (usually the `markdown` field from a scan result).
        title: Optional heading for this entry (e.g. "predict_delivery_2day 2026-06-23").

    Returns:
        {path, branch, date, appended} on success, or {error} on failure.
    """
    if not content or not content.strip():
        return {"error": "nothing to save (empty content)"}
    branch = _branch_name()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    folder = os.path.join(_PROJECT_DIR, "reports", branch, today)
    path = os.path.join(folder, f"{today}.md")
    stamp = datetime.now(IST).strftime("%H:%M:%S IST")
    heading = title.strip() or "report"
    block = f"\n\n---\n## {heading} — {stamp}\n\n{content.rstrip()}\n"
    try:
        os.makedirs(folder, exist_ok=True)
        existed = os.path.exists(path)
        with open(path, "a", encoding="utf-8") as fh:
            if not existed:
                fh.write(f"# Reports — {branch} — {today}\n")
            fh.write(block)
    except OSError as exc:
        log(f"save_report: write failed {path}: {exc}")
        return {"error": f"could not write report: {exc}"}
    rel = os.path.relpath(path, _PROJECT_DIR)
    log(f"save_report: appended to {rel}")
    return {"path": rel, "branch": branch, "date": today, "appended": True}


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
