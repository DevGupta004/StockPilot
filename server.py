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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bounded parallelism for the per-symbol scan. The cold daily fetch of a large
# universe (e.g. nifty500) is I/O-bound; a small pool cuts wall-time dramatically
# while staying gentle enough not to trip provider rate limits. Tune via env.
try:
    _SCAN_WORKERS = max(1, min(16, int(os.environ.get("STOCK_SCAN_WORKERS", "6"))))
except ValueError:
    _SCAN_WORKERS = 6

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
from analysis import patterns
from analysis import regime as regime_mod
from analysis import scoring
from analysis import tracker
from config import CONFIG, DISCLAIMER, named_universe
from data import market
from data.market import get_ohlcv, get_spot_price, prefilter_by_price
from data.news import get_news_sentiment as fetch_news_sentiment
from data.news import term_for
from data.catalysts import get_catalysts as fetch_catalysts
from data.catalysts import merge_into_news_events
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


def _analyze_many(symbols: list[str], min_confidence: float, horizon: int | None,
                  clock: dict, entry_today: bool = False
                  ) -> tuple[list[scoring.Candidate], list[str]]:
    """Analyze symbols concurrently (bounded pool). Returns (candidates, failed).

    Order of ``symbols`` is preserved in the output. Each symbol is fully isolated:
    one symbol raising never aborts the scan — it is recorded as failed and the rest
    continue. Network fetches inside ``_analyze_one`` are individually retry-wrapped.
    """
    results: list[scoring.Candidate | None] = [None] * len(symbols)
    failed: list[str] = []

    def _task(idx_sym: tuple[int, str]) -> None:
        idx, sym = idx_sym
        try:
            results[idx] = _analyze_one(sym, min_confidence, horizon, clock, entry_today)
        except Exception as exc:  # noqa: BLE001 - never let one symbol kill the scan
            log(f"pipeline: unexpected error for {sym}: {exc}")
            results[idx] = None

    workers = max(1, min(_SCAN_WORKERS, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_task, enumerate(symbols)))

    cands: list[scoring.Candidate] = []
    for sym, c in zip(symbols, results):
        if c is None:
            failed.append(sym)
        else:
            cands.append(c)
    return cands, failed


def _attach_live(picks: list[scoring.Candidate]) -> None:
    """Fetch the near-live quote for the FINAL shortlist only (cheap, block-safe)."""
    for c in picks:
        spot = get_spot_price(c.ticker)  # best-effort; None on failure
        if spot:
            c.live_price = spot["price"]
            c.live_as_of = spot["as_of"]
            c.live_kind = spot["kind"]


def _attach_catalysts(picks: list[scoring.Candidate]) -> None:
    """Enrich the FINAL shortlist with NSE corporate catalysts (free, block-safe).

    Per-name NSE calls are too heavy to run across a 500-symbol scan, so — like the
    live quote — we only pull catalysts for the handful that survive ranking. A fresh
    material filing (order win, board outcome, takeover/stake, fundraise) merges into
    the candidate's event flags and nudges confidence up; nothing is ever lowered.
    """
    for c in picks:
        cat = fetch_catalysts(c.ticker)  # best-effort; valid even if NSE unreachable
        c.catalysts = cat
        if cat.get("provider", "").startswith("none"):
            continue
        c.news_events = merge_into_news_events(c.news_events, cat)
        score = cat.get("catalyst_score", 0.0)
        if score > 0:
            # Up to +0.08 confidence for a fresh, material, primary-source catalyst.
            c.confidence = round(min(1.0, c.confidence + 0.08 * score), 4)
            c.confidence_pct = round(c.confidence * 100)
            c.label = scoring.grade(c.confidence, CONFIG.thresholds.min_confidence)
        fresh = [a["subject"] for a in cat.get("announcements", []) if a["material"]]
        if fresh:
            c.drivers = (c.drivers + [f"NSE catalyst: {fresh[0]}"])[:6]


def _attach_charts(picks: list[scoring.Candidate]) -> None:
    """Enrich the FINAL shortlist with chart structure + ASCII charts (block-safe).

    Re-pulls the (cached) daily frame per pick and runs algorithmic pattern detection
    plus ASCII rendering. Never raises — a pick with no data simply keeps chart=None.
    """
    for c in picks:
        try:
            df = get_ohlcv(c.ticker)
            if df is None or len(df) < 20:
                continue
            c.patterns = patterns.detect(df)
            c.chart = {
                "sparkline_30d": patterns.sparkline(df["Close"], 30),
                "ascii": patterns.ascii_block_chart(df["Close"], width=48, height=10),
                "window_bars": int(min(len(df), 144)),
            }
        except Exception as exc:  # noqa: BLE001 - chart is best-effort enrichment
            log(f"charts: enrichment failed for {c.ticker}: {exc}")


def _structure_penalty(feats: dict | None) -> float:
    """Multiplier (≤1) demoting clearly bearish chart structure so a strong raw signal
    on a downtrend/topping chart does not outrank a clean uptrend."""
    if not feats:
        return 1.0
    pen = 1.0
    if feats.get("trend") == "down":
        pen *= 0.85
    if feats.get("double_top"):
        pen *= 0.88
    if str(feats.get("structure", "")).startswith("lower"):
        pen *= 0.90
    return pen


def _blended_final(c: scoring.Candidate) -> float:
    """Confidence (60%) blended with pattern_score (40%), demoted by chart structure."""
    ps = (c.patterns or {}).get("pattern_score", 0.5)
    return round((0.60 * c.confidence + 0.40 * ps) * _structure_penalty(c.patterns), 4)


def _select_best(picks: list[scoring.Candidate]) -> scoring.Candidate | None:
    """Pick the best-of-N by blending confidence (60%) with pattern_score (40%),
    demoted by chart structure (downtrend/double-top). Winner flagged best=True."""
    if not picks:
        return None
    for c in picks:
        c.final_score = _blended_final(c)
    ranked = sorted(picks, key=lambda c: c.final_score or 0.0, reverse=True)
    winner = ranked[0]
    winner.best = True
    feats = winner.patterns or {}
    bits = [f"highest blended score {winner.final_score} "
            f"(confidence {winner.confidence_pct}%, "
            f"pattern {feats.get('pattern_score','-')})"]
    if feats.get("summary"):
        bits.append(f"chart: {feats['summary']}")
    winner.best_reason = "; ".join(bits)
    return winner


def _render_chart_block(p: dict) -> list[str]:
    """Per-pick ASCII chart + pattern bullets for the detailed markdown report."""
    lines: list[str] = []
    feats = p.get("patterns")
    chart = p.get("chart")
    if chart and chart.get("sparkline_30d"):
        lines.append(f"  - 30d: `{chart['sparkline_30d']}`")
    if feats:
        lines += patterns.pattern_table(feats)
    if chart and chart.get("ascii"):
        lines.append("")
        lines.append("```")
        lines.append(chart["ascii"])
        lines.append("```")
    return lines


def _render_markdown_detailed(result: dict, tool_name: str) -> str:
    """Rich report: the standard summary PLUS per-pick ASCII charts, detected
    patterns, and an explicit best-of-N verdict. Used by chart-enabled tools."""
    best = result.get("best_pick") or {}
    lines = [
        f"# {tool_name} (chart-analysed) — {result.get('generated_at', _now_ist())}",
        "",
        f"**{result.get('market_status', '')}**",
        "",
        result.get("header", ""),
        "",
    ]
    if best:
        lines += [
            f"## 🏆 Best pick: {best.get('ticker','')} "
            f"({best.get('confidence_pct','')}% · blended {best.get('final_score','')})",
            f"> {best.get('reason','')}",
            "",
        ]
    lines += [result.get("table", ""), "", "### Picks (with charts)"]
    for p in result.get("picks", []):
        flag = "🏆 " if p.get("best") else ""
        lines.append(
            f"- {flag}**#{p['rank']} {p['ticker']}** ({p.get('company','')}) — "
            f"{p['direction']} {p['confidence_pct']}% [{p['label']}]"
        )
        lines.append(
            f"  - Buy by {p['buy_by']} @ ₹{p['entry']['level']} → "
            f"target ₹{p['target']}, stop ₹{p['stop_loss']}, Sell by {p['sell_by']}"
        )
        pos = p.get("position") or {}
        if pos.get("shares"):
            lines.append(
                f"  - Size: {pos['shares']} sh, deploy ₹{pos.get('deploy')}, "
                f"risk ₹{pos.get('risk_amount')} ({pos.get('capital_pct')}% capital)"
            )
        lines += _render_chart_block(p)
        lines.append(f"  - {p.get('rationale','')}")
        lines.append("")
    lines += [
        f"_{result.get('data_freshness','')}_",
        "",
        f"> {result.get('disclaimer','')}",
    ]
    return "\n".join(lines)


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
    stale = any(str(c.provider).startswith("cache-stale") for c in picks)
    stale_bit = (" ⚠️ DATA STALE: newest bar is behind the expected session "
                 "(provider lag) — treat levels as indicative." if stale else "")
    return (
        f"Generated {_now_ist()}. {live_bit}"
        f"Bars = last DAILY close (newest {newest}, source {provider}).{stale_bit} "
        f"Not an official tick — confirm the live price in your broker before entry."
    )


def _run_scan(*, universe: str | None, min_confidence: float | None,
              max_price: float | None, capital: float | None,
              risk_per_trade: float | None, delivery_only: bool | None,
              horizon: int | None = None, entry_today: bool = False,
              tool_name: str = "daily_picks",
              enrich_charts: bool = False, auto_save: bool = False) -> dict:
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

    analyzed, failed = _analyze_many(
        symbols, min_confidence, horizon, clock, entry_today)
    cands: list[scoring.Candidate] = []
    over_price: list[str] = []
    shorts_skipped: list[str] = []
    for c in analyzed:
        if cap is not None and c.price > cap:
            over_price.append(c.ticker)
            continue
        if deliv and c.direction == "SHORT":
            shorts_skipped.append(c.ticker)  # delivery can't hold shorts
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

    if enrich_charts:
        # Pattern-aware selection: rank a WIDER shortlist by raw score, attach charts to
        # those, then pick the top 3 by the structure-demoted blend so a downtrend name
        # can't take a slot from a clean uptrend (bounded fetches — charts are cached).
        shortlist = scoring.rank_and_grade(cands, min_confidence, top_n=8)
        _attach_charts(shortlist)
        for c in shortlist:
            c.final_score = _blended_final(c)
        shortlist.sort(key=lambda c: c.final_score or 0.0, reverse=True)
        picks = shortlist[:3]
        for i, c in enumerate(picks, start=1):
            c.rank = i
    else:
        picks = scoring.rank_and_grade(cands, min_confidence)
    _attach_live(picks)        # live quote for the shortlist only (block-safe)
    _attach_catalysts(picks)   # NSE corporate catalysts for the shortlist only

    # Market regime: LONG-only delivery fights a falling tape. Shave confidence in a
    # RISK-OFF market and re-grade (picks still shown — project decision).
    regime_info = regime_mod.assess()
    if regime_info.get("regime") == regime_mod.RISK_OFF and deliv:
        factor = CONFIG.thresholds.risk_off_factor
        for c in picks:
            c.confidence = round(c.confidence * factor, 4)
            c.confidence_pct = round(c.confidence * 100)
            c.label = scoring.grade(c.confidence, min_confidence)

    for c in picks:
        c.position = scoring.size_position(
            c.entry["level"], c.stop_loss, c.direction, capital, risk_per_trade
        )
        if c.direction == "SHORT":
            c.position["delivery_note"] = (
                "SHORT not allowed as delivery (CNC) — NSE permits short selling "
                "intraday (MIS) only. Not executable as a 2-day delivery trade."
            )

    best: scoring.Candidate | None = None
    if enrich_charts:
        # Charts already attached to the shortlist above; re-blend (confidence changed
        # via catalysts/regime) and flag the winner.
        best = _select_best(picks)

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
    if regime_info.get("regime") == regime_mod.RISK_OFF:
        note = (f"⚠️ Market RISK-OFF ({regime_info.get('note','')}). LONG confidence "
                f"shaved ×{CONFIG.thresholds.risk_off_factor}. " + note)

    result = {
        "header": note,
        "regime": regime_info,
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

    if enrich_charts:
        if best is not None:
            result["best_pick"] = {
                "ticker": best.ticker,
                "company": best.company,
                "confidence_pct": best.confidence_pct,
                "final_score": best.final_score,
                "reason": best.best_reason,
            }
        result["markdown_detailed"] = _render_markdown_detailed(result, tool_name)
        if auto_save:
            today = datetime.now(IST).strftime("%Y-%m-%d")
            saved = _save_report_file(
                result["markdown_detailed"], f"{tool_name} (chart) {today}")
            result["saved_to"] = saved.get("path")
            result["save_error"] = saved.get("error")

    # Log picks to the forward-outcome ledger so accuracy can be measured later
    # (best-effort; never affects the response).
    try:
        n = tracker.log_picks(result, tool_name, _PROJECT_DIR, _branch_name())
        if n:
            result["logged_predictions"] = n
    except Exception as exc:  # noqa: BLE001
        log(f"tracker: log_picks failed: {exc}")
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

    Chart-analysed: each of the 3 picks is run through algorithmic chart-pattern
    detection (trend, support/resistance, double-bottom/top, breakout, 52w position)
    with an ASCII chart, and a best-of-3 is chosen by blending confidence (60%) with
    the bullish pattern score (40%). A detailed markdown record — including the ASCII
    charts and detected patterns — is auto-saved to reports/<branch>/<date>/. The
    result adds `best_pick`, `markdown_detailed`, and `saved_to`.

    Args mirror get_daily_picks; omit any to use the .env trading profile.
    """
    return _run_scan(
        universe=universe, min_confidence=min_confidence, max_price=max_price,
        capital=capital, risk_per_trade=risk_per_trade, delivery_only=True,
        horizon=1, entry_today=True, tool_name="predict_buy_today_sell_tomorrow",
        enrich_charts=True, auto_save=True,
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

    def _scan_one(sym: str):
        """Fetch + surge-filter + analyze one symbol. Returns (ratio, cand), 'fail', or None."""
        try:
            df = get_ohlcv(sym)  # cached per-day
            if df is None:
                return "fail"
            vs = ind.volume_surge(df)
            if vs is None or vs["surge_ratio"] < min_surge:
                return None
            if cap is not None and float(df["Close"].iloc[-1]) > cap:
                return None
            c = _analyze_one(sym, min_confidence, horizon=2, clock=clock)  # hits cache
            if c is None:
                return "fail"
            if deliv and c.direction == "SHORT":
                return None
            c.volume = vs
            return (vs["surge_ratio"], c)
        except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
            log(f"scan_volume_spikes: error for {sym}: {exc}")
            return "fail"

    workers = max(1, min(_SCAN_WORKERS, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, res in zip(symbols, pool.map(_scan_one, symbols)):
            if res == "fail":
                failed.append(sym)
            elif res is not None:
                surged.append(res)

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


def _oversold_table(picks: list[scoring.Candidate]) -> str:
    """Oversold-scan view: RSI / Stochastic / score columns alongside trade levels."""
    rows = [
        "| # | Stock | RSI | %K | <BB | Score | Signal | Now | Buy by | Buy | Target "
        "| Stop | Sell by | Qty |",
        "|---|-------|-----|-----|-----|-------|--------|-----|--------|-----|--------"
        "|------|---------|-----|",
    ]
    for c in picks:
        flag = "✅" if c.label == scoring.ACTIONABLE else "⚠️"
        short = " 🚫CNC" if c.direction == "SHORT" else ""
        o = c.oversold or {}
        qty = (c.position or {}).get("shares", "-")
        now = f"₹{c.live_price}" if c.live_price is not None else "—"
        band = "✓" if o.get("below_lower_band") else "—"
        rows.append(
            f"| {c.rank} | {c.ticker.replace('.NS', '')} | {o.get('rsi14','-')} "
            f"| {o.get('stoch_k','-')} | {band} | {o.get('oversold_score','-')} "
            f"| {flag} {c.confidence_pct}%{short} | {now} | {c.buy_by} "
            f"| ₹{c.entry['level']} | ₹{c.target} | ₹{c.stop_loss} | {c.sell_by} | {qty} |"
        )
    return "\n".join(rows)


@mcp.tool()
def scan_oversold(universe: str | None = None, top_n: int | None = None,
                  oversold_rsi: float | None = None,
                  oversold_stoch: float | None = None,
                  max_price: float | None = None, capital: float | None = None,
                  risk_per_trade: float | None = None,
                  delivery_only: bool | None = None) -> dict:
    """Find the most OVERSOLD stocks in the universe + a full trade plan for each.

    Scans the universe and flags names whose RSI-14 is at/below `oversold_rsi` (a
    classic oversold reading), deepening the score when Stochastic %K is also low and
    the close is beneath the lower Bollinger band. Survivors are ranked by how oversold
    they are and the top N are returned with the same delivery analysis as the other
    tools (entry/target/stop/Buy-by/Sell-by + a position sized to your capital).

    Honest note: oversold means STRETCHED TO THE DOWNSIDE — a mean-reversion (bounce)
    setup, NOT a guaranteed reversal. A falling knife can stay oversold for a while.
    Research signal only.

    Args:
        universe: Preset name / inline list / default watchlist (as get_daily_picks).
        top_n: How many oversold names to return. Omit to use OVERSOLD_TOP_N (.env,
            default 5).
        oversold_rsi: RSI-14 ceiling to qualify as oversold. Omit for OVERSOLD_RSI
            (.env, default 30).
        oversold_stoch: Stochastic %K ceiling that deepens the score. Omit for
            OVERSOLD_STOCH (.env, default 20).
        max_price / capital / risk_per_trade / delivery_only: as get_daily_picks; omit
            to use the .env trading profile.

    Returns:
        A dict with header, market_status, generated_at, picks (each with an `oversold`
        block: rsi14, stoch_k, below_lower_band, oversold_score, reasons), table,
        markdown, and a mandatory disclaimer.
    """
    min_confidence = CONFIG.thresholds.min_confidence
    top_n = CONFIG.thresholds.oversold_top_n if top_n is None else top_n
    rsi_max = CONFIG.thresholds.oversold_rsi if oversold_rsi is None else oversold_rsi
    stoch_max = (CONFIG.thresholds.oversold_stoch
                 if oversold_stoch is None else oversold_stoch)
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
    log(f"scan_oversold: scanning {len(symbols)}, rsi_max={rsi_max}, "
        f"stoch_max={stoch_max}, top_n={top_n}, session={clock['session']}")

    found: list[tuple[float, scoring.Candidate]] = []
    failed: list[str] = []

    def _scan_one(sym: str):
        """Fetch + oversold-filter + analyze one symbol. Returns (score, cand), 'fail', None."""
        try:
            df = get_ohlcv(sym)  # cached per-day
            if df is None:
                return "fail"
            if cap is not None and float(df["Close"].iloc[-1]) > cap:
                return None
            snap = ind.compute(df)
            ov = ind.oversold(snap, rsi_max, stoch_max)
            if not ov["is_oversold"]:
                return None
            c = _analyze_one(sym, min_confidence, horizon=2, clock=clock)  # hits cache
            if c is None:
                return "fail"
            if deliv and c.direction == "SHORT":
                return None
            c.oversold = ov
            return (ov["oversold_score"], c)
        except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
            log(f"scan_oversold: error for {sym}: {exc}")
            return "fail"

    workers = max(1, min(_SCAN_WORKERS, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, res in zip(symbols, pool.map(_scan_one, symbols)):
            if res == "fail":
                failed.append(sym)
            elif res is not None:
                found.append(res)

    if not found:
        return {
            "header": f"No stock was oversold (RSI-14 ≤ {rsi_max:.0f}) in this universe "
                      f"today.",
            "generated_at": _now_ist(),
            "market_status": clock["status_line"],
            "picks": [],
            "failed_symbols": failed,
            "disclaimer": DISCLAIMER,
        }

    found.sort(key=lambda x: x[0], reverse=True)  # most oversold first
    picks = [c for _, c in found[:top_n]]
    for i, c in enumerate(picks, start=1):
        c.rank = i
        c.label = scoring.grade(c.confidence, min_confidence)
    _attach_live(picks)
    for c in picks:
        c.position = scoring.size_position(
            c.entry["level"], c.stop_loss, c.direction, capital, risk_per_trade
        )

    actionable = sum(1 for c in picks if c.label == scoring.ACTIONABLE)
    note = (f"{len(picks)} oversold stock(s) (RSI-14 ≤ {rsi_max:.0f}), most oversold "
            f"first ({actionable} actionable). Oversold = stretched down, a bounce "
            f"setup — NOT a guaranteed reversal.")
    if deliv:
        note += " Delivery (CNC) mode: LONG-only."
    if cap is not None:
        note += f" Filtered to ≤ ₹{cap:.0f}."

    result = {
        "header": note,
        "generated_at": _now_ist(),
        "market_status": clock["status_line"],
        "data_freshness": _freshness_line(picks),
        "actionable_count": actionable,
        "horizon_days": 2,
        "capital": capital,
        "oversold_rsi": rsi_max,
        "oversold_stoch": stoch_max,
        "top_n": top_n,
        "picks": [c.to_dict() for c in picks],
        "table": _oversold_table(picks),
        "failed_symbols": failed,
        "disclaimer": DISCLAIMER,
    }
    result["markdown"] = _render_markdown(result, "scan_oversold")
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
    _attach_catalysts([cand])  # primary-source NSE catalysts (free); merges + nudges
    return {
        "ticker": ticker,
        "verdict": cand.to_dict(),
        "indicators": indicators,
        "news": news,
        "catalysts": cand.catalysts,
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
def get_catalysts(ticker: str) -> dict:
    """Primary-source corporate catalysts for one symbol — FREE, from NSE directly.

    Pulls the NSE corporate-announcements feed and bulk/block-deal tape (no API key)
    and classifies them into high-impact event flags (order_win, mna, results,
    fundraise, rating_buyback, regulatory) with a recency-weighted catalyst_score.
    This is the layer the technical+RSS stack is blind to: it catches the actual
    *trigger* behind a sharp move (an order win, board outcome, takeover/stake
    disclosure) the moment it is filed — before any market-wrap article exists.

    Args:
        ticker: NSE symbol (with or without .NS, e.g. "RAMCOSYS.NS" or "RAMCOSYS").

    Returns:
        A dict with provider, announcement_count, material_count, events flags,
        has_deal_activity, recency_hours (age of newest material filing),
        catalyst_score (0..1), up to 10 announcements, and bulk/block deals.
        Always valid — never raises; provider="none (unreachable)" if NSE is blocked.
    """
    ticker = ticker.upper().strip()
    result = fetch_catalysts(ticker)
    result["disclaimer"] = DISCLAIMER
    return result


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
def backtest(ticker: str, days: int = 90,
             min_confidence: float | None = None) -> dict:
    """Historical sanity-check using the SAME selection path as the live engine.

    Walks the last ``days`` of bars. At each bar it reproduces the live decision:
    technical stack → confidence (neutral sentiment, since point-in-time news is not
    reconstructable) gated against ``min_confidence``, plus the chart-pattern structure
    veto (skips clear downtrend/double-top), then derives entry/target/stop from the
    same ``_entry_exit`` rule (shallow ATR-bounded dip with buy-at-open fallback).
    Simulates the fill and exits at the first of target / stop / T+horizon close.
    Reports win-rate, target-hit / stop / fill rates, and average return.

    Caveat: omits news-sentiment and market-regime (neither is reconstructable per past
    bar), so it isolates the technical+pattern+levels core — closer to live than the old
    crude check, but still not a guarantee.

    Args:
        ticker: NSE symbol with the .NS suffix.
        days: Number of recent trading days to test over (default 90).
        min_confidence: Confidence bar; defaults to the .env profile.
    """
    ticker = ticker.upper().strip()
    bar = (CONFIG.thresholds.min_confidence
           if min_confidence is None else min_confidence)
    df = get_ohlcv(ticker, lookback_days=max(days + 80, 180))
    if df is None or len(df) < days + 30:
        return {"ticker": ticker, "error": "insufficient data for backtest",
                "disclaimer": DISCLAIMER}

    horizon = min(CONFIG.horizon_days, 2)
    trades: list[dict] = []
    window = df.iloc[-(days + 30):]

    for i in range(30, len(window) - horizon - 1):
        sub = window.iloc[: i + 1]
        try:
            t = ind.compute(sub)
        except Exception:  # noqa: BLE001
            continue
        tech, _, direction = scoring._technical_signals(t)
        if direction != "LONG":
            continue
        # Live gating: confidence bar (neutral news) + chart-structure veto.
        dq = _data_quality(sub)
        conf = scoring._confidence(tech, 0.5, {}, dq, direction)
        if conf < bar:
            continue
        feats = patterns.detect(sub)
        if feats.get("trend") == "down" and feats.get("double_top"):
            continue  # structure veto — the live engine would not surface this

        entry, target, stop, _, _ = scoring._entry_exit(t, "LONG", horizon)
        level = entry["level"]

        # Fill: dip to the level, else buy at next open ("buy at open on strength").
        nxt = window.iloc[i + 1]
        filled = float(nxt["Low"]) <= level
        fill_price = level if filled else float(nxt["Open"])

        exit_price, reason = None, "time"
        for d in range(1, horizon + 1):
            b = window.iloc[i + 1 + d] if (i + 1 + d) < len(window) else None
            if b is None:
                break
            if float(b["Low"]) <= stop:            # stop before target (conservative)
                exit_price, reason = stop, "stop"
                break
            if float(b["High"]) >= target:
                exit_price, reason = target, "target"
                break
            exit_price = float(b["Close"])
        if exit_price is None:
            continue
        ret = (exit_price - fill_price) / fill_price * 100.0
        trades.append({
            "date": str(window.index[i + 1].date()),
            "confidence": round(conf, 3),
            "filled": filled,
            "entry": round(fill_price, 2),
            "exit": round(exit_price, 2),
            "reason": reason,
            "return_pct": round(ret, 2),
        })

    if not trades:
        return {"ticker": ticker, "trades": 0,
                "note": f"no signals cleared confidence {bar:.2f} in the window",
                "disclaimer": DISCLAIMER}

    wins = [t for t in trades if t["return_pct"] > 0]
    tgt = [t for t in trades if t["reason"] == "target"]
    stp = [t for t in trades if t["reason"] == "stop"]
    filled = [t for t in trades if t["filled"]]
    avg = sum(t["return_pct"] for t in trades) / len(trades)
    return {
        "ticker": ticker,
        "min_confidence": round(bar, 2),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "target_hit_rate": round(len(tgt) / len(trades), 3),
        "stop_rate": round(len(stp) / len(trades), 3),
        "fill_rate": round(len(filled) / len(trades), 3),
        "avg_return_pct": round(avg, 3),
        "horizon_days": horizon,
        "results": trades[-20:],
        "note": "Aligned with live engine (technical+pattern+levels); excludes news + "
                "regime. Conservative same-bar resolution counts STOPPED.",
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


def _save_report_file(content: str, title: str = "") -> dict:
    """Append a markdown block to reports/<branch>/<date>/<date>.md. Shared by the
    save_report tool and the auto-save path of chart-enabled tools."""
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


@mcp.tool()
def score_predictions() -> dict:
    """Measure the engine's REAL accuracy from its own logged picks.

    Every predict/scan run records its picks (entry/target/stop/horizon + confidence,
    pattern, regime) to reports/<branch>/predictions.jsonl. This tool replays the actual
    price path for each pick whose holding window has elapsed, classifies the outcome
    (TARGET_HIT / STOPPED / TIME_EXIT), records whether the entry filled, and reports
    rolling stats: overall win-rate, win-rate BY CONFIDENCE BUCKET (50-60/60-70/70+),
    best-pick vs the rest, fill-rate, and average return.

    Use this to answer "how accurate is the tool, really?" — it needs picks that have
    aged past their sell-by, so run it a day or two after generating predictions.
    Conservative: a bar that touches both stop and target counts as STOPPED. Educational
    measurement only — NOT a performance guarantee.
    """
    return tracker.score(_PROJECT_DIR, _branch_name())


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
    return _save_report_file(content, title)


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
