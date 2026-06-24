#!/usr/bin/env python
"""Standalone health check for the stock-signals MCP server.

Run it anytime to confirm everything works end-to-end WITHOUT going through MCP:

    .venv/bin/python healthcheck.py            # quick (small universe)
    .venv/bin/python healthcheck.py --full      # also scans a live preset (nifty)

It exercises every data source and every tool, prints a PASS/FAIL line per check,
and exits non-zero if anything failed (so you can wire it into cron / CI). All the
server's own diagnostics still go to stderr; the PASS/FAIL summary goes to stdout.
"""

from __future__ import annotations

import json
import sys
import time

# Symbols cheap to fetch and almost always present.
_TEST = "RELIANCE,TCS,SBIN,ITC,WIPRO,INFY"

_passed = 0
_failed = 0


def check(name: str, fn) -> None:
    """Run one check; print PASS/FAIL + timing; record outcome."""
    global _passed, _failed
    t0 = time.time()
    try:
        detail = fn()
        dt = time.time() - t0
        print(f"PASS  {name:34} {dt:5.1f}s  {detail}")
        _passed += 1
    except Exception as exc:  # noqa: BLE001 - report, don't abort the run
        dt = time.time() - t0
        print(f"FAIL  {name:34} {dt:5.1f}s  {type(exc).__name__}: {exc}")
        _failed += 1


def _strict_json(obj) -> None:
    """Raise if the result isn't valid strict JSON (catches NaN/Inf leaks)."""
    json.dumps(obj, allow_nan=False)


def main() -> int:
    full = "--full" in sys.argv

    import data.market as market
    import server

    # --- Data sources (each independently) ---
    def src_yf():
        from data.market import YFinanceProvider
        df = YFinanceProvider().fetch("RELIANCE.NS", 270)
        assert df is not None and len(df) >= 30, "no/short frame"
        return f"{len(df)} bars"

    def src_yahoo():
        from data.market import YahooChartProvider
        df = YahooChartProvider().fetch("RELIANCE.NS", 270)
        assert df is not None and len(df) >= 30, "no/short frame"
        return f"{len(df)} bars"

    def src_ohlcv():
        df = market.get_ohlcv("TCS.NS")
        assert df is not None, "all providers failed"
        return f"{len(df)} bars via {df.attrs.get('provider')}"

    check("source: yfinance", src_yf)
    check("source: yahoo_chart (keyless)", src_yahoo)
    check("source: get_ohlcv (with fallback)", src_ohlcv)

    # --- News ---
    def chk_news():
        r = server.get_news_sentiment("INFY.NS")
        _strict_json(r)
        return f"{r['article_count']} articles via {r['provider']}"

    check("news sentiment", chk_news)

    # --- Tools (small universe, no price cap so picks always materialise) ---
    def tool(fn, **kw):
        def _run():
            r = fn(**kw)
            _strict_json(r)
            n = len(r.get("picks", []))
            f = len(r.get("failed_symbols", []))
            return f"{n} picks, {f} failed"
        return _run

    check("tool: get_daily_picks",
          tool(server.get_daily_picks, universe=_TEST, max_price=100000))
    check("tool: predict_delivery_2day",
          tool(server.predict_delivery_2day, universe=_TEST, max_price=100000))
    check("tool: predict_buy_today_sell_tomorrow",
          tool(server.predict_buy_today_sell_tomorrow, universe=_TEST, max_price=100000))
    check("tool: scan_volume_spikes",
          tool(server.scan_volume_spikes, universe=_TEST, min_surge=0.1, max_price=100000))
    check("tool: analyze_stock",
          lambda: (_strict_json(r := server.analyze_stock("RELIANCE.NS")),
                   r["verdict"]["label"])[1])
    check("tool: get_technicals",
          lambda: (_strict_json(r := server.get_technicals("TCS.NS")),
                   f"{r['direction']} {r['technical_score']}")[1])
    check("tool: backtest",
          lambda: (_strict_json(r := server.backtest("RELIANCE.NS", days=90)),
                   f"{r.get('trades', 0)} trades")[1])

    if full:
        check("live preset: nifty (50 symbols)",
              tool(server.get_daily_picks, universe="nifty"))

    print("-" * 60)
    print(f"{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
