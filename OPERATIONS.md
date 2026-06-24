# Operations & Monitoring — stock-signals MCP

How to verify the server works, watch its logs, and diagnose failures in production.

---

## 1. Quick verify — healthcheck script

One command exercises every data source and every tool, end-to-end, **without** going
through MCP. Use it after any change, on a schedule, or whenever picks look wrong.

```bash
cd /Users/dev/aiWorkSpace/stock-signal-mcp

.venv/bin/python healthcheck.py            # quick: sources + all 7 tools
.venv/bin/python healthcheck.py --full     # + a live nifty50 scan (50 symbols)
```

- PASS/FAIL line per check (with timing) on **stdout**; server diagnostics on **stderr**.
- Exits non-zero if anything failed → wire into cron / CI.
- Clean summary only: `.venv/bin/python healthcheck.py 2>/dev/null`
- Watch logs while it runs: drop the `2>/dev/null`.

Healthy output:

```
PASS  source: yfinance                     0.9s  280 bars
PASS  source: yahoo_chart (keyless)        0.3s  498 bars
PASS  source: get_ohlcv (with fallback)    0.0s  498 bars via cache
PASS  news sentiment                       0.0s  3 articles via marketaux
PASS  tool: get_daily_picks                2.4s  3 picks, 0 failed
... (all 7 tools) ...
------------------------------------------------------------
11 passed, 0 failed
```

---

## 2. Where the logs go

Everything the server prints goes to **stderr** (never stdout — stdout carries the
JSON-RPC protocol; a stray stdout write corrupts the MCP connection). See
[utils/log.py](utils/log.py).

### a) Claude Code's captured MCP stderr (when run as an MCP server)

Claude Code captures the server's stderr into per-session `.jsonl` files. Each `log()`
line shows up as a `"Server stderr:"` entry.

```bash
LOGDIR="/Users/dev/Library/Caches/claude-cli-nodejs/-Users-dev-aiWorkSpace-stock-signal-mcp/mcp-logs-stock-signals"

# Live tail, server lines only:
tail -f "$LOGDIR/"*.jsonl | grep "Server stderr"

# Newest session file:
ls -t "$LOGDIR/"*.jsonl | head -1
```

> A **stale** file (old timestamp, e.g. `universe size=25`) means the server has not
> been reloaded since the last code change. Reload it (see §4) — a fresh `.jsonl`
> appears on reconnect.

### b) Run the server standalone (watch raw stderr live)

```bash
PYTHONUNBUFFERED=1 .venv/bin/python server.py
```

Logs stream to your terminal; Ctrl-C to stop. Use this only for log-watching — while
standalone it is not serving Claude Code.

---

## 3. Reading the logs

| Log line | Meaning |
|----------|---------|
| `starting stock-signals MCP server (stdio)` | Boot OK |
| `providers: market=yfinance news=marketaux \| horizon=2d \| universe size=500` | Config loaded; universe resolved live |
| `daily_picks: scanning N/M symbols ...` | Scan running (N after price prefilter of M) |
| `yfinance insufficient data ... trying fallback` → `yahoo_chart` | **Failover working** (expected, not an error) |
| `news: <prov> unavailable ... trying fallback` | News source down; falling back |
| `retry[...]: attempt n/3 failed ... backing off` | Transient blip, self-healing |
| `market: ALL providers failed for X` | Real outage — **both** sources down for X |
| `prefilter: K/M symbols ≤ ₹P` | Price prefilter narrowed the universe |
| `cache: ...` | Best-effort cache note (never fatal) |

---

## 4. Reloading after a code change

The MCP server runs as a **separate process** — it will not see edited code until
reloaded.

- In Claude Code: reconnect MCP (`/mcp`) or restart Claude Code.
- Standalone: Ctrl-C and re-run `python server.py`.

Confirm the reload took: new `.jsonl` file with `universe size=500` (not `25`).

---

## 5. Data sources & reliability

Market data is tried in order, each independent; first success wins:

1. **yfinance** — primary (richest history).
2. **yahoo_chart** — keyless raw Yahoo chart API via our own httpx session/User-Agent.
   Independent of the yfinance library, so it survives yfinance's common failures
   (stale cookie/crumb, internal session rate-limit, library breakage).
3. **twelvedata** — only if `TWELVEDATA_API_KEY` is set; auto-skipped otherwise.

News: **marketaux** (needs `MARKETAUX_API_KEY`) → **rss** (keyless, VADER sentiment).
A transient news outage is **not** cached, so it does not poison the rest of the day.

Per-symbol OHLCV and news are cached **per calendar day** under `.cache/`, so the first
scan of the day is the slow one; later scans are near-instant.

### Tuning (env / `.env`)

| Var | Default | Effect |
|-----|---------|--------|
| `STOCK_SCAN_WORKERS` | 6 | Parallel scan threads (1–16). Lower if you see yfinance 429s. |
| `MARKET_PROVIDER` | yfinance | Which source to try first. |
| `STOCK_UNIVERSE` | nifty500 | Default universe when none passed. |
| `MAX_PRICE` | 500 | Only suggest stocks ≤ this (0/unset = no cap). |
| `MIN_CONFIDENCE` | 0.55 | ACTIONABLE vs LOW-CONFIDENCE bar. |

---

## 6. Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `0 candidates — every data fetch failed` | No network, or both sources down | Check connectivity; run `healthcheck.py`; check logs for `ALL providers failed`. |
| `0 candidates — no stock under ₹P qualified` | Price cap excluded everything | Raise `max_price` / `MAX_PRICE`. |
| `could not load the stock universe` | Live NSE archive fetch failed | Retry; NSE archives may be briefly down. |
| All inline symbols "failed" | Symbols lacked `.NS` | Fixed — handled automatically now. |
| First scan very slow | Cold daily fetch of a large universe | Expected; subsequent scans hit the per-day cache. |
| Logs file not updating | Server not reloaded | See §4. |
| yfinance `429 Too Many Requests` | Scan concurrency too high | Lower `STOCK_SCAN_WORKERS`; failover to `yahoo_chart` covers it. |

---

## 7. Scheduled health check (optional)

Run the healthcheck every weekday morning before market open and log the result:

```bash
# crontab -e  (08:45 IST, Mon–Fri)
45 8 * * 1-5 cd /Users/dev/aiWorkSpace/stock-signal-mcp && \
  .venv/bin/python healthcheck.py >> reports/healthcheck.log 2>&1
```

Non-zero exit = something is down before you trade.
