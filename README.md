# Short-Swing Stock Signal — MCP server

A daily research tool for Claude Code. It scans a configurable universe of stocks
(default: NSE / Indian market) and **always returns the top 3 short-swing trade
candidates**, each with a suggested entry/target/stop and a hard **2-trading-day**
time stop, combining technical analysis of historical price data with recent news
sentiment.

> ⚠️ **Disclaimer — read this.** This is an **educational research signal generator,
> NOT financial advice and NOT a prediction.** Short-term price moves are largely
> noise. Every result carries a one-line risk disclaimer and an honest quality grade.
> You can lose money. Do your own research and consult a licensed advisor.

---

## Core behaviour: always 3, graded honestly

- Outputs the **top 3** candidates ranked by a blended score.
- Each pick is labelled against `min_confidence`:
  - `confidence >= min_confidence` → **`ACTIONABLE`** (✅)
  - `confidence <  min_confidence` → **`LOW CONFIDENCE - NOT RECOMMENDED`** (⚠️)
- The header states how many cleared the bar, e.g.
  *"2 of 3 actionable today"* or *"0 of 3 cleared the bar — today is choppy, treat all
  three as ideas only, NOT trades."*
- Weak picks are never hidden and never dressed up as real signals. The label is
  mandatory and visually obvious.

---

## MCP tools

| Tool | Purpose |
|------|---------|
| `get_daily_picks(universe=None, min_confidence=0.55)` | **Main tool.** Top 3 graded picks + summary header. |
| `analyze_stock(ticker)` | Full technical + news breakdown + verdict for one symbol. |
| `get_technicals(ticker, lookback_days=180)` | Indicator snapshot + signal list. |
| `get_news_sentiment(ticker)` | Recent headlines with per-article sentiment + event flags. |
| `backtest(ticker, days=90)` | Quick historical sanity-check of the entry/exit logic. |

All tickers use the NSE `.NS` suffix (e.g. `RELIANCE.NS`).

---

## How it works (pipeline)

1. Load the universe from config (default: 25 NSE large-caps; fully configurable).
2. Per symbol: fetch ~9 months of daily OHLCV via the data provider, **cached per
   day** (local JSON) to respect rate limits.
3. Technical indicators: EMA 9/21/50 + MACD (trend), RSI-14 + Stochastic (momentum),
   ATR + Bollinger Bands (volatility), 20-day volume avg + spike + OBV (volume).
4. Recent news (last 24–72h) per symbol, per-article sentiment, aggregated, with
   high-impact event flags (earnings / regulatory / M&A).
5. Blend into one score (default **technical 0.6 / sentiment 0.4**, configurable).
   Long-bias by default; a strongly bearish stack flips to `SHORT` (flagged — mind
   Indian intraday / short-sell constraints).
6. Derive **entry** (level + timing), **target** (ATR/resistance-based), **stop-loss**,
   and a **hard time stop** (`exit by close of day+2`). The ≤ 2-day rule is enforced
   structurally in code, not as advice text.
7. **Confidence** (0–1) from signal agreement + sentiment alignment + data quality.
8. Rank all scored symbols, take the top 3, grade each against `min_confidence`.
   Symbols whose data fetch fails are excluded and reported in the header.

---

## Install

Requires **Python 3.11+**. Two options.

### Option A — `uv` (recommended)
```bash
cd stock-signal-mcp
uv venv --python 3.11
uv pip install -r requirements.txt
cp .env.example .env   # then fill in keys (optional — see below)
```

### Option B — venv + pip
```bash
cd stock-signal-mcp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> `pandas-ta` is **optional and not in `requirements.txt`** (current releases require
> Python ≥ 3.12). `indicators.py` hand-rolls every indicator as a fallback, so
> everything works on 3.11. On Python ≥ 3.12 you may add the (identical) engine with
> `uv pip install "pandas-ta>=0.3.14b0"`.

---

## API keys (both optional — it degrades gracefully)

| Key | Provider | Free tier | Get it |
|-----|----------|-----------|--------|
| `MARKETAUX_API_KEY` | Marketaux (news + sentiment) | ~100 req/day, covers India | https://www.marketaux.com → Sign up → Dashboard |
| `TWELVEDATA_API_KEY` | Twelve Data (price fallback) | ~800 calls/day, global | https://twelvedata.com → Sign up → API key |

- **No keys?** It still runs: price data comes from **yfinance** (no key) and news
  falls back to **RSS feeds** (Economic Times, Moneycontrol, Mint, Business Standard)
  scored locally with VADER.
- Keys only ever come from env / `.env` — nothing is hardcoded.

Provider selection (optional, in `.env`):
`MARKET_PROVIDER=yfinance|twelvedata`, `NEWS_PROVIDER=marketaux|rss`.

---

## Register with Claude Code

```bash
claude mcp add stock-signals \
  -e MARKETAUX_API_KEY=your_key \
  -e PYTHONUNBUFFERED=1 \
  -- uv run --with "mcp[cli]" python /full/path/to/stock-signal-mcp/server.py
```

If you installed into the project venv instead of using `uv run`, point at that python:
```bash
claude mcp add stock-signals \
  -e MARKETAUX_API_KEY=your_key -e PYTHONUNBUFFERED=1 \
  -- /full/path/to/stock-signal-mcp/.venv/bin/python /full/path/to/stock-signal-mcp/server.py
```

Verify:
```bash
claude mcp list          # should show stock-signals: connected
```
Then inside Claude Code run `/mcp` to confirm the 5 tools are exposed, and ask:
> "Run my daily stock picks."

> **Scope matters.** `claude mcp add` defaults to **local scope** — the server is only
> visible from the directory you ran the command in. If `claude mcp list` shows the
> server from the project folder but *not* from elsewhere, re-add it at user scope so
> it works everywhere:
> ```bash
> claude mcp add --scope user stock-signals \
>   -e PYTHONUNBUFFERED=1 \
>   -- /full/path/to/stock-signal-mcp/.venv/bin/python /full/path/to/stock-signal-mcp/server.py
> ```

---

## Using it in Claude Code

After the server shows `✔ Connected` in `claude mcp list`, **start (or restart) a
Claude Code session** so the tools load, then just talk to it in plain English —
Claude picks the right tool and arguments.

### Daily workflow (the main one)
> **"Run my daily stock picks."**

Claude calls `get_daily_picks` and returns the top 3 graded candidates with a header
like *"2 of 3 actionable today"*. Variations it understands:
- *"Daily picks, but only show me high-conviction ones"* → raises `min_confidence`.
- *"Run picks on RELIANCE.NS, TCS.NS, INFY.NS and SBIN.NS only"* → custom `universe`.
- *"Scan the nifty universe with a 0.6 confidence bar."*

### Delivery trading with a fixed budget (e.g. ₹1.5 lakh, stocks under ₹500)
`get_daily_picks` does **position sizing** for delivery (CNC) and can **filter by
price**:
> **"Run picks on stocks under ₹500 with ₹1.5 lakh capital."**

That maps to `universe="under500"`, `max_price=500`, `capital=150000`. Each pick then
includes a `position`: **shares to buy**, **rupees to deploy**, **rupees at risk** to
the stop, and **% of capital** used. Sizing is *risk-based* — by default each trade
risks `risk_per_trade` (2%) of capital to its stop, so quantity = (capital × 2%) ÷
(entry − stop), capped by what you can afford.

Tunable parameters:

| Param | Default | Meaning |
|-------|---------|---------|
| `universe` | configured watchlist | `"under500"`/`"cheap"` = liquid sub-₹500 names; or inline list |
| `min_confidence` | `0.55` | ACTIONABLE bar |
| `max_price` | none | only consider stocks ≤ this price (filtered on **live** price) |
| `capital` | `150000` | trading capital (₹) for sizing |
| `risk_per_trade` | `0.02` | fraction of capital risked to the stop per trade |

> ⚠️ **Delivery vs short selling.** Delivery (CNC) holds work for **LONG** picks held
> to T+2. A pick marked `SHORT` **cannot be a delivery trade** — NSE allows short
> selling **intraday (MIS) only** — so the tool flags SHORT picks with a
> `delivery_note`. For pure delivery trading, act on the LONG picks and ignore shorts.

### Reading a result
Each pick gives you: **rank · ticker · direction (LONG/SHORT) · confidence% · quality
label · entry (level + when) · target · stop-loss · time stop (T+2) · rationale**. The
label is the honest grade:
- ✅ **ACTIONABLE** — confidence ≥ your bar.
- ⚠️ **LOW CONFIDENCE - NOT RECOMMENDED** — below the bar; treat as an idea, not a trade.

The header tells you how many of the 3 cleared the bar. *"0 of 3 cleared the bar"* means
it's a choppy day — none of them are real signals, don't force a trade.

### Drilling into one name
- *"Analyze TCS.NS"* → `analyze_stock`: full technical + news + verdict for one symbol.
- *"Show me the technicals for INFY.NS over the last 90 days"* → `get_technicals`.
- *"What's the news sentiment on HDFCBANK.NS?"* → `get_news_sentiment`.
- *"Backtest the entry/exit logic on RELIANCE.NS over 120 days"* → `backtest`
  (crude sanity check of the rules, not a validated strategy).

### Tool → natural-language cheat sheet
| Say this | Tool called |
|----------|-------------|
| "run my daily picks" / "what are today's trades" | `get_daily_picks` |
| "analyze \<TICKER\>" / "full breakdown of \<TICKER\>" | `analyze_stock` |
| "technicals for \<TICKER\>" / "indicators on \<TICKER\>" | `get_technicals` |
| "news sentiment on \<TICKER\>" | `get_news_sentiment` |
| "backtest \<TICKER\>" | `backtest` |

### Good habits
- It's a **research/idea generator, not advice** — every result says so. Size positions
  yourself and respect the stop + the **hard T+2 time stop**.
- Re-running the same scan the same day is cheap: price + news are **cached per day**,
  so you won't burn API quota. Delete `.cache/` to force a fresh pull.
- If lots of symbols show up under `failed_symbols`, yfinance is likely rate-limiting —
  add `TWELVEDATA_API_KEY` (auto-fallback) or scan a smaller universe.

---

## Example output

### Normal day — some actionable
```
2 of 3 actionable today (min_confidence 0.50).

| # | Ticker       | Dir   | Conf | Label                 | Entry    | Target   | Stop     | Time stop |
|---|--------------|-------|------|-----------------------|----------|----------|----------|-----------|
| 1 | RELIANCE.NS  | SHORT | 43%  | ⚠️ LOW CONFIDENCE      | ₹1309.50 | ₹1253.20 | ₹1335.50 | T+2       |
| 2 | INFY.NS      | SHORT | 53%  | ✅ ACTIONABLE          | ₹1051.40 | ₹997.64  | ₹1087.24 | T+2       |
| 3 | ICICIBANK.NS | LONG  | 53%  | ✅ ACTIONABLE          | ₹1298.74 | ₹1362.70 | ₹1274.46 | T+2       |
```

### Choppy day — 0 of 3 cleared the bar
```
0 of 3 cleared the bar — today is choppy, treat all three as ideas only, NOT trades.
1 symbol(s) skipped (no data): TATAMOTORS.NS.

| # | Ticker       | Dir   | Conf | Label                 | Entry    | Target   | Stop     | Time stop |
|---|--------------|-------|------|-----------------------|----------|----------|----------|-----------|
| 1 | RELIANCE.NS  | SHORT | 43%  | ⚠️ LOW CONFIDENCE      | ₹1309.50 | ₹1253.20 | ₹1335.50 | T+2       |
| 2 | INFY.NS      | SHORT | 53%  | ⚠️ LOW CONFIDENCE      | ₹1051.40 | ₹997.64  | ₹1087.24 | T+2       |
| 3 | ICICIBANK.NS | LONG  | 53%  | ⚠️ LOW CONFIDENCE      | ₹1298.74 | ₹1362.70 | ₹1274.46 | T+2       |
```

(The choppy run above also demonstrates graceful degradation: `TATAMOTORS.NS` failed
its data fetch and was excluded with an explicit note rather than crashing the scan.)

---

## Configuration

Everything tunable lives in `config.py` and is overridable via env vars:

| Env var | Default | Meaning |
|---------|---------|---------|
| `STOCK_UNIVERSE` | 25 NSE large-caps | Comma-separated `.NS` symbols |
| `MIN_CONFIDENCE` | `0.55` | ACTIONABLE bar |
| `WEIGHT_TECHNICAL` / `WEIGHT_SENTIMENT` | `0.60` / `0.40` | Score blend |
| `LOOKBACK_DAYS` | `270` | Daily history pulled |
| `NEWS_LOOKBACK_HOURS` | `72` | News window |
| `VOLUME_SPIKE_MULT` | `1.5` | Volume-spike threshold vs 20-day avg |
| `ATR_TARGET_MULT` / `ATR_STOP_MULT` | `1.5` / `1.0` | Target/stop distance in ATRs |
| `CACHE_DIR` | `.cache` | Per-day cache location |

`HORIZON_DAYS = 2` is a hard cap and is clamped in code — it is not meant to be raised.

---

## Engineering notes

- **stdio-safe:** all logs go to **stderr** (`utils/log.py`); stdout is reserved for
  JSON-RPC. Run with `PYTHONUNBUFFERED=1`.
- **Never crashes on provider failure:** every network call is wrapped; a failed
  symbol is skipped (or falls back to the secondary provider) and reported.
- **Caching:** price + news cached per calendar day under `CACHE_DIR`.
- **Swappable providers:** data and news each sit behind a small interface
  (`data/market.py`, `data/news.py`).
# StockPilot
