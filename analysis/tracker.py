"""Forward outcome tracker — the feedback loop that makes accuracy measurable.

Every predict/scan run appends its picks to ``reports/<branch>/predictions.jsonl`` with
the levels it committed to (entry/target/stop/horizon) and the context (confidence,
pattern_score, regime). Later, ``score`` replays the *actual* price path for any pick
whose holding window has elapsed and classifies the outcome — TARGET_HIT / STOPPED /
TIME_EXIT — plus realized return and whether the entry ever filled. Rolling stats
(hit-rate, by confidence bucket, fill-rate, expectancy) are the honest answer to
"how accurate is this?".

Pure stdlib + the market layer; never raises out (best-effort instrumentation).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone

from data.market import get_ohlcv
from utils.log import log

LEDGER = "predictions.jsonl"


def _ledger_path(project_dir: str, branch: str) -> str:
    folder = os.path.join(project_dir, "reports", branch)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, LEDGER)


def _read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"tracker: read failed {path}: {exc}")
    return rows


def _write(path: str, rows: list[dict]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    except OSError as exc:
        log(f"tracker: write failed {path}: {exc}")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log_picks(result: dict, tool_name: str, project_dir: str, branch: str) -> int:
    """Append each pick to the ledger as status=open. Idempotent per
    (call_date, tool, ticker). Returns the number of new rows written."""
    picks = result.get("picks") or []
    if not picks:
        return 0
    path = _ledger_path(project_dir, branch)
    rows = _read(path)
    seen = {(r.get("call_date"), r.get("tool"), r.get("ticker")) for r in rows}
    call_date = str(result.get("generated_at", ""))[:10] or date.today().isoformat()
    regime = (result.get("regime") or {}).get("regime", "UNKNOWN")
    added = 0
    for p in picks:
        key = (call_date, tool_name, p.get("ticker"))
        if key in seen:
            continue
        pat = p.get("patterns") or {}
        rows.append({
            "id": f"{call_date}|{tool_name}|{p.get('ticker')}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "call_date": call_date,
            "tool": tool_name,
            "ticker": p.get("ticker"),
            "direction": p.get("direction"),
            "entry": p.get("entry", {}).get("level"),
            "target": p.get("target"),
            "stop": p.get("stop_loss"),
            "horizon": p.get("time_stop_days"),
            "confidence": p.get("confidence"),
            "confidence_pct": p.get("confidence_pct"),
            "pattern_score": pat.get("pattern_score"),
            "best": bool(p.get("best")),
            "regime": regime,
            "spot_at_call": p.get("live_price") or p.get("price"),
            "sell_by": p.get("sell_by"),
            "status": "open",
        })
        seen.add(key)
        added += 1
    if added:
        _write(path, rows)
        log(f"tracker: logged {added} pick(s) from {tool_name} to {LEDGER}")
    return added


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _evaluate(row: dict) -> dict | None:
    """Replay the actual path for one open pick. Returns an outcome dict, or None if
    the holding window has not produced enough bars yet (leave it open)."""
    ticker = row.get("ticker")
    entry, target, stop = row.get("entry"), row.get("target"), row.get("stop")
    horizon = int(row.get("horizon") or 2)
    if not all(isinstance(x, (int, float)) for x in (entry, target, stop)):
        return None
    df = get_ohlcv(ticker)
    if df is None or df.empty:
        return None
    try:
        cd = date.fromisoformat(row["call_date"])
    except Exception:  # noqa: BLE001
        return None
    dates = [d.date() for d in df.index]
    post = [i for i, d in enumerate(dates) if d > cd]
    if not post:
        return None
    e_i = post[0]
    last_i = min(e_i + horizon, len(df) - 1)
    if last_i - e_i < horizon and last_i != len(df) - 1:
        return None  # not enough bars yet
    window = df.iloc[e_i: last_i + 1]
    if window.empty:
        return None

    first = window.iloc[0]
    filled = float(first["Low"]) <= entry
    actual_entry = entry if filled else float(first["Open"])  # "else buy at open"

    outcome, exit_price = "TIME_EXIT", float(window.iloc[-1]["Close"])
    for _, bar in window.iterrows():
        lo, hi = float(bar["Low"]), float(bar["High"])
        if lo <= stop:               # conservative: stop checked before target
            outcome, exit_price = "STOPPED", stop
            break
        if hi >= target:
            outcome, exit_price = "TARGET_HIT", target
            break
    ret = (exit_price - actual_entry) / actual_entry * 100.0
    return {
        "status": "closed",
        "filled": filled,
        "actual_entry": round(actual_entry, 2),
        "outcome": outcome,
        "exit_price": round(exit_price, 2),
        "return_pct": round(ret, 2),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def _bucket(pct) -> str:
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return "unknown"
    if p >= 70:
        return "70+"
    if p >= 60:
        return "60-70"
    if p >= 50:
        return "50-60"
    return "<50"


def score(project_dir: str, branch: str) -> dict:
    """Close out every open pick whose window has elapsed, then return rolling stats."""
    path = _ledger_path(project_dir, branch)
    rows = _read(path)
    if not rows:
        return {"note": "no predictions logged yet", "total": 0}

    newly = 0
    for r in rows:
        if r.get("status") == "open":
            res = _evaluate(r)
            if res:
                r.update(res)
                newly += 1
    if newly:
        _write(path, rows)

    closed = [r for r in rows if r.get("status") == "closed"]
    if not closed:
        return {"note": "predictions logged but none have closed yet",
                "total": len(rows), "open": len(rows), "closed": 0}

    def _stats(group: list[dict]) -> dict:
        n = len(group)
        wins = [r for r in group if r.get("return_pct", 0) > 0]
        tgt = [r for r in group if r.get("outcome") == "TARGET_HIT"]
        stp = [r for r in group if r.get("outcome") == "STOPPED"]
        filled = [r for r in group if r.get("filled")]
        avg = sum(r.get("return_pct", 0) for r in group) / n if n else 0.0
        return {
            "n": n,
            "win_rate": round(len(wins) / n, 3) if n else 0,
            "target_hit_rate": round(len(tgt) / n, 3) if n else 0,
            "stop_rate": round(len(stp) / n, 3) if n else 0,
            "fill_rate": round(len(filled) / n, 3) if n else 0,
            "avg_return_pct": round(avg, 3),
        }

    buckets: dict[str, list[dict]] = {}
    for r in closed:
        buckets.setdefault(_bucket(r.get("confidence_pct")), []).append(r)

    best = [r for r in closed if r.get("best")]
    rest = [r for r in closed if not r.get("best")]

    return {
        "total": len(rows),
        "open": len([r for r in rows if r.get("status") == "open"]),
        "closed": len(closed),
        "newly_closed": newly,
        "overall": _stats(closed),
        "by_confidence_bucket": {k: _stats(v) for k, v in sorted(buckets.items())},
        "best_pick": _stats(best) if best else {"n": 0},
        "non_best": _stats(rest) if rest else {"n": 0},
        "recent": [
            {k: r.get(k) for k in ("call_date", "tool", "ticker", "confidence_pct",
                                   "outcome", "return_pct", "filled", "best")}
            for r in closed[-15:]
        ],
        "note": ("Realized outcomes from logged picks. Conservative: stop checked before "
                 "target within a bar. Short horizons stay noisy — read the bucketed "
                 "win-rate, not any single trade."),
    }
