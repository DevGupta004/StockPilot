"""Algorithmic chart-pattern detection + ASCII rendering.

Pure pandas/numpy — no plotting deps. Reads the same daily OHLC frame the indicator
stack uses and derives human-readable chart structure (trend, support/resistance,
double-bottom/top, breakout, consolidation, distance from the 52-week extremes) plus a
0..1 ``pattern_score`` summarising how constructive the picture is for a LONG.

Everything here is heuristic and deterministic. It is a second opinion on top of the
indicator score — NOT a prediction. Bars are daily closes; "52w" ≈ 252 trading days.
"""

from __future__ import annotations

import pandas as pd

_SPARK = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def sparkline(series: pd.Series, points: int = 30) -> str:
    """Unicode sparkline of the last ``points`` closes (8 levels, min→max scaled)."""
    vals = [float(v) for v in series.dropna().tail(points)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _SPARK[0] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def ascii_block_chart(series: pd.Series, width: int = 48, height: int = 10) -> str:
    """A compact fixed-size ASCII price chart (last ``width`` resampled closes).

    Rows are price buckets (high → low), columns are time (old → new). A '●' marks
    each column's level. Y-axis shows the high/low price. Text-only, diff-friendly.
    """
    vals = [float(v) for v in series.dropna().tail(width * 3)]
    if len(vals) < 3:
        return ""
    # Resample to at most ``width`` columns by simple striding (keep newest).
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(len(vals) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    grid = [[" "] * len(vals) for _ in range(height)]
    for col, v in enumerate(vals):
        # row 0 = top (high price); invert so higher price sits higher.
        lvl = int((v - lo) / span * (height - 1))
        row = (height - 1) - lvl
        grid[row][col] = "●"
    lines = []
    for r in range(height):
        if r == 0:
            axis = f"{hi:>8.2f} │"
        elif r == height - 1:
            axis = f"{lo:>8.2f} │"
        else:
            axis = " " * 8 + " │"
        lines.append(axis + "".join(grid[r]))
    lines.append(" " * 9 + "└" + "─" * len(vals))
    lines.append(" " * 10 + f"{len(vals)} bars (old → new)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _trend(close: pd.Series) -> tuple[str, float]:
    """Direction over ~20 bars via SMA20 slope, returned with a normalised slope."""
    if len(close) < 25:
        return "insufficient", 0.0
    sma = close.rolling(20).mean()
    now, prev = float(sma.iloc[-1]), float(sma.iloc[-20])
    if prev <= 0:
        return "sideways", 0.0
    chg = (now - prev) / prev
    if chg > 0.03:
        return "up", round(chg, 4)
    if chg < -0.03:
        return "down", round(chg, 4)
    return "sideways", round(chg, 4)


def _swing_structure(df: pd.DataFrame, lookback: int = 40) -> str:
    """Higher-highs/higher-lows (uptrend) vs lower-highs/lower-lows (downtrend)."""
    seg = df.tail(lookback)
    if len(seg) < 20:
        return "unclear"
    half = len(seg) // 2
    h1, h2 = seg["High"].iloc[:half], seg["High"].iloc[half:]
    l1, l2 = seg["Low"].iloc[:half], seg["Low"].iloc[half:]
    hh = float(h2.max()) > float(h1.max())
    hl = float(l2.min()) > float(l1.min())
    lh = float(h2.max()) < float(h1.max())
    ll = float(l2.min()) < float(l1.min())
    if hh and hl:
        return "higher-highs & higher-lows"
    if lh and ll:
        return "lower-highs & lower-lows"
    return "mixed/range-bound"


def _double_bottom(df: pd.DataFrame, lookback: int = 35, tol: float = 0.03) -> bool:
    """Two comparable troughs (within ``tol``) separated by a higher pivot."""
    lows = df["Low"].tail(lookback).reset_index(drop=True)
    if len(lows) < 15:
        return False
    i_min = int(lows.idxmin())
    v_min = float(lows.iloc[i_min])
    # look for a second trough at least 5 bars away, within tolerance
    for j in range(len(lows)):
        if abs(j - i_min) < 5:
            continue
        if abs(float(lows.iloc[j]) - v_min) <= tol * v_min:
            mid_lo, mid_hi = sorted((i_min, j))
            between = lows.iloc[mid_lo:mid_hi]
            if len(between) and float(between.max()) > v_min * (1 + tol):
                return True
    return False


def _double_top(df: pd.DataFrame, lookback: int = 35, tol: float = 0.03) -> bool:
    highs = df["High"].tail(lookback).reset_index(drop=True)
    if len(highs) < 15:
        return False
    i_max = int(highs.idxmax())
    v_max = float(highs.iloc[i_max])
    for j in range(len(highs)):
        if abs(j - i_max) < 5:
            continue
        if abs(float(highs.iloc[j]) - v_max) <= tol * v_max:
            mid_lo, mid_hi = sorted((i_max, j))
            between = highs.iloc[mid_lo:mid_hi]
            if len(between) and float(between.min()) < v_max * (1 - tol):
                return True
    return False


def detect(df: pd.DataFrame) -> dict:
    """Derive the chart-structure feature block + bullish ``pattern_score`` (0..1)."""
    close = df["Close"]
    last = float(close.iloc[-1])

    trend, slope = _trend(close)
    structure = _swing_structure(df)
    sup20 = float(df["Low"].tail(20).min())
    res20 = float(df["High"].tail(20).max())
    win52 = df.tail(252)
    hi52 = float(win52["High"].max())
    lo52 = float(win52["Low"].min())
    dist_hi = round((last - hi52) / hi52 * 100, 2) if hi52 else 0.0
    dist_lo = round((last - lo52) / lo52 * 100, 2) if lo52 else 0.0

    near_52w_high = dist_hi >= -3.0           # within 3% of the year high
    dbl_bottom = _double_bottom(df)
    dbl_top = _double_top(df)
    # A "breakout" needs room above: price at/over the 20-bar ceiling AND that ceiling
    # is not simply today's price (res20 == last == 52w high gives a circular, roomless
    # "breakout"). Require the close to clear the prior-19-bar high, leaving the latest
    # bar out of the reference.
    prior_high = float(df["High"].iloc[-20:-1].max()) if len(df) >= 21 else res20
    breakout = last >= res20 * 0.995 and last >= prior_high * 0.999

    # tightness of the last 10 bars (range as % of price) → consolidation
    rng10 = df.tail(10)
    band = (float(rng10["High"].max()) - float(rng10["Low"].min())) / last if last else 1
    consolidating = band < 0.05

    # extended/parabolic: far above the 20-day mean relative to recent range
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else last
    extended = sma20 > 0 and (last - sma20) / sma20 > 0.12

    tags: list[str] = []
    if trend == "up":
        tags.append("uptrend")
    elif trend == "down":
        tags.append("downtrend")
    else:
        tags.append("sideways")
    if structure.startswith("higher"):
        tags.append("higher-highs/higher-lows")
    if dbl_bottom:
        tags.append("double-bottom")
    if dbl_top:
        tags.append("double-top")
    if breakout:
        tags.append("20-day breakout")
    if near_52w_high:
        tags.append("near 52w high")
    if consolidating:
        tags.append("tight consolidation")
    if extended:
        tags.append("extended/parabolic")

    # Bullish pattern score, centred at 0.5.
    score = 0.5
    score += 0.15 if trend == "up" else (-0.15 if trend == "down" else 0.0)
    score += 0.10 if structure.startswith("higher") else (
        -0.10 if structure.startswith("lower") else 0.0)
    score += 0.10 if breakout else 0.0
    score += 0.08 if dbl_bottom else 0.0
    # Proximity to the 52w high is only constructive when NOT topping — at a double-top
    # it is resistance, not momentum, so the bonus is suppressed there.
    score += 0.05 if (near_52w_high and not dbl_top) else 0.0
    score -= 0.12 if dbl_top else 0.0
    score -= 0.08 if extended else 0.0          # blow-off risk
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "trend": trend,
        "trend_slope_pct": round(slope * 100, 2),
        "structure": structure,
        "support_20d": round(sup20, 2),
        "resistance_20d": round(res20, 2),
        "high_52w": round(hi52, 2),
        "low_52w": round(lo52, 2),
        "dist_from_52w_high_pct": dist_hi,
        "dist_from_52w_low_pct": dist_lo,
        "breakout": breakout,
        "double_bottom": dbl_bottom,
        "double_top": dbl_top,
        "consolidating": consolidating,
        "extended": extended,
        "tags": tags,
        "pattern_score": score,
        "summary": ", ".join(tags) if tags else "no clear pattern",
    }


def pattern_table(feats: dict) -> list[str]:
    """Markdown bullet lines describing the detected structure (for the .md report)."""
    return [
        f"  - Pattern: **{feats.get('summary','-')}** "
        f"(score {feats.get('pattern_score','-')})",
        f"  - Trend: {feats.get('trend','-')} "
        f"({feats.get('trend_slope_pct','-'):+}% / 20d), "
        f"structure: {feats.get('structure','-')}",
        f"  - Support ₹{feats.get('support_20d','-')} · "
        f"Resistance ₹{feats.get('resistance_20d','-')} · "
        f"52w ₹{feats.get('low_52w','-')}–₹{feats.get('high_52w','-')} "
        f"({feats.get('dist_from_52w_high_pct','-')}% from high)",
    ]
