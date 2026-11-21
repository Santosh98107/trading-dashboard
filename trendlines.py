"""trendlines.py - automatic trendlines (rising support, falling resistance) from confirmed swing points.

Pure numpy/pandas. Works only with candles up to the last row, so a live call and a backtest call on the same truncated data agree
(a swing point is "confirmed" only after `right` later bars, so the newest bars are never pivots).

Method
  1. Confirmed swing highs / lows (a pivot is the extreme of left+right+1 bars).
  2. Candidate lines through pairs of recent swing lows (rising = support) or swing highs (falling = resistance).
  3. A line is valid if no close before the last `break_window` bars violated it by more than `tol` (0.25 ATR).
  4. Best line = most touches (other pivots within `tol` of the line), then most recent anchor, then longest span.
  5. Status: INTACT, or BROKEN (a close beyond the line by `tol` inside the last `break_window` bars), plus RETEST when price returns to a broken line.
A line is "confirmed" with 3+ touches; with 2 touches it is only "provisional" and is drawn but never scored.
All thresholds are starting values chosen by design, not fitted to any data.
"""
import numpy as np
import pandas as pd

TOL_ATR = 0.25
BREAK_WINDOW = 8
MIN_GAP_BARS = 5
MAX_DIST_ATR = 10.0


def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _atr(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr = pd.Series(tr).rolling(n, min_periods=5).mean().to_numpy()
    a = tr[-1] if len(tr) else np.nan
    return float(a) if np.isfinite(a) and a > 0 else max(abs(float(c[-1])) * 0.001, 0.01)


def swing_points(h, l, left=4, right=4):
    highs, lows = [], []
    n = len(h)
    for i in range(left, n - right):
        if np.isfinite(h[i]) and h[i] == np.nanmax(h[i - left:i + right + 1]):
            highs.append((i, float(h[i])))
        if np.isfinite(l[i]) and l[i] == np.nanmin(l[i - left:i + right + 1]):
            lows.append((i, float(l[i])))
    return highs, lows


def _best_line(pivs, closes, kind, atr, n):
    tol = TOL_ATR * atr
    pts = pivs[-7:]
    best, best_key = None, None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (i1, p1), (i2, p2) = pts[a], pts[b]
            if i2 - i1 < MIN_GAP_BARS:
                continue
            slope = (p2 - p1) / (i2 - i1)
            if (kind == "support" and slope <= 0) or (kind == "resistance" and slope >= 0):
                continue
            xs = np.arange(i1, n)
            line = p1 + slope * (xs - i1)
            seg_c = closes[i1:n]
            valid_to = max(0, (n - i1) - BREAK_WINDOW)          # older closes must respect the line
            old = seg_c[:valid_to]
            old_line = line[:valid_to]
            if len(old) and (np.any(old < old_line - tol) if kind == "support" else np.any(old > old_line + tol)):
                continue
            value_now = float(line[-1])
            if abs(float(closes[-1]) - value_now) / atr > MAX_DIST_ATR:
                continue
            touches = sum(1 for (j, pj) in pts if j >= i1 and abs(pj - (p1 + slope * (j - i1))) <= tol)
            key = (touches, i2, i2 - i1)
            if best_key is None or key > best_key:
                best_key, best = key, (i1, p1, i2, p2, slope, touches, value_now, line, seg_c)
    return best


def detect_trendlines(df, lookback=120, pivot=4):
    """Return {"available", "atr", "lines": [...]}; each line has kind, status, touches, confirmed, value_now, dist_atr, anchors."""
    out = {"available": False, "atr": None, "lines": []}
    try:
        if df is None or len(df) < max(30, 2 * pivot + 12) or not {"High", "Low", "Close"}.issubset(df.columns):
            return out
        d = df.tail(int(lookback))
        off = len(df) - len(d)
        h, l, c = _col(d, "High"), _col(d, "Low"), _col(d, "Close")
        n = len(c)
        atr = _atr(h, l, c)
        highs, lows = swing_points(h, l, pivot, pivot)
        out.update(available=True, atr=atr)
        tol = TOL_ATR * atr
        for kind, pivs in (("support", lows), ("resistance", highs)):
            best = _best_line(pivs, c, kind, atr, n)
            if best is None:
                continue
            i1, p1, i2, p2, slope, touches, value_now, line, seg_c = best
            win = np.arange(max(i1, n - BREAK_WINDOW), n)
            rel = c[win] - (p1 + slope * (win - i1))
            viol = np.where(rel < -tol)[0] if kind == "support" else np.where(rel > tol)[0]
            status, bars_since = "INTACT", None
            retest = False
            if len(viol):
                status = "BROKEN"
                bars_since = int(n - 1 - win[viol[0]])
                retest = bool(bars_since > 0 and abs(c[-1] - value_now) <= tol)
            out["lines"].append({
                "kind": kind, "status": status, "touches": int(touches), "confirmed": bool(touches >= 3),
                "slope": float(slope), "value_now": float(value_now), "dist_atr": float((c[-1] - value_now) / atr),
                "bars_since_break": bars_since, "retest": retest,
                "i1": int(i1 + off), "p1": float(p1), "i2": int(i2 + off), "p2": float(p2), "last": int(n - 1 + off),
            })
    except Exception:
        return {"available": False, "atr": None, "lines": []}
    return out


def trendline_evidence(res, max_points=0.75):
    """(bullish, bearish, reasons) evidence from CONFIRMED lines only. Small, capped; the caller decides whether to use it."""
    bull = bear = 0.0
    why = []
    if not res or not res.get("available"):
        return 0.0, 0.0, why
    for ln in res["lines"]:
        if not ln["confirmed"]:
            continue
        d, st_ = ln["dist_atr"], ln["status"]
        if ln["kind"] == "support":
            if st_ == "INTACT" and 0 <= d <= 0.6:
                bull += 0.5; why.append(f"price at rising support ({ln['touches']} touches) (+0.5 bullish)")
            elif st_ == "BROKEN" and ln["retest"]:
                bear += 0.5; why.append("broken support retested from below (+0.5 bearish)")
            elif st_ == "BROKEN" and (ln["bars_since_break"] or 0) <= 3:
                bear += 0.75; why.append(f"rising support broken ({ln['touches']} touches) (+0.75 bearish)")
        else:
            if st_ == "INTACT" and -0.6 <= d <= 0:
                bear += 0.5; why.append(f"price at falling resistance ({ln['touches']} touches) (+0.5 bearish)")
            elif st_ == "BROKEN" and ln["retest"]:
                bull += 0.5; why.append("broken resistance retested from above (+0.5 bullish)")
            elif st_ == "BROKEN" and (ln["bars_since_break"] or 0) <= 3:
                bull += 0.75; why.append(f"falling resistance broken ({ln['touches']} touches) (+0.75 bullish)")
    return min(bull, max_points), min(bear, max_points), why


def describe(res):
    """Short plain-text lines for the UI."""
    if not res or not res.get("available") or not res.get("lines"):
        return ["No valid trendline right now (needs two swing points that price has respected)."]
    rows = []
    for ln in res["lines"]:
        nm = "Rising support" if ln["kind"] == "support" else "Falling resistance"
        tag = "confirmed" if ln["confirmed"] else "provisional (2 touches, not scored)"
        st_ = ln["status"] + (f", {ln['bars_since_break']} bars ago" if ln["bars_since_break"] is not None else "") + (", retesting" if ln["retest"] else "")
        rows.append(f"{nm}: {ln['value_now']:,.2f} now | price is {ln['dist_atr']:+.2f} ATR from it | {ln['touches']} touches ({tag}) | {st_}")
    return rows
