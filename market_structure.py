"""Live, non-repainting BOS / CHOCH engine.

* A swing high/low at bar j is only treated as known once `right` bars after it have CLOSED (default 3), exactly like a
  centred pivot would need - so nothing here ever uses a bar that had not closed when the decision was made.
* A break = the first CLOSE beyond the most recent confirmed swing level. Each level can only be broken once.
* BOS  = break in the direction of the prior structure (or with no clear prior structure).
  CHOCH = break against the prior structure (HH-HL then a close below the last swing low, or LH-LL then a close above the last swing high).
* Points (+2.0 BOS / +2.5 CHOCH) are awarded only while the break is within the last `max_age` candles AND price is still
  beyond the level (a break that already failed gets nothing).
* The whole thing is ONE causal pass: the answer at bar t computed from df[:t+1] is identical to the answer computed
  from a longer frame (tests/test_regression.py checks this).

`dedupe_institutional()` removes the break credit that institutional_upgrades.detect_market_structure already gives
(it scores "price is beyond the last swing" with no notion of when the break happened) so one break is never counted twice.
"""
import numpy as np
import pandas as pd

POINTS = {"BOS": 2.0, "CHOCH": 2.5}
LEFT, RIGHT, MAX_AGE = 3, 3, 5


def _arr(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def find_breaks(df, left=LEFT, right=RIGHT):
    """All BOS/CHOCH events, found with a single forward pass. Returns a list of dicts (ordered by bar)."""
    if df is None or len(df) < left + right + 3 or not {"High", "Low", "Close"}.issubset(df.columns):
        return []
    h, l, c = _arr(df, "High"), _arr(df, "Low"), _arr(df, "Close")
    idx = df.index
    n = len(df)
    piv_h, piv_l = [], []                    # (bar, price), appended in the order they became KNOWN
    spent_h, spent_l = set(), set()
    events = []
    for k in range(n):
        j = k - right                         # the bar that just became confirmable
        if j >= left:
            wh, wl = h[j - left:k + 1], l[j - left:k + 1]
            if np.isfinite(h[j]) and np.isfinite(wh).all() and h[j] == wh.max():
                piv_h.append((j, float(h[j])))
            if np.isfinite(l[j]) and np.isfinite(wl).all() and l[j] == wl.min():
                piv_l.append((j, float(l[j])))
        if k < 1 or not np.isfinite(c[k]):
            continue
        up = len(piv_h) >= 2 and len(piv_l) >= 2 and piv_h[-1][1] > piv_h[-2][1] and piv_l[-1][1] > piv_l[-2][1]
        down = len(piv_h) >= 2 and len(piv_l) >= 2 and piv_h[-1][1] < piv_h[-2][1] and piv_l[-1][1] < piv_l[-2][1]
        if piv_h and piv_h[-1][0] not in spent_h and c[k] > piv_h[-1][1]:
            lvl_bar, lvl = piv_h[-1]
            events.append({"bar": k, "time": idx[k], "kind": "CHOCH" if down else "BOS", "direction": "bull", "level": lvl, "level_bar": lvl_bar, "close": float(c[k])})
            spent_h.update(b for b, p in piv_h if p < c[k])
        elif piv_l and piv_l[-1][0] not in spent_l and c[k] < piv_l[-1][1]:
            lvl_bar, lvl = piv_l[-1]
            events.append({"bar": k, "time": idx[k], "kind": "CHOCH" if up else "BOS", "direction": "bear", "level": lvl, "level_bar": lvl_bar, "close": float(c[k])})
            spent_l.update(b for b, p in piv_l if p > c[k])
    return events


def current_break(df, left=LEFT, right=RIGHT, max_age=MAX_AGE, require_hold=True):
    """The break (if any) that counts for the latest candle."""
    out = {"available": False, "active": False, "kind": None, "direction": None, "level": None, "bars_ago": None,
           "time": None, "points": 0.0, "bullish": 0.0, "bearish": 0.0, "reason": "Not enough candles"}
    if df is None or len(df) < left + right + 8 or not {"High", "Low", "Close"}.issubset(df.columns):
        return out
    out["available"] = True
    out["reason"] = f"No BOS/CHOCH in the last {max_age} candles"
    events = find_breaks(df, left, right)
    t = len(df) - 1
    last_close = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])
    recent = [e for e in events if e["bar"] >= t - (max_age - 1)]
    if not recent:
        return out
    e = recent[-1]
    if require_hold and not ((e["direction"] == "bull" and last_close > e["level"]) or (e["direction"] == "bear" and last_close < e["level"])):
        out["reason"] = f"{e['kind']} {e['direction']} at {e['level']:.2f} already failed (price back through the level)"
        return out
    pts = POINTS[e["kind"]]
    out.update(active=True, kind=e["kind"], direction=e["direction"], level=e["level"], bars_ago=t - e["bar"], time=e["time"], points=pts,
               bullish=pts if e["direction"] == "bull" else 0.0, bearish=pts if e["direction"] == "bear" else 0.0,
               reason=f"{'Bullish' if e['direction'] == 'bull' else 'Bearish'} {e['kind']}: close beyond confirmed swing {e['level']:.2f}, {t - e['bar']} candle(s) ago")
    return out


def dedupe_institutional(components, weight=0.6):
    """Remove the BOS/CHOCH credit from institutional_upgrades' 'Market Structure' component (keep the plain trend part).
    '(BOS)' reasons keep the +-3 raw trend score, 'CHOCH' reasons keep 0 (no trend yet). Returns a new list."""
    out = []
    for comp in components or []:
        c = dict(comp)
        if c.get("name") == "Market Structure":
            reason = str(c.get("reason", ""))
            if "CHOCH" in reason or "BOS" in reason:
                sign = 1.0 if c.get("net", 0.0) >= 0 else -1.0
                raw = 0.0 if "CHOCH" in reason else 3.0 * sign
                net = raw * weight
                c.update(net=net, bullish=max(net, 0.0), bearish=max(-net, 0.0), reason=reason + " [break credit moved to the dated BOS/CHOCH engine]")
        out.append(c)
    return out
