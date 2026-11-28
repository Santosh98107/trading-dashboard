"""Trade plan, chase check and open-trade flip alert.

Everything here is a pure function of the candle table (no Streamlit, no network) and is DISPLAY ONLY, except
`extension_atr()` which the optional "anti-chase guard" in analyze_signal can use (default OFF).

Why: the signal engine reacts after a move has started. Professional traders plan the levels BEFORE price gets there:
"buy above X (or on a pullback into Y), stop Z, targets T1/T2/T3". build_plan() prepares exactly that for both sides from the
levels the app already knows (support / resistance, swing high-low, previous-day high-low, opening range, VWAP, EMA20, ATR).

Stop and targets follow the same convention the app already uses for its own signals: stop = risk R, T1/T2/T3 = 2R / 3R / 4R.
The stop here is placed beyond the recent swing (clamped between 0.8 and 2.0 ATR), instead of a flat 1 ATR.
"""
import re

import numpy as np
import pandas as pd

TARGET_R = (2.0, 3.0, 4.0)          # same as the engine's own signals
UP_NAMES = ("Resistance", "Swing high", "Prev-day high", "Opening-range high")
DOWN_NAMES = ("Support", "Swing low", "Prev-day low", "Opening-range low")


def _f(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def tf_minutes(timeframe):
    t = str(timeframe).strip().lower()
    if t.endswith("m") and t[:-1].isdigit():
        return float(t[:-1])
    if t.endswith("h") and t[:-1].isdigit():
        return float(t[:-1]) * 60.0
    return None                       # daily / unknown: session levels make no sense


def _atr(df):
    a = _f(df["ATR"].iloc[-1]) if "ATR" in df.columns else np.nan
    if np.isfinite(a) and a > 0:
        return a
    try:
        tr = (df["High"] - df["Low"]).tail(14).astype(float)
        a = float(tr.mean())
        return a if np.isfinite(a) and a > 0 else np.nan
    except Exception:
        return np.nan


def session_levels(df, timeframe, or_minutes=15):
    """Previous-day high/low and today's opening range. Empty dict for daily data."""
    out = {}
    try:
        if tf_minutes(timeframe) is None or df is None or df.empty:
            return out
        idx = pd.DatetimeIndex(df.index)
        days = idx.normalize()
        cur = days[-1]
        prev_days = days[days < cur].unique()
        if len(prev_days):
            p = df[days == prev_days[-1]]
            out["Prev-day high"], out["Prev-day low"] = _f(p["High"].max()), _f(p["Low"].min())
        today = df[days == cur]
        first = pd.Timestamp(today.index[0])
        win = today[(pd.DatetimeIndex(today.index) - first) < pd.Timedelta(minutes=or_minutes)]
        if len(today) and (pd.Timestamp(today.index[-1]) - first) >= pd.Timedelta(minutes=or_minutes):   # only once the range is complete
            out["Opening-range high"], out["Opening-range low"] = _f(win["High"].max()), _f(win["Low"].min())
    except Exception:
        return {}
    return {k: v for k, v in out.items() if np.isfinite(v)}


def extension_atr(df, side):
    """How far price is stretched from EMA20 in the trade direction, in ATR. 2.0 = two ATR beyond EMA20. nan when unknown."""
    try:
        atr = _atr(df)
        e = _f(df["EMA20"].iloc[-1]) if "EMA20" in df.columns else np.nan
        c = _f(df["Close"].iloc[-1])
        if not (np.isfinite(atr) and np.isfinite(e) and np.isfinite(c)):
            return np.nan
        return ((c - e) if str(side).upper() == "BUY" else (e - c)) / atr
    except Exception:
        return np.nan


def run_info(df, side):
    """How many candles in a row have already moved in the trade direction, and how far (in ATR)."""
    try:
        c = df["Close"].astype(float).to_numpy()
        s = 1.0 if str(side).upper() == "BUY" else -1.0
        n = 0
        while n < len(c) - 1 and s * (c[-1 - n] - c[-2 - n]) > 0:
            n += 1
        atr = _atr(df)
        move = s * (c[-1] - c[-1 - n]) if n else 0.0
        return {"candles": int(n), "move": float(move), "move_atr": float(move / atr) if np.isfinite(atr) and atr > 0 else np.nan}
    except Exception:
        return {"candles": 0, "move": 0.0, "move_atr": np.nan}


def chase_info(df, side, ext_limit=2.0):
    """Is the signal already late? Late = price stretched >= ext_limit ATR from EMA20, or 3+ candles in a row that moved >= 1.5 ATR."""
    ext = extension_atr(df, side)
    run = run_info(df, side)
    late = bool((np.isfinite(ext) and ext >= ext_limit) or (run["candles"] >= 3 and np.isfinite(run["move_atr"]) and run["move_atr"] >= 1.5))
    if late:
        text = (f"LATE: {run['candles']} candle(s) already moved {run['move_atr']:.1f} ATR in this direction and price is {ext:.1f} ATR from EMA20. "
                "Chasing here usually means a bad entry; wait for a pullback.") if np.isfinite(ext) and np.isfinite(run["move_atr"]) else "LATE: price is stretched."
    else:
        text = (f"Not stretched: {run['candles']} candle(s) in a row, {run['move_atr']:.1f} ATR, {ext:.1f} ATR from EMA20."
                if np.isfinite(ext) and np.isfinite(run["move_atr"]) else "Not enough data to judge the move.")
    return {"late": late, "extension_atr": ext, "run": run, "text": text}


def _side_plan(side, price, atr, df, levels, ext_atr, trigger_buffer_atr, stop_floor_atr, stop_cap_atr, swing_bars):
    s = 1.0 if side == "BUY" else -1.0
    ahead_names = UP_NAMES if s > 0 else DOWN_NAMES              # levels that must break for a breakout entry
    behind_names = DOWN_NAMES if s > 0 else UP_NAMES
    tiny = 0.05 * atr

    ahead = [(n, v) for n, v in levels.items() if n in ahead_names and s * (v - price) > tiny]
    trig_name, trig_val = (min(ahead, key=lambda t: s * (t[1] - price)) if ahead else (None, None))
    trigger = None if trig_val is None else float(trig_val + s * trigger_buffer_atr * atr)

    # pullback zone: nearest dynamic level (VWAP / EMA20) or a level price has already broken (now acting the other way), behind price
    back = [(n, v) for n, v in levels.items() if n in ("VWAP", "EMA20") and s * (price - v) > tiny]
    back += [(n, v) for n, v in levels.items() if n in ahead_names and 0 <= s * (price - v) <= 2.0 * atr and s * (price - v) > tiny]
    zone = None
    if back:
        zn, zv = min(back, key=lambda t: s * (price - t[1]))
        lo, hi = (zv - 0.25 * atr, zv) if s > 0 else (zv, zv + 0.25 * atr)
        zone = {"name": zn, "low": float(lo), "high": float(hi), "mid": float((lo + hi) / 2.0)}

    ext = s * (price - _f(levels.get("EMA20"))) / atr if np.isfinite(_f(levels.get("EMA20"))) else np.nan
    extended = bool(np.isfinite(ext) and ext >= ext_atr)
    in_zone = bool(zone and zone["low"] - 0.1 * atr <= price <= zone["high"] + 0.1 * atr)

    if extended:
        status = "EXTENDED"
    elif trigger is None:
        status = "BROKEN_OUT"
    else:
        status = "WAITING"

    entry = trigger if status == "WAITING" else (zone["mid"] if zone else None)
    entry_kind = "breakout" if status == "WAITING" else ("pullback" if zone else None)
    plan = {"side": side, "status": status, "trigger": trigger, "trigger_level": trig_name, "zone": zone, "in_zone": in_zone,
            "extension_atr": float(ext) if np.isfinite(ext) else None, "entry": None, "entry_kind": entry_kind}

    if entry is None:
        plan["note"] = ("Price has already moved beyond every known level and no pullback level is near. Wait; do not chase."
                        if status != "WAITING" else "No clear level ahead.")
        return plan

    try:
        w = df.tail(swing_bars)
        swing = float(w["Low"].min()) if s > 0 else float(w["High"].max())
    except Exception:
        swing = entry - s * stop_floor_atr * atr
    stop = (min(swing - 0.1 * atr, entry - stop_floor_atr * atr) if s > 0 else max(swing + 0.1 * atr, entry + stop_floor_atr * atr))
    stop = max(stop, entry - stop_cap_atr * atr) if s > 0 else min(stop, entry + stop_cap_atr * atr)
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        plan["note"] = "Could not size a stop."
        return plan
    targets = [float(entry + s * r * risk) for r in TARGET_R]
    in_way = sorted([(n, float(v)) for n, v in levels.items() if n in ahead_names and 0 < s * (v - entry) < s * (targets[0] - entry)],
                    key=lambda t: s * (t[1] - entry))
    plan.update(entry=float(entry), stop=float(stop), risk=float(risk), risk_pct=float(risk / entry * 100.0), targets=targets,
                rr=list(TARGET_R), levels_in_the_way=in_way)
    if in_way:
        n0, v0 = in_way[0]
        plan["note"] = f"{n0} at {v0:,.2f} sits only {abs(v0 - entry) / risk:.1f}R from the entry, before T1: room for the first target is limited."
    else:
        plan["note"] = "No known level between entry and T1."
    return plan


def build_plan(df, timeframe="3m", resistance=None, support=None, or_minutes=15, ext_atr=2.0, trigger_buffer_atr=0.1,
               stop_floor_atr=0.8, stop_cap_atr=2.0, swing_bars=10, level_bars=20):
    """Plan for BUY and SELL from the current candles. Never raises."""
    out = {"available": False, "reason": "Not enough candles for a plan"}
    try:
        if df is None or len(df) < 30 or not {"High", "Low", "Close"} <= set(df.columns):
            return out
        price = _f(df["Close"].iloc[-1])
        atr = _atr(df)
        if not (np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0):
            out["reason"] = "Price or ATR is not available"
            return out
        levels = {}
        r, sp = _f(resistance), _f(support)
        if np.isfinite(r):
            levels["Resistance"] = r
        if np.isfinite(sp):
            levels["Support"] = sp
        prior = df.iloc[-(level_bars + 1):-1]
        if len(prior):
            levels["Swing high"], levels["Swing low"] = _f(prior["High"].max()), _f(prior["Low"].min())
        levels.update(session_levels(df, timeframe, or_minutes))
        for col, name in (("VWAP", "VWAP"), ("EMA20", "EMA20")):
            if col in df.columns and np.isfinite(_f(df[col].iloc[-1])):
                levels[name] = _f(df[col].iloc[-1])
        levels = {k: v for k, v in levels.items() if np.isfinite(v)}
        common = dict(price=price, atr=atr, df=df, levels=levels, ext_atr=ext_atr, trigger_buffer_atr=trigger_buffer_atr,
                      stop_floor_atr=stop_floor_atr, stop_cap_atr=stop_cap_atr, swing_bars=swing_bars)
        return {"available": True, "price": price, "atr": atr, "levels": levels,
                "buy": _side_plan("BUY", **common), "sell": _side_plan("SELL", **common)}
    except Exception as exc:
        out["reason"] = f"Trade plan failed: {type(exc).__name__}: {exc}"
        return out


def plan_text(plan, atr, price):
    """One plain sentence for a side plan."""
    if not plan:
        return ""
    side = plan["side"]
    if plan.get("entry") is None:
        return plan.get("note", "")
    d = (plan["entry"] - price) / atr
    if plan["status"] == "WAITING":
        head = f"{side} above {plan['trigger']:,.2f} ({plan['trigger_level']} + buffer), {abs(d):.1f} ATR {'above' if d > 0 else 'below'} the price now"
    else:
        z = plan["zone"]
        head = (f"{plan['status'].replace('_', ' ').title()}: do not chase. {side} on a pullback into {z['low']:,.2f} - {z['high']:,.2f} ({z['name']})"
                + (" (price is in the zone now)" if plan["in_zone"] else ""))
    t = plan["targets"]
    return f"{head}. Stop {plan['stop']:,.2f} ({plan['risk_pct']:.2f}% / {plan['risk']:,.1f} pts). Targets {t[0]:,.2f} / {t[1]:,.2f} / {t[2]:,.2f}."


# ---------------- chase tracking across refreshes ----------------
def track_signal(state, key, signal_text, price, now_ts):
    """Remember when THIS page first saw the current BUY/SELL signal. Returns (age_seconds, price_when_first_seen) or (None, None).
    `state` is any dict (st.session_state). A new or cleared signal resets the clock."""
    sig = str(signal_text or "").upper()
    side = "BUY" if "BUY" in sig else "SELL" if "SELL" in sig else None
    rec = state.get(key)
    if side is None:
        state[key] = None
        return None, None
    if not rec or rec.get("side") != side:
        rec = {"side": side, "t": float(now_ts), "price": float(price)}
        state[key] = rec
    return max(0.0, float(now_ts) - rec["t"]), rec["price"]


# ---------------- open trade vs current signal ----------------
_OPT_RE = re.compile(r"Option paper BUY (CE|PE)", re.I)


def open_trade_direction(row):
    """(underlying, BUY|SELL) of an open journal row. Option paper rows are long CE (bullish) or long PE (bearish)."""
    sym = str(row.get("Symbol", "")).strip()
    notes = str(row.get("Notes", "") or "")
    if "[ikey=" in notes:
        m = _OPT_RE.search(notes)
        return (sym.split()[0].upper() if sym else ""), ("SELL" if m and m.group(1).upper() == "PE" else "BUY")
    side = str(row.get("Side", "")).upper()
    return sym.upper(), ("SELL" if side == "SELL" else "BUY")


def flip_alerts(open_rows, symbol, signal_text):
    """Messages for open trades on `symbol` whose direction no longer matches the live signal. Display only; nothing is exited."""
    sig = str(signal_text or "").upper()
    now_side = "BUY" if "BUY" in sig else "SELL" if "SELL" in sig else None
    out = []
    for row in open_rows or []:
        und, side = open_trade_direction(row)
        if und != str(symbol).upper() or side == now_side:
            continue
        label = row.get("Symbol", symbol)
        if now_side is None:
            out.append({"level": "warning", "text": f"Open {side} on {label}: the live signal is now NO TRADE. The setup that justified the trade is gone. "
                                                    "Check your stop and targets (nothing is exited automatically)."})
        else:
            out.append({"level": "error", "text": f"Open {side} on {label}, but the live signal is now {sig}. The engine has turned against this trade. "
                                                  "Review it now (nothing is exited automatically)."})
    return out
