"""Loss autopsy + approved lessons.

1. autopsy(trade) looks at the entry snapshot saved with a losing trade and lists which risk conditions were present.
2. rule_stats() checks each condition on ALL other closed trades (no manual calculation) and says how much data backs it.
3. The user APPROVES a finding -> it is stored as a lesson (WARN by default, BLOCK only if the user switches it).
4. active_hits() compares a new signal's snapshot with the approved lessons.

Findings are probable causes, not proof. Small samples are labelled as such. Nothing here changes the signal.
"""
import numpy as np
import pandas as pd

MIN_SAMPLE = 30


def as_bool(v, default=True):
    """Tables can hand back True/1/'True'/'1'/'false'... normalise them."""
    if isinstance(v, str):
        t = v.strip().lower()
        return t in ("true", "1", "yes", "y") if t else default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return bool(v)


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _g(row, k):
    try:
        return row.get(k) if hasattr(row, "get") else row[k]
    except Exception:
        return None


def _dt_minutes(entry_dt):
    try:
        d = pd.to_datetime(entry_dt, dayfirst=True, errors="coerce")
        return None if pd.isna(d) else d.hour * 60 + d.minute
    except Exception:
        return None


# each check: (row, side, entry_dt) -> (present True/False/None-when-data-missing, evidence text)
def _weak_trend(r, side, dt):
    a = _f(_g(r, "ADX"))
    return (None, "") if a is None else (a < 18, f"ADX was {a:.1f} (below 18 = weak / sideways trend)")


def _low_volume(r, side, dt):
    v = _f(_g(r, "RelativeVolume"))
    return (None, "") if v is None else (v < 0.8, f"Relative volume was {v:.2f} (below 0.8 = thin volume)")


def _mtf_conflict(r, side, dt):
    m = str(_g(r, "MTFTrendStatus") or "")
    if not m or m.lower() in ("unavailable", "nan"):
        return None, ""
    want = "bullish" if side == "BUY" else "bearish"
    return (want not in m.lower(), f"Higher timeframes were '{m}' while the trade was a {side}")


def _poor_rr(r, side, dt):
    v = _f(_g(r, "RiskRewardRatio"))
    return (None, "") if v is None else (v < 1.5, f"Risk:reward was {v:.2f} (below 1.5)")


def _exhaustion(r, side, dt):
    v = _f(_g(r, "RSI"))
    if v is None:
        return None, ""
    return ((v > 75) if side == "BUY" else (v < 25)), f"RSI was {v:.0f} at a {side} entry (stretched, move may be exhausted)"


def _against_vwap(r, side, dt):
    v = _f(_g(r, "DistanceVWAPPct"))
    if v is None:
        return None, ""
    bad = (v < 0) if side == "BUY" else (v > 0)
    return bad, f"Price was {abs(v):.2f}% {'below' if v < 0 else 'above'} VWAP at a {side} entry"


def _against_slope(r, side, dt):
    v = _f(_g(r, "TrendSlope"))
    if v is None:
        return None, ""
    return ((v < 0) if side == "BUY" else (v > 0)), f"Trend slope was {v:+.3f}, against the {side}"


def _late_entry(r, side, dt):
    v = _f(_g(r, "DistanceEMA20Pct"))
    return (None, "") if v is None else (abs(v) > 1.5, f"Price was {abs(v):.2f}% away from EMA20 (late / stretched entry)")


def _low_conf(r, side, dt):
    v = _f(_g(r, "ConfidenceAtEntry"))
    return (None, "") if v is None else (v < 55, f"Setup confidence was only {v:.0f}%")


def _tight_stop(r, side, dt):
    e, s, a = _f(_g(r, "EntryPrice")), _f(_g(r, "StopLoss")), _f(_g(r, "ATR"))
    if None in (e, s, a) or a <= 0:
        return None, ""
    d = abs(e - s) / a
    return d < 0.8, f"Stop was only {d:.2f} ATR from entry (very tight, easily hit by normal noise)"


def _time_window(r, side, dt):
    m = _dt_minutes(dt if dt is not None else _g(r, "DateTime"))
    if m is None:
        return None, ""
    early, late = m < 9 * 60 + 30, m >= 15 * 60
    return (early or late), f"Entry time {m // 60:02d}:{m % 60:02d} ({'first 15 minutes - noisy open' if early else 'last 30 minutes - closing noise' if late else 'normal hours'})"


CHECKS = {
    "weak_trend": ("Weak / sideways trend (ADX < 18)", _weak_trend),
    "low_volume": ("Thin volume (relative volume < 0.8)", _low_volume),
    "mtf_conflict": ("Higher timeframes not aligned with the trade", _mtf_conflict),
    "poor_rr": ("Poor risk:reward (< 1.5)", _poor_rr),
    "exhaustion": ("Stretched entry (RSI extreme)", _exhaustion),
    "against_vwap": ("Entered against VWAP", _against_vwap),
    "against_slope": ("Entered against the trend slope", _against_slope),
    "late_entry": ("Late entry (far from EMA20)", _late_entry),
    "low_conf": ("Low setup confidence (< 55%)", _low_conf),
    "tight_stop": ("Stop too tight (< 0.8 ATR)", _tight_stop),
    "time_window": ("Entered in the noisy open / close", _time_window),
}


def _side(row):
    return str(_g(row, "Side") or _g(row, "Signal") or "BUY").upper()


def check_present(code, row, side=None, entry_dt=None):
    side = side or _side(row)
    try:
        return CHECKS[code][1](row, side, entry_dt)
    except Exception:
        return None, ""


def closed_trades(journal):
    if journal is None or journal.empty or "Status" not in journal.columns:
        return pd.DataFrame()
    c = journal[journal["Status"].astype(str).str.upper() == "CLOSED"].copy()
    c["_pnl"] = pd.to_numeric(c.get("PnL", 0), errors="coerce").fillna(0.0)
    return c


def rule_stats(code, journal):
    """How trades with this condition did vs trades without it (all closed trades, option rows excluded)."""
    c = closed_trades(journal)
    if c.empty:
        return {"n": 0, "wins": 0, "win_rate": None, "pnl": 0.0, "n_other": 0, "win_rate_other": None, "enough": False}
    m_n = m_w = o_n = o_w = 0
    pnl = 0.0
    for _, r in c.iterrows():
        pres, _ = check_present(code, r)
        if pres is None:
            continue
        if pres:
            m_n += 1; m_w += int(r["_pnl"] > 0); pnl += float(r["_pnl"])
        else:
            o_n += 1; o_w += int(r["_pnl"] > 0)
    return {"n": m_n, "wins": m_w, "win_rate": (100.0 * m_w / m_n) if m_n else None, "pnl": round(pnl, 2),
            "n_other": o_n, "win_rate_other": (100.0 * o_w / o_n) if o_n else None, "enough": m_n >= MIN_SAMPLE}


def stats_text(st_):
    if not st_["n"]:
        return "No other closed trade has this condition recorded."
    t = (f"{st_['n']} closed trade(s) had this: win rate {st_['win_rate']:.0f}%, total P&L ₹{st_['pnl']:,.0f}"
         + (f"; without it {st_['win_rate_other']:.0f}% over {st_['n_other']} trade(s)." if st_["win_rate_other"] is not None else "."))
    return t + ("" if st_["enough"] else f"  ⚠ Only {st_['n']} trades (< {MIN_SAMPLE}): a hint, not proof.")


def autopsy(trade, journal=None):
    """List the risk conditions present at entry of one losing trade."""
    side = _side(trade)
    out = []
    for code, (title, _fn) in CHECKS.items():
        pres, ev = check_present(code, trade, side)
        if pres:
            item = {"code": code, "title": title, "evidence": ev, "side": side}
            if journal is not None:
                item["stats"] = rule_stats(code, journal)
            out.append(item)
    return out


LESSON_COLS = ["LessonID", "CreatedAt", "Code", "Side", "Title", "Evidence", "Mode", "Active", "SourceTrade"]


def new_lesson(code, side, evidence, source="", mode="WARN", now=None):
    now = now or pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")
    return {"LessonID": f"{code}-{side}-{pd.Timestamp.now().strftime('%y%m%d%H%M%S%f')[-9:]}", "CreatedAt": now, "Code": code, "Side": side,
            "Title": CHECKS[code][0], "Evidence": evidence, "Mode": mode, "Active": True, "SourceTrade": str(source)}


def add_lesson(lessons, lesson):
    df = lessons.copy() if lessons is not None and not lessons.empty else pd.DataFrame(columns=LESSON_COLS)
    dup = (df["Code"] == lesson["Code"]) & (df["Side"] == lesson["Side"]) if not df.empty else pd.Series(dtype=bool)
    if not df.empty and dup.any():
        df.loc[dup, "Active"] = True            # already approved: just make sure it is active
        return df, False
    return pd.concat([df, pd.DataFrame([lesson])], ignore_index=True), True


def active_hits(snapshot, side, lessons, entry_dt=None):
    """Approved lessons that match a new signal's snapshot. [{'id','title','evidence','mode'}]"""
    hits = []
    if lessons is None or lessons.empty or not isinstance(snapshot, dict):
        return hits
    for _, l in lessons.iterrows():
        if not as_bool(l.get("Active", True)) or str(l.get("Side", "")).upper() != str(side).upper() or l.get("Code") not in CHECKS:
            continue
        pres, ev = check_present(l["Code"], snapshot, str(side).upper(), entry_dt)
        if pres:
            hits.append({"id": l["LessonID"], "title": str(l.get("Title", "")), "evidence": ev, "mode": str(l.get("Mode", "WARN")).upper()})
    return hits
