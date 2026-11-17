"""trade_health.py - Trend Health, Exit Score and trend-based trailing advice for one open (or planned) trade.

Pure pandas/numpy, no Streamlit, so the page AND the background worker can use it and tests can run it on fake candles.

Exit Score (0-100, higher = more reason to leave) is a weighted blend of five "danger" readings, each 0..1:
    30% Momentum   RSI falling, MACD histogram weak/falling, new extreme on weaker RSI (divergence)
    25% Trend      price vs EMA20 (in ATR units), SuperTrend against, EMA20 vs EMA50
    20% Volume     last 3 bars vs normal, heavy adverse bar   (left out when the feed has no volume, e.g. Yahoo indices)
    15% Structure  close through the last swing low (BUY) / high (SELL), or under the previous bar's low / high
    10% Volatility an unusually large adverse candle, ATR expanding against the trade
Trend Health (0-100, higher = better) uses only the trend / structure / momentum readings.

States: STRONG_HOLD, HOLD, WEAKENING (early warning), PROTECT (score 40-70), EXIT (score > 70, structure break or SuperTrend flip).
All numbers are starting values chosen by design, NOT fitted to the user's data.
"""
import numpy as np
import pandas as pd

WEIGHTS = {"momentum": 0.30, "trend": 0.25, "volume": 0.20, "structure": 0.15, "volatility": 0.10}
DEFAULT_CFG = {"hold_max": 40.0, "exit_min": 70.0}
LABELS = {
    "STRONG_HOLD": "🟢 Strong Hold",
    "HOLD": "🟢 Hold",
    "WEAKENING": "⚠ Momentum Weakening",
    "PROTECT": "🟡 Profit Protection Mode",
    "EXIT": "🔴 Exit Recommended",
}
MESSAGES = {
    "STRONG_HOLD": "Trend is intact. Keep the trade, trail below EMA20.",
    "HOLD": "Trend is intact but not at its best. Keep the trade.",
    "WEAKENING": "Buyers/sellers are losing strength. Tighten the trailing stop - this is a warning, not an exit.",
    "PROTECT": "Move the stop to breakeven and trail beyond the last swing.",
    "EXIT": "The trade's structure is no longer valid.",
}
MIN_BARS = 40


# ---------------------------------------------------------------- indicators
def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _supertrend(high, low, close, period=10, mult=3.0):
    """Returns (direction array +1/-1, line array)."""
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().to_numpy()
    hl2 = ((high + low) / 2.0).to_numpy()
    c = close.to_numpy()
    n = len(c)
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(n, dtype=int)
    line = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr[i]):
            direction[i] = direction[i - 1]
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1] or np.isnan(fub[i - 1])) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1] or np.isnan(flb[i - 1])) else flb[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < flb[i] else 1
        else:
            direction[i] = 1 if c[i] > fub[i] else -1
        line[i] = flb[i] if direction[i] == 1 else fub[i]
    return direction, line


def prepare(df):
    """OHLCV frame -> arrays dict, or None when there are too few usable bars."""
    try:
        if df is None or len(df) < MIN_BARS:
            return None
        f = df[["High", "Low", "Close"]].astype(float).copy()
        f["Volume"] = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else np.nan
        f = f.dropna(subset=["High", "Low", "Close"])
        if len(f) < MIN_BARS:
            return None
        close = f["Close"]
        pc = close.shift(1)
        tr = pd.concat([f["High"] - f["Low"], (f["High"] - pc).abs(), (f["Low"] - pc).abs()], axis=1).max(axis=1)
        ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        st_dir, st_line = _supertrend(f["High"], f["Low"], close)
        return {
            "close": close.to_numpy(), "high": f["High"].to_numpy(), "low": f["Low"].to_numpy(),
            "vol": f["Volume"].to_numpy(), "ema20": close.ewm(span=20, adjust=False).mean().to_numpy(),
            "ema50": close.ewm(span=50, adjust=False).mean().to_numpy(), "rsi": _rsi(close).to_numpy(),
            "hist": hist.to_numpy(), "atr": tr.rolling(14).mean().to_numpy(), "st_dir": st_dir, "st_line": st_line,
            "n": len(f),
        }
    except Exception:
        return None


# ---------------------------------------------------------------- scoring
def _clip(x, lo=0.0, hi=1.0):
    return float(min(hi, max(lo, x)))


def _components(a, i, sgn):
    """Danger readings 0..1 at bar i for a trade of direction sgn (+1 BUY, -1 SELL). Volume may be None."""
    c, h, l, atr = a["close"], a["high"], a["low"], a["atr"][i]
    if not np.isfinite(atr) or atr <= 0:
        atr = max(abs(c[i]) * 0.001, 0.01)
    # momentum
    d1 = _clip(-sgn * (a["rsi"][i] - a["rsi"][i - 3]) / 10.0)
    hv, hd = a["hist"][i], a["hist"][i] - a["hist"][i - 3]
    d2 = 1.0 if sgn * hv < 0 else (0.5 if sgn * hd < 0 else 0.0)
    w = slice(i - 9, i + 1)
    new_extreme = sgn * c[i] >= np.max(sgn * c[w]) - 1e-12
    weaker_rsi = sgn * a["rsi"][i] < np.max(sgn * a["rsi"][i - 9:i]) - 5.0
    div = 1.0 if (new_extreme and weaker_rsi) else 0.0
    momentum = 0.40 * d1 + 0.35 * d2 + 0.25 * div
    # trend
    dist = sgn * (c[i] - a["ema20"][i]) / atr
    e1 = _clip((1.0 - dist) / 1.5)
    st_against = 1.0 if a["st_dir"][i] != sgn else 0.0
    cross = 1.0 if sgn * (a["ema20"][i] - a["ema50"][i]) < 0 else 0.0
    trend = 0.45 * e1 + 0.40 * st_against + 0.15 * cross
    # volume (None when the feed has no volume)
    volume = None
    v = a["vol"]
    base = v[max(0, i - 23):i - 2]
    base = base[np.isfinite(base)]
    if len(base) >= 10 and np.nanmean(base) > 0 and np.all(np.isfinite(v[i - 2:i + 1])):
        mean_base = float(np.mean(base))
        ratio = float(np.mean(v[i - 2:i + 1])) / mean_base
        adverse_heavy = 1.0 if (sgn * (c[i] - c[i - 1]) < 0 and v[i] / mean_base > 1.5) else 0.0
        volume = 0.6 * _clip((1.0 - ratio) / 0.5) + 0.4 * adverse_heavy
    # structure
    swing_low, swing_high = float(np.min(l[i - 8:i])), float(np.max(h[i - 8:i]))
    broken = (c[i] < swing_low) if sgn > 0 else (c[i] > swing_high)
    under_prev = (c[i] < l[i - 1]) if sgn > 0 else (c[i] > h[i - 1])
    structure = 1.0 if broken else (0.5 if under_prev else 0.0)
    # volatility
    adv = max(0.0, -sgn * (c[i] - c[i - 1])) / atr
    atr_prev = a["atr"][i - 5]
    expanding = bool(np.isfinite(atr_prev) and atr_prev > 0 and atr / atr_prev > 1.3 and adv > 0)
    volatility = _clip(_clip((adv - 0.5) / 1.5) + (0.3 if expanding else 0.0))
    return {
        "momentum": momentum, "trend": trend, "volume": volume, "structure": structure, "volatility": volatility,
        "_flags": {"rsi_falling": d1 >= 0.3, "divergence": bool(div), "volume_weak": (volume is not None and volume >= 0.5),
                   "structure_break": bool(broken), "st_flip": bool(st_against), "below_ema20": bool(dist < 0), "momentum_fading": bool(momentum >= 0.5)},
        "_levels": {"ema20": float(a["ema20"][i]), "atr": float(atr), "swing_low": swing_low, "swing_high": swing_high,
                    "st_line": float(a["st_line"][i]) if np.isfinite(a["st_line"][i]) else None},
    }


def _score(comp):
    used = {k: w for k, w in WEIGHTS.items() if comp.get(k) is not None}
    tot = sum(used.values())
    score = 100.0 * sum(comp[k] * w for k, w in used.items()) / tot
    hw = {"trend": 0.5, "structure": 0.25, "momentum": 0.25}
    health = 100.0 - 100.0 * sum(comp[k] * w for k, w in hw.items())
    return score, health


def _state(score, comp, cfg):
    f = comp["_flags"]
    reasons = []
    if f["structure_break"]:
        reasons.append("structure break")
    if f["st_flip"]:
        reasons.append("SuperTrend against")
    if score >= cfg["exit_min"]:
        reasons.append(f"Exit Score {score:.0f} >= {cfg['exit_min']:.0f}")
    if reasons:
        return "EXIT", ", ".join(reasons)
    if score >= cfg["hold_max"]:
        return "PROTECT", ""
    warn = sum([f["rsi_falling"], f["divergence"], f["volume_weak"], f["below_ema20"]])
    if warn >= 2 or f["momentum_fading"]:
        return "WEAKENING", ""
    return ("STRONG_HOLD" if score < cfg["hold_max"] * 0.6 and not f["below_ema20"] else "HOLD"), ""


def assess(df, side, entry=None, cfg=None, history=5, a=None):
    """Full reading for the latest bar plus the last `history` bars. Returns {"ok": False, ...} when data is insufficient."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    sgn = 1 if str(side).upper() == "BUY" else -1
    a = a or prepare(df)
    if a is None:
        return {"ok": False, "reason": f"need at least {MIN_BARS} candles"}
    n = a["n"]
    hist_list = []
    for i in range(max(12, n - max(1, int(history))), n):
        comp = _components(a, i, sgn)
        s, hl = _score(comp)
        hist_list.append({"score": round(s), "health": round(hl)})
    comp = _components(a, n - 1, sgn)
    score, health = _score(comp)
    state, why = _state(score, comp, cfg)
    lv = comp["_levels"]
    inv = lv["st_line"] if (lv["st_line"] is not None and a["st_dir"][n - 1] == sgn) else (lv["swing_low"] if sgn > 0 else lv["swing_high"])
    price = float(a["close"][n - 1])
    return {
        "ok": True, "side": "BUY" if sgn > 0 else "SELL", "price": price,
        "score": round(score), "health": round(health), "state": state, "label": LABELS[state], "message": MESSAGES[state],
        "exit_reason": why, "history": hist_list,
        "components": {k: (None if comp[k] is None else round(100 * comp[k])) for k in WEIGHTS},
        "flags": comp["_flags"], "levels": lv,
        "valid_level": float(inv),     # BUY: bullish while price stays above this; SELL: bearish while below
        "entry": None if entry is None else float(entry),
    }


# ---------------------------------------------------------------- trailing advice
def suggest_stop(side, entry, cur_stop, price, levels, state, cost_pts=0.0):
    """Tighten-only stop suggestion, or None.  HOLD states trail EMA20; WEAKENING tightens; PROTECT goes to breakeven / last swing."""
    try:
        is_long = str(side).upper() == "BUY"
        atr = float(levels["atr"])
        e20, sl_, sh_ = float(levels["ema20"]), float(levels["swing_low"]), float(levels["swing_high"])
        be = entry + (cost_pts if is_long else -cost_pts)
        if state in ("STRONG_HOLD", "HOLD"):
            cand = (e20 - 0.5 * atr) if is_long else (e20 + 0.5 * atr)
        elif state == "WEAKENING":
            cand = (e20 - 0.25 * atr) if is_long else (e20 + 0.25 * atr)
        elif state == "PROTECT":
            swing = (sl_ - 0.1 * atr) if is_long else (sh_ + 0.1 * atr)
            in_profit = price > be if is_long else price < be
            cand = (max(be, swing) if is_long else min(be, swing)) if in_profit else swing
        else:
            return None
        gap = 0.05 * atr
        if is_long:
            ok = cand > cur_stop and cand < price - gap
        else:
            ok = cand < cur_stop and cand > price + gap
        return float(cand) if ok else None
    except Exception:
        return None
