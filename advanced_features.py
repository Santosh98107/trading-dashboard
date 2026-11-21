"""Advanced context indicators (SHADOW MODE: shown, not used in scoring unless switched on).

Parity-safe: every value at bar t uses only data up to bar t (no look-ahead), so the
same numbers appear live and in a replay. Expects columns High, Low, Close, Volume and a
DatetimeIndex (IST or naive). Returns a DataFrame with new columns; never raises.
"""
import numpy as np
import pandas as pd


def _safe(df, col):
    return df[col].astype(float) if col in df.columns else pd.Series(np.nan, index=df.index)


def session_vwap(df):
    tp = (_safe(df, "High") + _safe(df, "Low") + _safe(df, "Close")) / 3
    v = _safe(df, "Volume").fillna(0)
    day = df.index.normalize()
    cv = (tp * v).groupby(day).cumsum()
    vv = v.groupby(day).cumsum().replace(0, np.nan)
    return cv / vv


def obv(df):
    c, v = _safe(df, "Close"), _safe(df, "Volume").fillna(0)
    return (np.sign(c.diff()).fillna(0) * v).cumsum()


def cmf(df, n=20):
    h, l, c, v = _safe(df, "High"), _safe(df, "Low"), _safe(df, "Close"), _safe(df, "Volume").fillna(0)
    rng = (h - l).replace(0, np.nan)
    mfv = ((c - l) - (h - c)) / rng * v
    return mfv.rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)


def choppiness(df, n=14):
    h, l, c = _safe(df, "High"), _safe(df, "Low"), _safe(df, "Close")
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    rng = (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def running_opening_range(df, minutes=30):
    """High/low of the first `minutes` of each day, built bar by bar (no full-day leakage)."""
    day = df.index.normalize()
    start = pd.Series(df.index, index=df.index).groupby(day).transform("min")
    inside = (df.index - start) < pd.Timedelta(minutes=minutes)
    hi = _safe(df, "High").where(inside).groupby(day).cummax().groupby(day).ffill()
    lo = _safe(df, "Low").where(inside).groupby(day).cummin().groupby(day).ffill()
    return hi, lo


def prev_day_levels(df):
    day = df.index.normalize()
    d = pd.DataFrame({"h": _safe(df, "High"), "l": _safe(df, "Low"), "c": _safe(df, "Close"), "day": day})
    agg = d.groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = agg.shift(1)
    m = pd.Series(day, index=df.index).map
    return m(prev["h"]), m(prev["l"]), m(prev["c"])


def cpr(df):
    ph, pl, pc = prev_day_levels(df)
    pivot = (ph + pl + pc) / 3
    bc = (ph + pl) / 2
    tc = 2 * pivot - bc
    return pivot, np.minimum(bc, tc), np.maximum(bc, tc)


def divergence_flags(df, n=14, lookback=10):
    """Per-bar divergence vs `lookback` bars ago: price makes a new extreme, RSI does not."""
    c = _safe(df, "Close")
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ph, pr = c.rolling(lookback).max().shift(1), rsi.rolling(lookback).max().shift(1)
    pl, pl_r = c.rolling(lookback).min().shift(1), rsi.rolling(lookback).min().shift(1)
    bear = (c > ph) & (rsi < pr)
    bull = (c < pl) & (rsi > pl_r)
    return bull.fillna(False), bear.fillna(False)


def add_advanced(df):
    out = df.copy()
    try:
        out["AF_VWAP"] = session_vwap(out)
        out["AF_OBV"] = obv(out)
        out["AF_CMF20"] = cmf(out)
        out["AF_CHOP"] = choppiness(out)
        out["AF_OR_HIGH"], out["AF_OR_LOW"] = running_opening_range(out)
        out["AF_PDH"], out["AF_PDL"], out["AF_PDC"] = prev_day_levels(out)
        out["AF_CPR_P"], out["AF_CPR_BC"], out["AF_CPR_TC"] = cpr(out)
        out["AF_BULL_DIV"], out["AF_BEAR_DIV"] = divergence_flags(out)
    except Exception:
        pass
    return out


def summarize(df):
    """Plain-language context for the last bar."""
    try:
        r = df.iloc[-1]
        c = float(r["Close"]); notes = []
        if pd.notna(r.get("AF_VWAP")):
            notes.append(f"Price {'above' if c >= r['AF_VWAP'] else 'below'} VWAP {r['AF_VWAP']:.1f}")
        if pd.notna(r.get("AF_CHOP")):
            ch = float(r["AF_CHOP"]); notes.append("Choppy market (" + f"{ch:.0f}" + ")" if ch > 61.8 else "Trending market (" + f"{ch:.0f}" + ")" if ch < 38.2 else f"Chop {ch:.0f} (neutral)")
        if pd.notna(r.get("AF_CMF20")):
            notes.append(f"Money flow {'positive' if r['AF_CMF20'] > 0 else 'negative'} ({r['AF_CMF20']:+.2f})")
        if pd.notna(r.get("AF_OR_HIGH")):
            notes.append("Above opening range high" if c > r["AF_OR_HIGH"] else "Below opening range low" if c < r["AF_OR_LOW"] else "Inside opening range")
        if bool(r.get("AF_BEAR_DIV")): notes.append("Bearish RSI divergence")
        if bool(r.get("AF_BULL_DIV")): notes.append("Bullish RSI divergence")
        return notes
    except Exception:
        return []


def score_adjust(df, side, enabled=False):
    """Optional +/- points (max 1.0). 0 unless enabled. Not replayed in backtests."""
    if not enabled:
        return 0.0
    try:
        r = df.iloc[-1]; s = 0.0; sg = 1 if side == "BUY" else -1
        if pd.notna(r.get("AF_VWAP")): s += 0.3 * sg * (1 if r["Close"] >= r["AF_VWAP"] else -1)
        if pd.notna(r.get("AF_CMF20")): s += 0.3 * sg * (1 if r["AF_CMF20"] > 0 else -1)
        if pd.notna(r.get("AF_CHOP")) and r["AF_CHOP"] > 61.8: s -= 0.4
        return max(-1.0, min(1.0, s))
    except Exception:
        return 0.0
