"""Institutional analysis module (fixed).

Key changes versus the previous version
---------------------------------------
* Liquidity sweep direction is correct: a BUY-side sweep (stop run above highs
  that closes back below) is a bearish trap and blocks BUY entries; a SELL-side
  sweep is a bullish reversal context and blocks SELL entries. Only the side that
  would be trapped is blocked.
* FVG / order-block detection looks for BOTH sides and ignores mitigated zones.
* Volume profile is vectorised and computes a real 70 % value area.
* Relative strength returns None (neutral/unavailable) instead of a fake 5.0.
* ML module uses stationary features, a chronological split and reports
  out-of-sample metrics. It is informational only and never a risk override.
* Sector tickers are valid Yahoo symbols, downloaded in one batch and cached;
  failures stay NaN instead of a fake 0.0 %.
* Adds equal highs/lows, BOS/CHOCH flags and premium/discount zones.
* `institutional_score_components()` turns the diagnostics into configurable,
  auditable (bullish, bearish) point contributions used by the main engine.
"""
import numpy as np
import pandas as pd

try:  # optional – keeps the module importable in tests / non-UI contexts
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


def _cache(ttl):
    """st.cache_data when Streamlit is present, otherwise a no-op decorator."""
    def deco(fn):
        if st is not None and hasattr(st, "cache_data"):
            try:
                return st.cache_data(ttl=ttl, show_spinner=False)(fn)
            except Exception:
                return fn
        return fn
    return deco


# =========================================================
# CONFIGURABLE WEIGHTS (points added to bullish / bearish totals)
# =========================================================
INSTITUTIONAL_WEIGHTS = {
    "market_structure": 0.6,   # raw score +-5  -> +-3.0
    "liquidity": 0.6,          # raw score +-5  -> +-3.0
    "order_block": 0.2,        # raw score +-10 -> +-2.0
    "fvg": 0.2,                # raw score +-10 -> +-2.0
    "volume_profile": 0.2,     # raw score +-5  -> +-1.0
    "relative_strength": 0.4,  # (RS-5) in +-5  -> +-2.0
    "premium_discount": 0.4,   # raw score +-5  -> +-2.0
}
INSTITUTIONAL_SCORE_CAP = 8.0


def _num_series(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


# =========================================================
# 1. MARKET STRUCTURE (+ BOS / CHOCH, swing points)
# =========================================================
def find_swings(df, left=3, right=3):
    """Confirmed swing highs/lows as (index, 'H'|'L', price). Needs `right` bars after."""
    highs, lows = _num_series(df, "High"), _num_series(df, "Low")
    n = len(df)
    swings = []
    for i in range(left, n - right):
        hw = highs[i - left:i + right + 1]
        lw = lows[i - left:i + right + 1]
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(hw):
            swings.append((i, "H", float(highs[i])))
        if np.isfinite(lows[i]) and lows[i] == np.nanmin(lw):
            swings.append((i, "L", float(lows[i])))
    return swings


def detect_market_structure(df, left=3, right=3):
    neutral = {"structure": "NEUTRAL", "last_hh": None, "last_hl": None, "last_lh": None,
               "last_ll": None, "bos": False, "choch": False, "buy_confirmed": False,
               "sell_confirmed": False, "score": 0, "last_swing_high": None,
               "last_swing_low": None, "available": False}
    if df is None or len(df) < (left + right + 8):
        return neutral
    closes = _num_series(df, "Close")
    swings = find_swings(df, left, right)
    highs_s = [(i, p) for i, k, p in swings if k == "H"]
    lows_s = [(i, p) for i, k, p in swings if k == "L"]
    if len(highs_s) < 2 or len(lows_s) < 2:
        neutral["available"] = True
        return neutral

    hh_flag = highs_s[-1][1] > highs_s[-2][1]
    hl_flag = lows_s[-1][1] > lows_s[-2][1]
    lh_flag = highs_s[-1][1] < highs_s[-2][1]
    ll_flag = lows_s[-1][1] < lows_s[-2][1]
    last_h, last_l = highs_s[-1][1], lows_s[-1][1]
    cur = closes[-1]

    bos = choch = False
    structure, score = "NEUTRAL", 0
    uptrend = hh_flag and hl_flag
    downtrend = lh_flag and ll_flag

    if cur > last_h:
        bos = True
        if downtrend:           # break against prior bearish structure = CHOCH
            choch = True
            structure, score = "Bullish CHOCH (reversal break)", 4
        else:
            structure, score = "HH-HL Bullish (BOS)", 5
    elif cur < last_l:
        bos = True
        if uptrend:
            choch = True
            structure, score = "Bearish CHOCH (reversal break)", -4
        else:
            structure, score = "LH-LL Bearish (BOS)", -5
    elif uptrend:
        structure, score = "HH-HL Bullish", 3
    elif downtrend:
        structure, score = "LH-LL Bearish", -3

    ema9 = df["EMA9"].iloc[-1] if "EMA9" in df.columns else np.nan
    ema21 = df["EMA21"].iloc[-1] if "EMA21" in df.columns else np.nan
    adx = df["ADX"].iloc[-1] if "ADX" in df.columns else np.nan
    have = np.isfinite(ema9) and np.isfinite(ema21) and np.isfinite(adx)
    return {
        "structure": structure,
        "last_hh": highs_s[-1][1] if hh_flag else None,
        "last_hl": lows_s[-1][1] if hl_flag else None,
        "last_lh": highs_s[-1][1] if lh_flag else None,
        "last_ll": lows_s[-1][1] if ll_flag else None,
        "bos": bos, "choch": choch,
        "buy_confirmed": bool(have and score > 0 and ema9 > ema21 and adx > 20),
        "sell_confirmed": bool(have and score < 0 and ema9 < ema21 and adx > 20),
        "score": score, "last_swing_high": last_h, "last_swing_low": last_l,
        "available": True,
    }


# =========================================================
# 2. LIQUIDITY: sweeps, buy/sell-side liquidity, equal highs/lows
# =========================================================
def detect_equal_levels(df, tolerance_pct=0.05, left=3, right=3):
    """Equal highs / lows = clustered swing points (resting liquidity)."""
    out = {"equal_highs": [], "equal_lows": []}
    if df is None or len(df) < 20:
        return out
    swings = find_swings(df, left, right)
    for kind, key in (("H", "equal_highs"), ("L", "equal_lows")):
        pts = [p for _, k, p in swings if k == kind][-8:]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if pts[a] > 0 and abs(pts[a] - pts[b]) / pts[a] * 100 <= tolerance_pct:
                    out[key].append(round((pts[a] + pts[b]) / 2, 2))
    out["equal_highs"] = sorted(set(out["equal_highs"]))
    out["equal_lows"] = sorted(set(out["equal_lows"]))
    return out


def detect_liquidity_sweep(df, lookback=20):
    base = {"sweep_detected": False, "type": "NONE", "risk": "LOW", "block_entry": False,
            "block_buy": False, "block_sell": False, "score": 0, "available": False,
            "buy_side_liquidity": None, "sell_side_liquidity": None, "equal_levels": {}}
    if df is None or len(df) < lookback + 1:
        return base
    prior = df.iloc[-(lookback + 1):-1]
    recent_high = float(pd.to_numeric(prior["High"], errors="coerce").max())
    recent_low = float(pd.to_numeric(prior["Low"], errors="coerce").min())
    last = df.iloc[-1]
    h, l, c, o = (float(last["High"]), float(last["Low"]), float(last["Close"]), float(last["Open"]))
    if not all(np.isfinite([recent_high, recent_low, h, l, c, o])):
        return base
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    rng = max(h - l, 1e-9)

    buy_side_sweep = h > recent_high and c < recent_high     # stop-run above highs, rejected
    sell_side_sweep = l < recent_low and c > recent_low      # stop-run below lows, rejected
    upper_reject = upper_wick > 2 * max(body, 0.1 * rng) and upper_wick / rng > 0.5
    lower_reject = lower_wick > 2 * max(body, 0.1 * rng) and lower_wick / rng > 0.5

    res = dict(base)
    res.update({"available": True, "buy_side_liquidity": recent_high,
                "sell_side_liquidity": recent_low,
                "equal_levels": detect_equal_levels(df)})
    if buy_side_sweep:
        res.update(sweep_detected=True, type="BUY_SIDE_SWEEP", risk="HIGH",
                   block_entry=True, block_buy=True, score=-5)
    elif sell_side_sweep:
        res.update(sweep_detected=True, type="SELL_SIDE_SWEEP", risk="HIGH",
                   block_entry=True, block_sell=True, score=5)
    elif upper_reject:
        res.update(sweep_detected=True, type="UPPER_WICK_REJECTION", risk="MEDIUM",
                   block_buy=False, score=-2)
    elif lower_reject:
        res.update(sweep_detected=True, type="LOWER_WICK_REJECTION", risk="MEDIUM",
                   block_sell=False, score=2)
    return res


# =========================================================
# 3. SECTOR STRENGTH (valid Yahoo tickers, one batch, cached)
# =========================================================
SECTORS = {
    "Banking": "^NSEBANK", "IT": "^CNXIT", "Pharma": "^CNXPHARMA",
    "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY", "Financial Services": "^CNXFIN",
}


def _close_series(raw, ticker):
    """Extract one ticker's Close from a yf.download() result of any column layout."""
    if raw is None or getattr(raw, "empty", True):
        return pd.Series(dtype=float)
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            for lvl in range(raw.columns.nlevels):
                if ticker in raw.columns.get_level_values(lvl):
                    sub = raw.xs(ticker, axis=1, level=lvl, drop_level=True)
                    if "Close" in sub.columns:
                        return pd.to_numeric(sub["Close"], errors="coerce").dropna()
            return pd.Series(dtype=float)
        if "Close" in raw.columns:
            return pd.to_numeric(raw["Close"], errors="coerce").dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


@_cache(300)
def get_sector_performance():
    """Return (DataFrame, nifty_change_pct|None). Unavailable values are NaN, never 0."""
    tickers = ["^NSEI"] + list(SECTORS.values())
    raw = None
    if yf is not None:
        try:
            raw = yf.download(tickers, period="10d", interval="1d", group_by="ticker",
                              auto_adjust=False, progress=False, threads=True, timeout=10)
        except Exception:
            raw = None

    def chg(ticker):
        s = _close_series(raw, ticker)
        if len(s) >= 2 and s.iloc[-2] != 0:
            return float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100.0)
        return np.nan

    nifty = chg("^NSEI")
    rows = []
    for name, tk in SECTORS.items():
        p = chg(tk)
        rows.append({"Sector": name, "Ticker": tk,
                     "Performance %": round(p, 2) if np.isfinite(p) else np.nan,
                     "Relative Strength": round(p - nifty, 2) if np.isfinite(p) and np.isfinite(nifty) else np.nan,
                     "Status": "OK" if np.isfinite(p) else "No data"})
    df_res = pd.DataFrame(rows).sort_values("Relative Strength", ascending=False, na_position="last")
    return df_res, (round(nifty, 2) if np.isfinite(nifty) else None)


# =========================================================
# 4. RELATIVE STRENGTH vs real benchmark
# =========================================================
def calculate_relative_strength(stock_df, benchmark_df, window=20):
    """0-10 score (5 = in line). None when either series is missing -> neutral."""
    if stock_df is None or benchmark_df is None:
        return None
    try:
        s = pd.to_numeric(stock_df["Close"], errors="coerce").dropna()
        b = pd.to_numeric(benchmark_df["Close"], errors="coerce").dropna()
        if len(s) <= window or len(b) <= window:
            return None
        s0, b0 = s.iloc[-(window + 1)], b.iloc[-(window + 1)]
        if s0 == 0 or b0 == 0:
            return None
        s_ret = (s.iloc[-1] - s0) / s0
        b_ret = (b.iloc[-1] - b0) / b0
        return round(float(np.clip(5.0 + (s_ret - b_ret) * 100.0, 0.0, 10.0)), 2)
    except Exception:
        return None


# =========================================================
# 5. OPEN-INTEREST POSITIONING
# =========================================================
def analyze_oi_positioning(price_change, oi_change):
    if price_change is None or oi_change is None:
        return "Unavailable", "Neutral", 0
    if price_change > 0 and oi_change > 0:
        return "Long Build Up", "Bullish", 10
    if price_change < 0 and oi_change > 0:
        return "Short Build Up", "Bearish", -10
    if price_change > 0 and oi_change < 0:
        return "Short Covering", "Bullish", 5
    if price_change < 0 and oi_change < 0:
        return "Long Unwinding", "Bearish", -5
    return "Neutral", "Neutral", 0


# =========================================================
# 6. ORDER BLOCKS (unmitigated only)
# =========================================================
def detect_order_blocks(df):
    base = {"bullish_ob": None, "bearish_ob": None, "retest": "NONE", "score": 0, "available": False}
    if df is None or len(df) < 10:
        return base
    o, c = _num_series(df, "Open"), _num_series(df, "Close")
    h, l = _num_series(df, "High"), _num_series(df, "Low")
    n = len(df)
    bullish_ob = bearish_ob = None
    for i in range(n - 3, 1, -1):  # last bar excluded as a candidate
        if bullish_ob is None and o[i] > c[i] and c[i + 1] > o[i + 1] and c[i + 2] > h[i]:
            zone = (l[i], h[i])
            if not np.any(c[i + 3:] < zone[0]):      # not mitigated (closed through)
                bullish_ob = (float(zone[0]), float(zone[1]))
        if bearish_ob is None and c[i] > o[i] and c[i + 1] < o[i + 1] and c[i + 2] < l[i]:
            zone = (l[i], h[i])
            if not np.any(c[i + 3:] > zone[1]):
                bearish_ob = (float(zone[0]), float(zone[1]))
        if bullish_ob and bearish_ob:
            break
    cur = c[-1]
    retest, score = "NONE", 0
    in_bull = bool(bullish_ob and bullish_ob[0] <= cur <= bullish_ob[1])
    in_bear = bool(bearish_ob and bearish_ob[0] <= cur <= bearish_ob[1])
    if in_bull and not in_bear:
        retest, score = "BULLISH_OB_RETEST", 10
    elif in_bear and not in_bull:
        retest, score = "BEARISH_OB_RETEST", -10
    return {"bullish_ob": bullish_ob, "bearish_ob": bearish_ob, "retest": retest,
            "score": score, "available": True}


# =========================================================
# 7. FAIR VALUE GAPS (both sides, unfilled only)
# =========================================================
def detect_fvg(df):
    base = {"bullish_fvg": None, "bearish_fvg": None, "retest": "NONE", "score": 0, "available": False}
    if df is None or len(df) < 5:
        return base
    h, l, c = _num_series(df, "High"), _num_series(df, "Low"), _num_series(df, "Close")
    n = len(df)
    bull = bear = None
    for i in range(n - 1, 1, -1):
        if bull is None and l[i] > h[i - 2]:
            zone = (float(h[i - 2]), float(l[i]))
            if not np.any(l[i + 1:] <= zone[0]):       # gap not fully filled
                bull = zone
        if bear is None and h[i] < l[i - 2]:
            zone = (float(h[i]), float(l[i - 2]))
            if not np.any(h[i + 1:] >= zone[1]):
                bear = zone
        if bull and bear:
            break
    cur = c[-1]
    in_bull = bool(bull and bull[0] <= cur <= bull[1])
    in_bear = bool(bear and bear[0] <= cur <= bear[1])
    retest, score = "NONE", 0
    if in_bull and not in_bear:
        retest, score = "BULLISH_FVG_RETEST", 10
    elif in_bear and not in_bull:
        retest, score = "BEARISH_FVG_RETEST", -10
    return {"bullish_fvg": bull, "bearish_fvg": bear, "retest": retest, "score": score, "available": True}


# =========================================================
# 8. PREMIUM / DISCOUNT ZONES
# =========================================================
def calculate_premium_discount(df, lookback=50):
    base = {"zone": "UNAVAILABLE", "equilibrium": None, "range_high": None,
            "range_low": None, "score": 0, "available": False}
    if df is None or len(df) < 20:
        return base
    win = df.tail(lookback)
    hi, lo = float(pd.to_numeric(win["High"], errors="coerce").max()), float(pd.to_numeric(win["Low"], errors="coerce").min())
    if not np.isfinite(hi + lo) or hi <= lo:
        return base
    eq = (hi + lo) / 2.0
    cur = float(df["Close"].iloc[-1])
    pos = (cur - lo) / (hi - lo)
    if pos >= 0.75:
        zone, score = "PREMIUM", -5
    elif pos <= 0.25:
        zone, score = "DISCOUNT", 5
    elif pos > 0.5:
        zone, score = "UPPER EQUILIBRIUM", -1
    else:
        zone, score = "LOWER EQUILIBRIUM", 1
    return {"zone": zone, "equilibrium": round(eq, 2), "range_high": hi, "range_low": lo,
            "position_pct": round(pos * 100, 1), "score": score, "available": True}


# =========================================================
# 9. VOLUME PROFILE (vectorised, real 70 % value area)
# =========================================================
def calculate_volume_profile(df, bins=24, value_area=0.70):
    base = {"poc": None, "vah": None, "val": None, "signal": "NEUTRAL", "score": 0, "available": False}
    if df is None or len(df) < 10 or "Volume" not in df.columns:
        return base
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    if vol.sum() <= 0 or (vol > 0).sum() < max(5, int(len(vol) * 0.5)):
        return base  # no representative volume (e.g. index feed) -> neutral
    typ = ((_num_series(df, "High") + _num_series(df, "Low") + _num_series(df, "Close")) / 3.0)
    lo, hi = float(np.nanmin(_num_series(df, "Low"))), float(np.nanmax(_num_series(df, "High")))
    if not np.isfinite(hi + lo) or hi <= lo:
        return base
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(typ, bins=edges, weights=vol)
    centers = (edges[:-1] + edges[1:]) / 2
    poc_i = int(np.argmax(hist))
    lo_i = hi_i = poc_i
    acc, total = hist[poc_i], hist.sum()
    while acc < total * value_area and (lo_i > 0 or hi_i < len(hist) - 1):
        left = hist[lo_i - 1] if lo_i > 0 else -1
        right = hist[hi_i + 1] if hi_i < len(hist) - 1 else -1
        if right >= left and hi_i < len(hist) - 1:
            hi_i += 1; acc += hist[hi_i]
        elif lo_i > 0:
            lo_i -= 1; acc += hist[lo_i]
        else:
            break
    poc, val, vah = float(centers[poc_i]), float(edges[lo_i]), float(edges[hi_i + 1])
    cur = float(df["Close"].iloc[-1])
    band = (hi - lo) / bins
    if abs(cur - poc) <= band:          # sitting on the POC = no directional information
        sig, score = "NEUTRAL", 0
    elif cur > vah:
        sig, score = "BUY", 5            # accepted above value
    elif cur < val:
        sig, score = "SELL", -5          # accepted below value
    elif cur > poc:
        sig, score = "BUY", 2
    else:
        sig, score = "SELL", -2
    return {"poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2),
            "signal": sig, "score": score, "available": True}


# =========================================================
# 10. ML (informational; chronological validation; never a blocker override)
# =========================================================
ML_FEATURES = ["ret1", "ret5", "rsi", "adx", "atr_pct", "dist_ema20", "macd_hist_norm", "rel_vol"]


def build_ml_features(df):
    f = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["Close"], errors="coerce")
    f["ret1"] = close.pct_change(1)
    f["ret5"] = close.pct_change(5)
    f["rsi"] = df["RSI"] if "RSI" in df.columns else np.nan
    f["adx"] = df["ADX"] if "ADX" in df.columns else np.nan
    f["atr_pct"] = (df["ATR"] / close) if "ATR" in df.columns else np.nan
    f["dist_ema20"] = ((close - df["EMA20"]) / df["EMA20"]) if "EMA20" in df.columns else np.nan
    f["macd_hist_norm"] = (df["MACD_Hist"] / close) if "MACD_Hist" in df.columns else np.nan
    f["rel_vol"] = df["RelativeVolume"] if "RelativeVolume" in df.columns else np.nan
    return f.replace([np.inf, -np.inf], np.nan)


def predict_trade_probability(df, horizon=1, min_rows=120):
    """Train on the past, evaluate on a later holdout, predict the latest bar.

    Output is a model-estimated probability that the next `horizon` bar closes
    higher. `confidence` is "Low" unless the out-of-sample AUC beats chance.
    """
    neutral = {"signal": "NEUTRAL", "probability": 50.0, "confidence": "Low",
               "oos_accuracy": None, "oos_auc": None, "train_rows": 0, "test_rows": 0,
               "note": "Insufficient data or scikit-learn unavailable", "available": False}
    if not SKLEARN_AVAILABLE or df is None or len(df) < min_rows:
        return neutral
    feats = build_ml_features(df)
    cols = [c for c in ML_FEATURES if c in feats.columns and feats[c].notna().sum() > 0.5 * len(feats)]
    if len(cols) < 4:
        return neutral
    close = pd.to_numeric(df["Close"], errors="coerce")
    future_ret = close.shift(-horizon) / close - 1.0            # label uses the future: training only
    data = feats[cols].copy()
    data["y"] = (future_ret > 0).astype(float).where(future_ret.notna())
    latest_x = data[cols].iloc[[-1]]
    if latest_x.isna().any(axis=1).iloc[0]:
        return neutral
    train_all = data.dropna()
    train_all = train_all.iloc[:-horizon] if horizon > 0 else train_all  # purge overlap with live bar
    if len(train_all) < 60 or train_all["y"].nunique() < 2:
        return neutral
    split = int(len(train_all) * 0.75)
    gap = max(horizon, 1)
    tr, te = train_all.iloc[:split], train_all.iloc[split + gap:]
    if len(te) < 15 or tr["y"].nunique() < 2:
        return neutral
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        model = RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=5,
                                       random_state=42, n_jobs=1)
        model.fit(tr[cols], tr["y"])
        pred = model.predict(te[cols])
        proba = model.predict_proba(te[cols])[:, 1]
        acc = float(accuracy_score(te["y"], pred))
        auc = float(roc_auc_score(te["y"], proba)) if te["y"].nunique() > 1 else None
        final = RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=5,
                                       random_state=42, n_jobs=1)
        final.fit(train_all[cols], train_all["y"])
        prob = float(final.predict_proba(latest_x)[0][1] * 100)
    except Exception as exc:
        out = dict(neutral)
        out["note"] = f"ML failed: {type(exc).__name__}"
        return out
    edge = auc is not None and auc >= 0.55
    sig = "NEUTRAL"
    if edge and prob >= 60:
        sig = "BUY"
    elif edge and prob <= 40:
        sig = "SELL"
    return {"signal": sig, "probability": round(prob, 2),
            "confidence": "Medium" if edge else "Low",
            "oos_accuracy": round(acc * 100, 1), "oos_auc": round(auc, 3) if auc is not None else None,
            "train_rows": int(len(tr)), "test_rows": int(len(te)), "available": True,
            "note": "Holdout metrics are chronological; estimate only, not a guarantee. Informational: never overrides risk blockers."}


# =========================================================
# 11. AI TRADE COACH
# =========================================================
def ai_loss_analysis(journal_df):
    if journal_df is None or getattr(journal_df, "empty", True) or "PnL" not in journal_df.columns:
        return pd.DataFrame()
    pnl = pd.to_numeric(journal_df["PnL"], errors="coerce")
    losses = journal_df[pnl < 0].copy()
    if losses.empty:
        return pd.DataFrame()
    rows = []
    for _, r in losses.iterrows():
        reason = str(r.get("LossReason", "") or "").strip()
        rsi = pd.to_numeric(pd.Series([r.get("RSI")]), errors="coerce").iloc[0]
        if not reason:
            if pd.notna(rsi) and rsi > 70:
                reason = "Overbought entry (RSI > 70)"
            elif pd.notna(rsi) and rsi < 30:
                reason = "Oversold entry (RSI < 30)"
            else:
                reason = "Unclassified – insufficient entry metrics saved"
        low = reason.lower()
        if "sideways" in low or "weak trend" in low:
            fix = "Require ADX above the mandatory threshold before entry"
        elif "volume" in low:
            fix = "Require relative volume >= 1.2x"
        elif "conflict" in low:
            fix = "Wait for multi-timeframe alignment"
        elif "risk" in low:
            fix = "Skip setups below the minimum risk/reward"
        elif "overbought" in low or "late" in low or "exhaust" in low:
            fix = "Wait for an EMA pullback instead of chasing"
        else:
            fix = "Review entry timing, news and execution manually"
        rows.append({"DateTime": r.get("DateTime", "N/A"), "Symbol": r.get("Symbol", "N/A"),
                     "PnL": r.get("PnL"), "Loss Reason": reason, "Suggested Fix": fix})
    return pd.DataFrame(rows)


# =========================================================
# DIAGNOSTICS + SCORE COMPONENTS (used by the main signal engine)
# =========================================================
def collect_institutional_diagnostics(df, benchmark_df=None):
    required = {"Open", "High", "Low", "Close", "Volume"}
    empty = {"available": False, "message": "Market OHLCV data unavailable or incomplete",
             "market_structure": None, "liquidity": None, "relative_strength": None,
             "order_blocks": None, "fvg": None, "volume_profile": None, "premium_discount": None}
    if df is None or getattr(df, "empty", True) or not required.issubset(set(df.columns)):
        return empty
    try:
        return {
            "available": True,
            "message": "Computed from the candles already loaded by the app",
            "market_structure": detect_market_structure(df),
            "liquidity": detect_liquidity_sweep(df),
            "relative_strength": calculate_relative_strength(df, benchmark_df),
            "benchmark_available": benchmark_df is not None and len(benchmark_df) > 20,
            "order_blocks": detect_order_blocks(df),
            "fvg": detect_fvg(df),
            "volume_profile": calculate_volume_profile(df),
            "premium_discount": calculate_premium_discount(df),
        }
    except Exception as exc:
        out = dict(empty)
        out["message"] = f"Institutional diagnostics unavailable: {type(exc).__name__}: {exc}"
        return out


def institutional_score_components(diag, weights=None, cap=INSTITUTIONAL_SCORE_CAP):
    """Turn diagnostics into audited contributions.

    Returns (components, net_score, block_buy, block_sell). Each component is a dict
    with name, bullish, bearish, available, reason. Unavailable pieces contribute 0.
    """
    w = dict(INSTITUTIONAL_WEIGHTS)
    if weights:
        w.update(weights)
    comps = []

    def add(name, raw, key, available, reason):
        pts = float(raw) * w.get(key, 0.0) if available else 0.0
        comps.append({"name": name, "bullish": max(pts, 0.0), "bearish": max(-pts, 0.0),
                      "net": pts, "available": bool(available), "reason": reason})

    if not diag or not diag.get("available"):
        return [], 0.0, False, False

    ms = diag.get("market_structure") or {}
    add("Market Structure", ms.get("score", 0), "market_structure", ms.get("available", False),
        ms.get("structure", "n/a"))
    ls = diag.get("liquidity") or {}
    add("Liquidity Sweep", ls.get("score", 0), "liquidity", ls.get("available", False),
        ls.get("type", "NONE"))
    ob = diag.get("order_blocks") or {}
    add("Order Block", ob.get("score", 0), "order_block", ob.get("available", False), ob.get("retest", "NONE"))
    fvg = diag.get("fvg") or {}
    add("Fair Value Gap", fvg.get("score", 0), "fvg", fvg.get("available", False), fvg.get("retest", "NONE"))
    vp = diag.get("volume_profile") or {}
    add("Volume Profile", vp.get("score", 0), "volume_profile", vp.get("available", False),
        f"POC {vp.get('poc')} / {vp.get('signal', 'NEUTRAL')}" if vp.get("available") else "Volume not representative")
    pd_ = diag.get("premium_discount") or {}
    add("Premium/Discount", pd_.get("score", 0), "premium_discount", pd_.get("available", False),
        pd_.get("zone", "UNAVAILABLE"))
    rs = diag.get("relative_strength")
    add("Relative Strength", (rs - 5.0) if rs is not None else 0.0, "relative_strength",
        rs is not None, f"RS {rs} vs benchmark" if rs is not None else "Benchmark unavailable")

    net = sum(c["net"] for c in comps)
    net = float(max(-cap, min(cap, net)))
    return comps, net, bool(ls.get("block_buy")), bool(ls.get("block_sell"))


# =========================================================
# UI RENDERER
# =========================================================
def render_institutional_dashboard(df, symbol, benchmark_df, journal_df=None, option_metrics=None):
    if st is None:
        return
    diagnostics = collect_institutional_diagnostics(df, benchmark_df)
    if not diagnostics.get("available"):
        st.warning(diagnostics.get("message", "Institutional diagnostics unavailable"))
        return
    ms, ls = diagnostics["market_structure"], diagnostics["liquidity"]
    rs_score = diagnostics["relative_strength"]
    ob, fvg, vp = diagnostics["order_blocks"], diagnostics["fvg"], diagnostics["volume_profile"]
    pdz = diagnostics["premium_discount"]
    comps, net, block_buy, block_sell = institutional_score_components(diagnostics)

    if block_buy:
        st.error(f"⛔ BUY entries blocked: {ls['type']} (stop-run above recent highs, rejected).")
    if block_sell:
        st.error(f"⛔ SELL entries blocked: {ls['type']} (stop-run below recent lows, rejected).")
    st.caption("These institutional components ARE part of the main signal score "
               f"(net contribution {net:+.1f}, capped at ±{INSTITUTIONAL_SCORE_CAP:g}).")
    if comps:
        st.dataframe(pd.DataFrame(comps)[["name", "bullish", "bearish", "available", "reason"]],
                     use_container_width=True, hide_index=True)

    tabs = st.tabs(["Structure", "Liquidity", "Sectors", "Relative Strength", "OI Positioning",
                    "Order Blocks", "FVG", "Volume Profile", "Premium/Discount", "ML (informational)", "AI Coach"])
    with tabs[0]:
        st.write(f"Structure: **{ms['structure']}** | BOS: {ms.get('bos')} | CHOCH: {ms.get('choch')}")
        st.write(f"HH: {ms['last_hh']} | HL: {ms['last_hl']} | LH: {ms['last_lh']} | LL: {ms['last_ll']}")
    with tabs[1]:
        st.write(f"Sweep: **{ls['type']}** | Risk: **{ls['risk']}** | Buy-side liquidity: {ls.get('buy_side_liquidity')} | Sell-side liquidity: {ls.get('sell_side_liquidity')}")
        eq = ls.get("equal_levels") or {}
        st.write(f"Equal highs: {eq.get('equal_highs') or 'none'} | Equal lows: {eq.get('equal_lows') or 'none'}")
    with tabs[2]:
        sec_df, n_perf = get_sector_performance()
        st.write(f"NIFTY 1-day change: {n_perf if n_perf is not None else 'unavailable'}%")
        st.dataframe(sec_df, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.metric(f"RS score vs benchmark ({symbol})", rs_score if rs_score is not None else "N/A")
        st.caption("0–10 scale, 5 = in line. N/A means the benchmark was unavailable and the component contributes 0.")
    with tabs[4]:
        pcr = (option_metrics or {}).get("pcr")
        cc, pc = (option_metrics or {}).get("call_change_oi"), (option_metrics or {}).get("put_change_oi")
        if option_metrics and pcr is not None:
            st.write(f"PCR: **{pcr}** | OI bias: **{option_metrics.get('oi_bias', 'N/A')}** | Call ΔOI: {cc} | Put ΔOI: {pc}")
        else:
            st.info("Option-chain data unavailable. OI positioning is not inferred without real OI.")
    with tabs[5]:
        st.write(f"Retest: {ob['retest']} | Bullish OB: {ob['bullish_ob']} | Bearish OB: {ob['bearish_ob']}")
    with tabs[6]:
        st.write(f"Retest: {fvg['retest']} | Bullish FVG: {fvg['bullish_fvg']} | Bearish FVG: {fvg['bearish_fvg']}")
    with tabs[7]:
        if vp.get("available"):
            st.metric("POC", vp["poc"])
            st.write(f"VAH: {vp['vah']} | VAL: {vp['val']}")
        else:
            st.info("Volume unavailable or not representative for this instrument.")
    with tabs[8]:
        st.write(f"Zone: **{pdz['zone']}** | Equilibrium: {pdz['equilibrium']} | Position in range: {pdz.get('position_pct', 'N/A')}%")
    with tabs[9]:
        ml = predict_trade_probability(df)
        st.write(f"Signal: **{ml['signal']}** | Model-estimated up-probability: **{ml['probability']}%** | Confidence: {ml['confidence']}")
        st.write(f"Out-of-sample accuracy: {ml['oos_accuracy']} | OOS AUC: {ml['oos_auc']} | Train/Test rows: {ml['train_rows']}/{ml['test_rows']}")
        st.caption(ml["note"])
    with tabs[10]:
        coaching = ai_loss_analysis(journal_df)
        if not coaching.empty:
            st.dataframe(coaching, use_container_width=True, hide_index=True)
        else:
            st.info("No losing trades found in journal for coaching.")
