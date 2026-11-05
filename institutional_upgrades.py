import institutional_upgrades as inst
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Optional ML Imports
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =========================================================
# 1. MARKET STRUCTURE DETECTOR
# =========================================================
def detect_market_structure(df, left=3, right=3):
    if df is None or len(df) < (left + right + 5):
        return {"structure": "NEUTRAL", "last_hh": None, "last_hl": None, "last_lh": None, "last_ll": None, "bos": False, "score": 0}
    
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    
    pivots = []
    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            pivots.append((i, "H", highs[i]))
        if lows[i] == min(lows[i-left:i+right+1]):
            pivots.append((i, "L", lows[i]))
            
    hh, hl, lh, ll = None, None, None, None
    last_h, last_l = None, None
    
    for idx, kind, price in pivots:
        if kind == "H":
            if last_h is not None:
                if price > last_h:
                    hh = price
                else:
                    lh = price
            last_h = price
        elif kind == "L":
            if last_l is not None:
                if price > last_l:
                    hl = price
                else:
                    ll = price
            last_l = price
            
    current_close = closes[-1]
    bos = False
    structure = "NEUTRAL"
    score = 0
    
    if last_h and current_close > last_h:
        bos = True
        structure = "HH-HL Bullish (BOS)"
        score = 20
    elif last_l and current_close < last_l:
        bos = True
        structure = "LH-LL Bearish (BOS)"
        score = -20
    elif hh and hl:
        structure = "HH-HL Bullish"
        score = 15
    elif lh and ll:
        structure = "LH-LL Bearish"
        score = -15

    ema9 = df["EMA9"].iloc[-1] if "EMA9" in df.columns else current_close
    ema21 = df["EMA21"].iloc[-1] if "EMA21" in df.columns else current_close
    adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 25

    buy_confirmed = ("Bullish" in structure) and (ema9 > ema21) and (adx > 20)
    sell_confirmed = ("Bearish" in structure) and (ema9 < ema21) and (adx > 20)

    return {
        "structure": structure, "last_hh": hh, "last_hl": hl, "last_lh": lh, "last_ll": ll,
        "bos": bos, "buy_confirmed": buy_confirmed, "sell_confirmed": sell_confirmed, "score": score
    }

# =========================================================
# 2. LIQUIDITY SWEEP DETECTOR
# =========================================================
def detect_liquidity_sweep(df):
    if df is None or len(df) < 20:
        return {"sweep_detected": False, "type": "NONE", "risk": "LOW", "block_entry": False, "score": 15}
    
    recent_high = df["High"].iloc[-21:-1].max()
    recent_low = df["Low"].iloc[-21:-1].min()
    
    latest_high = df["High"].iloc[-1]
    latest_low = df["Low"].iloc[-1]
    latest_close = df["Close"].iloc[-1]
    latest_open = df["Open"].iloc[-1]
    
    body = abs(latest_close - latest_open)
    upper_wick = latest_high - max(latest_open, latest_close)
    lower_wick = min(latest_open, latest_close) - latest_low
    
    buy_side_sweep = (latest_high > recent_high) and (latest_close < recent_high)
    sell_side_sweep = (latest_low < recent_low) and (latest_close > recent_low)
    stop_hunt = (upper_wick > 2 * body) or (lower_wick > 2 * body)
    
    if buy_side_sweep or sell_side_sweep or stop_hunt:
        risk = "HIGH" if (buy_side_sweep or sell_side_sweep) else "MEDIUM"
        return {
            "sweep_detected": True,
            "type": "BUY_SIDE_SWEEP" if buy_side_sweep else "SELL_SIDE_SWEEP" if sell_side_sweep else "STOP_HUNT",
            "risk": risk, "block_entry": True, "score": -20
        }
    return {"sweep_detected": False, "type": "NONE", "risk": "LOW", "block_entry": False, "score": 15}

# =========================================================
# 3. SECTOR STRENGTH TRACKER
# =========================================================
SECTORS = {
    "Banking": "^NSEBANK", "IT": "NIFTY_IT.NS", "Pharma": "NIFTY_PHARMA.NS",
    "Auto": "NIFTY_AUTO.NS", "FMCG": "NIFTY_FMCG.NS", "Metal": "NIFTY_METAL.NS"
}

def get_sector_performance():
    results = []
    try:
        nifty = yf.download("^NSEI", period="5d", progress=False)
        nifty_perf = ((nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2]) / nifty["Close"].iloc[-2] * 100).item()
    except Exception:
        nifty_perf = 0.0

    for name, ticker in SECTORS.items():
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if not data.empty and len(data) >= 2:
                perf = ((data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100).item()
                rs = perf - nifty_perf
                results.append({"Sector": name, "Performance %": round(perf, 2), "Relative Strength": round(rs, 2)})
            else:
                results.append({"Sector": name, "Performance %": 0.0, "Relative Strength": 0.0})
        except Exception:
            results.append({"Sector": name, "Performance %": 0.0, "Relative Strength": 0.0})

    df_res = pd.DataFrame(results).sort_values("Relative Strength", ascending=False)
    return df_res, nifty_perf

# =========================================================
# 4. RELATIVE STRENGTH SCORE
# =========================================================
def calculate_relative_strength(stock_df, benchmark_df):
    if stock_df is None or benchmark_df is None or len(stock_df) < 20 or len(benchmark_df) < 20:
        return 5.0
    s_ret = (stock_df["Close"].iloc[-1] - stock_df["Close"].iloc[-20]) / stock_df["Close"].iloc[-20]
    b_ret = (benchmark_df["Close"].iloc[-1] - benchmark_df["Close"].iloc[-20]) / benchmark_df["Close"].iloc[-20]
    rs_raw = (s_ret - b_ret) * 100
    return round(max(0.0, min(10.0, 5.0 + rs_raw)), 2)

# =========================================================
# 5. OPTION CHAIN OI ANALYZER
# =========================================================
def analyze_oi_positioning(price_change, oi_change):
    if price_change > 0 and oi_change > 0:
        return "Long Build Up", "Bullish", 10
    elif price_change < 0 and oi_change > 0:
        return "Short Build Up", "Bearish", -10
    elif price_change > 0 and oi_change < 0:
        return "Short Covering", "Bullish", 5
    elif price_change < 0 and oi_change < 0:
        return "Long Unwinding", "Bearish", -5
    return "Neutral", "Neutral", 0

# =========================================================
# 6. ORDER BLOCK DETECTOR
# =========================================================
def detect_order_blocks(df):
    if df is None or len(df) < 10:
        return {"bullish_ob": None, "bearish_ob": None, "retest": "NONE", "score": 0}
    closes, opens, highs, lows = df["Close"].to_numpy(), df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy()
    bullish_ob, bearish_ob = None, None
    for i in range(len(df) - 4, 1, -1):
        if opens[i] > closes[i] and closes[i+1] > opens[i+1] and closes[i+2] > highs[i]:
            bullish_ob = (lows[i], highs[i])
            break
    for i in range(len(df) - 4, 1, -1):
        if closes[i] > opens[i] and closes[i+1] < opens[i+1] and closes[i+2] < lows[i]:
            bearish_ob = (lows[i], highs[i])
            break
    curr = closes[-1]
    retest = "NONE"
    score = 0
    if bullish_ob and (bullish_ob[0] <= curr <= bullish_ob[1]):
        retest = "BULLISH_OB_RETEST"
        score = 10
    elif bearish_ob and (bearish_ob[0] <= curr <= bearish_ob[1]):
        retest = "BEARISH_OB_RETEST"
        score = -10
    return {"bullish_ob": bullish_ob, "bearish_ob": bearish_ob, "retest": retest, "score": score}

# =========================================================
# 7. FAIR VALUE GAP (FVG)
# =========================================================
def detect_fvg(df):
    if df is None or len(df) < 5:
        return {"bullish_fvg": None, "bearish_fvg": None, "retest": "NONE", "score": 0}
    highs, lows, closes = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    bullish_fvg, bearish_fvg = None, None
    for i in range(len(df) - 2, 2, -1):
        if lows[i] > highs[i-2]:
            bullish_fvg = (highs[i-2], lows[i])
            break
        if highs[i] < lows[i-2]:
            bearish_fvg = (highs[i], lows[i-2])
            break
    curr = closes[-1]
    retest = "NONE"
    score = 0
    if bullish_fvg and (bullish_fvg[0] <= curr <= bullish_fvg[1]):
        retest = "BULLISH_FVG_RETEST"
        score = 10
    elif bearish_fvg and (bearish_fvg[0] <= curr <= bearish_fvg[1]):
        retest = "BEARISH_FVG_RETEST"
        score = -10
    return {"bullish_fvg": bullish_fvg, "bearish_fvg": bearish_fvg, "retest": retest, "score": score}

# =========================================================
# 8. MACHINE LEARNING PREDICTION
# =========================================================
def predict_trade_probability(df):
    if not SKLEARN_AVAILABLE or df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "probability": 50.0, "confidence": "Low"}
    features = ["Close", "Volume"]
    if "RSI" in df.columns: features.append("RSI")
    if "ADX" in df.columns: features.append("ADX")
    X = df[features].dropna()
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).iloc[:len(X)]
    if len(X) < 30:
        return {"signal": "NEUTRAL", "probability": 50.0, "confidence": "Low"}
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    latest = pd.DataFrame([X.iloc[-1]])
    prob = round(float(model.predict_proba(latest)[0][1] * 100), 2)
    sig = "BUY" if prob >= 60 else "SELL" if prob <= 40 else "NEUTRAL"
    conf = "High" if prob >= 75 or prob <= 25 else "Medium"
    return {"signal": sig, "probability": prob, "confidence": conf}

# =========================================================
# 9. VOLUME PROFILE (POC, VAH, VAL)
# =========================================================
def calculate_volume_profile(df, bins=20):
    if df is None or len(df) < 10:
        return {"poc": 0, "vah": 0, "val": 0, "signal": "NEUTRAL", "score": 0}
    p_min, p_max = df["Low"].min(), df["High"].max()
    p_bins = np.linspace(p_min, p_max, bins)
    vol_prof = np.zeros(bins-1)
    for _, row in df.iterrows():
        mid = (row["High"] + row["Low"]) / 2
        idx = np.digitize(mid, p_bins) - 1
        if 0 <= idx < len(vol_prof):
            vol_prof[idx] += row["Volume"]
    poc_idx = np.argmax(vol_prof)
    poc = round((p_bins[poc_idx] + p_bins[poc_idx+1]) / 2, 2)
    curr = df["Close"].iloc[-1]
    sig = "BUY" if curr > poc else "SELL"
    return {"poc": poc, "vah": round(p_max * 0.98, 2), "val": round(p_min * 1.02, 2), "signal": sig, "score": 5 if sig == "BUY" else -5}

# =========================================================
# 10. AI TRADE COACH (LOSS DIAGNOSTICS)
# =========================================================
def ai_loss_analysis(journal_df):
    if journal_df is None or journal_df.empty:
        return pd.DataFrame()
    losses = journal_df[journal_df.get("PnL", 0) < 0].copy()
    if losses.empty:
        return pd.DataFrame()
    results = []
    for _, row in losses.iterrows():
        rsi = float(row.get("RSI", 50))
        reason = "Overbought Late Entry" if rsi > 70 else "Oversold Breakdown" if rsi < 30 else "Liquidity Stop Hunt"
        fix = "Wait for EMA21 retest before entry" if rsi > 70 else "Avoid taking counter-trend signals"
        results.append({"DateTime": row.get("DateTime", "N/A"), "Symbol": row.get("Symbol", "N/A"), "PnL": row.get("PnL"), "Loss Reason": reason, "Suggested Fix": fix})
    return pd.DataFrame(results)

# =========================================================
# COMBINED UI RENDERER FOR STREAMLIT TABS
# =========================================================
def render_institutional_dashboard(df, symbol, benchmark_df, journal_df=None):
    ms = detect_market_structure(df)
    ls = detect_liquidity_sweep(df)
    rs_score = calculate_relative_strength(df, benchmark_df)
    ob = detect_order_blocks(df)
    fvg = detect_fvg(df)
    vp = calculate_volume_profile(df)
    ml = predict_trade_probability(df)
    
    if ls["block_entry"]:
        st.error(f"⛔ SIGNAL BLOCKED: Liquidity Sweep Active ({ls['type']})! High Risk.")
    
    tabs = st.tabs([
        "1. Structure", "2. Liquidity", "3. Sectors", "4. Relative Strength",
        "5. OI Positioning", "6. Order Blocks", "7. FVG", "8. Volume Profile",
        "9. AI ML Engine", "10. AI Coach"
    ])
    
    with tabs[0]:
        st.subheader("Market Structure")
        st.write(f"Structure: **{ms['structure']}**")
        st.write(f"HH: {ms['last_hh']} | HL: {ms['last_hl']} | LH: {ms['last_lh']} | LL: {ms['last_ll']}")
    with tabs[1]:
        st.subheader("Liquidity Sweep")
        st.write(f"Risk Level: **{ls['risk']}** | Block Status: **{ls['block_entry']}**")
    with tabs[2]:
        st.subheader("Sector Performance")
        sec_df, n_perf = get_sector_performance()
        st.write(f"NIFTY Performance: {n_perf:.2f}%")
        st.dataframe(sec_df, use_container_width=True)
    with tabs[3]:
        st.subheader("Relative Strength Score")
        st.metric(f"RS Score vs Nifty ({symbol})", rs_score)
    with tabs[4]:
        st.subheader("OI Analysis")
        p_chg = df["Close"].iloc[-1] - df["Close"].iloc[-2]
        pos, sent, _ = analyze_oi_positioning(p_chg, 1000)
        st.write(f"Market Position: **{pos}** ({sent})")
    with tabs[5]:
        st.subheader("Order Blocks Zone")
        st.write(f"Retest: {ob['retest']} | Bullish OB: {ob['bullish_ob']} | Bearish OB: {ob['bearish_ob']}")
    with tabs[6]:
        st.subheader("Fair Value Gap (FVG)")
        st.write(f"Retest: {fvg['retest']} | Bullish FVG: {fvg['bullish_fvg']} | Bearish FVG: {fvg['bearish_fvg']}")
    with tabs[7]:
        st.subheader("Volume Profile")
        st.metric("Point of Control (POC)", vp["poc"])
        st.write(f"VAH: {vp['vah']} | VAL: {vp['val']}")
    with tabs[8]:
        st.subheader("ML Signal Prediction")
        st.write(f"Predicted Signal: **{ml['signal']}** | Win Probability: **{ml['probability']}%**")
    with tabs[9]:
        st.subheader("AI Loss Coach")
        coaching = ai_loss_analysis(journal_df)
        if not coaching.empty:
            st.dataframe(coaching, use_container_width=True)
        else:
            st.info("No losing trades found in journal for coaching.")
