import os
import hashlib
from datetime import datetime, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Elite Trading Dashboard Ultimate+", layout="wide")

JOURNAL_FILE = "trade_journal.csv"
SIGNAL_STATE_FILE = "last_signal_state.txt"
PAPER_STATE_FILE = "paper_trade_state.csv"
USERS_FILE = "users.csv"
SESSION_SUMMARY_FILE = "daily_summary.csv"


TOP_INDIAN_COMPANIES = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "TITAN": "TITAN.NS"
}

BANKING_STOCKS = {
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "PNB": "PNB.NS",
    "INDUSINDBK": "INDUSINDBK.NS"
}

INDEX_STOCKS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

stocks = {
    **INDEX_STOCKS,
    **TOP_INDIAN_COMPANIES,
    **BANKING_STOCKS
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_users():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame([
            {"username": "admin", "password": hash_password("admin123")}
        ]).to_csv(USERS_FILE, index=False)


def login_panel():
    init_users()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.sidebar.subheader("🔐 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            try:
                users = pd.read_csv(USERS_FILE)
                entered_hash = hash_password(password)
                valid = ((users["username"] == username) & (users["password"] == entered_hash)).any()

                if valid:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.sidebar.success("Login successful")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                st.sidebar.error(f"Login error: {e}")
        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        cols = [
            "DateTime", "Symbol", "Timeframe", "Signal", "EntryPrice",
            "ExitPrice", "Quantity", "StopLoss", "Target1", "Target2",
            "TrailingStop", "Status", "PnL", "PnLPercent", "Notes", "TradeMode"
        ]
        pd.DataFrame(columns=cols).to_csv(JOURNAL_FILE, index=False)


def init_paper_state():
    if not os.path.exists(PAPER_STATE_FILE):
        pd.DataFrame([{
            "Balance": 100000.0,
            "UsedMargin": 0.0,
            "OpenPositions": 0
        }]).to_csv(PAPER_STATE_FILE, index=False)


def init_daily_summary():
    if not os.path.exists(SESSION_SUMMARY_FILE):
        pd.DataFrame(columns=[
            "Date", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]).to_csv(SESSION_SUMMARY_FILE, index=False)


def load_journal():
    init_journal()
    return pd.read_csv(JOURNAL_FILE)


def save_journal(df):
    df.to_csv(JOURNAL_FILE, index=False)


def load_paper_state():
    init_paper_state()
    return pd.read_csv(PAPER_STATE_FILE)


def save_paper_state(df):
    df.to_csv(PAPER_STATE_FILE, index=False)


def add_trade_to_journal(symbol, timeframe, signal, entry_price, quantity, stop_loss, target1, target2, trailing_stop, notes="", trade_mode="MANUAL"):
    journal = load_journal()
    new_row = {
        "DateTime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Symbol": symbol,
        "Timeframe": timeframe,
        "Signal": signal,
        "EntryPrice": entry_price,
        "ExitPrice": np.nan,
        "Quantity": quantity,
        "StopLoss": stop_loss,
        "Target1": target1,
        "Target2": target2,
        "TrailingStop": trailing_stop,
        "Status": "OPEN",
        "PnL": np.nan,
        "PnLPercent": np.nan,
        "Notes": notes,
        "TradeMode": trade_mode
    }
    journal = pd.concat([journal, pd.DataFrame([new_row])], ignore_index=True)
    save_journal(journal)


def close_trade_in_journal(index_id, exit_price):
    journal = load_journal()
    if index_id in journal.index:
        row = journal.loc[index_id]
        entry = float(row["EntryPrice"])
        qty = float(row["Quantity"])
        signal = str(row["Signal"]).upper()

        if "BUY" in signal or "CALL" in signal:
            pnl = (exit_price - entry) * qty
            pnl_pct = ((exit_price - entry) / entry) * 100 if entry != 0 else 0
        else:
            pnl = (entry - exit_price) * qty
            pnl_pct = ((entry - exit_price) / entry) * 100 if entry != 0 else 0

        journal.at[index_id, "ExitPrice"] = exit_price
        journal.at[index_id, "Status"] = "CLOSED"
        journal.at[index_id, "PnL"] = round(pnl, 2)
        journal.at[index_id, "PnLPercent"] = round(pnl_pct, 2)
        save_journal(journal)


def update_daily_summary():
    init_daily_summary()
    journal = load_journal()
    closed = journal[journal["Status"] == "CLOSED"].copy()

    if closed.empty:
        return

    closed["DateOnly"] = pd.to_datetime(
        closed["DateTime"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    ).dt.date

    today = datetime.now().date()
    day_df = closed[closed["DateOnly"] == today]

    if day_df.empty:
        return

    total = len(day_df)
    wins = (pd.to_numeric(day_df["PnL"], errors="coerce").fillna(0) > 0).sum()
    losses = total - wins
    pnl = pd.to_numeric(day_df["PnL"], errors="coerce").fillna(0).sum()
    win_rate = (wins / total * 100) if total > 0 else 0

    summary = pd.read_csv(SESSION_SUMMARY_FILE)
    summary = summary[summary["Date"] != str(today)]

    new_row = {
        "Date": str(today),
        "TotalTrades": total,
        "Wins": wins,
        "Losses": losses,
        "RealizedPnL": round(pnl, 2),
        "WinRate": round(win_rate, 2)
    }

    summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
    summary.to_csv(SESSION_SUMMARY_FILE, index=False)


def paper_trade_entry(price, quantity):
    state = load_paper_state()
    bal = float(state.loc[0, "Balance"])
    used = float(state.loc[0, "UsedMargin"])
    cost = price * quantity

    if bal >= cost:
        state.loc[0, "Balance"] = bal - cost
        state.loc[0, "UsedMargin"] = used + cost
        state.loc[0, "OpenPositions"] = int(state.loc[0, "OpenPositions"]) + 1
        save_paper_state(state)
        return True
    return False


def paper_trade_exit(entry_price, exit_price, quantity, signal):
    state = load_paper_state()
    bal = float(state.loc[0, "Balance"])
    used = float(state.loc[0, "UsedMargin"])
    invested = entry_price * quantity

    if "BUY" in signal.upper() or "CALL" in signal.upper():
        pnl = (exit_price - entry_price) * quantity
    else:
        pnl = (entry_price - exit_price) * quantity

    state.loc[0, "Balance"] = bal + invested + pnl
    state.loc[0, "UsedMargin"] = max(0.0, used - invested)
    state.loc[0, "OpenPositions"] = max(0, int(state.loc[0, "OpenPositions"]) - 1)
    save_paper_state(state)
    return pnl


def fetch_data(ticker, interval, period):
    try:
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df[needed].dropna().copy()
        df["Volume"] = df["Volume"].fillna(1)
        df["Volume"] = df["Volume"].replace(0, 1)
        return df
    except Exception:
        return pd.DataFrame()


def resample_to_10m(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("10min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    k = 100 * ((df["Close"] - low_min) / (high_max - low_min).replace(0, 1e-10))
    d = k.rolling(d_window).mean()
    return k, d


def compute_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm_raw = high.diff()
    minus_dm_raw = -low.diff()

    plus_dm = np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0.0)
    minus_dm = np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().replace(0, 1e-10)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def add_candlestick_patterns(df):
    df = df.copy()

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    body = (c - o).abs()
    rng = (h - l).replace(0, 1e-10)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_body = body.shift(1)

    df["Doji"] = body <= rng * 0.1
    df["LongLeggedDoji"] = df["Doji"] & (upper > body * 2) & (lower > body * 2)
    df["DragonflyDoji"] = df["Doji"] & (lower > body * 2) & (upper <= body)
    df["GravestoneDoji"] = df["Doji"] & (upper > body * 2) & (lower <= body)

    df["Hammer"] = (lower >= body * 2) & (upper <= body)
    df["HangingMan"] = df["Hammer"] & (c < o)
    df["InvertedHammer"] = (upper >= body * 2) & (lower <= body)
    df["ShootingStar"] = df["InvertedHammer"] & (c < o)

    df["BullishMarubozu"] = (c > o) & (upper <= body * 0.1) & (lower <= body * 0.1)
    df["BearishMarubozu"] = (c < o) & (upper <= body * 0.1) & (lower <= body * 0.1)

    df["BullishEngulfing"] = (
        (prev_c < prev_o) &
        (c > o) &
        (o <= prev_c) &
        (c >= prev_o)
    )

    df["BearishEngulfing"] = (
        (prev_c > prev_o) &
        (c < o) &
        (o >= prev_c) &
        (c <= prev_o)
    )

    df["BullishHarami"] = (
        (prev_c < prev_o) &
        (c > o) &
        (o > prev_c) &
        (c < prev_o)
    )

    df["BearishHarami"] = (
        (prev_c > prev_o) &
        (c < o) &
        (o < prev_c) &
        (c > prev_o)
    )

    df["PiercingPattern"] = (
        (prev_c < prev_o) &
        (c > o) &
        (o < prev_l) &
        (c > (prev_o + prev_c) / 2) &
        (c < prev_o)
    )

    df["DarkCloudCover"] = (
        (prev_c > prev_o) &
        (c < o) &
        (o > prev_h) &
        (c < (prev_o + prev_c) / 2) &
        (c > prev_o)
    )

    df["SpinningTop"] = (body / rng < 0.3) & (upper > body) & (lower > body)

    df["InsideBar"] = (h < prev_h) & (l > prev_l)
    df["OutsideBar"] = (h > prev_h) & (l < prev_l)

    df["BullishKicker"] = (prev_c < prev_o) & (c > o) & (o > prev_o)
    df["BearishKicker"] = (prev_c > prev_o) & (c < o) & (o < prev_o)

    df["MorningStar"] = (
        (prev_c.shift(1) < prev_o.shift(1)) &
        (prev_body < prev_body.rolling(5).mean()) &
        (c > o) &
        (c > ((prev_o.shift(1) + prev_c.shift(1)) / 2))
    )

    df["EveningStar"] = (
        (prev_c.shift(1) > prev_o.shift(1)) &
        (prev_body < prev_body.rolling(5).mean()) &
        (c < o) &
        (c < ((prev_o.shift(1) + prev_c.shift(1)) / 2))
    )

    df["ThreeWhiteSoldiers"] = (
        (c.shift(2) > o.shift(2)) &
        (c.shift(1) > o.shift(1)) &
        (c > o) &
        (c.shift(1) > c.shift(2)) &
        (c > c.shift(1))
    )

    df["ThreeBlackCrows"] = (
        (c.shift(2) < o.shift(2)) &
        (c.shift(1) < o.shift(1)) &
        (c < o) &
        (c.shift(1) < c.shift(2)) &
        (c < c.shift(1))
    )

    return df


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    vol_cum = df["Volume"].cumsum()
    df["VWAP"] = ((df["Close"] * df["Volume"]).cumsum() / vol_cum.replace(0, 1))
    df["VWAP"] = df["VWAP"].replace([np.inf, -np.inf], np.nan)
    df["VWAP"] = df["VWAP"].fillna(df["Close"])

    df["RSI"] = compute_rsi(df["Close"], 14)
    df["RSI"] = df["RSI"].replace([np.inf, -np.inf], np.nan).fillna(50).clip(lower=1, upper=99)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["SignalLine"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["SMA20"] = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Spike"] = df["Volume"] > (1.5 * df["Volume_MA20"])
    df["RVOL"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)

    df["PivotHigh"] = df["High"].rolling(20).max()
    df["PivotLow"] = df["Low"].rolling(20).min()
    df["Breakout"] = (df["Close"] > df["PivotHigh"].shift(1)) & (df["Volume_Spike"])
    df["Breakdown"] = (df["Close"] < df["PivotLow"].shift(1)) & (df["Volume_Spike"])

    df["RetestBull"] = (df["Close"] > df["PivotHigh"].shift(1)) & (df["Low"] <= df["PivotHigh"].shift(1))
    df["RetestBear"] = (df["Close"] < df["PivotLow"].shift(1)) & (df["High"] >= df["PivotLow"].shift(1))

    k, d = compute_stochastic(df)
    df["StochK"] = k.rolling(3).mean()
    df["StochD"] = d.rolling(3).mean()
    df["ADX"], df["PlusDI"], df["MinusDI"] = compute_adx(df)

    candle_body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-10)
    df["BodyStrength"] = candle_body / candle_range

    df["BuyMarker"] = np.where(
        (df["EMA20"] > df["EMA50"]) & (df["EMA20"].shift(1) <= df["EMA50"].shift(1)),
        df["Low"] * 0.995,
        np.nan
    )

    df["SellMarker"] = np.where(
        (df["EMA20"] < df["EMA50"]) & (df["EMA20"].shift(1) >= df["EMA50"].shift(1)),
        df["High"] * 1.005,
        np.nan
    )

    df = add_candlestick_patterns(df)
    return df


def get_detected_patterns(latest):
    pattern_map = {
        "Doji": "🟡 Doji",
        "LongLeggedDoji": "🟡 Long-Legged Doji",
        "DragonflyDoji": "🟢 Dragonfly Doji",
        "GravestoneDoji": "🔴 Gravestone Doji",
        "Hammer": "🟢 Hammer",
        "HangingMan": "🔴 Hanging Man",
        "InvertedHammer": "🟢 Inverted Hammer",
        "ShootingStar": "🔴 Shooting Star",
        "BullishMarubozu": "🟢 Bullish Marubozu",
        "BearishMarubozu": "🔴 Bearish Marubozu",
        "BullishEngulfing": "🟢 Bullish Engulfing",
        "BearishEngulfing": "🔴 Bearish Engulfing",
        "BullishHarami": "🟢 Bullish Harami",
        "BearishHarami": "🔴 Bearish Harami",
        "PiercingPattern": "🟢 Piercing Pattern",
        "DarkCloudCover": "🔴 Dark Cloud Cover",
        "SpinningTop": "🟡 Spinning Top",
        "InsideBar": "🟡 Inside Bar",
        "OutsideBar": "🟡 Outside Bar",
        "BullishKicker": "🟢 Bullish Kicker",
        "BearishKicker": "🔴 Bearish Kicker",
        "MorningStar": "🟢 Morning Star",
        "EveningStar": "🔴 Evening Star",
        "ThreeWhiteSoldiers": "🟢 Three White Soldiers",
        "ThreeBlackCrows": "🔴 Three Black Crows",
        "RetestBull": "🚀 Bullish Retest",
        "RetestBear": "🔻 Bearish Retest"
    }

    found = []
    for key, label in pattern_map.items():
        if bool(latest.get(key, False)):
            found.append(label)
    return found


def get_higher_timeframe(tf):
    if tf == "1m":
        return "5m", "30d"
    if tf == "5m":
        return "15m", "30d"
    if tf == "10m":
        return "30m", "60d"
    if tf == "15m":
        return "30m", "60d"
    if tf == "30m":
        return "60m", < 18:
        score -= 10

    if volume_spike:
        score += 5
    if breakout:
        score += 6
    if breakdown:
        score += 6
    if htf_bias in ["bullish", "bearish"]:
        score += 5
    if sideways:
        score -= 12

    if rr_ratio is not None:
        if rr_ratio >= 2:
            score += 6
        elif rr_ratio < 1.2:
            score -= 8

    if backtest_win_rate >= 60:
        score += 8
    elif backtest_win_rate < 45:
        score -= 8

    return int(max(5, min(score, 95)))


def suggest_option_strike(symbol, spot_price, signal_type):
    if symbol == "NIFTY":
        step = 50
    elif symbol == "BANKNIFTY":
        step = 100
    else:
        step = 10

    atm = round(spot_price / step) * step

    if signal_type == "buy":
        return {
            "type": "CALL",
            "atm": atm,
            "otm": atm + step,
            "text": f"Suggested CALL strikes: ATM {atm} CE, OTM {atm + step} CE"
        }
    elif signal_type == "sell":
        return {
            "type": "PUT",
            "atm": atm,
            "otm": atm - step,
            "text": f"Suggested PUT strikes: ATM {atm} PE, OTM {atm - step} PE"
        }

    return {
        "type": "NONE",
        "atm": None,
        "otm": None,
        "text": "No option strike suggestion"
    }


def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def calculate_pcr_metrics(total_put_oi, total_call_oi, total_put_vol, total_call_vol):
    pcr_oi = None
    pcr_vol = None
    pcr_score = 0
    pcr_bias = "NEUTRAL"

    try:
        if total_call_oi and total_call_oi != 0:
            pcr_oi = total_put_oi / total_call_oi
        if total_call_vol and total_call_vol != 0:
            pcr_vol = total_put_vol / total_call_vol
    except Exception:
        pass

    if pcr_oi is not None:
        if pcr_oi > 1.2:
            pcr_score += 2
        elif pcr_oi > 1.0:
            pcr_score += 1
        elif pcr_oi < 0.8:
            pcr_score -= 2
        elif pcr_oi < 1.0:
            pcr_score -= 1

    if pcr_vol is not None:
        if pcr_vol > 1.2:
            pcr_score += 1
        elif pcr_vol < 0.8:
            pcr_score -= 1

    if pcr_score >= 2:
        pcr_bias = "BULLISH"
    elif pcr_score <= -2:
        pcr_bias = "BEARISH"

    return {
        "pcr_oi": None if pcr_oi is None else round(pcr_oi, 4),
        "pcr_vol": None if pcr_vol is None else round(pcr_vol, 4),
        "pcr_score": pcr_score,
        "pcr_bias": pcr_bias
    }


def calculate_macro_score(crude_now, crude_prev, brent_now, brent_prev, usdinr_now, usdinr_prev):
    score = 0

    try:
        if crude_now is not None and crude_prev is not None:
            score += 1 if crude_now < crude_prev else -1
        if brent_now is not None and brent_prev is not None:
            score += 1 if brent_now < brent_prev else -1
        if usdinr_now is not None and usdinr_prev is not None:
            score += 1 if usdinr_now < usdinr_prev else -1
    except Exception:
        pass

    bias = "NEUTRAL"
    if score >= 2:
        bias = "BULLISH"
    elif score <= -2:
        bias = "BEARISH"

    return {"macro_score": score, "macro_bias": bias}


def calculate_breadth_score(top_green, top_red, breadth_pct):
    score = 0

    try:
        if breadth_pct is not None:
            if breadth_pct >= 60:
                score += 2
            elif breadth_pct >= 50:
                score += 1
            elif breadth_pct < 40:
                score -= 2
            elif breadth_pct < 50:
                score -= 1

        if top_green is not None and top_red is not None:
            if top_green > top_red:
                score += 1
            elif top_red > top_green:
                score -= 1
    except Exception:
        pass

    bias = "NEUTRAL"
    if score >= 2:
        bias = "BULLISH"
    elif score <= -2:
        bias = "BEARISH"

    return {"breadth_score": score, "breadth_bias": bias}


def calculate_bank_score(hdfc, icici, sbin, axis, kotak):
    vals = [hdfc, icici, sbin, axis, kotak]
    pos = 0
    neg = 0

    for v in vals:
        if v is None or pd.isna(v):
            continue
        if v > 0:
            pos += 1
        elif v < 0:
            neg += 1

    score = 0
    if pos >= 4:
        score += 2
    elif pos >= 3:
        score += 1

    if neg >= 4:
        score -= 2
    elif neg >= 3:
        score -= 1

    bias = "NEUTRAL"
    if score >= 2:
        bias = "BULLISH"
    elif score <= -2:
        bias = "BEARISH"

    return {
        "bank_score": score,
        "bank_bias": bias,
        "positive_banks": pos,
        "negative_banks": neg
    }


def calculate_volume_score(latest):
    score = 0
    bias = "NEUTRAL"

    rvol = latest.get("RVOL", np.nan)
    volume_spike = bool(latest.get("Volume_Spike", False))
    close_price = latest.get("Close", np.nan)
    vwap = latest.get("VWAP", np.nan)

    if pd.notna(rvol):
        if rvol >= 2.0:
            score += 2
        elif rvol >= 1.2:
            score += 1
        elif rvol < 0.8:
            score -= 1

    if volume_spike:
        score += 1

    if pd.notna(close_price) and pd.notna(vwap):
        if close_price > vwap and score >= 2:
            bias = "BULLISH"
        elif close_price < vwap and score >= 2:
            bias = "BEARISH"
        elif score <= -1:
            bias = "WEAK"

    return {"volume_score": score, "volume_bias": bias}


def calculate_regime(latest):
    adx = latest.get("ADX", np.nan)
    atr = latest.get("ATR", np.nan)
    close_price = latest.get("Close", np.nan)

    score = 0
    regime = "NEUTRAL"

    if pd.notna(adx):
        if adx >= 25:
            score += 2
        elif adx < 15:
            score -= 1

    if pd.notna(atr) and pd.notna(close_price) and close_price != 0:
        atr_pct = (atr / close_price) * 100
        if atr_pct > 1.5:
            score += 1

    if score >= 2:
        regime = "TRENDING"
    elif score <= -1:
        regime = "SIDEWAYS"

    return {"regime": regime, "regime_score": score}


def fetch_geopolitical_news(news_api_key=None):
    if not news_api_key:
        return [{"title": "No live API key provided", "source": "System", "sentiment": "Neutral"}]

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "geopolitics OR war OR sanctions OR crude oil OR middle east OR fed OR china taiwan OR russia ukraine",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": news_api_key
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "No title")
            source = item.get("source", {}).get("name", "Unknown")
            sentiment = "Neutral"

            lower_title = title.lower()
            negative_words = ["war", "attack", "sanction", "conflict", "missile", "tension", "crisis"]
            positive_words = ["deal", "ceasefire", "agreement", "peace", "recovery"]

            if any(word in lower_title for word in negative_words):
                sentiment = "Negative"
            elif any(word in lower_title for word in positive_words):
                sentiment = "Positive"

            articles.append({
                "title": title,
                "source": source,
                "sentiment": sentiment
            })

        return articles if articles else [{"title": "No geopolitical news found", "source": "API", "sentiment": "Neutral"}]
    except Exception as e:
        return [{"title": f"News fetch error: {e}", "source": "System", "sentiment": "Neutral"}]


def calculate_news_score(news_items):
    score = 0
    for item in news_items:
        sentiment = str(item.get("sentiment", "Neutral")).
