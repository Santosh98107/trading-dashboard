import os
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Elite Trading Dashboard Ultimate+", layout="wide")

# ===== BASE PATHS =====
BASE_DIR = Path(__file__).resolve().parent

# ===== FILE PATHS =====
JOURNAL_FILE = BASE_DIR / "trade_journal.csv"
SIGNAL_STATE_FILE = BASE_DIR / "last_signal_state.txt"
PAPER_STATE_FILE = BASE_DIR / "paper_trade_state.csv"
USERS_FILE = BASE_DIR / "users.csv"
SESSION_SUMMARY_FILE = BASE_DIR / "daily_summary.csv"

# ===== STOCK DICTIONARIES =====
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

stocks = {**INDEX_STOCKS, **TOP_INDIAN_COMPANIES, **BANKING_STOCKS}

# ===== AUTHENTICATION =====
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users():
    try:
        users = pd.read_csv(USERS_FILE)
        if "username" not in users.columns or "password" not in users.columns:
            raise ValueError("Missing username/password columns")
        users["username"] = users["username"].astype(str).str.strip()
        users["password"] = users["password"].astype(str).str.strip()
        return users
    except Exception:
        return pd.DataFrame(columns=["username", "password"])


def init_users():
    default_user = pd.DataFrame([
        {"username": DEFAULT_USERNAME, "password": hash_password(DEFAULT_PASSWORD)}
    ])

    if not USERS_FILE.exists():
        default_user.to_csv(USERS_FILE, index=False)
        return

    try:
        users = load_users()
        admin_exists = ((users["username"] == DEFAULT_USERNAME) & (users["password"] == hash_password(DEFAULT_PASSWORD))).any()
        if users.empty or not admin_exists:
            merged = pd.concat([users, default_user], ignore_index=True)
            merged = merged.drop_duplicates(subset=["username"], keep="last")
            merged.to_csv(USERS_FILE, index=False)
    except Exception:
        default_user.to_csv(USERS_FILE, index=False)


def login_panel():
    init_users()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.sidebar.subheader("🔐 Login")
        username = str(st.sidebar.text_input("Username", value="")).strip()
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            try:
                users = load_users()
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

# ===== FILE INITIALIZATION =====
def init_journal():
    if not JOURNAL_FILE.exists():
        cols = [
            "DateTime", "Symbol", "Timeframe", "Signal", "EntryPrice", "ExitPrice", "Quantity",
            "StopLoss", "Target1", "Target2", "TrailingStop", "Status", "PnL", "PnLPercent",
            "Notes", "TradeMode"
        ]
        pd.DataFrame(columns=cols).to_csv(JOURNAL_FILE, index=False)


def init_paper_state():
    if not PAPER_STATE_FILE.exists():
        pd.DataFrame([{"Balance": 100000.0, "UsedMargin": 0.0, "OpenPositions": 0}]).to_csv(PAPER_STATE_FILE, index=False)


def init_daily_summary():
    if not SESSION_SUMMARY_FILE.exists():
        pd.DataFrame(columns=["Date", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"]).to_csv(SESSION_SUMMARY_FILE, index=False)


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

# ===== JOURNAL OPERATIONS =====
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

# ===== DATA FETCHING =====
def fetch_data(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return pd.DataFrame()
        df = df[needed].dropna().copy()
        df["Volume"] = df["Volume"].fillna(1).replace(0, 1)
        return df
    except Exception:
        return pd.DataFrame()

# ===== TECHNICAL INDICATORS =====
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

# ===== CANDLESTICK PATTERNS =====
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

    # Doji patterns
    df["Doji"] = body <= rng * 0.1
    df["LongLeggedDoji"] = df["Doji"] & (upper > body * 2) & (lower > body * 2)
    df["DragonflyDoji"] = df["Doji"] & (lower > body * 2) & (upper <= body)
    df["GravestoneDoji"] = df["Doji"] & (upper > body * 2) & (lower <= body)

    # Hammer patterns
    df["Hammer"] = (lower >= body * 2) & (upper <= body)
    df["HangingMan"] = df["Hammer"] & (c < o)
    df["InvertedHammer"] = (upper >= body * 2) & (lower <= body)
    df["ShootingStar"] = df["InvertedHammer"] & (c < o)

    # Marubozu patterns
    df["BullishMarubozu"] = (c > o) & (upper <= body * 0.1) & (lower <= body * 0.1)
    df["BearishMarubozu"] = (c < o) & (upper <= body * 0.1) & (lower <= body * 0.1)

    # Engulfing patterns
    df["BullishEngulfing"] = (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    df["BearishEngulfing"] = (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)

    # Harami patterns
    df["BullishHarami"] = (prev_c < prev_o) & (c > o) & (o > prev_c) & (c < prev_o)
    df["BearishHarami"] = (prev_c > prev_o) & (c < o) & (o < prev_c) & (c > prev_o)

    # Piercing & Dark Cloud
    df["PiercingPattern"] = (prev_c < prev_o) & (c > o) & (o < prev_l) & (c > (prev_o + prev_c) / 2) & (c < prev_o)
    df["DarkCloudCover"] = (prev_c > prev_o) & (c < o) & (o > prev_h) & (c < (prev_o + prev_c) / 2) & (c > prev_o)

    # Spinning Top
    df["SpinningTop"] = (body / rng < 0.3) & (upper > body) & (lower > body)

    # Inside/Outside bars
    df["InsideBar"] = (h < prev_h) & (l > prev_l)
    df["OutsideBar"] = (h > prev_h) & (l < prev_l)

    # Kicker patterns
    df["BullishKicker"] = (prev_c < prev_o) & (c > o) & (o > prev_o)
    df["BearishKicker"] = (prev_c > prev_o) & (c < o) & (o < prev_o)

    # Morning Star & Evening Star
    df["MorningStar"] = (prev_c.shift(1) < prev_o.shift(1)) & (prev_body < prev_body.rolling(5).mean()) & (c > o) & (c > ((prev_o.shift(1) + prev_c.shift(1)) / 2))
    df["EveningStar"] = (prev_c.shift(1) > prev_o.shift(1)) & (prev_body < prev_body.rolling(5).mean()) & (c < o) & (c < ((prev_o.shift(1) + prev_c.shift(1)) / 2))

    # Three White Soldiers & Three Black Crows
    df["ThreeWhiteSoldiers"] = (c.shift(2) > o.shift(2)) & (c.shift(1) > o.shift(1)) & (c > o) & (c.shift(1) > c.shift(2)) & (c > c.shift(1))
    df["ThreeBlackCrows"] = (c.shift(2) < o.shift(2)) & (c.shift(1) < o.shift(1)) & (c < o) & (c.shift(1) < c.shift(2)) & (c < c.shift(1))

    return df

# ===== ALL INDICATORS =====
def add_indicators(df):
    df = df.copy()

    # Moving averages
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # VWAP
    vol_cum = df["Volume"].cumsum()
    df["VWAP"] = ((df["Close"] * df["Volume"]).cumsum() / vol_cum.replace(0, 1))
    df["VWAP"] = df["VWAP"].replace([np.inf, -np.inf], np.nan).fillna(df["Close"])

    # RSI
    df["RSI"] = compute_rsi(df["Close"], 14).replace([np.inf, -np.inf], np.nan).fillna(50).clip(lower=1, upper=99)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["SignalLine"]

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Bollinger Bands
    df["SMA20"] = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20

    # Volume analysis
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Spike"] = df["Volume"] > (1.5 * df["Volume_MA20"])
    df["RVOL"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)

    # Pivot analysis
    df["PivotHigh"] = df["High"].rolling(20).max()
    df["PivotLow"] = df["Low"].rolling(20).min()
    df["Breakout"] = (df["Close"] > df["PivotHigh"].shift(1)) & (df["Volume_Spike"])
    df["Breakdown"] = (df["Close"] < df["PivotLow"].shift(1)) & (df["Volume_Spike"])
    df["RetestBull"] = (df["Close"] > df["PivotHigh"].shift(1)) & (df["Low"] <= df["PivotHigh"].shift(1))
    df["RetestBear"] = (df["Close"] < df["PivotLow"].shift(1)) & (df["High"] >= df["PivotLow"].shift(1))

    # Stochastic
    k, d = compute_stochastic(df)
    df["StochK"] = k.rolling(3).mean()
    df["StochD"] = d.rolling(3).mean()

    # ADX
    df["ADX"], df["PlusDI"], df["MinusDI"] = compute_adx(df)

    # Candlestick patterns
    df = add_candlestick_patterns(df)

    return df

# ===== PATTERN DETECTION =====
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

# ===== MAIN APP =====
def main():
    login_panel()

    st.title("🚀 Elite Trading Dashboard Ultimate+")
    st.subheader("Professional Trading & Analysis Terminal")

    st.sidebar.header("⚙️ Settings")
    symbol_name = st.sidebar.selectbox("📊 Select Stock", list(stocks.keys()), index=0)
    symbol = stocks[symbol_name]
    timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1h", "4h", "1d"], index=2)

    period_map = {"1h": "60d", "4h": "90d", "1d": "1y"}
    period = period_map.get(timeframe, "90d")

    df = fetch_data(symbol, timeframe, period)
    if df.empty:
        st.error("❌ Could not fetch data. Try again later.")
        return

    df = add_indicators(df)
    latest = df.iloc[-1].to_dict()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Price", f"₹{latest.get('Close', 0):.2f}")
    col2.metric("📊 RSI", f"{latest.get('RSI', 50):.1f}")
    col3.metric("🔊 Volume", f"{latest.get('Volume', 0):,.0f}")
    col4.metric("🎯 ADX", f"{latest.get('ADX', 0):.1f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="red")))
    fig.update_layout(title=f"{symbol_name} - {timeframe}", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Detected Patterns")
    patterns = get_detected_patterns(latest)
    if patterns:
        for pattern in patterns:
            st.success(pattern)
    else:
        st.info("No patterns detected")

    st.subheader("📓 Trade Journal")
    journal = load_journal()
    if not journal.empty:
        st.dataframe(journal.tail(10), use_container_width=True)
    else:
        st.info("No trades yet")

    st.subheader("📊 Paper Trading")
    state = load_paper_state()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Balance", f"₹{state.loc[0, 'Balance']:.2f}")
    col2.metric("💸 Used Margin", f"₹{state.loc[0, 'UsedMargin']:.2f}")
    col3.metric("📍 Open Positions", int(state.loc[0, 'OpenPositions']))


if __name__ == "__main__":
    main()
