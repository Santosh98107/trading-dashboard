import os
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
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
AUTH_STATE_FILE = BASE_DIR / "auth_state.txt"

# ===== STOCK DICTIONARIES =====
TOP_10_INDIAN_COMPANIES = {
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

FOREX_PAIRS = {
    "USD/INR": "USDINR=X"
}

MARKET_GROUPS = {
    "Top 10 Indian Companies": TOP_10_INDIAN_COMPANIES,
    "Banking Stocks": BANKING_STOCKS,
    "Indices": INDEX_STOCKS,
    "Forex": FOREX_PAIRS,
}

stocks = {**INDEX_STOCKS, **TOP_10_INDIAN_COMPANIES, **BANKING_STOCKS, **FOREX_PAIRS}

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


def read_server_session():
    try:
        if AUTH_STATE_FILE.exists():
            value = AUTH_STATE_FILE.read_text(encoding="utf-8").strip()
            if value:
                return value
    except Exception:
        pass
    return ""


def write_server_session(username: str):
    try:
        AUTH_STATE_FILE.write_text(str(username).strip(), encoding="utf-8")
    except Exception:
        pass


def clear_server_session():
    try:
        if AUTH_STATE_FILE.exists():
            AUTH_STATE_FILE.unlink()
    except Exception:
        pass


def is_valid_login(username: str, password: str) -> bool:
    users = load_users()
    entered_hash = hash_password(password)
    return bool(((users["username"] == str(username).strip()) & (users["password"] == entered_hash)).any())


def login_panel():
    init_users()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    saved_user = read_server_session()
    if not st.session_state.logged_in and saved_user and saved_user == DEFAULT_USERNAME:
        st.session_state.logged_in = True
        st.session_state.username = saved_user

    if not st.session_state.logged_in:
        st.sidebar.subheader("🔐 Login")
        username = str(st.sidebar.text_input("Username", value="")).strip()
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            try:
                if is_valid_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    write_server_session(username)
                    st.sidebar.success("Login successful")
                    st.rerun()
                else:
                    clear_server_session()
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                clear_server_session()
                st.sidebar.error(f"Login error: {e}")
        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        clear_server_session()
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


@st.cache_data(ttl=300)
def fetch_data_cached(ticker, interval, period):
    return fetch_data(ticker, interval, period)


def fetch_pcr_data(symbol="NIFTY"):
    try:
        symbol_url = symbol.upper()
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol_url}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        records = payload.get("records", {}).get("data", [])
        if not records:
            return None
        total_calls_oi = 0
        total_puts_oi = 0
        for item in records:
            ce = item.get("CE") or {}
            pe = item.get("PE") or {}
            total_calls_oi += float(ce.get("openInterest", 0) or 0)
            total_puts_oi += float(pe.get("openInterest", 0) or 0)
        if total_calls_oi == 0:
            return None
        return round(total_puts_oi / total_calls_oi, 2)
    except Exception:
        return None

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


def compute_heikin_ashi(df):
    df = df.copy()
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    df["HA_Open"] = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    df["HA_Open"] = df["HA_Open"].fillna(df["Open"])
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)
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

    # Trend levels
    df["TrendSupport"] = df["Low"].rolling(20).min()
    df["TrendResistance"] = df["High"].rolling(20).max()

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


def detect_trend_pattern(df):
    if df.empty or len(df) < 20:
        return "Trend not enough data"

    close = df["Close"].iloc[-1]
    ema20 = df["EMA20"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]

    if pd.notna(close) and pd.notna(ema20) and pd.notna(ema50):
        if close > ema20 > ema50:
            return "Bullish Uptrend ✅"
        if close < ema20 < ema50:
            return "Bearish Downtrend ❌"
        return "Sideways / Consolidation ➖"
    return "Trend not enough data"

# ===== MAIN APP =====
def main():
    login_panel()

    st.title("🚀 Elite Trading Dashboard Ultimate+")
    st.subheader("Professional Trading & Analysis Terminal")

    st.sidebar.header("⚙️ Settings")
    market_group = st.sidebar.selectbox("📊 Market Group", list(MARKET_GROUPS.keys()), index=0)
    symbol_name = st.sidebar.selectbox("📈 Select Asset", list(MARKET_GROUPS[market_group].keys()), index=0)
    symbol = MARKET_GROUPS[market_group][symbol_name]
    timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1h", "4h", "1d"], index=2)

    period_map = {"1h": "60d", "4h": "90d", "1d": "1y"}
    period = period_map.get(timeframe, "90d")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_seconds = st.sidebar.slider("Refresh Every (sec)", min_value=15, max_value=180, value=60, step=5)
    if auto_refresh:
        components.html(
            f"""
            <script>
                setTimeout(function(){{ window.location.reload(); }}, {refresh_seconds * 1000});
            </script>
            """,
            height=0,
            scrolling=False,
        )

    df = fetch_data_cached(symbol, timeframe, period)
    if df.empty:
        st.error("❌ Could not fetch data. Try again later.")
        return

    df = add_indicators(df)
    latest = df.iloc[-1].to_dict()
    trend_state = detect_trend_pattern(df)

    pcr_value = None
    if market_group in ["Indices", "Top 10 Indian Companies", "Banking Stocks", "Forex"]:
        pcr_value = fetch_pcr_data("NIFTY")

    st.caption(f"Market: {market_group} | Asset: {symbol_name} | Symbol: {symbol}")

    col1, col2, col3, col4, col5 = st.columns(5)
    if "USDINR" in symbol:
        price_label = f"₹{latest.get('Close', 0):.4f}"
    else:
        price_label = f"₹{latest.get('Close', 0):.2f}"
    col1.metric("📈 Price", price_label)
    col2.metric("📊 RSI", f"{latest.get('RSI', 50):.1f}")
    col3.metric("🔊 Volume", f"{latest.get('Volume', 0):,.0f}")
    col4.metric("🎯 ADX", f"{latest.get('ADX', 0):.1f}")
    col5.metric("🧠 PCR", f"{pcr_value:.2f}" if pcr_value is not None else "N/A")

    st.subheader("📌 Trend & Market Context")
    st.success(trend_state)
    st.write(f"- Support Zone: ₹{latest.get('TrendSupport', 0):.2f}")
    st.write(f"- Resistance Zone: ₹{latest.get('TrendResistance', 0):.2f}")
    st.write(f"- MACD: {latest.get('MACD', 0):.4f} | Signal: {latest.get('SignalLine', 0):.4f} | Histogram: {latest.get('MACD_Hist', 0):.4f}")

    use_heikin_ashi = st.checkbox("Show Heikin Ashi chart", value=False)
    chart_df = compute_heikin_ashi(df) if use_heikin_ashi else df.copy()

    fig = go.Figure()
    if use_heikin_ashi:
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["HA_Open"],
            high=chart_df["HA_High"],
            low=chart_df["HA_Low"],
            close=chart_df["HA_Close"],
            name="Heikin Ashi"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price"
        ))

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], name="EMA20", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], name="EMA50", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["TrendSupport"], name="Trend Support", line=dict(color="lime", dash="dot")))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["TrendResistance"], name="Trend Resistance", line=dict(color="orange", dash="dot")))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"], name="BB Upper", line=dict(color="purple", dash="dash")))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"], name="BB Lower", line=dict(color="purple", dash="dash")))
    fig.update_layout(title=f"{symbol_name} - {timeframe} | {market_group}", template="plotly_dark", height=560, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Bollinger Bands & MACD")
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    macd = latest.get("MACD", 0)
    hist = latest.get("MACD_Hist", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BB Upper", f"₹{bb_upper:.2f}")
    col2.metric("BB Lower", f"₹{bb_lower:.2f}")
    col3.metric("MACD", f"{macd:.4f}")
    col4.metric("MACD Hist", f"{hist:.4f}")

    st.subheader("🎯 Detected Patterns")
    patterns = get_detected_patterns(latest)
    if patterns:
        for pattern in patterns:
            st.success(pattern)
    else:
        st.info("No patterns detected")

    if pcr_value is not None:
        st.subheader("📉 Put-Call Ratio (PCR)")
        st.metric("NIFTY PCR", f"{pcr_value:.2f}")
        if pcr_value > 1.2:
            st.info("PCR suggests bullish put interest; markets may be defensive.")
        elif pcr_value < 0.8:
            st.info("PCR suggests strong call interest; momentum may be bullish.")
        else:
            st.info("PCR is near neutral; trend is balanced.")
    else:
        st.subheader("📉 Put-Call Ratio (PCR)")
        st.info("PCR is not available from the live feed in this session.")

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
