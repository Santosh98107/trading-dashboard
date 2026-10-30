```python
import os
from datetime import datetime, time, timedelta, timezone
import threading
import time as pytime
import queue

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import pyotp
from SmartApi import SmartConnect
from streamlit_autorefresh import st_autorefresh

# Optional WebSocket import
SMART_WS_AVAILABLE = False
try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    SMART_WS_AVAILABLE = True
except Exception:
    SMART_WS_AVAILABLE = False


st.set_page_config(page_title="Elite Trading Dashboard Ultimate V6", layout="wide")

JOURNAL_FILE = "trade_journal.csv"
SIGNAL_STATE_FILE = "last_signal_state.txt"
PAPER_STATE_FILE = "paper_trade_state.csv"
USERS_FILE = "users.csv"
SESSION_SUMMARY_FILE = "daily_summary.csv"
RISK_STATE_FILE = "risk_state.csv"

MAX_WS_RETRIES = 5
WS_RETRY_DELAY_SEC = 5

# ----------------------------
# GLOBAL WS STORE
# ----------------------------
WS_STATE_LOCK = threading.Lock()
WS_SHARED_STATE = {
    "running": False,
    "connected": False,
    "error": "",
    "last_tick": None,
    "live_price": None,
    "retries": 0,
    "should_run": False,
    "thread_started": False,
    "symbol": None,
    "socket": None,
    "last_data_time": None
}


# ----------------------------
# INIT FILES
# ----------------------------
def init_users():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame([
            {"username": "admin", "password": "admin123"}
        ]).to_csv(USERS_FILE, index=False)
        return

    try:
        users = pd.read_csv(USERS_FILE)
        if "username" not in users.columns or "password" not in users.columns:
            raise ValueError("Invalid users file structure")
    except Exception:
        pd.DataFrame([
            {"username": "admin", "password": "admin123"}
        ]).to_csv(USERS_FILE, index=False)


def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        cols = [
            "DateTime", "Symbol", "Timeframe", "Signal", "EntryPrice",
            "ExitPrice", "Quantity", "StopLoss", "Target1", "Target2",
            "TrailingStop", "Status", "PnL", "PnLPercent", "Notes"
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


def init_risk_state():
    if not os.path.exists(RISK_STATE_FILE):
        pd.DataFrame([{
            "DailyLossLimit": 2000.0,
            "MaxTradesPerDay": 5,
            "RiskPerTradePct": 1.0,
            "TradingEnabled": True
        }]).to_csv(RISK_STATE_FILE, index=False)


# ----------------------------
# LOGIN
# ----------------------------
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
                match = users[
                    (users["username"].astype(str).str.strip() == username.strip()) &
                    (users["password"].astype(str).str.strip() == password.strip())
                ]
                if not match.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                st.sidebar.error(f"Login error: {e}")

        with st.sidebar.expander("Reset Password"):
            reset_user = st.text_input("Username for reset", key="reset_user")
            old_pass = st.text_input("Old Password", type="password", key="old_pass")
            new_pass = st.text_input("New Password", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirm New Password", type="password", key="confirm_pass")

            if st.button("Reset Password"):
                try:
                    if not reset_user or not old_pass or not new_pass or not confirm_pass:
                        st.sidebar.warning("Please fill all password reset fields.")
                    elif new_pass != confirm_pass:
                        st.sidebar.error("New password and confirm password do not match.")
                    else:
                        users = pd.read_csv(USERS_FILE)
                        match_idx = users[
                            (users["username"].astype(str).str.strip() == reset_user.strip()) &
                            (users["password"].astype(str).str.strip() == old_pass.strip())
                        ].index

                        if len(match_idx) == 0:
                            st.sidebar.error("Username / old password invalid.")
                        else:
                            users.loc[match_idx[0], "password"] = new_pass.strip()
                            users.to_csv(USERS_FILE, index=False)
                            st.sidebar.success("Password reset successful. Please login.")
                except Exception as e:
                    st.sidebar.error(f"Password reset error: {e}")

        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# ----------------------------
# FILE HELPERS
# ----------------------------
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


def load_risk_state():
    init_risk_state()
    return pd.read_csv(RISK_STATE_FILE)


def save_risk_state(df):
    df.to_csv(RISK_STATE_FILE, index=False)


def add_trade_to_journal(symbol, timeframe, signal, entry_price, quantity, stop_loss, target1, target2, trailing_stop, notes=""):
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
        "Notes": notes
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
            pnl_pct = ((exit_price - entry) / entry) * 100 if entry else 0
        else:
            pnl = (entry - exit_price) * qty
            pnl_pct = ((entry - exit_price) / entry) * 100 if entry else 0

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


def get_today_closed_trades():
    journal = load_journal()
    if journal.empty:
        return pd.DataFrame()

    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return closed

    closed["DateOnly"] = pd.to_datetime(
        closed["DateTime"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    ).dt.date

    today = datetime.now().date()
    return closed[closed["DateOnly"] == today].copy()


def risk_check_before_trade():
    risk_state = load_risk_state()
    journal_today = get_today_closed_trades()

    trading_enabled = bool(risk_state.loc[0, "TradingEnabled"])
    daily_loss_limit = float(risk_state.loc[0, "DailyLossLimit"])
    max_trades_per_day = int(risk_state.loc[0, "MaxTradesPerDay"])

    if not trading_enabled:
        return False, "Trading disabled by Risk Management module."

    realized_pnl_today = pd.to_numeric(journal_today.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    trades_today = len(journal_today)

    if realized_pnl_today <= -abs(daily_loss_limit):
        return False, f"Daily loss limit hit. Realized PnL today: {round(realized_pnl_today, 2)}"

    if trades_today >= max_trades_per_day:
        return False, f"Max trades per day reached ({max_trades_per_day})."

    return True, "Risk checks passed."


def calculate_position_size(entry_price, stop_loss, paper_balance):
    risk_state = load_risk_state()
    risk_pct = float(risk_state.loc[0, "RiskPerTradePct"])

    if entry_price is None or stop_loss is None:
        return 1

    risk_amount = paper_balance * (risk_pct / 100.0)
    per_unit_risk = abs(float(entry_price) - float(stop_loss))

    if per_unit_risk <= 0:
        return 1

    qty = int(risk_amount // per_unit_risk)
    return max(1, qty)


# ----------------------------
# PAPER TRADE
# ----------------------------
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


# ----------------------------
# MARKET DATA
# ----------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker, interval, period):
    try:
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[needed].copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df


def resample_ohlcv(df, rule):
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()

    if df.empty:
        return df

    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return resampled


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(lower=1, upper=99)


def compute_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    denom = (high_max - low_min).replace(0, 1e-10)
    k = 100 * ((df["Close"] - low_min) / denom)
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


def compute_vwap(df):
    df = df.copy()

    if df.empty:
        df["VWAP"] = np.nan
        return df

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()

    for col in ["High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = df["Volume"].fillna(0)

    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    if df["Volume"].sum() <= 0:
        df["VWAP"] = tp.expanding().mean()
        df["VWAP"] = df["VWAP"].bfill().ffill()
        return df

    trade_date = pd.Series(df.index.date, index=df.index)
    cum_pv = (tp * df["Volume"]).groupby(trade_date).cumsum()
    cum_vol = df["Volume"].groupby(trade_date).cumsum()

    df["VWAP"] = np.where(
        cum_vol > 0,
        cum_pv / cum_vol,
        tp.expanding().mean()
    )

    df["VWAP"] = pd.to_numeric(df["VWAP"], errors="coerce")
    df["VWAP"] = df["VWAP"].replace([np.inf, -np.inf], np.nan)
    df["VWAP"] = df["VWAP"].bfill().ffill()

    return df


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df = compute_vwap(df)

    df["RSI"] = compute_rsi(df["Close"], 14)

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

    df["PivotHigh"] = df["High"].rolling(20).max()
    df["PivotLow"] = df["Low"].rolling(20).min()
    df["Breakout"] = df["Close"] > df["PivotHigh"].shift(1)
    df["Breakdown"] = df["Close"] < df["PivotLow"].shift(1)

    df["RetestBull"] = (
        df["Breakout"].shift(1)
        & (df["Low"] <= df["PivotHigh"].shift(1))
        & (df["Close"] > df["PivotHigh"].shift(1))
    )

    df["RetestBear"] = (
        df["Breakdown"].shift(1)
        & (df["High"] >= df["PivotLow"].shift(1))
        & (df["Close"] < df["PivotLow"].shift(1))
    )

    k, d = compute_stochastic(df)
    df["StochK"] = k.rolling(3).mean()
    df["StochD"] = d.rolling(3).mean()

    df["ADX"], df["PlusDI"], df["MinusDI"] = compute_adx(df)

    candle_body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-10)

    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["BullishMarubozu"] = (
        (df["Close"] > df["Open"])
        & (upper_shadow < candle_body * 0.1)
        & (lower_shadow < candle_body * 0.1)
    )

    df["BearishMarubozu"] = (
        (df["Close"] < df["Open"])
        & (upper_shadow < candle_body * 0.1)
        & (lower_shadow < candle_body * 0.1)
    )

    df["BullishEngulfing"] = (
        (df["Close"] > df["Open"])
        & (df["Close"].shift(1) < df["Open"].shift(1))
        & (df["Open"] <= df["Close"].shift(1))
        & (df["Close"] >= df["Open"].shift(1))
    )

    df["BearishEngulfing"] = (
        (df["Close"] < df["Open"])
        & (df["Close"].shift(1) > df["Open"].shift(1))
        & (df["Open"] >= df["Close"].shift(1))
        & (df["Close"] <= df["Open"].shift(1))
    )

    df["PiercingPattern"] = (
        (df["Close"].shift(1) < df["Open"].shift(1))
        & (df["Close"] > df["Open"])
        & (df["Close"] > ((df["Open"].shift(1) + df["Close"].shift(1)) / 2))
    )

    df["DarkCloudCover"] = (
        (df["Close"].shift(1) > df["Open"].shift(1))
        & (df["Close"] < df["Open"])
        & (df["Close"] < ((df["Open"].shift(1) + df["Close"].shift(1)) / 2))
    )

    df["TweezerBottom"] = (
        abs(df["Low"] - df["Low"].shift(1)) <= (df["ATR"] * 0.1)
    )

    df["BullishHarami"] = (
        (df["Close"].shift(1) < df["Open"].shift(1))
        & (df["Close"] > df["Open"])
        & (df["Open"] > df["Close"].shift(1))
        & (df["Close"] < df["Open"].shift(1))
    )

    df["BearishHarami"] = (
        (df["Close"].shift(1) > df["Open"].shift(1))
        & (df["Close"] < df["Open"])
        & (df["Open"] < df["Close"].shift(1))
        & (df["Close"] > df["Open"].shift(1))
    )

    df["Doji"] = candle_body <= (candle_range * 0.1)

    df["TweezerTop"] = (
        abs(df["High"] - df["High"].shift(1)) <= (df["ATR"] * 0.1)
    )

    df["Hammer"] = (
        (lower_shadow > candle_body * 2)
        & (upper_shadow < candle_body)
    )

    df["ShootingStar"] = (
        (upper_shadow > candle_body * 2)
        & (lower_shadow < candle_body)
    )

    df["InvertedHammer"] = (
        (upper_shadow > candle_body * 2)
        & (lower_shadow < candle_body * 0.5)
        & (df["Close"] > df["Open"])
    )

    df["HangingMan"] = (
        (lower_shadow > candle_body * 2)
        & (upper_shadow < candle_body * 0.5)
        & (df["Close"] < df["Open"])
    )

    df["DragonflyDoji"] = (
        (candle_body <= candle_range * 0.05)
        & (lower_shadow > candle_body * 3)
    )

    df["GravestoneDoji"] = (
        (candle_body <= candle_range * 0.05)
        & (upper_shadow > candle_body * 3)
    )

    df["ThreeWhiteSoldiers"] = (
        (df["Close"] > df["Open"])
        & (df["Close"].shift(1) > df["Open"].shift(1))
        & (df["Close"].shift(2) > df["Open"].shift(2))
        & (df["Close"] > df["Close"].shift(1))
        & (df["Close"].shift(1) > df["Close"].shift(2))
    )

    df["ThreeBlackCrows"] = (
        (df["Close"] < df["Open"])
        & (df["Close"].shift(1) < df["Open"].shift(1))
        & (df["Close"].shift(2) < df["Open"].shift(2))
        & (df["Close"] < df["Close"].shift(1))
        & (df["Close"].shift(1) < df["Close"].shift(2))
    )

    df["MorningStar"] = (
        (df["Close"].shift(2) < df["Open"].shift(2))
        & (abs(df["Close"].shift(1) - df["Open"].shift(1)) < candle_body.shift(1))
        & (df["Close"] > df["Open"])
    )

    df["EveningStar"] = (
        (df["Close"].shift(2) > df["Open"].shift(2))
        & (abs(df["Close"].shift(1) - df["Open"].shift(1)) < candle_body.shift(1))
        & (df["Close"] < df["Open"])
    )

    df["BodyStrength"] = candle_body / candle_range

    df["BuyMarker"] = np.where(
        (df["EMA20"] > df["EMA50"]) &
        (df["EMA20"].shift(1) <= df["EMA50"].shift(1)),
        df["Low"] * 0.995,
        np.nan
    )

    df["SellMarker"] = np.where(
        (df["EMA20"] < df["EMA50"]) &
        (df["EMA20"].shift(1) >= df["EMA50"].shift(1)),
        df["High"] * 1.005,
        np.nan
    )

    return df


# ----------------------------
# SIGNAL UTILS
# ----------------------------
def get_higher_timeframe(tf):
    if tf == "1m":
        return "3m", "7d"
    if tf == "3m":
        return "5m", "30d"
    if tf == "5m":
        return "15m", "30d"
    if tf == "10m":
        return "30m", "60d"
    if tf == "15m":
        return "30m", "60d"
    if tf == "30m":
        return "1h", "90d"
    if tf == "1h":
        return "1d", "1y"
    return "1d", "1y"


def session_filter(df, tf):
    if tf == "1d":
        return {"valid": True, "message": "Daily timeframe - session filter skipped"}

    try:
        last_ts = pd.to_datetime(df.index[-1])
        current_t = last_ts.time()

        if current_t < time(9, 20):
            return {"valid": False, "message": "Avoid first few minutes after market open"}
        if time(12, 15) <= current_t <= time(13, 15):
            return {"valid": False, "message": "Lunch/session low momentum zone"}

        return {"valid": True, "message": "Session timing looks okay"}
    except Exception:
        return {"valid": True, "message": "Session filter unavailable"}


def calculate_win_chance(bullish, bearish, adx, volume_spike, breakout, breakdown, htf_bias, sideways, rr_ratio, backtest_win_rate):
    score = 50
    score += (bullish - bearish) * 4

    if adx > 25:
        score += 8
    elif adx < 18:
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


@st.cache_data(ttl=60)
def get_pcr():
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        resp = None
        for _ in range(3):
            try:
                resp = session.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
            except Exception:
                continue

        if resp is None or resp.status_code != 200:
            return None

        data = resp.json()
        total_ce = 0
        total_pe = 0

        for item in data.get("records", {}).get("data", []):
            if "CE" in item:
                total_ce += item["CE"].get("openInterest", 0)
            if "PE" in item:
                total_pe += item["PE"].get("openInterest", 0)

        if total_ce <= 0:
            return None

        return round(total_pe / total_ce, 2)

    except Exception:
        return None


def analyze_signal(df, htf_bias=None, timeframe_label="1d", backtest_win_rate=50):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close_price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema20_slope = ema20 - float(prev["EMA20"])
    vwap = float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else float(close_price)
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["SignalLine"])
    atr = float(latest["ATR"]) if pd.notna(latest["ATR"]) else 0.0
    stoch_k = float(latest["StochK"]) if pd.notna(latest["StochK"]) else 50.0
    stoch_d = float(latest["StochD"]) if pd.notna(latest["StochD"]) else 50.0
    adx = float(latest["ADX"]) if pd.notna(latest["ADX"]) else 0.0
    support = float(latest["PivotLow"]) if pd.notna(latest["PivotLow"]) else close_price
    resistance = float(latest["PivotHigh"]) if pd.notna(latest["PivotHigh"]) else close_price

    distance_to_resistance = ((resistance - close_price) / close_price) * 100 if close_price else 0
    distance_to_support = ((close_price - support) / close_price) * 100 if close_price else 0
    trend_strength = abs(ema20 - ema50) / close_price * 100 if close_price else 0

    volume_spike = bool(latest["Volume_Spike"])
    breakout = bool(latest["Breakout"])
    breakdown = bool(latest["Breakdown"])
    body_strength = float(latest["BodyStrength"]) if pd.notna(latest["BodyStrength"]) else 0

    hammer = bool(latest.get("Hammer", False))
    shooting_star = bool(latest.get("ShootingStar", False))
    bull_maru = bool(latest.get("BullishMarubozu", False))
    bear_maru = bool(latest.get("BearishMarubozu", False))
    doji = bool(latest.get("Doji", False))
    bull_engulf = bool(latest.get("BullishEngulfing", False))
    bear_engulf = bool(latest.get("BearishEngulfing", False))
    retest_bull = bool(latest.get("RetestBull", False))
    retest_bear = bool(latest.get("RetestBear", False))
    plus_di = float(latest["PlusDI"]) if pd.notna(latest["PlusDI"]) else 0
    minus_di = float(latest["MinusDI"]) if pd.notna(latest["MinusDI"]) else 0

    bullish = 0
    bearish = 0
    reasons = []

    pcr = get_pcr()
    if pcr is not None:
        if pcr > 1.2:
            bullish += 1
            reasons.append(f"PCR Bullish ({pcr})")
        elif pcr < 0.8:
            bearish += 1
            reasons.append(f"PCR Bearish ({pcr})")

    if ema20 > ema50:
        bullish += 1
        reasons.append("EMA20 > EMA50 bullish trend")
    else:
        bearish += 1
        reasons.append("EMA20 < EMA50 bearish trend")

    if close_price > ema20 > ema50:
        bullish += 1
        reasons.append("Perfect bullish EMA alignment")

    if close_price < ema20 < ema50:
        bearish += 1
        reasons.append("Perfect bearish EMA alignment")

    if close_price > ema20:
        bullish += 1
        reasons.append("Price above EMA20")
    else:
        bearish += 1
        reasons.append("Price below EMA20")

    if pd.notna(vwap):
        if close_price > vwap:
            bullish += 1
            reasons.append("Price above VWAP")
        else:
            bearish += 1
            reasons.append("Price below VWAP")

    if rsi > 60:
        bullish += 1
        reasons.append("RSI bullish strength")
    elif rsi < 40:
        bearish += 1
        reasons.append("RSI bearish weakness")
    else:
        reasons.append("RSI neutral")

    if rsi > float(prev["RSI"]):
        bullish += 1
        reasons.append("RSI momentum increasing")
    elif rsi < float(prev["RSI"]):
        bearish += 1
        reasons.append("RSI momentum decreasing")

    if macd > macd_signal:
        bullish += 1
        reasons.append("MACD above signal")
    else:
        bearish += 1
        reasons.append("MACD below signal")

    if latest["MACD_Hist"] > prev["MACD_Hist"]:
        bullish += 1
        reasons.append("MACD momentum increasing")
    elif latest["MACD_Hist"] < prev["MACD_Hist"]:
        bearish += 1
        reasons.append("MACD momentum decreasing")

    if ema20_slope > 0:
        bullish += 1
        reasons.append("EMA20 rising")
    if ema20_slope < 0:
        bearish += 1
        reasons.append("EMA20 falling")

    if adx > 20:
        reasons.append("ADX trend strength okay")
        if plus_di > minus_di:
            bullish += 1
        else:
            bearish += 1
    else:
        reasons.append("ADX low / sideways risk")

    if adx > 30:
        if plus_di > minus_di:
            bullish += 1
            reasons.append("Strong bullish trend")
        elif minus_di > plus_di:
            bearish += 1
            reasons.append("Strong bearish trend")

    if stoch_k > stoch_d and stoch_k < 80:
        bullish += 1
        reasons.append("Stochastic bullish crossover")
    elif stoch_k < stoch_d and stoch_k > 20:
        bearish += 1
        reasons.append("Stochastic bearish crossover")

    if prev["EMA20"] <= prev["EMA50"] and latest["EMA20"] > latest["EMA50"]:
        bullish += 2
        reasons.append("Fresh bullish EMA crossover")

    if prev["EMA20"] >= prev["EMA50"] and latest["EMA20"] < latest["EMA50"]:
        bearish += 2
        reasons.append("Fresh bearish EMA crossover")

    if breakout:
        bullish += 2
        reasons.append("Resistance breakout")
        if volume_spike and htf_bias == "bullish":
            bullish += 3
            reasons.append("High quality breakout")

    if breakdown:
        bearish += 2
        reasons.append("Support breakdown")
        if volume_spike and htf_bias == "bearish":
            bearish += 3
            reasons.append("High quality breakdown")

    if retest_bull:
        bullish += 3
        reasons.append("Bullish breakout retest confirmed")

    if retest_bear:
        bearish += 3
        reasons.append("Bearish breakdown retest confirmed")

    if hammer:
        bullish += 2
        reasons.append("Hammer pattern detected")

    if bool(latest.get("InvertedHammer", False)):
        bullish += 2
        reasons.append("Inverted Hammer detected")

    if bool(latest.get("HangingMan", False)):
        bearish += 2
        reasons.append("Hanging Man detected")

    if bool(latest.get("DragonflyDoji", False)):
        bullish += 2
        reasons.append("Dragonfly Doji detected")

    if bool(latest.get("GravestoneDoji", False)):
        bearish += 2
        reasons.append("Gravestone Doji detected")

    if bull_maru:
        bullish += 4
        bearish = max(0, bearish - 2)
        reasons.append("Strong Bullish Marubozu detected")

    if shooting_star:
        bearish += 2
        reasons.append("Shooting Star detected")

    if bear_maru:
        bearish += 4
        bullish = max(0, bullish - 2)
        reasons.append("Strong Bearish Marubozu detected")

    if bool(latest.get("ThreeWhiteSoldiers", False)):
        bullish += 4
        reasons.append("Three White Soldiers detected")

    if bool(latest.get("ThreeBlackCrows", False)):
        bearish += 4
        reasons.append("Three Black Crows detected")

    if doji:
        reasons.append("Doji pattern (market indecision)")

    if bull_engulf:
        bullish += 3
        reasons.append("Bullish Engulfing detected")

    if bear_engulf:
        bearish += 3
        reasons.append("Bearish Engulfing detected")

    if bool(latest.get("PiercingPattern", False)):
        bullish += 3
        reasons.append("Piercing Pattern detected")

    if bool(latest.get("DarkCloudCover", False)):
        bearish += 3
        reasons.append("Dark Cloud Cover detected")

    if bool(latest.get("BullishHarami", False)):
        bullish += 3
        reasons.append("Bullish Harami detected")

    if bool(latest.get("BearishHarami", False)):
        bearish += 3
        reasons.append("Bearish Harami detected")

    if bool(latest.get("TweezerBottom", False)):
        bullish += 2
        reasons.append("Tweezer Bottom detected")

    if bool(latest.get("TweezerTop", False)):
        bearish += 2
        reasons.append("Tweezer Top detected")

    if bool(latest.get("MorningStar", False)):
        bullish += 4
        reasons.append("Morning Star detected")

    if bool(latest.get("EveningStar", False)):
        bearish += 4
        reasons.append("Evening Star detected")

    if bull_engulf and volume_spike:
        bullish += 2
        reasons.append("Bullish Engulfing with Volume Confirmation")

    if bear_engulf and volume_spike:
        bearish += 2
        reasons.append("Bearish Engulfing with Volume Confirmation")

    if volume_spike:
        reasons.append("Volume spike present")
        if close_price > ema20:
            bullish += 1
        else:
            bearish += 1

    if body_strength > 0.6:
        reasons.append("Strong candle body")
        if close_price > ema20:
            bullish += 1
        else:
            bearish += 1
    else:
        reasons.append("Weak candle body")

    if bullish > bearish and distance_to_resistance < 0.5:
        bullish -= 2
        reasons.append("Too close to resistance")

    if bearish > bullish and distance_to_support < 0.5:
        bearish -= 2
        reasons.append("Too close to support")

    if htf_bias == "bullish":
        bullish += 2
        reasons.append("Higher timeframe bullish confirmation")
    elif htf_bias == "bearish":
        bearish += 2
        reasons.append("Higher timeframe bearish confirmation")

    if bullish > bearish and htf_bias == "bearish":
        bullish -= 2
        reasons.append("Higher timeframe conflict")

    if bearish > bullish and htf_bias == "bullish":
        bearish -= 2
        reasons.append("Higher timeframe conflict")

    sideways = bool(adx < 18 and trend_strength < 0.3)
    session_info = session_filter(df, timeframe_label)

    signal_text = "NO TRADE"
    signal_type = "neutral"
    strength = "Low"
    stop_loss = None
    partial_exit = None
    target1 = None
    target2 = None
    trailing_stop = None
    rr_ratio = None

    if atr and atr > 0:
        rr_ratio = 2.0

    if not sideways and session_info["valid"] and atr > 0:
        if bullish >= 8 and (bullish - bearish) >= 3 and rsi > 55 and macd > macd_signal and adx > 25:
            signal_text = "STRONG BUY"
            signal_type = "buy"
            strength = "High"
            stop_loss = close_price - atr
            partial_exit = close_price + (0.8 * atr)
            target1 = close_price + atr
            target2 = close_price + (2 * atr)
            trailing_stop = close_price - (0.5 * atr)

        elif bearish >= 8 and (bearish - bullish) >= 3 and rsi < 45 and macd < macd_signal and adx > 25:
            signal_text = "STRONG SELL"
            signal_type = "sell"
            strength = "High"
            stop_loss = close_price + atr
            partial_exit = close_price - (0.8 * atr)
            target1 = close_price - atr
            target2 = close_price - (2 * atr)
            trailing_stop = close_price + (0.5 * atr)

        elif bullish > bearish and bullish >= 6:
            signal_text = "BUY BIAS"
            signal_type = "buy"
            strength = "Medium"
            stop_loss = close_price - atr
            partial_exit = close_price + (0.7 * atr)
            target1 = close_price + atr
            trailing_stop = close_price - (0.4 * atr)

        elif bearish > bullish and bearish >= 6:
            signal_text = "SELL BIAS"
            signal_type = "sell"
            strength = "Medium"
            stop_loss = close_price + atr
            partial_exit = close_price - (0.7 * atr)
            target1 = close_price - atr
            trailing_stop = close_price + (0.4 * atr)

    confidence = min(100, max(40, int((max(bullish, bearish) / 15) * 100)))

    win_chance = calculate_win_chance(
        bullish, bearish, adx, volume_spike, breakout, breakdown,
        htf_bias, sideways, rr_ratio, backtest_win_rate
    )

    if bear_maru:
        win_chance = max(5, win_chance - 10)
    if bull_maru:
        win_chance = min(95, win_chance + 10)

    return {
        "close_price": close_price,
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "adx": adx,
        "support": support,
        "resistance": resistance,
        "volume_spike": volume_spike,
        "breakout": breakout,
        "breakdown": breakdown,
        "body_strength": body_strength,
        "sideways": sideways,
        "signal_text": signal_text,
        "signal_type": signal_type,
        "strength": strength,
        "confidence": confidence,
        "win_chance": win_chance,
        "stop_loss": stop_loss,
        "partial_exit": partial_exit,
        "target1": target1,
        "target2": target2,
        "trailing_stop": trailing_stop,
        "rr_ratio": rr_ratio,
        "reasons": reasons,
        "bullish_score": bullish,
        "bearish_score": bearish,
        "session_message": session_info["message"]
    }


def simple_backtest(df, symbol=""):
    trades = []

    df = df.copy()
    required_cols = [
        "Open", "High", "Low", "Close",
        "EMA20", "EMA50", "VWAP", "RSI",
        "MACD", "SignalLine", "ATR"
    ]

    for col in required_cols:
        if col not in df.columns:
            empty_df = pd.DataFrame(columns=[
                "Date", "Time", "Symbol", "Signal", "Entry",
                "SL", "Target", "Exit", "Result", "PnL"
            ])
            return 0, 0, 0, 0.0, empty_df

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "EMA20", "EMA50", "RSI", "MACD", "SignalLine", "ATR"]).copy()

    if len(df) < 30:
        empty_df = pd.DataFrame(columns=[
            "Date", "Time", "Symbol", "Signal", "Entry",
            "SL", "Target", "Exit", "Result", "PnL"
        ])
        return 0, 0, 0, 0.0, empty_df

    for i in range(20, len(df) - 1):
        latest = df.iloc[i]
        ts = df.index[i]

        vwap_value = latest["VWAP"] if pd.notna(latest["VWAP"]) else latest["Close"]

        buy_signal = (
            latest["EMA20"] > latest["EMA50"] and
            latest["Close"] >= vwap_value and
            latest["RSI"] > 52 and
            latest["MACD"] > latest["SignalLine"]
        )

        sell_signal = (
            latest["EMA20"] < latest["EMA50"] and
            latest["Close"] <= vwap_value and
            latest["RSI"] < 48 and
            latest["MACD"] < latest["SignalLine"]
        )

        atr = latest["ATR"]
        if pd.isna(atr) or atr <= 0:
            continue

        entry = float(latest["Close"])
        future = df.iloc[i + 1:i + 11]

        if future.empty:
            continue

        if buy_signal:
            signal = "BUY"
            sl = float(entry - atr)
            target = float(entry + atr)
            exit_price = None
            result = None

            for _, row in future.iterrows():
                if row["Low"] <= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                if row["High"] >= target:
                    exit_price = target
                    result = "WIN"
                    break

            if exit_price is None:
                exit_price = float(future.iloc[-1]["Close"])
                result = "WIN" if exit_price > entry else "LOSS"

            pnl = round(exit_price - entry, 2)

            trades.append({
                "Date": pd.to_datetime(ts).strftime("%d-%m-%Y"),
                "Time": pd.to_datetime(ts).strftime("%H:%M:%S"),
                "Symbol": symbol,
                "Signal": signal,
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "Target": round(target, 2),
                "Exit": round(exit_price, 2),
                "Result": result,
                "PnL": pnl
            })

        elif sell_signal:
            signal = "SELL"
            sl = float(entry + atr)
            target = float(entry - atr)
            exit_price = None
            result = None

            for _, row in future.iterrows():
                if row["High"] >= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                if row["Low"] <= target:
                    exit_price = target
                    result = "WIN"
                    break

            if exit_price is None:
                exit_price = float(future.iloc[-1]["Close"])
                result = "WIN" if exit_price < entry else "LOSS"

            pnl = round(entry - exit_price, 2)

            trades.append({
                "Date": pd.to_datetime(ts).strftime("%d-%m-%Y"),
                "Time": pd.to_datetime(ts).strftime("%H:%M:%S"),
                "Symbol": symbol,
                "Signal": signal,
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "Target": round(target, 2),
                "Exit": round(exit_price, 2),
                "Result": result,
                "PnL": pnl
            })

    trades_df = pd.DataFrame(trades, columns=[
        "Date", "Time", "Symbol", "Signal", "Entry",
        "SL", "Target", "Exit", "Result", "PnL"
    ])

    total = len(trades_df)
    wins = len(trades_df[trades_df["Result"] == "WIN"])
    losses = len(trades_df[trades_df["Result"] == "LOSS"])
    win_rate = (wins / total * 100) if total else 0.0

    return wins, losses, total, round(win_rate, 2), trades_df


# ----------------------------
# ANGEL ONE
# ----------------------------
def map_angel_interval(timeframe):
    mapping = {
        "1m": "ONE_MINUTE",
        "3m": "ONE_MINUTE",
        "5m": "FIVE_MINUTE",
        "10m": "FIVE_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE",
        "1h": "ONE_HOUR",
        "1d": "ONE_DAY"
    }
    return mapping.get(timeframe, "FIVE_MINUTE")


def angel_login(api_key, client_code, pin, totp_secret, force_refresh=False):
    try:
        cache_key = f"{client_code}_{api_key}"

        if (
            not force_refresh
            and "angel_api" in st.session_state
            and "angel_session" in st.session_state
            and st.session_state.get("angel_cache_key") == cache_key
        ):
            return st.session_state["angel_api"], st.session_state["angel_session"]

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_code, pin, totp)

        if session and session.get("status"):
            st.session_state["angel_api"] = smart_api
            st.session_state["angel_session"] = session
            st.session_state["angel_cache_key"] = cache_key
            return smart_api, session

        return None, None

    except Exception as e:
        st.error(f"Angel Login Failed: {e}")
        return None, None


def get_angel_profile(api_key, client_code, pin, totp_secret):
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        if angel is None or session is None:
            return None

        refresh_token = session["data"].get("refreshToken")
        if not refresh_token:
            return None

        return angel.getProfile(refresh_token)
    except Exception as e:
        return {"status": False, "message": str(e)}


def get_angel_candles(api_key, client_code, pin, totp_secret, symbol_name, interval="FIVE_MINUTE"):
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        if angel is None or not session or not session.get("data"):
            return None

        symbol_token = ANGEL_TOKENS.get(symbol_name)
        if not symbol_token:
            return {"status": False, "message": f"Missing Angel token for {symbol_name}"}

        ist = timezone(timedelta(hours=5, minutes=30))
        to_date = datetime.now(ist)
        from_date = to_date - timedelta(days=7)

        params = {
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        candles = angel.getCandleData(params)
        return candles

    except Exception as e:
        return {"status": False, "message": str(e)}


def candles_to_df(candles):
    if not isinstance(candles, dict) or not candles.get("data"):
        return pd.DataFrame()

    df = pd.DataFrame(
        candles["data"],
        columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
    )

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df


# ----------------------------
# ANGEL WEBSOCKET - STABLE VERSION
# ----------------------------
def ws_set_state(**kwargs):
    with WS_STATE_LOCK:
        for k, v in kwargs.items():
            WS_SHARED_STATE[k] = v


def ws_get_state():
    with WS_STATE_LOCK:
        return dict(WS_SHARED_STATE)


def sync_ws_state_to_session():
    state = ws_get_state()
    st.session_state["ws_running"] = state["running"]
    st.session_state["ws_connected"] = state["connected"]
    st.session_state["ws_error"] = state["error"]
    st.session_state["ws_last_tick"] = state["last_tick"]
    st.session_state["live_price"] = state["live_price"]
    st.session_state["ws_retries"] = state["retries"]
    st.session_state["ws_should_run"] = state["should_run"]
    st.session_state["ws_thread_started"] = state["thread_started"]


def init_ws_state():
    defaults = {
        "ws_running": False,
        "ws_connected": False,
        "ws_error": "",
        "ws_last_tick": None,
        "ws_thread_started": False,
        "live_price": None,
        "ws_retries": 0,
        "ws_should_run": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    sync_ws_state_to_session()


def stop_angel_websocket():
    try:
        ws_set_state(
            should_run=False,
            connected=False,
            running=False,
            error="WebSocket stop requested by user."
        )

        state = ws_get_state()
        sws = state.get("socket")

        if sws is not None:
            try:
                if hasattr(sws, "close_connection"):
                    sws.close_connection()
                elif hasattr(sws, "close"):
                    sws.close()
            except Exception:
                pass

        ws_set_state(
            socket=None,
            thread_started=False
        )

    except Exception as e:
        ws_set_state(error=f"Stop error: {e}")


def extract_ltp_from_message(message):
    ltp = None

    if not isinstance(message, dict):
        return None

    possible_keys = [
        "last_traded_price",
        "ltp",
        "lastTradedPrice",
        "last_traded_price_int",
        "close"
    ]

    for key in possible_keys:
        if key in message and message.get(key) is not None:
            ltp = message.get(key)
            break

    if ltp is None:
        return None

    try:
        ltp_f = float(ltp)
        if ltp_f > 10000:
            ltp_f = ltp_f / 100.0
        return ltp_f
    except Exception:
        return None


def start_angel_websocket(api_key, client_code, pin, totp_secret, symbol_name):
    init_ws_state()

    if not SMART_WS_AVAILABLE:
        ws_set_state(error="SmartWebSocketV2 not available in installed SmartApi package.")
        sync_ws_state_to_session()
        return

    symbol_token = ANGEL_TOKENS.get(symbol_name)
    if not symbol_token:
        ws_set_state(error=f"Missing token for {symbol_name}")
        sync_ws_state_to_session()
        return

    state = ws_get_state()

    if state.get("thread_started") and state.get("symbol") == symbol_name and state.get("should_run"):
        sync_ws_state_to_session()
        return

    if state.get("thread_started") and state.get("symbol") != symbol_name:
        stop_angel_websocket()
        pytime.sleep(1)

    ws_set_state(
        should_run=True,
        thread_started=True,
        symbol=symbol_name,
        error=""
    )

    def run_socket():
        retry_count = 0

        while True:
            state_now = ws_get_state()
            if not state_now.get("should_run"):
                break

            if retry_count >= MAX_WS_RETRIES:
                ws_set_state(
                    error="Max WebSocket reconnect attempts reached.",
                    running=False,
                    connected=False,
                    retries=retry_count,
                    thread_started=False,
                    socket=None
                )
                break

            try:
                angel, session = angel_login(
                    api_key,
                    client_code,
                    pin,
                    totp_secret,
                    force_refresh=True
                )

                if angel is None or session is None or not session.get("data"):
                    raise Exception("Angel login failed during WebSocket reconnect")

                auth_token = session["data"]["jwtToken"]
                feed_token_resp = angel.getfeedToken()

                if isinstance(feed_token_resp, dict):
                    feed_token = feed_token_resp.get("data", {}).get("feedToken")
                    if not feed_token:
                        feed_token = feed_token_resp.get("feedToken")
                else:
                    feed_token = feed_token_resp

                if not feed_token:
                    raise Exception("Feed token not received")

                sws = SmartWebSocketV2(auth_token, api_key, client_code, feed_token)
                ws_set_state(
                    socket=sws,
                    running=True,
                    connected=False,
                    retries=retry_count,
                    error=""
                )

                correlation_id = f"elite_dashboard_feed_{symbol_name}"
                mode = 1
                token_list = [{"exchangeType": 1, "tokens": [str(symbol_token)]}]

                def on_data(wsapp, message):
                    ltp = extract_ltp_from_message(message)
                    current_state = ws_get_state()
                    ws_set_state(
                        last_tick=message,
                        connected=True,
                        error="",
                        live_price=ltp if ltp is not None else current_state.get("live_price"),
                        last_data_time=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    )

                def on_open(wsapp):
                    try:
                        ws_set_state(
                            connected=True,
                            running=True,
                            error=""
                        )
                        sws.subscribe(correlation_id, mode, token_list)
                    except Exception as e:
                        ws_set_state(
                            error=f"Subscribe error: {e}",
                            connected=False
                        )

                def on_error(wsapp, error):
                    ws_set_state(
                        error=f"WS error: {error}",
                        connected=False,
                        running=False
                    )

                def on_close(wsapp, *args):
                    reason = ""
                    if args:
                        reason = " | ".join([str(x) for x in args if x is not None])
                    ws_set_state(
                        connected=False,
                        running=False,
                        error=f"WS closed{': ' + reason if reason else ''}"
                    )

                sws.on_open = on_open
                sws.on_data = on_data
                sws.on_error = on_error
                sws.on_close = on_close

                try:
                    sws.connect()
                finally:
                    ws_set_state(
                        connected=False,
                        running=False,
                        socket=None
                    )

            except Exception as e:
                ws_set_state(
                    error=f"Reconnect attempt {retry_count + 1}/{MAX_WS_RETRIES}: {e}",
                    connected=False,
                    running=False,
                    socket=None
                )

            retry_count += 1
            ws_set_state(retries=retry_count)

            state_after = ws_get_state()
            if not state_after.get("should_run"):
                break

            pytime.sleep(WS_RETRY_DELAY_SEC)

        ws_set_state(
            running=False,
            connected=False,
            socket=None,
            thread_started=False
        )

    thread = threading.Thread(target=run_socket, daemon=True)
    thread.start()

    sync_ws_state_to_session()


# ----------------------------
# TELEGRAM
# ----------------------------
def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


# ----------------------------
# SNAPSHOT TABLES
# ----------------------------
@st.cache_data(ttl=300)
def fetch_market_snapshot(symbol_map):
    rows = []
    for name, ticker in symbol_map.items():
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 2:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            change_pct = ((close - prev_close) / prev_close) * 100 if prev_close else 0.0

            rows.append({
                "Name": name,
                "CMP": round(close, 2),
                "% Change": round(change_pct, 2)
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def get_intraday_trend_label(df):
    if df.empty or len(df) < 20:
        return "----"

    df = add_indicators(df).bfill().ffill()
    if df.empty:
        return "----"

    latest = df.iloc[-1]

    if latest["EMA20"] > latest["EMA50"]:
        return "BULLISH"
    elif latest["EMA20"] < latest["EMA50"]:
        return "BEARISH"
    return "NEUTRAL"


def safe_fetch_intraday_trend(ticker, minutes="1m"):
    try:
        df = fetch_data(ticker, "1m", "7d")
        if df.empty:
            return "----"

        if minutes == "2m":
            df = resample_ohlcv(df, "2min")
        elif minutes == "3m":
            df = resample_ohlcv(df, "3min")

        return get_intraday_trend_label(df)
    except Exception:
        return "----"


@st.cache_data(ttl=300)
def fetch_fno_style_scanner(symbol_map):
    rows = []
    sno = 1

    for name, ticker in symbol_map.items():
        try:
            df = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            cmp_value = "----"
            change_str = "----"
            direction = "----"

            if not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if "Close" in df.columns:
                    close = float(df["Close"].iloc[-1])
                    prev_close = float(df["Close"].iloc[-2])

                    cmp_value = f"{close:.2f}"

                    if prev_close:
                        change_pct = ((close - prev_close) / prev_close) * 100
                        change_str = f"{change_pct:+.2f}%"
                        direction = "BULLISH" if change_pct >= 0 else "BEARISH"

            trend_1m = safe_fetch_intraday_trend(ticker, "1m")
            trend_2m = safe_fetch_intraday_trend(ticker, "2m")
            trend_3m = safe_fetch_intraday_trend(ticker, "3m")

            rows.append({
                "S.NO": sno,
                "STOCKS": name,
                "CMP": cmp_value,
                "% Change": change_str,
                "1m": trend_1m,
                "2m": trend_2m,
                "3m": trend_3m,
                "Direction": direction
            })
            sno += 1

        except Exception:
            rows.append({
                "S.NO": sno,
                "STOCKS": name,
                "CMP": "----",
                "% Change": "----",
                "1m": "----",
                "2m": "----",
                "3m": "----",
                "Direction": "----"
            })
            sno += 1

    return pd.DataFrame(rows, columns=[
        "S.NO", "STOCKS", "CMP", "% Change", "1m", "2m", "3m", "Direction"
    ])


def summarize_fno_bias(fno_df):
    valid = fno_df[fno_df["Direction"].isin(["BULLISH", "BEARISH"])].copy()

    if valid.empty:
        return {
            "overall_bias": "N/A",
            "bullish_pct": 0,
            "bearish_pct": 0,
            "strongest_gainer": "N/A",
            "strongest_loser": "N/A"
        }

    bullish_count = (valid["Direction"] == "BULLISH").sum()
    bearish_count = (valid["Direction"] == "BEARISH").sum()
    total = len(valid)

    bullish_pct = round((bullish_count / total) * 100)
    bearish_pct = round((bearish_count / total) * 100)

    overall_bias = "BULLISH" if bullish_count >= bearish_count else "BEARISH"

    valid["ChangeNum"] = valid["% Change"].astype(str).str.replace("%", "", regex=False)
    valid["ChangeNum"] = pd.to_numeric(valid["ChangeNum"], errors="coerce")

    gainers = valid.dropna(subset=["ChangeNum"]).sort_values("ChangeNum", ascending=False)
    losers = valid.dropna(subset=["ChangeNum"]).sort_values("ChangeNum", ascending=True)

    strongest_gainer = "N/A"
    strongest_loser = "N/A"

    if not gainers.empty:
        strongest_gainer = f"{gainers.iloc[0]['STOCKS']} ({gainers.iloc[0]['% Change']})"

    if not losers.empty:
        strongest_loser = f"{losers.iloc[0]['STOCKS']} ({losers.iloc[0]['% Change']})"

    return {
        "overall_bias": overall_bias,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "strongest_gainer": strongest_gainer,
        "strongest_loser": strongest_loser
    }


# ----------------------------
# MTF SIGNAL
# ----------------------------
def get_data_for_timeframe(data_source, selected, ticker, tf, angel_creds):
    if data_source == "Yahoo":
        if tf == "1m":
            data = fetch_data(ticker, "1m", "7d")
        elif tf == "3m":
            data = fetch_data(ticker, "1m", "7d")
            if not data.empty:
                data = resample_ohlcv(data, "3min")
        elif tf == "5m":
            data = fetch_data(ticker, "5m", "30d")
        elif tf == "10m":
            data = fetch_data(ticker, "5m", "30d")
            if not data.empty:
                data = resample_ohlcv(data, "10min")
        elif tf == "15m":
            data = fetch_data(ticker, "15m", "30d")
        elif tf == "30m":
            data = fetch_data(ticker, "30m", "60d")
        elif tf == "1h":
            data = fetch_data(ticker, "60m", "90d")
        else:
            data = fetch_data(ticker, "1d", "1y")
        return data

    api_key, client_code, pin, totp_secret = angel_creds
    interval = map_angel_interval(tf)
    candles = get_angel_candles(api_key, client_code, pin, totp_secret, selected, interval)
    data = candles_to_df(candles)

    if data.empty:
        return data

    if tf == "3m":
        data = resample_ohlcv(data, "3min")
    elif tf == "10m":
        data = resample_ohlcv(data, "10min")

    return data


def get_mtf_signal(data_source, selected, ticker, tf, angel_creds):
    data = get_data_for_timeframe(data_source, selected, ticker, tf, angel_creds)
    if data.empty:
        return "⚪ N/A"

    data = add_indicators(data)
    data = data.bfill().ffill()

    if len(data) < 5:
        return "⚪ N/A"

    latest = data.iloc[-1]
    if latest["EMA20"] > latest["EMA50"]:
        return "🟢 Bullish"
    elif latest["EMA20"] < latest["EMA50"]:
        return "🔴 Bearish"
    return "⚪ Neutral"


# ----------------------------
# APP START
# ----------------------------
login_panel()
init_journal()
init_paper_state()
init_daily_summary()
init_risk_state()
init_ws_state()

stocks = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS"
}

ANGEL_TOKENS = {
    "SBIN": "3045",
    "RELIANCE": "2885",
    "INFY": "1594",
    "TCS": "11536",
    "NIFTY": "99926000",
    "BANKNIFTY": "99926009",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "ITC": "1660"
}

global_indices = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "USD/INR": "INR=X"
}

indian_indices = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA"
}

major_stocks = {
    "RELIANCE": "RELIANCE.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "SBI": "SBIN.NS",
    "AXIS BANK": "AXISBANK.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "HCLTECH": "HCLTECH.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS",
    "NESTLE INDIA": "NESTLEIND.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "POWER GRID": "POWERGRID.NS"
}

st.title("📈 Elite Trading Dashboard Ultimate V6")
st.caption("Journal + Paper Trading + Alerts + MTF + Backtest + Global/India Market Snapshot + Angel Ready")

debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

selected = st.selectbox("Select Stock / Index", list(stocks.keys()))
data_source = st.radio("Data Source", ["Yahoo", "Angel"], horizontal=True)
timeframe = st.selectbox("Select Time Frame", ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"])
auto_refresh = st.checkbox("Enable Auto Refresh", value=True)
refresh_seconds = st.selectbox("Refresh Every (seconds)", [10, 20, 30, 60], index=2)

with st.expander("Telegram Alert Settings"):
    telegram_enabled = st.checkbox("Enable Telegram Alerts", value=False)
    telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID")
    auto_send_telegram = st.checkbox("Auto-send on New Signal", value=False)

with st.expander("Angel One API Settings"):
    angel_enabled = st.checkbox("Enable Angel One API", value=(data_source == "Angel"))
    angel_api_key = st.text_input("Angel API Key", type="password")
    angel_client_code = st.text_input("Angel Client Code")
    angel_pin = st.text_input("Angel PIN", type="password")
    angel_totp_secret = st.text_input("Angel TOTP Secret", type="password")
    ws_enable = st.checkbox("Enable Angel WebSocket", value=False)

    if st.button("Test Angel Login"):
        angel, session = angel_login(
            angel_api_key, angel_client_code, angel_pin, angel_totp_secret, force_refresh=True
        )
        if angel and session:
            st.success("✅ Angel One Login Successful")
        else:
            st.error("❌ Angel login failed")

    if st.button("Stop WebSocket"):
        stop_angel_websocket()
        sync_ws_state_to_session()
        st.warning("WebSocket stop requested.")

effective_refresh_seconds = refresh_seconds
if data_source == "Angel" and ws_enable:
    effective_refresh_seconds = max(refresh_seconds, 20)

if auto_refresh:
    st_autorefresh(interval=effective_refresh_seconds * 
