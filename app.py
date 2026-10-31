# ==============================================================================
# File: Elite_Trading_Dashboard_Pro.py
# Complete Merged Final Code
# ==============================================================================

import os
import json
import math
import shutil
import hashlib
from datetime import datetime, timedelta, time, timezone
import threading
import time as pytime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import pyotp
from SmartApi import SmartConnect
from streamlit_autorefresh import st_autorefresh

SMART_WS_AVAILABLE = False
try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    SMART_WS_AVAILABLE = True
except Exception:
    SMART_WS_AVAILABLE = False

st.set_page_config(page_title="Elite Trading Dashboard Pro", layout="wide")

# =========================================================
# CONFIGURATION & PRODUCTION SETTINGS
# =========================================================
DATA_DIR = "elite_data"
BACKUP_DIR = os.path.join(DATA_DIR, "backup")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

JOURNAL_FILE = os.path.join(DATA_DIR, "trade_journal.csv")
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
PAPER_STATE_FILE = os.path.join(DATA_DIR, "paper_trade_state.csv")
DAILY_SUMMARY_FILE = os.path.join(DATA_DIR, "daily_summary.csv")
WEEKLY_SUMMARY_FILE = os.path.join(DATA_DIR, "weekly_summary.csv")
MONTHLY_SUMMARY_FILE = os.path.join(DATA_DIR, "monthly_summary.csv")
SIGNAL_AUDIT_FILE = os.path.join(DATA_DIR, "signal_audit.csv")
ERROR_LOG_FILE = os.path.join(DATA_DIR, "app_error_log.csv")
SIGNAL_STATE_FILE = os.path.join(DATA_DIR, "last_signal_state.txt")
RISK_STATE_FILE = os.path.join(DATA_DIR, "risk_state.csv")
WEEKLY_RISK_FILE = os.path.join(DATA_DIR, "weekly_risk_state.csv")
MONTHLY_RISK_FILE = os.path.join(DATA_DIR, "monthly_risk_state.csv")
OPTION_CHAIN_FILE = os.path.join(DATA_DIR, "option_chain_cache.json")
NEWS_CACHE_FILE = os.path.join(DATA_DIR, "news_cache.json")

# Mandatory Production Constraints
MANDATORY_ADX = 20
MANDATORY_MTF_ALIGN_COUNT = 2
STRICT_TRADE_ONLY_STRONG = False
DISABLE_BUY_BIAS_TRADING = False
DISABLE_SELL_BIAS_TRADING = False

# Risk & Strategy Defaults
MIN_RR_REQUIRED = 2.0
IDEAL_RR = 3.0
SIGNAL_EXPIRY_MINUTES = 3
HIGH_QUALITY_SIGNAL_MIN_SCORE = 75
MAX_SIGNAL_DUPLICATE_WINDOW_MIN = 10
NO_TRADE_NEAR_SR_PCT = 0.5
GAP_FILTER_PCT = 0.75
HIGH_VOL_ATR_PCT = 1.8
SIDEWAYS_ADX_MAX = 18
TRENDING_ADX_MIN = 25
MIN_VALIDATION_SCORE_FOR_ALERT = 70
DEFAULT_INITIAL_BALANCE = 100000.0
DATA_INTEGRITY_MIN_ROWS = 30
BACKUP_RETENTION_COUNT = 30
USE_POLLING_AS_PRIMARY = True

MAX_WS_RETRIES = 5
WS_RETRY_DELAY_SEC = 5

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
    "last_data_time": None,
    "log": []
}

# =========================================================
# ASSET DICTIONARIES
# =========================================================
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

# =========================================================
# BASIC UTILITIES & FILE MANAGEMENT
# =========================================================
def now_str():
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def backup_file(path):
    try:
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = os.path.basename(path)
            shutil.copy(path, os.path.join(BACKUP_DIR, f"{ts}_{name}"))
            enforce_backup_retention()
    except Exception as e:
        log_app_error("backup_file", e)


def enforce_backup_retention():
    try:
        files = []
        for f in os.listdir(BACKUP_DIR):
            full = os.path.join(BACKUP_DIR, f)
            if os.path.isfile(full):
                files.append((full, os.path.getmtime(full)))
        files = sorted(files, key=lambda x: x[1], reverse=True)
        if len(files) > BACKUP_RETENTION_COUNT:
            for old_file, _ in files[BACKUP_RETENTION_COUNT:]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    log_app_error("enforce_backup_retention_remove", e)
    except Exception as e:
        log_app_error("enforce_backup_retention", e)


def log_app_error(source, message):
    try:
        row = pd.DataFrame([{
            "DateTime": now_str(),
            "Source": str(source),
            "Message": str(message)
        }])
        if os.path.exists(ERROR_LOG_FILE):
            old = pd.read_csv(ERROR_LOG_FILE)
            old = pd.concat([old, row], ignore_index=True)
            old.to_csv(ERROR_LOG_FILE, index=False)
        else:
            row.to_csv(ERROR_LOG_FILE, index=False)
    except Exception:
        pass


def safe_read_csv(path, default_df=None):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        return default_df.copy() if default_df is not None else pd.DataFrame()
    except Exception as e:
        log_app_error("safe_read_csv", f"{path} -> {e}")
        return default_df.copy() if default_df is not None else pd.DataFrame()


def safe_write_csv(df, path):
    try:
        backup_file(path)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        log_app_error("safe_write_csv", f"{path} -> {e}")
        return False


def data_integrity_check(df, required_cols=None, min_rows=1):
    try:
        if df is None or df.empty:
            return False, "Empty dataframe"
        if len(df) < min_rows:
            return False, f"Insufficient rows: {len(df)}"
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return False, f"Missing columns: {missing}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


# =========================================================
# INITIALIZERS
# =========================================================
def init_users():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame([{
            "username": "admin",
            "password": hash_password("admin123")
        }])
        safe_write_csv(df, USERS_FILE)
        return
    try:
        users = pd.read_csv(USERS_FILE)
        if "username" not in users.columns or "password" not in users.columns:
            raise ValueError("Invalid users file structure")
    except Exception:
        df = pd.DataFrame([{
            "username": "admin",
            "password": hash_password("admin123")
        }])
        safe_write_csv(df, USERS_FILE)


def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        cols = [
            "DateTime", "Symbol", "Timeframe", "Signal", "EntryPrice", "ExitPrice",
            "Quantity", "StopLoss", "Target1", "Target2", "Target3", "TrailingStop",
            "Status", "PnL", "PnLPercent", "Notes", "Result", "T1_Hit", "T2_Hit", "T3_Hit",
            "TradeQualityScore", "ValidationScore", "StrategyGrade"
        ]
        safe_write_csv(pd.DataFrame(columns=cols), JOURNAL_FILE)


def init_paper_state():
    if not os.path.exists(PAPER_STATE_FILE):
        df = pd.DataFrame([{
            "Balance": DEFAULT_INITIAL_BALANCE,
            "UsedMargin": 0.0,
            "OpenPositions": 0,
            "InitialBalance": DEFAULT_INITIAL_BALANCE
        }])
        safe_write_csv(df, PAPER_STATE_FILE)


def init_daily_summary():
    if not os.path.exists(DAILY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "Date", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), DAILY_SUMMARY_FILE)


def init_weekly_summary():
    if not os.path.exists(WEEKLY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "WeekKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), WEEKLY_SUMMARY_FILE)


def init_monthly_summary():
    if not os.path.exists(MONTHLY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "MonthKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), MONTHLY_SUMMARY_FILE)


def init_risk_state():
    if not os.path.exists(RISK_STATE_FILE):
        df = pd.DataFrame([{
            "DailyLossLimit": 2000.0,
            "MaxTradesPerDay": 5,
            "RiskPerTradePct": 1.0,
            "TradingEnabled": True,
            "DailyLock": False,
            "CapitalProtectionPct": 20.0
        }])
        safe_write_csv(df, RISK_STATE_FILE)


def init_weekly_risk():
    if not os.path.exists(WEEKLY_RISK_FILE):
        df = pd.DataFrame([{
            "WeekKey": datetime.now().strftime("%Y-W%U"),
            "WeeklyLossLimit": 5000.0,
            "WeeklyPnL": 0.0,
            "WeeklyLock": False
        }])
        safe_write_csv(df, WEEKLY_RISK_FILE)


def init_monthly_risk():
    if not os.path.exists(MONTHLY_RISK_FILE):
        df = pd.DataFrame([{
            "MonthKey": datetime.now().strftime("%Y-%m"),
            "MonthlyLossLimit": 15000.0,
            "MonthlyPnL": 0.0,
            "MonthlyLock": False
        }])
        safe_write_csv(df, MONTHLY_RISK_FILE)


def init_signal_audit():
    if not os.path.exists(SIGNAL_AUDIT_FILE):
        df = pd.DataFrame(columns=[
            "DateTime", "Symbol", "Timeframe", "Signal", "SignalType", "Price",
            "ValidationScore", "TradeQualityScore", "StrategyGrade",
            "Confidence", "WinChance", "ReasonCount", "ExpiryTime"
        ])
        safe_write_csv(df, SIGNAL_AUDIT_FILE)


def load_journal():
    init_journal()
    return safe_read_csv(JOURNAL_FILE, pd.DataFrame())


def save_journal(df):
    safe_write_csv(df, JOURNAL_FILE)


def load_paper_state():
    init_paper_state()
    return safe_read_csv(PAPER_STATE_FILE, pd.DataFrame([{
        "Balance": DEFAULT_INITIAL_BALANCE,
        "UsedMargin": 0.0,
        "OpenPositions": 0,
        "InitialBalance": DEFAULT_INITIAL_BALANCE
    }]))


def save_paper_state(df):
    safe_write_csv(df, PAPER_STATE_FILE)


def load_risk_state():
    init_risk_state()
    return safe_read_csv(RISK_STATE_FILE)


def save_risk_state(df):
    safe_write_csv(df, RISK_STATE_FILE)


def load_weekly_risk():
    init_weekly_risk()
    df = safe_read_csv(WEEKLY_RISK_FILE)
    current_key = datetime.now().strftime("%Y-W%U")
    if df.empty or str(df.loc[0, "WeekKey"]) != current_key:
        df = pd.DataFrame([{
            "WeekKey": current_key,
            "WeeklyLossLimit": 5000.0,
            "WeeklyPnL": 0.0,
            "WeeklyLock": False
        }])
        save_weekly_risk(df)
    return df


def save_weekly_risk(df):
    safe_write_csv(df, WEEKLY_RISK_FILE)


def load_monthly_risk():
    init_monthly_risk()
    df = safe_read_csv(MONTHLY_RISK_FILE)
    current_key = datetime.now().strftime("%Y-%m")
    if df.empty or str(df.loc[0, "MonthKey"]) != current_key:
        df = pd.DataFrame([{
            "MonthKey": current_key,
            "MonthlyLossLimit": 15000.0,
            "MonthlyPnL": 0.0,
            "MonthlyLock": False
        }])
        save_monthly_risk(df)
    return df


def save_monthly_risk(df):
    safe_write_csv(df, MONTHLY_RISK_FILE)


def load_signal_audit():
    init_signal_audit()
    return safe_read_csv(SIGNAL_AUDIT_FILE, pd.DataFrame())


def save_signal_audit(df):
    safe_write_csv(df, SIGNAL_AUDIT_FILE)


# =========================================================
# AUTHENTICATION
# =========================================================
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
                hashed = hash_password(password.strip())
                match = users[
                    (users["username"].astype(str).str.strip() == username.strip()) &
                    (users["password"].astype(str).str.strip() == hashed)
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
                        st.sidebar.error("New passwords do not match.")
                    else:
                        users = pd.read_csv(USERS_FILE)
                        old_hash = hash_password(old_pass.strip())
                        match_idx = users[
                            (users["username"].astype(str).str.strip() == reset_user.strip()) &
                            (users["password"].astype(str).str.strip() == old_hash)
                        ].index

                        if len(match_idx) == 0:
                            st.sidebar.error("Username or old password incorrect.")
                        else:
                            users.loc[match_idx[0], "password"] = hash_password(new_pass.strip())
                            safe_write_csv(users, USERS_FILE)
                            st.sidebar.success("Password reset successful. Please login.")
                except Exception as e:
                    st.sidebar.error(f"Password reset error: {e}")

        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# =========================================================
# DATA FETCH ENGINE
# =========================================================
@st.cache_data(ttl=30)
def fetch_data(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in needed):
            return pd.DataFrame()

        df = df[needed].copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        log_app_error("fetch_data", e)
        return pd.DataFrame()


def resample_ohlcv(df, rule):
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()
        if df.empty:
            return df
        return df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()
    except Exception as e:
        log_app_error("resample_ohlcv", e)
        return pd.DataFrame()


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
            not force_refresh and
            "angel_api" in st.session_state and
            "angel_session" in st.session_state and
            st.session_state.get("angel_cache_key") == cache_key
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
        log_app_error("angel_login", e)
        return None, None


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
        from_date = to_date - timedelta(days=10)

        params = {
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        return angel.getCandleData(params)
    except Exception as e:
        log_app_error("get_angel_candles", e)
        return {"status": False, "message": str(e)}


def candles_to_df(candles):
    try:
        if not isinstance(candles, dict) or not candles.get("data"):
            return pd.DataFrame()
        df = pd.DataFrame(candles["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"]).set_index("Datetime")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return df
    except Exception as e:
        log_app_error("candles_to_df", e)
        return pd.DataFrame()


def get_angel_ltp_polling(api_key, client_code, pin, totp_secret, symbol_name):
    try:
        candles = get_angel_candles(api_key, client_code, pin, totp_secret, symbol_name, "ONE_MINUTE")
        df = candles_to_df(candles)
        if not df.empty:
            return float(df["Close"].iloc[-1])
        return None
    except Exception as e:
        log_app_error("get_angel_ltp_polling", e)
        return None


# =========================================================
# WEBSOCKET SUBSCRIPTION HANDLERS
# =========================================================
def ws_log(msg):
    with WS_STATE_LOCK:
        ts = datetime.now().strftime("%H:%M:%S")
        WS_SHARED_STATE["log"].append(f"[{ts}] {msg}")
        WS_SHARED_STATE["log"] = WS_SHARED_STATE["log"][-100:]


def ws_set_state(**kwargs):
    with WS_STATE_LOCK:
        for k, v in kwargs.items():
            WS_SHARED_STATE[k] = v


def ws_get_state():
    with WS_STATE_LOCK:
        return dict(WS_SHARED_STATE)


def sync_ws_state_to_session():
    state = ws_get_state()
    for k, v in {
        "ws_running": state["running"],
        "ws_connected": state["connected"],
        "ws_error": state["error"],
        "ws_last_tick": state["last_tick"],
        "live_price": state["live_price"],
        "ws_retries": state["retries"],
        "ws_should_run": state["should_run"],
        "ws_thread_started": state["thread_started"],
        "ws_log": state["log"]
    }.items():
        st.session_state[k] = v


def init_ws_state():
    defaults = {
        "ws_running": False,
        "ws_connected": False,
        "ws_error": "",
        "ws_last_tick": None,
        "ws_thread_started": False,
        "live_price": None,
        "ws_retries": 0,
        "ws_should_run": False,
        "ws_log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    sync_ws_state_to_session()


def stop_angel_websocket():
    try:
        ws_log("Stop requested")
        ws_set_state(should_run=False, connected=False, running=False, error="WebSocket stopped")
        state = ws_get_state()
        sws = state.get("socket")
        if sws is not None:
            try:
                if hasattr(sws, "close_connection"):
                    sws.close_connection()
                elif hasattr(sws, "close"):
                    sws.close()
            except Exception as e:
                ws_log(f"Socket close error: {e}")
        ws_set_state(socket=None, thread_started=False)
    except Exception as e:
        ws_set_state(error=f"Stop error: {e}")
        log_app_error("stop_angel_websocket", e)


def extract_ltp_from_message(message):
    if not isinstance(message, dict):
        return None
    keys = ["last_traded_price", "ltp", "lastTradedPrice", "last_traded_price_int", "close"]
    for key in keys:
        if key in message and message.get(key) is not None:
            try:
                val = float(message[key])
                if val > 10000:
                    val /= 100.0
                return val
            except Exception:
                return None
    return None


def start_angel_websocket(api_key, client_code, pin, totp_secret, symbol_name):
    init_ws_state()
    if not SMART_WS_AVAILABLE:
        ws_set_state(error="SmartWebSocketV2 not available")
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

    ws_set_state(should_run=True, thread_started=True, symbol=symbol_name, error="", retries=0)

    def run_socket():
        retry_count = 0
        while True:
            state_now = ws_get_state()
            if not state_now.get("should_run"):
                break
            if retry_count >= MAX_WS_RETRIES:
                ws_set_state(error="Max reconnect attempts reached", running=False, connected=False, retries=retry_count, thread_started=False, socket=None)
                break

            try:
                angel, session = angel_login(api_key, client_code, pin, totp_secret, force_refresh=True)
                if angel is None or session is None or not session.get("data"):
                    raise Exception("Angel login failed during reconnect")

                auth_token = session["data"].get("jwtToken")
                if not auth_token:
                    raise Exception("JWT missing")

                feed_token_resp = angel.getfeedToken()
                if isinstance(feed_token_resp, dict):
                    feed_token = feed_token_resp.get("data", {}).get("feedToken") or feed_token_resp.get("feedToken")
                else:
                    feed_token = feed_token_resp

                if not feed_token:
                    raise Exception("Feed token missing")

                sws = SmartWebSocketV2(auth_token, api_key, client_code, feed_token)
                ws_set_state(socket=sws, running=True, connected=False, retries=retry_count, error="")

                correlation_id = f"elite_feed_{symbol_name}"
                mode = 1
                token_list = [{"exchangeType": 1, "tokens": [str(symbol_token)]}]

                def on_data(wsapp, message):
                    ltp = extract_ltp_from_message(message)
                    current_state = ws_get_state()
                    ws_set_state(
                        last_tick=message,
                        connected=True,
                        running=True,
                        error="",
                        live_price=ltp if ltp is not None else current_state.get("live_price"),
                        last_data_time=now_str()
                    )

                def on_open(wsapp):
                    try:
                        ws_set_state(connected=True, running=True, error="")
                        sws.subscribe(correlation_id, mode, token_list)
                    except Exception as e:
                        ws_set_state(error=f"Subscribe error: {e}", connected=False)

                def on_error(wsapp, error):
                    ws_set_state(error=f"WS error: {error}", connected=False, running=False)

                def on_close(wsapp, *args):
                    reason = " | ".join([str(x) for x in args if x is not None]) if args else ""
                    ws_set_state(connected=False, running=False, error=f"WS closed {reason}")

                sws.on_open = on_open
                sws.on_data = on_data
                sws.on_error = on_error
                sws.on_close = on_close

                try:
                    sws.connect()
                finally:
                    ws_set_state(connected=False, running=False, socket=None)

            except Exception as e:
                ws_set_state(
                    error=f"Reconnect attempt {retry_count + 1}/{MAX_WS_RETRIES}: {e}",
                    connected=False, running=False, socket=None
                )
                log_app_error("start_angel_websocket", e)

            retry_count += 1
            ws_set_state(retries=retry_count)
            if not ws_get_state().get("should_run"):
                break
            pytime.sleep(WS_RETRY_DELAY_SEC)

        ws_set_state(running=False, connected=False, socket=None, thread_started=False)

    threading.Thread(target=run_socket, daemon=True).start()
    sync_ws_state_to_session()


# =========================================================
# SNAPSHOT & SCANNER ENGINES (FROM APP.PY)
# =========================================================
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
        except Exception as e:
            log_app_error("fetch_market_snapshot", e)
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
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
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


# =========================================================
# INDICATORS ENGINE
# =========================================================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(1, 99)


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

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().replace(0, 1e-10)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
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
        df["VWAP"] = tp.expanding().mean().bfill().ffill()
        return df

    trade_date = pd.Series(df.index.date, index=df.index)
    cum_pv = (tp * df["Volume"]).groupby(trade_date).cumsum()
    cum_vol = df["Volume"].groupby(trade_date).cumsum()
    df["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, tp.expanding().mean())
    df["VWAP"] = pd.to_numeric(df["VWAP"], errors="coerce").replace([np.inf, -np.inf], np.nan).bfill().ffill()
    return df


def convert_to_heikin_ashi(df):
    ha = df.copy()
    if ha.empty:
        return ha
    ha["HA_Close"] = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4
    ha_open = [(ha["Open"].iloc[0] + ha["Close"].iloc[0]) / 2]
    for i in range(1, len(ha)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"] = ha[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    return ha


def compute_supertrend(df, period=10, multiplier=3):
    data = df.copy()
    if data.empty:
        data["SuperTrend"] = np.nan
        data["SuperTrendDirection"] = 0
        return data

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(data)):
        if pd.notna(close.iloc[i - 1]) and pd.notna(final_upperband.iloc[i - 1]):
            if close.iloc[i - 1] <= final_upperband.iloc[i - 1]:
                final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i - 1])
            else:
                final_upperband.iloc[i] = upperband.iloc[i]

        if pd.notna(close.iloc[i - 1]) and pd.notna(final_lowerband.iloc[i - 1]):
            if close.iloc[i - 1] >= final_lowerband.iloc[i - 1]:
                final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i - 1])
            else:
                final_lowerband.iloc[i] = lowerband.iloc[i]

    supertrend = pd.Series(index=data.index, dtype=float)
    direction = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        if i == 0:
            supertrend.iloc[i] = final_upperband.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_upper = final_upperband.iloc[i - 1]
        prev_lower = final_lowerband.iloc[i - 1]
        curr_close = close.iloc[i]

        if pd.notna(prev_upper) and curr_close > prev_upper:
            direction.iloc[i] = 1
        elif pd.notna(prev_lower) and curr_close < prev_lower:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = final_lowerband.iloc[i] if direction.iloc[i] == 1 else final_upperband.iloc[i]

    data["SuperTrend"] = supertrend
    data["SuperTrendDirection"] = direction.fillna(0).astype(int)
    return data


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

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

    df = compute_supertrend(df, period=10, multiplier=3)

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
        df["Breakout"].shift(1) &
        (df["Low"] <= df["PivotHigh"].shift(1)) &
        (df["Close"] > df["PivotHigh"].shift(1))
    )

    df["RetestBear"] = (
        df["Breakdown"].shift(1) &
        (df["High"] >= df["PivotLow"].shift(1)) &
        (df["Close"] < df["PivotLow"].shift(1))
    )

    k, d = compute_stochastic(df)
    df["StochK"] = k.rolling(3).mean()
    df["StochD"] = d.rolling(3).mean()
    df["ADX"], df["PlusDI"], df["MinusDI"] = compute_adx(df)

    candle_body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-10)
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["Hammer"] = ((lower_shadow > candle_body * 2) & (upper_shadow < candle_body))
    df["ShootingStar"] = ((upper_shadow > candle_body * 2) & (lower_shadow < candle_body))
    df["BullishEngulfing"] = (
        (df["Close"] > df["Open"]) &
        (df["Close"].shift(1) < df["Open"].shift(1)) &
        (df["Open"] <= df["Close"].shift(1)) &
        (df["Close"] >= df["Open"].shift(1))
    )
    df["BearishEngulfing"] = (
        (df["Close"] < df["Open"]) &
        (df["Close"].shift(1) > df["Open"].shift(1)) &
        (df["Open"] >= df["Close"].shift(1)) &
        (df["Close"] <= df["Open"].shift(1))
    )
    df["Doji"] = candle_body <= (candle_range * 0.1)
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
    return df


# =========================================================
# NEWS FILTER API
# =========================================================
@st.cache_data(ttl=300)
def get_news_events(news_api_key, symbol):
    if not news_api_key:
        return {"allow_trade": True, "message": "News API key not configured", "events": []}
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "sortBy": "publishedAt",
            "pageSize": 10,
            "language": "en",
            "apiKey": news_api_key
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return {"allow_trade": True, "message": f"News API unavailable ({resp.status_code})", "events": []}

        data = resp.json()
        articles = data.get("articles", [])
        important = []
        keywords = ["results", "earnings", "fed", "inflation", "policy", "rbi", "rate", "war", "crash", "merger", "acquisition", "guidance"]

        now_dt = datetime.utcnow()
        for a in articles:
            title = (a.get("title") or "").lower()
            published = a.get("publishedAt")
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pub_dt = None

            recent = False
            if pub_dt is not None:
                recent = (now_dt - pub_dt) <= timedelta(hours=6)

            if any(k in title for k in keywords) and recent:
                important.append({
                    "title": a.get("title"),
                    "source": a.get("source", {}).get("name", ""),
                    "publishedAt": published
                })

        allow_trade = len(important) == 0
        return {
            "allow_trade": allow_trade,
            "message": "No high-impact recent news" if allow_trade else "High-impact recent news detected",
            "events": important
        }
    except Exception as e:
        log_app_error("get_news_events", e)
        return {"allow_trade": True, "message": "News filter fallback", "events": []}


# =========================================================
# OPTION CHAIN / PCR / OI ENGINE
# =========================================================
@st.cache_data(ttl=120)
def get_option_chain_snapshot(symbol="NIFTY"):
    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept-Language": "en-US,en;q=0.9"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        with open(OPTION_CHAIN_FILE, "w") as f:
            json.dump({"symbol": symbol, "data": data}, f)
        return data
    except Exception as e:
        log_app_error("get_option_chain_snapshot", e)
        try:
            if os.path.exists(OPTION_CHAIN_FILE):
                with open(OPTION_CHAIN_FILE, "r") as f:
                    cache = json.load(f)
                    if cache.get("symbol") == symbol:
                        return cache.get("data")
        except Exception:
            pass
        return None


def get_dynamic_option_chain_symbol(selected_symbol):
    if selected_symbol == "BANKNIFTY":
        return "BANKNIFTY"
    return "NIFTY"


def parse_option_chain_metrics(symbol="NIFTY"):
    data = get_option_chain_snapshot(symbol)
    if not data:
        return {
            "pcr": None,
            "oi_bias": "Neutral",
            "long_unwinding": False,
            "short_covering": False,
            "call_oi": 0,
            "put_oi": 0
        }
    try:
        total_ce, total_pe = 0, 0
        ce_change_oi, pe_change_oi = 0, 0

        for item in data.get("records", {}).get("data", []):
            if "CE" in item:
                total_ce += item["CE"].get("openInterest", 0)
                ce_change_oi += item["CE"].get("changeinOpenInterest", 0)
            if "PE" in item:
                total_pe += item["PE"].get("openInterest", 0)
                pe_change_oi += item["PE"].get("changeinOpenInterest", 0)

        pcr = round(total_pe / total_ce, 2) if total_ce > 0 else None
        oi_bias = "Bullish" if total_pe > total_ce else "Bearish" if total_ce > total_pe else "Neutral"
        long_unwinding = ce_change_oi < 0 and pe_change_oi > 0
        short_covering = ce_change_oi > 0 and pe_change_oi < 0

        return {
            "pcr": pcr,
            "oi_bias": oi_bias,
            "long_unwinding": long_unwinding,
            "short_covering": short_covering,
            "call_oi": total_ce,
            "put_oi": total_pe
        }
    except Exception as e:
        log_app_error("parse_option_chain_metrics", e)
        return {
            "pcr": None,
            "oi_bias": "Neutral",
            "long_unwinding": False,
            "short_covering": False,
            "call_oi": 0,
            "put_oi": 0
        }


# =========================================================
# SIGNAL ANALYSIS HELPERS
# =========================================================
def detect_gap_filter(df):
    try:
        if len(df) < 2:
            return {"gap_up": False, "gap_down": False, "gap_pct": 0.0, "allow_trade": True}
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev_close = float(prev["Close"])
        current_open = float(latest["Open"])
        gap_pct = ((current_open - prev_close) / prev_close) * 100 if prev_close else 0.0
        return {
            "gap_up": gap_pct >= GAP_FILTER_PCT,
            "gap_down": gap_pct <= -GAP_FILTER_PCT,
            "gap_pct": round(gap_pct, 2),
            "allow_trade": abs(gap_pct) < 2.5
        }
    except Exception as e:
        log_app_error("detect_gap_filter", e)
        return {"gap_up": False, "gap_down": False, "gap_pct": 0.0, "allow_trade": True}


def detect_market_regime(df):
    try:
        latest = df.iloc[-1]
        close_price = float(latest["Close"])
        adx = float(latest["ADX"]) if pd.notna(latest["ADX"]) else 0.0
        atr = float(latest["ATR"]) if pd.notna(latest["ATR"]) else 0.0
        ema20 = float(latest["EMA20"])
        ema50 = float(latest["EMA50"])

        atr_pct = (atr / close_price) * 100 if close_price else 0.0
        ema_spread_pct = (abs(ema20 - ema50) / close_price) * 100 if close_price else 0.0

        trending = adx >= TRENDING_ADX_MIN and ema_spread_pct >= 0.3
        sideways = adx <= SIDEWAYS_ADX_MAX and ema_spread_pct < 0.3
        high_volatility = atr_pct >= HIGH_VOL_ATR_PCT

        if trending and high_volatility:
            regime = "Trending High Volatility"
        elif trending:
            regime = "Trending"
        elif sideways:
            regime = "Sideways"
        elif high_volatility:
            regime = "High Volatility"
        else:
            regime = "Balanced"

        return {
            "regime": regime,
            "trending": trending,
            "sideways": sideways,
            "high_volatility": high_volatility,
            "atr_pct": round(atr_pct, 2),
            "ema_spread_pct": round(ema_spread_pct, 2)
        }
    except Exception as e:
        log_app_error("detect_market_regime", e)
        return {
            "regime": "Unknown",
            "trending": False,
            "sideways": False,
            "high_volatility": False,
            "atr_pct": 0.0,
            "ema_spread_pct": 0.0
        }


def classify_strategy_grade(score):
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def strategy_health_monitor(validation_score, win_rate, profit_factor, max_drawdown):
    health = 50
    if validation_score >= 80:
        health += 20
    elif validation_score >= 65:
        health += 10
    if win_rate >= 60:
        health += 15
    elif win_rate < 45:
        health -= 10
    if profit_factor >= 1.8:
        health += 10
    elif profit_factor < 1.2:
        health -= 10
    if max_drawdown <= 5:
        health += 10
    elif max_drawdown > 15:
        health -= 15

    health = max(0, min(100, health))
    if health >= 85:
        label = "Excellent"
    elif health >= 70:
        label = "Healthy"
    elif health >= 55:
        label = "Average"
    elif health >= 40:
        label = "Weak"
    else:
        label = "Critical"
    return health, label


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
            return {"valid": False, "message": "Avoid first few minutes after open"}
        if time(12, 15) <= current_t <= time(13, 15):
            return {"valid": False, "message": "Lunch low momentum zone"}
        return {"valid": True, "message": "Session timing okay"}
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


# =========================================================
# CORE SIGNAL ENGINE: DO NOT REPLACE
# =========================================================
def analyze_signal(
    df,
    htf_bias=None,
    timeframe_label="1d",
    backtest_win_rate=50,
    mtf_bull_count=0,
    mtf_bear_count=0,
    option_metrics=None,
    news_info=None
):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    ha_df = convert_to_heikin_ashi(df.tail(10))
    ha_latest = ha_df.iloc[-1]
    ha_prev = ha_df.iloc[-2]

    ha_bullish = ha_latest["HA_Close"] > ha_latest["HA_Open"]
    ha_bearish = ha_latest["HA_Close"] < ha_latest["HA_Open"]
    ha_trend_up = ha_bullish and (ha_prev["HA_Close"] > ha_prev["HA_Open"])
    ha_trend_down = ha_bearish and (ha_prev["HA_Close"] < ha_prev["HA_Open"])

    close_price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"]) if pd.notna(latest["EMA200"]) else close_price
    ema20_slope = ema20 - float(prev["EMA20"])
    vwap = float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else close_price
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["SignalLine"])
    atr = float(latest["ATR"]) if pd.notna(latest["ATR"]) else 0.0
    stoch_k = float(latest["StochK"]) if pd.notna(latest["StochK"]) else 50.0
    stoch_d = float(latest["StochD"]) if pd.notna(latest["StochD"]) else 50.0
    adx = float(latest["ADX"]) if pd.notna(latest["ADX"]) else 0.0
    support = float(latest["PivotLow"]) if pd.notna(latest["PivotLow"]) else close_price
    resistance = float(latest["PivotHigh"]) if pd.notna(latest["PivotHigh"]) else close_price
    supertrend_dir = int(latest["SuperTrendDirection"]) if pd.notna(latest["SuperTrendDirection"]) else 0
    plus_di = float(latest["PlusDI"]) if pd.notna(latest["PlusDI"]) else 0
    minus_di = float(latest["MinusDI"]) if pd.notna(latest["MinusDI"]) else 0

    distance_to_resistance = ((resistance - close_price) / close_price) * 100 if close_price else 0
    distance_to_support = ((close_price - support) / close_price) * 100 if close_price else 0

    volume_spike = bool(latest["Volume_Spike"])
    breakout = bool(latest["Breakout"])
    breakdown = bool(latest["Breakdown"])
    body_strength = float(latest["BodyStrength"]) if pd.notna(latest["BodyStrength"]) else 0

    bullish = 0
    bearish = 0
    reasons = []

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

    if close_price > ema200:
        bullish += 2
        reasons.append("Price above EMA200")
    elif close_price < ema200:
        bearish += 2
        reasons.append("Price below EMA200")

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
    elif ema20_slope < 0:
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
        else:
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

    if supertrend_dir == 1:
        bullish += 3
        reasons.append("SuperTrend bullish")
    elif supertrend_dir == -1:
        bearish += 3
        reasons.append("SuperTrend bearish")

    if ha_bullish:
        bullish += 1
        reasons.append("Heikin Ashi bullish candle")
    if ha_bearish:
        bearish += 1
        reasons.append("Heikin Ashi bearish candle")
    if ha_trend_up:
        bullish += 2
        reasons.append("Heikin Ashi bullish continuation")
    if ha_trend_down:
        bearish += 2
        reasons.append("Heikin Ashi bearish continuation")

    if breakout:
        bullish += 2
        reasons.append("Resistance breakout")
        if volume_spike:
            bullish += 2
            reasons.append("Breakout with volume")

    if breakdown:
        bearish += 2
        reasons.append("Support breakdown")
        if volume_spike:
            bearish += 2
            reasons.append("Breakdown with volume")

    if bool(latest.get("RetestBull", False)):
        bullish += 3
        reasons.append("Bullish retest confirmed")
    if bool(latest.get("RetestBear", False)):
        bearish += 3
        reasons.append("Bearish retest confirmed")

    if bool(latest.get("Hammer", False)):
        bullish += 2
        reasons.append("Hammer pattern")
    if bool(latest.get("ShootingStar", False)):
        bearish += 2
        reasons.append("Shooting Star pattern")
    if bool(latest.get("BullishEngulfing", False)):
        bullish += 3
        reasons.append("Bullish Engulfing")
    if bool(latest.get("BearishEngulfing", False)):
        bearish += 3
        reasons.append("Bearish Engulfing")
    if bool(latest.get("Doji", False)):
        reasons.append("Doji indecision")

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

    if bullish > bearish and distance_to_resistance < NO_TRADE_NEAR_SR_PCT:
        bullish -= 2
        reasons.append("Too close to resistance")
    if bearish > bullish and distance_to_support < NO_TRADE_NEAR_SR_PCT:
        bearish -= 2
        reasons.append("Too close to support")

    if htf_bias == "bullish":
        bullish += 3
        reasons.append("Higher timeframe bullish confirmation")
    elif htf_bias =
