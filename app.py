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

st.set_page_config(page_title="Elite Trading Dashboard Final Ultimate", layout="wide")

# =========================================================
# FILES / CONFIG
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

MAX_WS_RETRIES = 5
WS_RETRY_DELAY_SEC = 5

MIN_RR_REQUIRED = 2.0
IDEAL_RR = 3.0
STRICT_TRADE_ONLY_STRONG = False
DISABLE_BUY_BIAS_TRADING = True
DISABLE_SELL_BIAS_TRADING = True
MANDATORY_ADX = 20
MANDATORY_MTF_ALIGN_COUNT = 2
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
# BASIC UTILITIES
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
# INIT FILES
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


# =========================================================
# LOAD / SAVE
# =========================================================
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
# AUTH
# =========================================================
def login_panel():
    init_users()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.sidebar.subheader("Login")
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
        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# =========================================================
# DATA FETCH
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
# OPTIONAL WEBSOCKET
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
    ws_log("Entered start_angel_websocket")

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
# INDICATORS
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
    return df


# =========================================================
# NEWS FILTER API
# =========================================================
@st.cache_data(ttl=300)
def get_news_events(news_api_key, symbol):
    if not news_api_key:
        return {"allow_trade": True, "message": "News API key not configured", "events": []}
    try:
        query = symbol
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
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
# OPTION CHAIN / PCR / OI
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
    elif htf_bias == "bearish":
        bearish += 3
        reasons.append("Higher timeframe bearish confirmation")

    regime = detect_market_regime(df)
    gap_info = detect_gap_filter(df)
    session_info = session_filter(df, timeframe_label)
    news_info = news_info or {"allow_trade": True, "message": "News filter unavailable", "events": []}

    sideways = bool(regime["sideways"])
    mtf_strict_bull_ok = mtf_bull_count >= MANDATORY_MTF_ALIGN_COUNT
    mtf_strict_bear_ok = mtf_bear_count >= MANDATORY_MTF_ALIGN_COUNT

    option_metrics = option_metrics or {
        "pcr": None,
        "oi_bias": "Neutral",
        "long_unwinding": False,
        "short_covering": False
    }

    pcr = option_metrics.get("pcr")
    oi_bias = option_metrics.get("oi_bias")
    long_unwinding = option_metrics.get("long_unwinding", False)
    short_covering = option_metrics.get("short_covering", False)

    if pcr is not None:
        if pcr > 1.2:
            bullish += 1
            reasons.append(f"PCR bullish ({pcr})")
        elif pcr < 0.8:
            bearish += 1
            reasons.append(f"PCR bearish ({pcr})")

    if oi_bias == "Bullish":
        bullish += 1
        reasons.append("OI bias bullish")
    elif oi_bias == "Bearish":
        bearish += 1
        reasons.append("OI bias bearish")

    if short_covering:
        bullish += 1
        reasons.append("Short covering detected")
    if long_unwinding:
        bearish += 1
        reasons.append("Long unwinding detected")

    signal_text = "NO TRADE"
    signal_type = "neutral"
    strength = "Low"
    stop_loss = None
    partial_exit = None
    target1 = None
    target2 = None
    target3 = None
    trailing_stop = None
    rr_ratio = None

    if atr and atr > 0:
        rr_ratio = IDEAL_RR

    buy_mandatory_ok = all([
        adx >= MANDATORY_ADX,
        close_price > ema200,
        supertrend_dir == 1,
        ha_trend_up,
        volume_spike,
        htf_bias == "bullish",
        mtf_strict_bull_ok,
        not sideways,
        session_info["valid"],
        gap_info["allow_trade"],
        news_info["allow_trade"]
    ])

    sell_mandatory_ok = all([
        adx >= MANDATORY_ADX,
        close_price < ema200,
        supertrend_dir == -1,
        ha_trend_down,
        volume_spike,
        htf_bias == "bearish",
        mtf_strict_bear_ok,
        not sideways,
        session_info["valid"],
        gap_info["allow_trade"],
        news_info["allow_trade"]
    ])

    if atr and atr > 0:
        buy_sl = close_price - atr
        buy_t1 = close_price + 2 * atr
        buy_t2 = close_price + 3 * atr
        buy_t3 = close_price + 4 * atr
        buy_rr = (buy_t1 - close_price) / (close_price - buy_sl) if close_price > buy_sl else None

        sell_sl = close_price + atr
        sell_t1 = close_price - 2 * atr
        sell_t2 = close_price - 3 * atr
        sell_t3 = close_price - 4 * atr
        sell_rr = (close_price - sell_t1) / (sell_sl - close_price) if sell_sl > close_price else None

        if bullish >= 10 and (bullish - bearish) >= 4 and rsi > 55 and macd > macd_signal and buy_mandatory_ok and buy_rr is not None and buy_rr >= MIN_RR_REQUIRED:
            signal_text = "STRONG BUY"
            signal_type = "buy"
            strength = "High"
            stop_loss = buy_sl
            target1 = buy_t1
            target2 = buy_t2
            target3 = buy_t3
            partial_exit = target1
            trailing_stop = close_price - (0.5 * atr)
            rr_ratio = round(buy_rr, 2)

        elif bearish >= 10 and (bearish - bullish) >= 4 and rsi < 45 and macd < macd_signal and sell_mandatory_ok and sell_rr is not None and sell_rr >= MIN_RR_REQUIRED:
            signal_text = "STRONG SELL"
            signal_type = "sell"
            strength = "High"
            stop_loss = sell_sl
            target1 = sell_t1
            target2 = sell_t2
            target3 = sell_t3
            partial_exit = target1
            trailing_stop = close_price + (0.5 * atr)
            rr_ratio = round(sell_rr, 2)

        elif not STRICT_TRADE_ONLY_STRONG:
            if bullish > bearish and bullish >= 7 and supertrend_dir == 1 and not DISABLE_BUY_BIAS_TRADING:
                signal_text = "BUY BIAS"
                signal_type = "buy"
                strength = "Medium"
            elif bearish > bullish and bearish >= 7 and supertrend_dir == -1 and not DISABLE_SELL_BIAS_TRADING:
                signal_text = "SELL BIAS"
                signal_type = "sell"
                strength = "Medium"

    confidence = min(100, max(40, int((max(bullish, bearish) / 18) * 100)))
    if supertrend_dir == 1 and bullish > bearish:
        confidence += 3
    elif supertrend_dir == -1 and bearish > bullish:
        confidence += 3
    if ha_trend_up and bullish > bearish:
        confidence += 3
    elif ha_trend_down and bearish > bullish:
        confidence += 3
    confidence = min(100, confidence)

    validation_score = 0
    validation_score += min(25, max(0, (max(bullish, bearish) * 2)))
    validation_score += 10 if adx >= MANDATORY_ADX else 0
    validation_score += 10 if volume_spike else 0
    validation_score += 10 if ((signal_type == "buy" and close_price > ema200) or (signal_type == "sell" and close_price < ema200)) else 0
    validation_score += 10 if ((signal_type == "buy" and supertrend_dir == 1) or (signal_type == "sell" and supertrend_dir == -1)) else 0
    validation_score += 10 if ((signal_type == "buy" and ha_trend_up) or (signal_type == "sell" and ha_trend_down)) else 0
    validation_score += 10 if ((signal_type == "buy" and htf_bias == "bullish") or (signal_type == "sell" and htf_bias == "bearish")) else 0
    validation_score += 10 if ((signal_type == "buy" and mtf_strict_bull_ok) or (signal_type == "sell" and mtf_strict_bear_ok)) else 0
    validation_score += 5 if rr_ratio is not None and rr_ratio >= MIN_RR_REQUIRED else 0
    validation_score += 5 if oi_bias != "Neutral" else 0
    if not gap_info["allow_trade"]:
        validation_score -= 10
    if sideways:
        validation_score -= 15
    if not news_info["allow_trade"]:
        validation_score -= 10
    validation_score = max(0, min(100, validation_score))

    strategy_grade = classify_strategy_grade(validation_score)
    trade_quality_score = int((validation_score + confidence + min(95, backtest_win_rate)) / 3)
    trade_quality_score = max(0, min(100, trade_quality_score))

    win_chance = calculate_win_chance(
        bullish, bearish, adx, volume_spike, breakout, breakdown, htf_bias, sideways, rr_ratio, backtest_win_rate
    )

    no_trade_reason = None
    if signal_type in ["buy", "sell"] and rr_ratio is not None and rr_ratio < MIN_RR_REQUIRED:
        no_trade_reason = "Risk Reward below 1:2"
    elif signal_type == "buy" and not buy_mandatory_ok:
        no_trade_reason = "BUY mandatory confirmations failed"
        signal_text = "NO TRADE"
        signal_type = "neutral"
    elif signal_type == "sell" and not sell_mandatory_ok:
        no_trade_reason = "SELL mandatory confirmations failed"
        signal_text = "NO TRADE"
        signal_type = "neutral"
    elif validation_score < MIN_VALIDATION_SCORE_FOR_ALERT and signal_type in ["buy", "sell"]:
        no_trade_reason = "Validation score below threshold"
        signal_text = "NO TRADE"
        signal_type = "neutral"
    elif signal_type == "buy" and distance_to_resistance < NO_TRADE_NEAR_SR_PCT:
        no_trade_reason = "No trade zone near resistance"
        signal_text = "NO TRADE"
        signal_type = "neutral"
    elif signal_type == "sell" and distance_to_support < NO_TRADE_NEAR_SR_PCT:
        no_trade_reason = "No trade zone near support"
        signal_text = "NO TRADE"
        signal_type = "neutral"

    if no_trade_reason:
        reasons.append(no_trade_reason)

    return {
        "close_price": close_price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
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
        "target3": target3,
        "trailing_stop": trailing_stop,
        "rr_ratio": rr_ratio,
        "reasons": reasons,
        "bullish_score": bullish,
        "bearish_score": bearish,
        "session_message": session_info["message"],
        "supertrend_direction": supertrend_dir,
        "ha_bullish": ha_bullish,
        "ha_bearish": ha_bearish,
        "ha_trend_up": ha_trend_up,
        "ha_trend_down": ha_trend_down,
        "validation_score": validation_score,
        "strategy_grade": strategy_grade,
        "trade_quality_score": trade_quality_score,
        "market_regime": regime["regime"],
        "trending_market": regime["trending"],
        "sideways_market": regime["sideways"],
        "high_volatility_market": regime["high_volatility"],
        "atr_pct": regime["atr_pct"],
        "gap_up": gap_info["gap_up"],
        "gap_down": gap_info["gap_down"],
        "gap_pct": gap_info["gap_pct"],
        "mtf_bull_ok": mtf_strict_bull_ok,
        "mtf_bear_ok": mtf_strict_bear_ok,
        "no_trade_zone": (
            (signal_type == "buy" and distance_to_resistance < NO_TRADE_NEAR_SR_PCT) or
            (signal_type == "sell" and distance_to_support < NO_TRADE_NEAR_SR_PCT)
        ),
        "oi_bias": oi_bias,
        "pcr": pcr,
        "long_unwinding": long_unwinding,
        "short_covering": short_covering,
        "news_message": news_info.get("message", "")
    }


# =========================================================
# SIGNAL AUDIT / DUPLICATE / EXPIRY
# =========================================================
def add_signal_audit(symbol, timeframe, analysis):
    try:
        audit = load_signal_audit()
        expiry_time = datetime.now() + timedelta(minutes=SIGNAL_EXPIRY_MINUTES)
        row = {
            "DateTime": now_str(),
            "Symbol": symbol,
            "Timeframe": timeframe,
            "Signal": analysis["signal_text"],
            "SignalType": analysis["signal_type"],
            "Price": round(float(analysis["close_price"]), 2),
            "ValidationScore": analysis.get("validation_score", 0),
            "TradeQualityScore": analysis.get("trade_quality_score", 0),
            "StrategyGrade": analysis.get("strategy_grade", "D"),
            "Confidence": analysis.get("confidence", 0),
            "WinChance": analysis.get("win_chance", 0),
            "ReasonCount": len(analysis.get("reasons", [])),
            "ExpiryTime": expiry_time.strftime("%d-%m-%Y %H:%M:%S")
        }
        audit = pd.concat([audit, pd.DataFrame([row])], ignore_index=True)
        save_signal_audit(audit)
    except Exception as e:
        log_app_error("add_signal_audit", e)


def is_duplicate_signal(symbol, timeframe, analysis):
    try:
        audit = load_signal_audit()
        if audit.empty:
            return False
        audit["DateTimeParsed"] = pd.to_datetime(audit["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        cutoff = datetime.now() - timedelta(minutes=MAX_SIGNAL_DUPLICATE_WINDOW_MIN)
        recent = audit[
            (audit["DateTimeParsed"] >= cutoff) &
            (audit["Symbol"] == symbol) &
            (audit["Timeframe"] == timeframe) &
            (audit["Signal"] == analysis["signal_text"])
        ]
        return not recent.empty
    except Exception as e:
        log_app_error("is_duplicate_signal", e)
        return False


def get_signal_expiry_remaining(symbol, timeframe, signal_text):
    try:
        audit = load_signal_audit()
        if audit.empty:
            return None
        rows = audit[
            (audit["Symbol"] == symbol) &
            (audit["Timeframe"] == timeframe) &
            (audit["Signal"] == signal_text)
        ].copy()
        if rows.empty:
            return None
        rows["ExpiryParsed"] = pd.to_datetime(rows["ExpiryTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        expiry = rows["ExpiryParsed"].max()
        if pd.isna(expiry):
            return None
        diff = expiry - datetime.now()
        return max(0, int(diff.total_seconds()))
    except Exception as e:
        log_app_error("get_signal_expiry_remaining", e)
        return None


# =========================================================
# RISK / PAPER TRADING
# =========================================================
def get_today_closed_trades():
    journal = load_journal()
    if journal.empty:
        return pd.DataFrame()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return closed
    closed["DateOnly"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce").dt.date
    today = datetime.now().date()
    return closed[closed["DateOnly"] == today].copy()


def get_week_closed_trades():
    journal = load_journal()
    if journal.empty:
        return pd.DataFrame()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return closed
    closed["DateOnly"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    start_week = datetime.now() - timedelta(days=datetime.now().weekday())
    start_week = datetime(start_week.year, start_week.month, start_week.day)
    return closed[closed["DateOnly"] >= start_week].copy()


def get_month_closed_trades():
    journal = load_journal()
    if journal.empty:
        return pd.DataFrame()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return closed
    closed["DateOnly"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    start_month = datetime(datetime.now().year, datetime.now().month, 1)
    return closed[closed["DateOnly"] >= start_month].copy()


def update_risk_locks():
    try:
        daily_trades = get_today_closed_trades()
        weekly_trades = get_week_closed_trades()
        monthly_trades = get_month_closed_trades()

        risk_state = load_risk_state()
        weekly = load_weekly_risk()
        monthly = load_monthly_risk()
        paper = load_paper_state()

        daily_pnl = pd.to_numeric(daily_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        weekly_pnl = pd.to_numeric(weekly_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        monthly_pnl = pd.to_numeric(monthly_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

        daily_limit = float(risk_state.loc[0, "DailyLossLimit"])
        weekly_limit = float(weekly.loc[0, "WeeklyLossLimit"])
        monthly_limit = float(monthly.loc[0, "MonthlyLossLimit"])
        capital_protection_pct = float(risk_state.loc[0, "CapitalProtectionPct"])

        balance = float(paper.loc[0, "Balance"])
        initial_balance = float(paper.loc[0, "InitialBalance"])
        capital_floor = initial_balance * (1 - capital_protection_pct / 100)

        risk_state.loc[0, "DailyLock"] = bool(daily_pnl <= -abs(daily_limit) or balance <= capital_floor)
        weekly.loc[0, "WeeklyPnL"] = weekly_pnl
        weekly.loc[0, "WeeklyLock"] = bool(weekly_pnl <= -abs(weekly_limit))
        monthly.loc[0, "MonthlyPnL"] = monthly_pnl
        monthly.loc[0, "MonthlyLock"] = bool(monthly_pnl <= -abs(monthly_limit))

        save_risk_state(risk_state)
        save_weekly_risk(weekly)
        save_monthly_risk(monthly)
    except Exception as e:
        log_app_error("update_risk_locks", e)


def risk_check_before_trade():
    update_risk_locks()
    risk_state = load_risk_state()
    weekly_risk = load_weekly_risk()
    monthly_risk = load_monthly_risk()
    journal_today = get_today_closed_trades()

    trading_enabled = bool(risk_state.loc[0, "TradingEnabled"])
    daily_lock = bool(risk_state.loc[0, "DailyLock"])
    weekly_lock = bool(weekly_risk.loc[0, "WeeklyLock"])
    monthly_lock = bool(monthly_risk.loc[0, "MonthlyLock"])
    max_trades_per_day = int(risk_state.loc[0, "MaxTradesPerDay"])
    trades_today = len(journal_today)

    if not trading_enabled:
        return False, "Trading disabled by risk management"
    if daily_lock:
        return False, "Daily lock active"
    if weekly_lock:
        return False, "Weekly lock active"
    if monthly_lock:
        return False, "Monthly lock active"
    if trades_today >= max_trades_per_day:
        return False, f"Max trades per day reached ({max_trades_per_day})"
    return True, "Risk checks passed"


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


def add_trade_to_journal(symbol, timeframe, signal, entry_price, quantity, stop_loss, target1, target2, target3, trailing_stop, analysis, notes=""):
    journal = load_journal()
    new_row = {
        "DateTime": now_str(),
        "Symbol": symbol,
        "Timeframe": timeframe,
        "Signal": signal,
        "EntryPrice": entry_price,
        "ExitPrice": np.nan,
        "Quantity": quantity,
        "StopLoss": stop_loss,
        "Target1": target1,
        "Target2": target2,
        "Target3": target3,
        "TrailingStop": trailing_stop,
        "Status": "OPEN",
        "PnL": np.nan,
        "PnLPercent": np.nan,
        "Notes": notes,
        "Result": "",
        "T1_Hit": False,
        "T2_Hit": False,
        "T3_Hit": False,
        "TradeQualityScore": analysis.get("trade_quality_score", 0),
        "ValidationScore": analysis.get("validation_score", 0),
        "StrategyGrade": analysis.get("strategy_grade", "D")
    }
    journal = pd.concat([journal, pd.DataFrame([new_row])], ignore_index=True)
    save_journal(journal)


def close_trade_in_journal(index_id, exit_price, result=None, t1_hit=None, t2_hit=None, t3_hit=None):
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
        journal.at[index_id, "Result"] = result if result else ("WIN" if pnl > 0 else "LOSS")
        if t1_hit is not None:
            journal.at[index_id, "T1_Hit"] = bool(t1_hit)
        if t2_hit is not None:
            journal.at[index_id, "T2_Hit"] = bool(t2_hit)
        if t3_hit is not None:
            journal.at[index_id, "T3_Hit"] = bool(t3_hit)
        save_journal(journal)


def auto_paper_trade_exit(current_price):
    try:
        journal = load_journal()
        if journal.empty:
            return 0

        open_trades = journal[journal["Status"] == "OPEN"].copy()
        if open_trades.empty:
            return 0

        closed_count = 0
        for idx, row in open_trades.iterrows():
            signal = str(row["Signal"]).upper()
            entry = float(row["EntryPrice"])
            sl = float(row["StopLoss"]) if pd.notna(row["StopLoss"]) else None
            t1 = float(row["Target1"]) if pd.notna(row["Target1"]) else None
            t2 = float(row["Target2"]) if pd.notna(row["Target2"]) else None
            t3 = float(row["Target3"]) if pd.notna(row["Target3"]) else None
            qty = float(row["Quantity"])

            t1_hit = False
            t2_hit = False
            t3_hit = False
            exit_price = None
            result = None

            if "BUY" in signal:
                if t1 is not None and current_price >= t1:
                    t1_hit = True
                if t2 is not None and current_price >= t2:
                    t2_hit = True
                if t3 is not None and current_price >= t3:
                    t3_hit = True

                if sl is not None and current_price <= sl:
                    exit_price = sl
                    result = "LOSS"
                elif t3 is not None and current_price >= t3:
                    exit_price = t3
                    result = "WIN"
                elif t2 is not None and current_price >= t2:
                    exit_price = t2
                    result = "WIN"
                elif t1 is not None and current_price >= t1:
                    exit_price = t1
                    result = "WIN"

            else:
                if t1 is not None and current_price <= t1:
                    t1_hit = True
                if t2 is not None and current_price <= t2:
                    t2_hit = True
                if t3 is not None and current_price <= t3:
                    t3_hit = True

                if sl is not None and current_price >= sl:
                    exit_price = sl
                    result = "LOSS"
                elif t3 is not None and current_price <= t3:
                    exit_price = t3
                    result = "WIN"
                elif t2 is not None and current_price <= t2:
                    exit_price = t2
                    result = "WIN"
                elif t1 is not None and current_price <= t1:
                    exit_price = t1
                    result = "WIN"

            if exit_price is not None:
                pnl = paper_trade_exit(entry, exit_price, qty, signal)
                close_trade_in_journal(idx, exit_price, result=result, t1_hit=t1_hit, t2_hit=t2_hit, t3_hit=t3_hit)
                closed_count += 1

        return closed_count
    except Exception as e:
        log_app_error("auto_paper_trade_exit", e)
        return 0


# =========================================================
# BACKTEST / WALK FORWARD / SHARPE
# =========================================================
def compute_equity_curve_from_trades(trades_df, initial_capital=100000):
    if trades_df.empty or "PnL" not in trades_df.columns:
        return pd.DataFrame(columns=["TradeNo", "Equity", "DrawdownPct", "ReturnPct"]), 0.0

    pnl_series = pd.to_numeric(trades_df["PnL"], errors="coerce").fillna(0)
    equity = initial_capital + pnl_series.cumsum()
    peak = equity.cummax()
    drawdown_pct = ((peak - equity) / peak.replace(0, np.nan)) * 100

    prev_equity = equity.shift(1).fillna(initial_capital)
    return_pct = np.where(prev_equity != 0, ((equity - prev_equity) / prev_equity) * 100, 0)

    curve = pd.DataFrame({
        "TradeNo": range(1, len(pnl_series) + 1),
        "Equity": equity.values,
        "DrawdownPct": pd.Series(drawdown_pct).fillna(0).values,
        "ReturnPct": pd.Series(return_pct).fillna(0).values
    })
    max_dd = float(curve["DrawdownPct"].max()) if not curve.empty else 0.0
    return curve, round(max_dd, 2)


def compute_streaks(results):
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0

    for r in results:
        if r == "WIN":
            current_win += 1
            current_loss = 0
        elif r == "LOSS":
            current_loss += 1
            current_win = 0
        else:
            current_win = 0
            current_loss = 0
        max_win_streak = max(max_win_streak, current_win)
        max_loss_streak = max(max_loss_streak, current_loss)

    return max_win_streak, max_loss_streak


def compute_trade_statistics(trades_df, initial_capital=100000):
    if trades_df.empty:
        return {
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "winning_streak": 0,
            "losing_streak": 0,
            "t1_hit_rate": 0.0,
            "t2_hit_rate": 0.0,
            "t3_hit_rate": 0.0,
            "profitability_score": 0,
            "sharpe_ratio": 0.0
        }

    pnl = pd.to_numeric(trades_df["PnL"], errors="coerce").fillna(0)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    win_rate = (len(wins) / len(pnl)) * 100 if len(pnl) else 0.0
    expectancy = ((win_rate / 100) * avg_win) - (((100 - win_rate) / 100) * avg_loss)

    results = trades_df["Result"].astype(str).tolist() if "Result" in trades_df.columns else []
    winning_streak, losing_streak = compute_streaks(results)

    t1_hit_rate = float((trades_df.get("T1_Hit", pd.Series(dtype=bool)) == True).mean() * 100) if "T1_Hit" in trades_df.columns else 0.0
    t2_hit_rate = float((trades_df.get("T2_Hit", pd.Series(dtype=bool)) == True).mean() * 100) if "T2_Hit" in trades_df.columns else 0.0
    t3_hit_rate = float((trades_df.get("T3_Hit", pd.Series(dtype=bool)) == True).mean() * 100) if "T3_Hit" in trades_df.columns else 0.0

    equity_curve_df, _ = compute_equity_curve_from_trades(trades_df, initial_capital=initial_capital)
    sharpe_ratio = 0.0
    if not equity_curve_df.empty and "ReturnPct" in equity_curve_df.columns:
        returns = pd.to_numeric(equity_curve_df["ReturnPct"], errors="coerce").fillna(0)
        if returns.std() and returns.std() > 0:
            sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(len(returns)))

    profitability_score = 50
    if profit_factor >= 2:
        profitability_score += 20
    elif profit_factor >= 1.5:
        profitability_score += 10
    elif profit_factor < 1.2:
        profitability_score -= 10
    if expectancy > 0:
        profitability_score += 15
    if win_rate >= 55:
        profitability_score += 10
    if avg_win > avg_loss:
        profitability_score += 10

    profitability_score = max(0, min(100, profitability_score))

    return {
        "profit_factor": round(profit_factor, 2) if profit_factor != 999.0 else 999.0,
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "expectancy": round(float(expectancy), 2),
        "winning_streak": int(winning_streak),
        "losing_streak": int(losing_streak),
        "t1_hit_rate": round(t1_hit_rate, 2),
        "t2_hit_rate": round(t2_hit_rate, 2),
        "t3_hit_rate": round(t3_hit_rate, 2),
        "profitability_score": int(profitability_score),
        "sharpe_ratio": round(sharpe_ratio, 2)
    }


def simple_backtest(df, symbol="", timeframe_label="", option_metrics=None, news_info=None):
    trades = []
    df = df.copy()
    required_cols = [
        "Open", "High", "Low", "Close", "EMA20", "EMA50", "EMA200", "VWAP", "RSI",
        "MACD", "SignalLine", "ATR", "ADX", "SuperTrendDirection", "Volume_Spike"
    ]
    for col in required_cols:
        if col not in df.columns:
            return 0, 0, 0, 0.0, pd.DataFrame()

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "EMA20", "EMA50", "EMA200", "RSI", "MACD", "SignalLine", "ATR", "ADX"]).copy()

    if len(df) < 50:
        return 0, 0, 0, 0.0, pd.DataFrame()

    for i in range(30, len(df) - 10):
        mini_df = df.iloc[:i + 1].copy()
        latest = mini_df.iloc[-1]
        ts = mini_df.index[-1]
        htf_bias = "bullish" if latest["EMA20"] > latest["EMA50"] else "bearish"

        analysis = analyze_signal(
            mini_df,
            htf_bias=htf_bias,
            timeframe_label=timeframe_label if timeframe_label else "1d",
            backtest_win_rate=50,
            mtf_bull_count=4 if htf_bias == "bullish" else 0,
            mtf_bear_count=4 if htf_bias == "bearish" else 0,
            option_metrics=option_metrics,
            news_info=news_info
        )

        if analysis["signal_text"] not in ["STRONG BUY", "STRONG SELL"]:
            continue

        entry = float(latest["Close"])
        sl = analysis["stop_loss"]
        t1 = analysis["target1"]
        t2 = analysis["target2"]
        t3 = analysis["target3"]

        if sl is None or t1 is None:
            continue

        future = df.iloc[i + 1:i + 11]
        if future.empty:
            continue

        signal = "BUY" if analysis["signal_type"] == "buy" else "SELL"
        exit_price = None
        result = None
        t1_hit = False
        t2_hit = False
        t3_hit = False

        for _, row in future.iterrows():
            if signal == "BUY":
                if row["Low"] <= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                if row["High"] >= t1:
                    t1_hit = True
                if row["High"] >= t2:
                    t2_hit = True
                if row["High"] >= t3:
                    t3_hit = True
                if row["High"] >= t3:
                    exit_price = t3
                    result = "WIN"
                    break
                elif row["High"] >= t2:
                    exit_price = t2
                    result = "WIN"
                    break
                elif row["High"] >= t1:
                    exit_price = t1
                    result = "WIN"
                    break
            else:
                if row["High"] >= sl:
                    exit_price = sl
                    result = "LOSS"
                    break
                if row["Low"] <= t1:
                    t1_hit = True
                if row["Low"] <= t2:
                    t2_hit = True
                if row["Low"] <= t3:
                    t3_hit = True
                if row["Low"] <= t3:
                    exit_price = t3
                    result = "WIN"
                    break
                elif row["Low"] <= t2:
                    exit_price = t2
                    result = "WIN"
                    break
                elif row["Low"] <= t1:
                    exit_price = t1
                    result = "WIN"
                    break

        if exit_price is None:
            exit_price = float(future.iloc[-1]["Close"])
            if signal == "BUY":
                result = "WIN" if exit_price > entry else "LOSS"
            else:
                result = "WIN" if exit_price < entry else "LOSS"

        pnl = round((exit_price - entry), 2) if signal == "BUY" else round((entry - exit_price), 2)

        trades.append({
            "Date": pd.to_datetime(ts).strftime("%d-%m-%Y"),
            "Time": pd.to_datetime(ts).strftime("%H:%M:%S"),
            "Symbol": symbol,
            "Timeframe": timeframe_label,
            "Signal": signal,
            "Entry": round(entry, 2),
            "SL": round(sl, 2),
            "T1": round(t1, 2),
            "T2": round(t2, 2) if t2 is not None else np.nan,
            "T3": round(t3, 2) if t3 is not None else np.nan,
            "Exit": round(exit_price, 2),
            "Result": result,
            "PnL": pnl,
            "T1_Hit": t1_hit,
            "T2_Hit": t2_hit,
            "T3_Hit": t3_hit
        })

    trades_df = pd.DataFrame(trades)
    total = len(trades_df)
    wins = len(trades_df[trades_df["Result"] == "WIN"]) if total else 0
    losses = len(trades_df[trades_df["Result"] == "LOSS"]) if total else 0
    win_rate = (wins / total * 100) if total else 0.0
    return wins, losses, total, round(win_rate, 2), trades_df


def walk_forward_validation(df, symbol="", timeframe_label="", option_metrics=None, news_info=None):
    try:
        if df.empty or len(df) < 150:
            return {"folds": 0, "avg_win_rate": 0.0, "avg_profit_factor": 0.0, "avg_expectancy": 0.0, "table": pd.DataFrame()}

        fold_results = []
        total_len = len(df)
        train_size = int(total_len * 0.6)
        test_size = int(total_len * 0.2)

        start = 0
        fold_no = 1
        while (start + train_size + test_size) <= total_len:
            train_df = df.iloc[start:start + train_size].copy()
            test_df = df.iloc[start + train_size:start + train_size + test_size].copy()

            _, _, _, _, _ = simple_backtest(
                train_df,
                symbol=symbol,
                timeframe_label=timeframe_label,
                option_metrics=option_metrics,
                news_info=news_info
            )

            wins, losses, total, win_rate, trades_df = simple_backtest(
                test_df,
                symbol=symbol,
                timeframe_label=timeframe_label,
                option_metrics=option_metrics,
                news_info=news_info
            )

            stats = compute_trade_statistics(trades_df)
            fold_results.append({
                "Fold": fold_no,
                "Trades": total,
                "Wins": wins,
                "Losses": losses,
                "WinRate": win_rate,
                "ProfitFactor": stats["profit_factor"],
                "Expectancy": stats["expectancy"]
            })

            fold_no += 1
            start += test_size

        result_df = pd.DataFrame(fold_results)
        if result_df.empty:
            return {"folds": 0, "avg_win_rate": 0.0, "avg_profit_factor": 0.0, "avg_expectancy": 0.0, "table": pd.DataFrame()}

        return {
            "folds": len(result_df),
            "avg_win_rate": round(result_df["WinRate"].mean(), 2),
            "avg_profit_factor": round(result_df["ProfitFactor"].replace(999.0, 5.0).mean(), 2),
            "avg_expectancy": round(result_df["Expectancy"].mean(), 2),
            "table": result_df
        }
    except Exception as e:
        log_app_error("walk_forward_validation", e)
        return {"folds": 0, "avg_win_rate": 0.0, "avg_profit_factor": 0.0, "avg_expectancy": 0.0, "table": pd.DataFrame()}


# =========================================================
# REPORTS / ANALYTICS
# =========================================================
def update_daily_summary():
    init_daily_summary()
    journal = load_journal()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return

    closed["DateOnly"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce").dt.date
    today = datetime.now().date()
    day_df = closed[closed["DateOnly"] == today]
    if day_df.empty:
        return

    total = len(day_df)
    wins = (pd.to_numeric(day_df["PnL"], errors="coerce").fillna(0) > 0).sum()
    losses = total - wins
    pnl = pd.to_numeric(day_df["PnL"], errors="coerce").fillna(0).sum()
    win_rate = (wins / total * 100) if total > 0 else 0

    summary = safe_read_csv(DAILY_SUMMARY_FILE, pd.DataFrame(columns=["Date", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"]))
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
    safe_write_csv(summary, DAILY_SUMMARY_FILE)


def update_weekly_summary():
    init_weekly_summary()
    journal = load_journal()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return

    closed["DT"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    current_week = datetime.now().strftime("%Y-W%U")
    start_week = datetime.now() - timedelta(days=datetime.now().weekday())
    start_week = datetime(start_week.year, start_week.month, start_week.day)
    week_df = closed[closed["DT"] >= start_week]
    if week_df.empty:
        return

    total = len(week_df)
    wins = (pd.to_numeric(week_df["PnL"], errors="coerce").fillna(0) > 0).sum()
    losses = total - wins
    pnl = pd.to_numeric(week_df["PnL"], errors="coerce").fillna(0).sum()
    win_rate = (wins / total * 100) if total else 0

    summary = safe_read_csv(WEEKLY_SUMMARY_FILE, pd.DataFrame(columns=["WeekKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"]))
    summary = summary[summary["WeekKey"] != current_week]
    new_row = {
        "WeekKey": current_week,
        "TotalTrades": total,
        "Wins": wins,
        "Losses": losses,
        "RealizedPnL": round(pnl, 2),
        "WinRate": round(win_rate, 2)
    }
    summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
    safe_write_csv(summary, WEEKLY_SUMMARY_FILE)


def update_monthly_summary():
    init_monthly_summary()
    journal = load_journal()
    closed = journal[journal["Status"] == "CLOSED"].copy()
    if closed.empty:
        return

    closed["DT"] = pd.to_datetime(closed["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    current_month = datetime.now().strftime("%Y-%m")
    start_month = datetime(datetime.now().year, datetime.now().month, 1)
    month_df = closed[closed["DT"] >= start_month]
    if month_df.empty:
        return

    total = len(month_df)
    wins = (pd.to_numeric(month_df["PnL"], errors="coerce").fillna(0) > 0).sum()
    losses = total - wins
    pnl = pd.to_numeric(month_df["PnL"], errors="coerce").fillna(0).sum()
    win_rate = (wins / total * 100) if total else 0

    summary = safe_read_csv(MONTHLY_SUMMARY_FILE, pd.DataFrame(columns=["MonthKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"]))
    summary = summary[summary["MonthKey"] != current_month]
    new_row = {
        "MonthKey": current_month,
        "TotalTrades": total,
        "Wins": wins,
        "Losses": losses,
        "RealizedPnL": round(pnl, 2),
        "WinRate": round(win_rate, 2)
    }
    summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
    safe_write_csv(summary, MONTHLY_SUMMARY_FILE)


def symbol_wise_performance(journal_df):
    if journal_df.empty:
        return pd.DataFrame()
    df = journal_df[journal_df["Status"] == "CLOSED"].copy()
    if df.empty:
        return pd.DataFrame()
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0)
    grouped = df.groupby("Symbol").agg(
        Trades=("Symbol", "count"),
        TotalPnL=("PnL", "sum"),
        Wins=("PnL", lambda x: (x > 0).sum()),
        Losses=("PnL", lambda x: (x <= 0).sum())
    ).reset_index()
    grouped["WinRate"] = np.where(grouped["Trades"] > 0, grouped["Wins"] / grouped["Trades"] * 100, 0)
    return grouped.sort_values("TotalPnL", ascending=False)


def timeframe_wise_performance(journal_df):
    if journal_df.empty:
        return pd.DataFrame()
    df = journal_df[journal_df["Status"] == "CLOSED"].copy()
    if df.empty:
        return pd.DataFrame()
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0)
    grouped = df.groupby("Timeframe").agg(
        Trades=("Timeframe", "count"),
        TotalPnL=("PnL", "sum"),
        Wins=("PnL", lambda x: (x > 0).sum()),
        Losses=("PnL", lambda x: (x <= 0).sum())
    ).reset_index()
    grouped["WinRate"] = np.where(grouped["Trades"] > 0, grouped["Wins"] / grouped["Trades"] * 100, 0)
    return grouped.sort_values("WinRate", ascending=False)


def best_worst_time_of_day(backtest_trades_df):
    if backtest_trades_df.empty or "Time" not in backtest_trades_df.columns:
        return None, None, pd.DataFrame()
    df = backtest_trades_df.copy()
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0)
    grp = df.groupby("Hour").agg(Trades=("Hour", "count"), AvgPnL=("PnL", "mean"), TotalPnL=("PnL", "sum")).reset_index()
    if grp.empty:
        return None, None, grp
    best = grp.sort_values("AvgPnL", ascending=False).iloc[0]
    worst = grp.sort_values("AvgPnL", ascending=True).iloc[0]
    return best, worst, grp


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        log_app_error("send_telegram_alert", e)
        return False


# =========================================================
# MTF HELPERS
# =========================================================
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
    data = add_indicators(data).bfill().ffill()
    if len(data) < 5:
        return "⚪ N/A"
    latest = data.iloc[-1]
    if latest["EMA20"] > latest["EMA50"]:
        return "🟢 Bullish"
    elif latest["EMA20"] < latest["EMA50"]:
        return "🔴 Bearish"
    return "⚪ Neutral"


# =========================================================
# OPTION STRIKE
# =========================================================
def suggest_option_strike(symbol, spot_price, signal_type):
    if symbol == "NIFTY":
        step = 50
    elif symbol == "BANKNIFTY":
        step = 100
    else:
        step = 10

    atm = round(spot_price / step) * step
    if signal_type == "buy":
        return {"type": "CALL", "atm": atm, "otm": atm + step, "text": f"Suggested CALL strikes: ATM {atm} CE, OTM {atm + step} CE"}
    elif signal_type == "sell":
        return {"type": "PUT", "atm": atm, "otm": atm - step, "text": f"Suggested PUT strikes: ATM {atm} PE, OTM {atm - step} PE"}
    return {"type": "NONE", "atm": None, "otm": None, "text": "No option strike suggestion"}


# =========================================================
# APP INIT
# =========================================================
login_panel()
init_journal()
init_paper_state()
init_daily_summary()
init_weekly_summary()
init_monthly_summary()
init_risk_state()
init_weekly_risk()
init_monthly_risk()
init_signal_audit()
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

st.title("Elite Trading Dashboard Final Ultimate")
st.caption("Polling First | Walk Forward | Auto Exit | News Filter | Dynamic Option Chain")

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
    auto_send_telegram = st.checkbox("Auto-send only High Quality New Signals", value=False)

with st.expander("News Filter API"):
    news_api_key = st.text_input("NewsAPI Key", type="password")
    news_filter_enabled = st.checkbox("Enable News Filter", value=False)

with st.expander("Angel One API Settings"):
    angel_api_key = st.text_input("Angel API Key", type="password")
    angel_client_code = st.text_input("Angel Client Code")
    angel_pin = st.text_input("Angel PIN", type="password")
    angel_totp_secret = st.text_input("Angel TOTP Secret", type="password")
    polling_mode = st.checkbox("Use Angel LTP Polling Mode", value=True)
    ws_enable = st.checkbox("Enable Angel WebSocket (Optional)", value=False)

    if st.button("Test Angel Login"):
        angel, session = angel_login(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, force_refresh=True)
        if angel and session:
            st.success("Angel One Login Successful")
        else:
            st.error("Angel login failed")

    if st.button("Stop WebSocket"):
        stop_angel_websocket()
        sync_ws_state_to_session()
        st.warning("WebSocket stop requested.")

ticker = stocks[selected]

if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, key="elite_ultimate_refresh")

if timeframe == "1m":
    yf_interval, yf_period = "1m", "7d"
elif timeframe == "3m":
    yf_interval, yf_period = "1m", "7d"
elif timeframe == "5m":
    yf_interval, yf_period = "5m", "30d"
elif timeframe == "10m":
    yf_interval, yf_period = "5m", "30d"
elif timeframe == "15m":
    yf_interval, yf_period = "15m", "30d"
elif timeframe == "30m":
    yf_interval, yf_period = "30m", "60d"
elif timeframe == "1h":
    yf_interval, yf_period = "60m", "90d"
else:
    yf_interval, yf_period = "1d", "1y"

data = pd.DataFrame()

if "last_good_candles" not in st.session_state:
    st.session_state["last_good_candles"] = None

if data_source == "Angel":
    if not (angel_api_key and angel_client_code and angel_pin and angel_totp_secret):
        st.warning("Please fill Angel credentials to use Angel data source.")
        st.stop()

    angel_interval = map_angel_interval(timeframe)
    candles = get_angel_candles(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected, angel_interval)
    data = candles_to_df(candles)

    if data.empty:
        last_good = st.session_state.get("last_good_candles")
        data = candles_to_df(last_good)
        if not data.empty:
            st.warning("Using last successful Angel candle data.")
    else:
        st.session_state["last_good_candles"] = candles

    if ws_enable and not USE_POLLING_AS_PRIMARY:
        start_angel_websocket(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected)
        sync_ws_state_to_session()
else:
    data = fetch_data(ticker, yf_interval, yf_period)

if data.empty:
    st.error("No data found")
    st.stop()

if timeframe == "3m":
    data = resample_ohlcv(data, "3min")
elif timeframe == "10m":
    data = resample_ohlcv(data, "10min")

ok, msg = data_integrity_check(data, ["Open", "High", "Low", "Close", "Volume"], min_rows=DATA_INTEGRITY_MIN_ROWS)
if not ok:
    st.error(f"Data integrity failed: {msg}")
    st.stop()

data = add_indicators(data).bfill().ffill()
backtest_data = data.copy()

if len(data) < 30:
    st.warning(f"Only {len(data)} rows available after indicator calculation.")
    st.stop()

htf_interval, _ = get_higher_timeframe(timeframe)
angel_creds = (angel_api_key, angel_client_code, angel_pin, angel_totp_secret)
htf_data = get_data_for_timeframe(data_source, selected, ticker, htf_interval, angel_creds)

if not htf_data.empty:
    htf_data = add_indicators(htf_data).bfill().ffill()
    if len(htf_data) > 20:
        htf_latest = htf_data.iloc[-1]
        htf_bias = "bullish" if htf_latest["EMA20"] > htf_latest["EMA50"] else "bearish"
    else:
        htf_bias = None
else:
    htf_bias = None

mtf_table = pd.DataFrame({
    "Timeframe": ["1m", "3m", "5m", "15m"],
    "Trend": [
        get_mtf_signal(data_source, selected, ticker, "1m", angel_creds),
        get_mtf_signal(data_source, selected, ticker, "3m", angel_creds),
        get_mtf_signal(data_source, selected, ticker, "5m", angel_creds),
        get_mtf_signal(data_source, selected, ticker, "15m", angel_creds)
    ]
})

bull_count = sum("Bullish" in sig for sig in mtf_table["Trend"])
bear_count = sum("Bearish" in sig for sig in mtf_table["Trend"])

dynamic_oc_symbol = get_dynamic_option_chain_symbol(selected)
option_metrics = parse_option_chain_metrics(dynamic_oc_symbol)

news_info = {"allow_trade": True, "message": "News filter disabled", "events": []}
if news_filter_enabled:
    news_info = get_news_events(news_api_key, selected)

wins, losses, total_trades, backtest_win_rate, backtest_trades_df = simple_backtest(
    backtest_data,
    symbol=selected,
    timeframe_label=timeframe,
    option_metrics=option_metrics,
    news_info=news_info
)

analysis = analyze_signal(
    data,
    htf_bias=htf_bias,
    timeframe_label=timeframe,
    backtest_win_rate=backtest_win_rate,
    mtf_bull_count=bull_count,
    mtf_bear_count=bear_count,
    option_metrics=option_metrics,
    news_info=news_info
)

walk_forward_result = walk_forward_validation(
    backtest_data,
    symbol=selected,
    timeframe_label=timeframe,
    option_metrics=option_metrics,
    news_info=news_info
)

stats = compute_trade_statistics(backtest_trades_df, initial_capital=DEFAULT_INITIAL_BALANCE)
equity_curve_df, max_drawdown = compute_equity_curve_from_trades(backtest_trades_df, initial_capital=DEFAULT_INITIAL_BALANCE)

health_score, health_label = strategy_health_monitor(
    analysis["validation_score"],
    backtest_win_rate,
    stats["profit_factor"] if stats["profit_factor"] != 999.0 else 5.0,
    max_drawdown
)

best_hour, worst_hour, hour_perf_df = best_worst_time_of_day(backtest_trades_df)

if bull_count >= 3:
    mtf_grade = "Strong Bullish Alignment"
elif bear_count >= 3:
    mtf_grade = "Strong Bearish Alignment"
else:
    mtf_grade = "Mixed Timeframes"

mtf_confirmation = "Neutral"
if analysis["signal_type"] == "buy" and htf_bias == "bullish":
    mtf_confirmation = "Bullish Alignment"
elif analysis["signal_type"] == "sell" and htf_bias == "bearish":
    mtf_confirmation = "Bearish Alignment"
elif analysis["signal_type"] in ["buy", "sell"]:
    mtf_confirmation = "Timeframe Conflict"

quality_score = (analysis["win_chance"] + analysis["confidence"] + analysis["validation_score"]) / 3
if quality_score >= 85:
    signal_quality = "Excellent"
elif quality_score >= 70:
    signal_quality = "Strong"
elif quality_score >= 55:
    signal_quality = "Moderate"
else:
    signal_quality = "Weak"

live_price = None
if data_source == "Angel" and polling_mode:
    live_price = get_angel_ltp_polling(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected)
elif data_source == "Angel" and ws_enable:
    sync_ws_state_to_session()
    live_price = st.session_state.get("live_price")

current_display_price = live_price if live_price else analysis["close_price"]

closed_auto = auto_paper_trade_exit(current_display_price)

option_suggestion = suggest_option_strike(selected, analysis["close_price"], analysis["signal_type"])
expiry_seconds = get_signal_expiry_remaining(selected, timeframe, analysis["signal_text"])

alert_text = (
    f"{selected} | {timeframe} | {analysis['signal_text']} | "
    f"Grade: {analysis['strategy_grade']} | Validation: {analysis['validation_score']} | "
    f"Quality: {analysis['trade_quality_score']} | Confidence: {analysis['confidence']}% | "
    f"Win Chance: {analysis['win_chance']}% | Price: {round(current_display_price, 2)} | "
    f"RR: {analysis['rr_ratio']} | OC: {dynamic_oc_symbol} | {option_suggestion['text']}"
)

current_signal_key = f"{selected}|{timeframe}|{analysis['signal_text']}|{round(analysis['close_price'], 2)}"
last_signal_key = ""
if os.path.exists(SIGNAL_STATE_FILE):
    with open(SIGNAL_STATE_FILE, "r") as f:
        last_signal_key = f.read().strip()

if analysis["signal_type"] in ["buy", "sell"] and not is_duplicate_signal(selected, timeframe, analysis):
    add_signal_audit(selected, timeframe, analysis)

if telegram_enabled and telegram_bot_token and telegram_chat_id:
    if st.button("Send Telegram Alert Now"):
        sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_text)
        if sent:
            st.success("Telegram alert sent successfully.")
        else:
            st.error("Failed to send Telegram alert.")

    if (
        auto_send_telegram and
        current_signal_key != last_signal_key and
        analysis["signal_type"] != "neutral" and
        analysis["trade_quality_score"] >= HIGH_QUALITY_SIGNAL_MIN_SCORE and
        analysis["validation_score"] >= MIN_VALIDATION_SCORE_FOR_ALERT and
        not is_duplicate_signal(selected, timeframe, analysis)
    ):
        sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_text)
        if sent:
            with open(SIGNAL_STATE_FILE, "w") as f:
                f.write(current_signal_key)

paper_state = load_paper_state()
paper_balance = float(paper_state.loc[0, "Balance"])
risk_ok, risk_msg = risk_check_before_trade()

st.subheader("Live Signal Analytics")

if data_source == "Angel":
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Polling Mode", "Primary" if polling_mode else "Off")
    with p2:
        st.metric("WebSocket Mode", "Optional" if ws_enable else "Off")
    with p3:
        st.metric("Live Price", round(live_price, 2) if live_price else "-")
    with p4:
        st.metric("Auto Closed Trades", closed_auto)

if analysis["signal_type"] == "buy":
    st.success(alert_text)
elif analysis["signal_type"] == "sell":
    st.error(alert_text)
else:
    st.warning(alert_text)

if expiry_seconds is not None:
    st.info(f"Signal Expiry Remaining: {expiry_seconds} sec")

st.subheader("Strategy Validation Dashboard")
sv1, sv2, sv3, sv4, sv5, sv6 = st.columns(6)
with sv1:
    st.metric("Validation Score", analysis["validation_score"])
with sv2:
    st.metric("Strategy Grade", analysis["strategy_grade"])
with sv3:
    st.metric("Trade Quality", analysis["trade_quality_score"])
with sv4:
    st.metric("Health Score", health_score)
with sv5:
    st.metric("Health Label", health_label)
with sv6:
    st.metric("Market Regime", analysis["market_regime"])

st.subheader("Multi Timeframe Alignment")
st.dataframe(mtf_table, use_container_width=True)
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Bullish TF", bull_count)
with m2:
    st.metric("Bearish TF", bear_count)
with m3:
    st.metric("MTF Status", mtf_confirmation)

if "Bullish" in mtf_grade:
    st.success(mtf_grade)
elif "Bearish" in mtf_grade:
    st.error(mtf_grade)
else:
    st.warning(mtf_grade)

top1, top2, top3, top4 = st.columns(4)
with top1:
    st.metric("Current Price", round(current_display_price, 2))
with top2:
    st.metric("Signal", analysis["signal_text"])
with top3:
    st.metric("Win Chance %", analysis["win_chance"])
with top4:
    st.metric("Confidence %", analysis["confidence"])

a1, a2, a3, a4, a5, a6 = st.columns(6)
with a1:
    st.metric("ADX", round(analysis["adx"], 2))
with a2:
    st.metric("EMA200 Confirm", "Yes" if ((analysis["signal_type"] == "buy" and analysis["close_price"] > analysis["ema200"]) or (analysis["signal_type"] == "sell" and analysis["close_price"] < analysis["ema200"])) else "No")
with a3:
    st.metric("SuperTrend", "Bullish" if analysis["supertrend_direction"] == 1 else "Bearish" if analysis["supertrend_direction"] == -1 else "Neutral")
with a4:
    st.metric("RR Ratio", analysis["rr_ratio"] if analysis["rr_ratio"] else "-")
with a5:
    st.metric("Dynamic OC", dynamic_oc_symbol)
with a6:
    st.metric("News Filter", "Blocked" if not news_info["allow_trade"] else "Clear")

b1, b2, b3, b4, b5, b6 = st.columns(6)
with b1:
    st.metric("VWAP", round(float(analysis["vwap"]), 2) if analysis["vwap"] is not None else "N/A")
with b2:
    st.metric("EMA20", round(analysis["ema20"], 2))
with b3:
    st.metric("EMA50", round(analysis["ema50"], 2))
with b4:
    st.metric("EMA200", round(analysis["ema200"], 2))
with b5:
    st.metric("RSI", round(analysis["rsi"], 2))
with b6:
    st.metric("ATR", round(analysis["atr"], 2))

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Support", round(analysis["support"], 2))
with c2:
    st.metric("Resistance", round(analysis["resistance"], 2))
with c3:
    st.metric("Gap %", analysis["gap_pct"])
with c4:
    st.metric("OI Bias", analysis["oi_bias"])
with c5:
    st.metric("PCR", analysis["pcr"] if analysis["pcr"] is not None else "N/A")
with c6:
    st.metric("No Trade Zone", "Yes" if analysis["no_trade_zone"] else "No")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Trending Market", "Yes" if analysis["trending_market"] else "No")
with r2:
    st.metric("Sideways Market", "Yes" if analysis["sideways_market"] else "No")
with r3:
    st.metric("High Volatility", "Yes" if analysis["high_volatility_market"] else "No")
with r4:
    st.metric("Signal Quality", signal_quality)

st.subheader("Risk Management")
rk1, rk2, rk3, rk4, rk5 = st.columns(5)
risk_state = load_risk_state()
weekly_risk = load_weekly_risk()
monthly_risk = load_monthly_risk()
with rk1:
    st.metric("Paper Balance", round(paper_balance, 2))
with rk2:
    st.metric("Daily Lock", "Yes" if bool(risk_state.loc[0, "DailyLock"]) else "No")
with rk3:
    st.metric("Weekly Lock", "Yes" if bool(weekly_risk.loc[0, "WeeklyLock"]) else "No")
with rk4:
    st.metric("Monthly Lock", "Yes" if bool(monthly_risk.loc[0, "MonthlyLock"]) else "No")
with rk5:
    st.metric("Risk Status", risk_msg)

if analysis["signal_type"] in ["buy", "sell"] and analysis["stop_loss"] is not None:
    qty = calculate_position_size(analysis["close_price"], analysis["stop_loss"], paper_balance)
    t1c, t2c, t3c, t4c, t5c, t6c = st.columns(6)
    with t1c:
        st.metric("Stop Loss", round(analysis["stop_loss"], 2))
    with t2c:
        st.metric("Partial Exit", round(analysis["partial_exit"], 2) if analysis["partial_exit"] else "-")
    with t3c:
        st.metric("Target 1", round(analysis["target1"], 2) if analysis["target1"] else "-")
    with t4c:
        st.metric("Target 2", round(analysis["target2"], 2) if analysis["target2"] else "-")
    with t5c:
        st.metric("Target 3", round(analysis["target3"], 2) if analysis["target3"] else "-")
    with t6c:
        st.metric("Auto Position Size", qty)

    if st.button("Paper Trade Entry"):
        if not risk_ok:
            st.error(risk_msg)
        else:
            ok_entry = paper_trade_entry(analysis["close_price"], qty)
            if ok_entry:
                add_trade_to_journal(
                    selected, timeframe, analysis["signal_text"], analysis["close_price"], qty,
                    analysis["stop_loss"], analysis["target1"], analysis["target2"], analysis["target3"],
                    analysis["trailing_stop"], analysis, notes="Paper trade entry from dashboard"
                )
                st.success("Paper trade entered")
            else:
                st.error("Insufficient paper balance")

with st.expander("Why this signal?"):
    for reason in analysis["reasons"]:
        st.write("-", reason)

if news_filter_enabled:
    st.subheader("News Filter Events")
    if news_info["events"]:
        st.dataframe(pd.DataFrame(news_info["events"]), use_container_width=True)
    else:
        st.info(news_info["message"])

st.subheader("Backtest Analytics")
ba1, ba2, ba3, ba4, ba5, ba6 = st.columns(6)
with ba1:
    st.metric("Backtest Wins", wins)
with ba2:
    st.metric("Backtest Losses", losses)
with ba3:
    st.metric("Total Trades", total_trades)
with ba4:
    st.metric("Backtest Win Rate", f"{backtest_win_rate:.1f}%")
with ba5:
    st.metric("Profit Factor", stats["profit_factor"])
with ba6:
    st.metric("Max Drawdown %", max_drawdown)

bb1, bb2, bb3, bb4, bb5, bb6 = st.columns(6)
with bb1:
    st.metric("Winning Streak", stats["winning_streak"])
with bb2:
    st.metric("Losing Streak", stats["losing_streak"])
with bb3:
    st.metric("Average Win", stats["avg_win"])
with bb4:
    st.metric("Average Loss", stats["avg_loss"])
with bb5:
    st.metric("Expectancy", stats["expectancy"])
with bb6:
    st.metric("Sharpe Ratio", stats["sharpe_ratio"])

bc1, bc2, bc3, bc4 = st.columns(4)
with bc1:
    st.metric("T1 Hit Rate %", stats["t1_hit_rate"])
with bc2:
    st.metric("T2 Hit Rate %", stats["t2_hit_rate"])
with bc3:
    st.metric("T3 Hit Rate %", stats["t3_hit_rate"])
with bc4:
    st.metric("Profitability Score", stats["profitability_score"])

st.subheader("Walk Forward Validation")
wf1, wf2, wf3, wf4 = st.columns(4)
with wf1:
    st.metric("Folds", walk_forward_result["folds"])
with wf2:
    st.metric("Avg Win Rate", walk_forward_result["avg_win_rate"])
with wf3:
    st.metric("Avg Profit Factor", walk_forward_result["avg_profit_factor"])
with wf4:
    st.metric("Avg Expectancy", walk_forward_result["avg_expectancy"])

if not walk_forward_result["table"].empty:
    st.dataframe(walk_forward_result["table"], use_container_width=True)
else:
    st.info("Walk forward validation data unavailable.")

st.subheader("Equity Curve")
if not equity_curve_df.empty:
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity_curve_df["TradeNo"], y=equity_curve_df["Equity"], mode="lines", name="Equity"))
    eq_fig.update_layout(height=320, template="plotly_dark", title="Equity Curve")
    st.plotly_chart(eq_fig, use_container_width=True)
else:
    st.info("No equity curve data available.")

st.subheader("Best / Worst Time of Day")
if best_hour is not None and worst_hour is not None:
    bt1, bt2 = st.columns(2)
    with bt1:
        st.success(f"Best Hour: {int(best_hour['Hour'])}:00 | AvgPnL: {round(best_hour['AvgPnL'], 2)}")
    with bt2:
        st.error(f"Worst Hour: {int(worst_hour['Hour'])}:00 | AvgPnL: {round(worst_hour['AvgPnL'], 2)}")
    st.dataframe(hour_perf_df, use_container_width=True)
else:
    st.info("Time of day analysis unavailable.")

st.subheader("Backtest Trade Log")
if not backtest_trades_df.empty:
    st.dataframe(backtest_trades_df, use_container_width=True)
else:
    st.info("No backtest trades found.")

journal = load_journal()
update_daily_summary()
update_weekly_summary()
update_monthly_summary()

st.subheader("Journal Analytics Dashboard")
closed_journal = journal[journal["Status"] == "CLOSED"].copy() if not journal.empty else pd.DataFrame()
if not closed_journal.empty:
    sym_perf = symbol_wise_performance(journal)
    tf_perf = timeframe_wise_performance(journal)

    j1, j2 = st.columns(2)
    with j1:
        st.markdown("**Symbol-wise Performance**")
        st.dataframe(sym_perf, use_container_width=True)
    with j2:
        st.markdown("**Timeframe-wise Performance**")
        st.dataframe(tf_perf, use_container_width=True)

    st.markdown("**Closed Trade Journal**")
    st.dataframe(closed_journal, use_container_width=True)
else:
    st.info("No closed trades in journal yet.")

st.subheader("Daily / Weekly / Monthly Reports")
rep1, rep2, rep3 = st.columns(3)
with rep1:
    st.markdown("**Daily Summary**")
    st.dataframe(safe_read_csv(DAILY_SUMMARY_FILE), use_container_width=True)
with rep2:
    st.markdown("**Weekly Summary**")
    st.dataframe(safe_read_csv(WEEKLY_SUMMARY_FILE), use_container_width=True)
with rep3:
    st.markdown("**Monthly Summary**")
    st.dataframe(safe_read_csv(MONTHLY_SUMMARY_FILE), use_container_width=True)

st.subheader("Option Suggestion")
if option_suggestion["type"] == "CALL":
    st.success(option_suggestion["text"])
elif option_suggestion["type"] == "PUT":
    st.error(option_suggestion["text"])
else:
    st.warning(option_suggestion["text"])

if debug_mode:
    st.subheader("Debug Panel")
   
