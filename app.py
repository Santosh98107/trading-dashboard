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
try:
    import pyotp
except Exception:
    pyotp = None
try:
    from SmartApi import SmartConnect
except Exception:
    SmartConnect = None
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

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
DISABLE_BUY_BIAS_TRADING = False
DISABLE_SELL_BIAS_TRADING = False
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
            "TradeQualityScore", "ValidationScore", "StrategyGrade",
            "RemainingQuantity", "RealizedPnL", "AvgExitPrice", "T1_Quantity", "T2_Quantity", "T3_Quantity"
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
        if SmartConnect is None or pyotp is None:
            log_app_error("angel_login", "smartapi-python and pyotp are required for Angel One")
            return None, None
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


@st.cache_data(ttl=86400, show_spinner=False)
def load_angel_instrument_master():
    """Download Angel One's public instrument master and normalize token fields."""
    urls = [
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
        "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            payload = r.json()
            df = pd.DataFrame(payload)
            if not df.empty:
                return df
        except Exception as exc:
            log_app_error("load_angel_instrument_master", f"{url}: {exc}")
    return pd.DataFrame()


def resolve_angel_instrument(symbol_name, exchange="NSE"):
    """Resolve exchange token/trading symbol from the current master; static map is fallback."""
    if str(symbol_name).upper() == "SENSEX":
        exchange = "BSE"
    aliases = {
        "NIFTY": ["NIFTY", "NIFTY 50"], "BANKNIFTY": ["BANKNIFTY", "NIFTY BANK"],
        "SENSEX": ["SENSEX"], "HDFCBANK": ["HDFCBANK"], "ICICIBANK": ["ICICIBANK"],
        "RELIANCE": ["RELIANCE"], "SBIN": ["SBIN"], "INFY": ["INFY"], "TCS": ["TCS"],
        "ITC": ["ITC"], "AXISBANK": ["AXISBANK"], "KOTAKBANK": ["KOTAKBANK"],
        "LT": ["LT"], "TATAMOTORS": ["TATAMOTORS"], "MARUTI": ["MARUTI"],
        "HCLTECH": ["HCLTECH"], "WIPRO": ["WIPRO"], "TECHM": ["TECHM"],
        "M&M": ["M&M"], "BAJAJ-AUTO": ["BAJAJ-AUTO"], "HEROMOTOCO": ["HEROMOTOCO"],
        "BHARTIARTL": ["BHARTIARTL"], "SUNPHARMA": ["SUNPHARMA"]
    }
    candidates = aliases.get(str(symbol_name).upper(), [str(symbol_name).upper()])
    master = load_angel_instrument_master()
    if not master.empty:
        cols = {str(c).lower(): c for c in master.columns}
        token_col = cols.get("token") or cols.get("symboltoken")
        symbol_col = cols.get("symbol")
        exch_col = cols.get("exch_seg") or cols.get("exchange")
        name_col = cols.get("name")
        if token_col and symbol_col:
            work = master.copy()
            if exch_col:
                work = work[work[exch_col].astype(str).str.upper().eq(exchange.upper())]
            sym = work[symbol_col].astype(str).str.upper()
            for candidate in candidates:
                mask = sym.eq(candidate + "-EQ") | sym.eq(candidate)
                if name_col and candidate in ("NIFTY", "BANKNIFTY", "SENSEX"):
                    mask = mask | work[name_col].astype(str).str.upper().eq(candidate)
                found = work[mask]
                if not found.empty:
                    row = found.iloc[0]
                    return {"token": str(row[token_col]), "tradingsymbol": str(row[symbol_col]), "exchange": exchange}
    token = ANGEL_TOKENS.get(symbol_name)
    if token:
        ts = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "SENSEX": "SENSEX"}.get(symbol_name, symbol_name + "-EQ")
        exch = "BSE" if symbol_name == "SENSEX" else exchange
        return {"token": str(token), "tradingsymbol": ts, "exchange": exch}
    return None


def get_angel_candles(api_key, client_code, pin, totp_secret, symbol_name, interval="FIVE_MINUTE"):
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        if angel is None or not session or not session.get("data"):
            return None

        instrument = resolve_angel_instrument(symbol_name)
        if not instrument:
            return {"status": False, "message": f"Instrument token not found in Angel master for {symbol_name}"}
        symbol_token = instrument["token"]

        ist = timezone(timedelta(hours=5, minutes=30))
        to_date = datetime.now(ist)
        from_date = to_date - timedelta(days=10)

        params = {
            "exchange": instrument["exchange"],
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


def get_angel_ltp_quote(api_key, client_code, pin, totp_secret, symbol_name):
    """Fetch the latest quote through Angel One's REST LTP endpoint (polling primary)."""
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        instrument = resolve_angel_instrument(symbol_name)
        if angel is None or not session or not instrument:
            return None
        result = angel.ltpData(instrument["exchange"], instrument["tradingsymbol"], instrument["token"])
        if isinstance(result, dict) and result.get("status") and isinstance(result.get("data"), dict):
            value = result["data"].get("ltp")
            return float(value) if value is not None else None
        return None
    except Exception as e:
        log_app_error("get_angel_ltp_quote", e)
        return None


def get_angel_ltp_polling(api_key, client_code, pin, totp_secret, symbol_name):
    """Primary polling: REST LTP quote, then latest 1-minute candle close as fallback."""
    try:
        quote = get_angel_ltp_quote(api_key, client_code, pin, totp_secret, symbol_name)
        if quote is not None and quote > 0:
            return quote
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
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["ADR14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["PrevHigh"] = df["High"].shift(1)
    df["PrevLow"] = df["Low"].shift(1)
    df["PrevClose"] = df["Close"].shift(1)
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20
    df["BB_Width_Pct"] = ((df["BB_Upper"] - df["BB_Lower"]) / df["SMA20"].replace(0, np.nan)) * 100
    df["BB_Squeeze"] = df["BB_Width_Pct"] <= df["BB_Width_Pct"].rolling(50, min_periods=15).quantile(0.20)

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
    prev_body = (df["Close"].shift(1) - df["Open"].shift(1)).abs()
    bullish = df["Close"] > df["Open"]
    bearish = df["Close"] < df["Open"]
    prev_bull = df["Close"].shift(1) > df["Open"].shift(1)
    prev_bear = df["Close"].shift(1) < df["Open"].shift(1)
    df["Hammer"] = (lower_shadow >= candle_body * 2) & (upper_shadow <= candle_body * 0.5) & (candle_body / candle_range <= 0.4)
    df["HangingMan"] = df["Hammer"] & (df["Close"] > df["Close"].rolling(10).mean())
    df["InvertedHammer"] = (upper_shadow >= candle_body * 2) & (lower_shadow <= candle_body * 0.5) & (candle_body / candle_range <= 0.4)
    df["ShootingStar"] = df["InvertedHammer"] & (df["Close"] > df["Close"].rolling(10).mean())
    df["BullishEngulfing"] = bullish & prev_bear & (df["Open"] <= df["Close"].shift(1)) & (df["Close"] >= df["Open"].shift(1))
    df["BearishEngulfing"] = bearish & prev_bull & (df["Open"] >= df["Close"].shift(1)) & (df["Close"] <= df["Open"].shift(1))
    df["Doji"] = candle_body <= candle_range * 0.1
    df["SpinningTop"] = (candle_body / candle_range <= 0.3) & (upper_shadow > candle_body) & (lower_shadow > candle_body)
    df["Marubozu"] = (candle_body / candle_range >= 0.9) & (upper_shadow / candle_range <= 0.05) & (lower_shadow / candle_range <= 0.05)
    df["InsideBar"] = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    df["OutsideBar"] = (df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))
    df["TweezerTop"] = (abs(df["High"] - df["High"].shift(1)) <= candle_range * 0.05) & bearish & prev_bull
    df["TweezerBottom"] = (abs(df["Low"] - df["Low"].shift(1)) <= candle_range * 0.05) & bullish & prev_bear
    df["PiercingLine"] = prev_bear & bullish & (df["Open"] < df["Low"].shift(1)) & (df["Close"] > (df["Open"].shift(1) + df["Close"].shift(1)) / 2) & (df["Close"] < df["Open"].shift(1))
    df["DarkCloudCover"] = prev_bull & bearish & (df["Open"] > df["High"].shift(1)) & (df["Close"] < (df["Open"].shift(1) + df["Close"].shift(1)) / 2) & (df["Close"] > df["Open"].shift(1))
    df["MorningStar"] = prev_bear.shift(1).fillna(False) & (prev_body < candle_range.shift(1) * 0.35) & bullish & (df["Close"] > (df["Open"].shift(2) + df["Close"].shift(2)) / 2)
    df["EveningStar"] = prev_bull.shift(1).fillna(False) & (prev_body < candle_range.shift(1) * 0.35) & bearish & (df["Close"] < (df["Open"].shift(2) + df["Close"].shift(2)) / 2)
    df["ThreeWhiteSoldiers"] = bullish & bullish.shift(1) & bullish.shift(2) & (df["Close"] > df["Close"].shift(1)) & (df["Close"].shift(1) > df["Close"].shift(2))
    df["ThreeBlackCrows"] = bearish & bearish.shift(1) & bearish.shift(2) & (df["Close"] < df["Close"].shift(1)) & (df["Close"].shift(1) < df["Close"].shift(2))
    df["BodyStrength"] = candle_body / candle_range
    # Transparent price-action labels; these describe the bar and do not forecast direction.
    df["BullishRejection"] = (lower_shadow >= candle_body * 1.5) & (df["Close"] > df["Open"])
    df["BearishRejection"] = (upper_shadow >= candle_body * 1.5) & (df["Close"] < df["Open"])
    df["InsideBarBreakout"] = df["InsideBar"].shift(1).fillna(False) & (df["Close"] > df["High"].shift(1))
    df["InsideBarBreakdown"] = df["InsideBar"].shift(1).fillna(False) & (df["Close"] < df["Low"].shift(1))
    df["PriceAction"] = np.select(
        [df["Breakout"], df["Breakdown"], df["BullishEngulfing"], df["BearishEngulfing"],
         df["BullishRejection"], df["BearishRejection"], df["InsideBarBreakout"], df["InsideBarBreakdown"]],
        ["Resistance breakout", "Support breakdown", "Bullish engulfing", "Bearish engulfing",
         "Bullish rejection", "Bearish rejection", "Inside-bar upside break", "Inside-bar downside break"],
        default="Range / no clear trigger"
    )
    pattern_cols = ["Hammer", "HangingMan", "InvertedHammer", "ShootingStar", "BullishEngulfing", "BearishEngulfing", "Doji", "SpinningTop", "Marubozu", "InsideBar", "OutsideBar", "TweezerTop", "TweezerBottom", "PiercingLine", "DarkCloudCover", "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows"]
    def _pattern_label(row):
        found = [name for name in pattern_cols if bool(row.get(name, False))]
        return ", ".join(found) if found else "No major pattern"
    df["CandlestickPattern"] = df.apply(_pattern_label, axis=1)
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

# =========================================================
# GLOBAL / INDIAN MARKET SNAPSHOTS + MAJOR STOCK SCANNER
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_snapshot(symbol_map):
    """Latest daily close and previous-close percentage change."""
    rows = []
    for name, ticker in symbol_map.items():
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty or len(df) < 2:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                continue
            closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(closes) < 2:
                continue
            close, prev_close = float(closes.iloc[-1]), float(closes.iloc[-2])
            pct = ((close - prev_close) / prev_close) * 100 if prev_close else 0.0
            rows.append({"Name": name, "CMP": round(close, 2), "% Change": round(pct, 2),
                         "Trend": "BULLISH" if pct > 0 else "BEARISH" if pct < 0 else "FLAT"})
        except Exception as exc:
            log_app_error("fetch_market_snapshot:" + str(name), exc)
    return pd.DataFrame(rows, columns=["Name", "CMP", "% Change", "Trend"])


def get_intraday_trend_label(df):
    if df is None or df.empty or len(df) < 20:
        return "N/A"
    try:
        work = add_indicators(df.copy()).ffill().bfill()
        latest = work.iloc[-1]
        if latest.get("EMA20", 0) > latest.get("EMA50", 0):
            return "BULLISH"
        if latest.get("EMA20", 0) < latest.get("EMA50", 0):
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "N/A"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_fno_style_scanner(symbol_map):
    """F&O-style scanner; Yahoo prices are spot, not live futures quotes."""
    rows = []
    for sno, (name, ticker) in enumerate(symbol_map.items(), start=1):
        try:
            intraday = fetch_data(ticker, "1m", "7d")
            daily = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, threads=False)
            if isinstance(daily.columns, pd.MultiIndex):
                daily.columns = daily.columns.get_level_values(0)
            closes = pd.to_numeric(daily["Close"], errors="coerce").dropna() if not daily.empty and "Close" in daily else pd.Series(dtype=float)
            cmp_value, change_str, direction = "N/A", "N/A", "N/A"
            if len(closes) >= 2:
                cmp_value = f"{float(closes.iloc[-1]):.2f}"
                pct = ((float(closes.iloc[-1]) / float(closes.iloc[-2])) - 1) * 100 if float(closes.iloc[-2]) else 0.0
                change_str = f"{pct:+.2f}%"
                direction = "BULLISH" if pct > 0 else "BEARISH" if pct < 0 else "NEUTRAL"
            if intraday is None or intraday.empty:
                t1 = t2 = t3 = "N/A"
            else:
                t1 = get_intraday_trend_label(intraday)
                t2 = get_intraday_trend_label(resample_ohlcv(intraday, "2min"))
                t3 = get_intraday_trend_label(resample_ohlcv(intraday, "3min"))
            rows.append({"S.NO": sno, "STOCKS": name, "CMP": cmp_value, "% Change": change_str,
                         "1m": t1, "2m": t2, "3m": t3, "Direction": direction})
        except Exception as exc:
            log_app_error("fetch_fno_style_scanner:" + str(name), exc)
            rows.append({"S.NO": sno, "STOCKS": name, "CMP": "N/A", "% Change": "N/A",
                         "1m": "N/A", "2m": "N/A", "3m": "N/A", "Direction": "N/A"})
    return pd.DataFrame(rows, columns=["S.NO", "STOCKS", "CMP", "% Change", "1m", "2m", "3m", "Direction"])


def summarize_fno_bias(fno_df):
    result = {"overall_bias": "N/A", "bullish_pct": 0, "bearish_pct": 0,
              "strongest_gainer": "N/A", "strongest_loser": "N/A"}
    if fno_df is None or fno_df.empty or "Direction" not in fno_df:
        return result
    valid = fno_df[fno_df["Direction"].isin(["BULLISH", "BEARISH"])].copy()
    if valid.empty:
        return result
    bulls = int((valid["Direction"] == "BULLISH").sum())
    bears = int((valid["Direction"] == "BEARISH").sum())
    valid["ChangeNum"] = pd.to_numeric(valid["% Change"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    gainers = valid.dropna(subset=["ChangeNum"]).sort_values("ChangeNum", ascending=False)
    losers = valid.dropna(subset=["ChangeNum"]).sort_values("ChangeNum", ascending=True)
    result.update({"overall_bias": "BULLISH" if bulls > bears else "BEARISH" if bears > bulls else "MIXED",
                   "bullish_pct": round(bulls / len(valid) * 100), "bearish_pct": round(bears / len(valid) * 100),
                   "strongest_gainer": (str(gainers.iloc[0]["STOCKS"]) + " " + str(gainers.iloc[0]["% Change"])) if not gainers.empty else "N/A",
                   "strongest_loser": (str(losers.iloc[0]["STOCKS"]) + " " + str(losers.iloc[0]["% Change"])) if not losers.empty else "N/A"})
    return result

@st.cache_data(ttl=600, show_spinner=False)
def get_macro_news_events():
    """Public GDELT headline context for India policy, RBI and geopolitical developments."""
    queries = [
        '(India government OR RBI OR finance ministry OR SEBI) (policy OR announcement OR decision)',
        '(India OR South Asia) (geopolitical OR border OR sanctions OR conflict OR trade disruption)'
    ]
    articles = []
    try:
        for query in queries:
            params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 10,
                      "sort": "HybridRel", "timespan": "2d", "sourcelang": "English"}
            response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=12)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("articles", []):
                url = item.get("url", "")
                if url and all(existing.get("url") != url for existing in articles):
                    articles.append({"Date": item.get("seendate", ""), "Headline": item.get("title", ""),
                                     "Source": item.get("domain", ""), "URL": url})
        return {"available": bool(articles), "articles": articles[:15], "source": "GDELT public news index",
                "message": "Headlines are informational; verify material announcements with official releases."}
    except Exception as exc:
        log_app_error("get_macro_news_events", exc)
        return {"available": False, "articles": [], "source": "GDELT", "message": f"News feed unavailable: {exc}"}


@st.cache_data(ttl=300, show_spinner=False)
def get_fii_dii_activity():
    """Fetch NSE FII/DII activity. Returns unavailable status rather than fabricated values."""
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*",
               "Referer": "https://www.nseindia.com/reports/fii-dii"}
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            category = str(row.get("category", row.get("clientType", row.get("type", ""))))
            buy = row.get("buyValue", row.get("buy_value", row.get("buy", None)))
            sell = row.get("sellValue", row.get("sell_value", row.get("sell", None)))
            net = row.get("netValue", row.get("net_value", row.get("net", None)))
            if buy is not None or sell is not None or net is not None:
                out.append({"Category": category, "Buy": buy, "Sell": sell, "Net": net})
        return {"available": bool(out), "rows": out, "source": "NSE India", "as_of": now_str(),
                "message": "Live NSE response" if out else "NSE response format/data unavailable"}
    except Exception as exc:
        log_app_error("get_fii_dii_activity", exc)
        return {"available": False, "rows": [], "source": "NSE India", "as_of": now_str(),
                "message": f"FII/DII feed unavailable: {exc}"}


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
        existing = {}
        try:
            if os.path.exists(OPTION_CHAIN_FILE):
                with open(OPTION_CHAIN_FILE, "r") as fh:
                    existing = json.load(fh)
        except Exception:
            existing = {}
        existing["symbol"] = symbol
        existing["data"] = data
        existing["last_fetch"] = now_str()
        with open(OPTION_CHAIN_FILE, "w") as f:
            json.dump(existing, f)
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
            "put_oi": 0,
            "call_day_change_oi": 0, "put_day_change_oi": 0,
            "call_snapshot_delta_oi": None, "put_snapshot_delta_oi": None,
            "oi_snapshot_time": None
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

        # Snapshot-to-snapshot OI delta is separate from NSE's day change-in-OI field.
        snapshot_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
        current_snapshot = {"call_oi": int(total_ce), "put_oi": int(total_pe),
                            "call_day_change": int(ce_change_oi), "put_day_change": int(pe_change_oi),
                            "timestamp": now_str()}
        previous_snapshot = {}
        try:
            if os.path.exists(OPTION_CHAIN_FILE):
                with open(OPTION_CHAIN_FILE, "r") as fh:
                    cached = json.load(fh)
                previous_snapshot = cached.get("metrics", {}).get(snapshot_key, {})
        except Exception:
            previous_snapshot = {}
        call_oi_delta = int(total_ce) - int(previous_snapshot.get("call_oi", total_ce))
        put_oi_delta = int(total_pe) - int(previous_snapshot.get("put_oi", total_pe))
        try:
            all_cache = {}
            if os.path.exists(OPTION_CHAIN_FILE):
                with open(OPTION_CHAIN_FILE, "r") as fh:
                    all_cache = json.load(fh)
            all_cache.setdefault("metrics", {})[snapshot_key] = current_snapshot
            all_cache["metrics"][snapshot_key]["timestamp"] = now_str()
            with open(OPTION_CHAIN_FILE, "w") as fh:
                json.dump(all_cache, fh)
        except Exception as cache_exc:
            log_app_error("option_oi_delta_cache", cache_exc)
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
            "put_oi": total_pe,
            "call_day_change_oi": ce_change_oi,
            "put_day_change_oi": pe_change_oi,
            "call_snapshot_delta_oi": call_oi_delta,
            "put_snapshot_delta_oi": put_oi_delta,
            "oi_snapshot_time": now_str()
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

    # Volatility compression + range expansion confirmation.
    bb_squeeze = bool(latest.get("BB_Squeeze", False))
    bb_upper = float(latest.get("BB_Upper", close_price)) if pd.notna(latest.get("BB_Upper", np.nan)) else close_price
    bb_lower = float(latest.get("BB_Lower", close_price)) if pd.notna(latest.get("BB_Lower", np.nan)) else close_price
    if bb_squeeze:
        reasons.append("Bollinger Band squeeze: volatility compression")
        if close_price > bb_upper and volume_spike:
            bullish += 2
            reasons.append("Upside Bollinger expansion with volume")
        elif close_price < bb_lower and volume_spike:
            bearish += 2
            reasons.append("Downside Bollinger expansion with volume")

    if bool(latest.get("BullishRejection", False)):
        bullish += 1
        reasons.append("Bullish price-action rejection")
    if bool(latest.get("BearishRejection", False)):
        bearish += 1
        reasons.append("Bearish price-action rejection")

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
    fii_dii_bias = news_info.get("fii_dii_bias", "Neutral")
    if fii_dii_bias == "Bullish":
        bullish += 1
        reasons.append("FII/DII combined net flow positive")
    elif fii_dii_bias == "Bearish":
        bearish += 1
        reasons.append("FII/DII combined net flow negative")
    if news_info.get("macro_event_present", False):
        reasons.append("Recent government/geopolitical headline detected; verify source and context")
    if news_info.get("macro_risk_caution", False):
        reasons.append("Macro/geopolitical risk caution: signal confidence reduced")

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

    # Snapshot-to-snapshot OI change is supplementary; first snapshot is a neutral baseline.
    call_delta = option_metrics.get("call_snapshot_delta_oi")
    put_delta = option_metrics.get("put_snapshot_delta_oi")
    if call_delta is not None and put_delta is not None:
        if int(put_delta) > int(call_delta) and (pcr is None or pcr >= 1.0):
            bullish += 1
            reasons.append("Snapshot OI change supports put-side buildup")
        elif int(call_delta) > int(put_delta) and (pcr is None or pcr <= 1.0):
            bearish += 1
            reasons.append("Snapshot OI change supports call-side buildup")

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
        news_info["allow_trade"],
        not news_info.get("macro_risk_caution", False)
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
        news_info["allow_trade"],
        not news_info.get("macro_risk_caution", False)
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
    if news_info.get("macro_event_present", False):
        confidence = max(40, confidence - 3)
    if news_info.get("macro_risk_caution", False):
        confidence = max(40, confidence - 7)

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


def paper_trade_exit(entry_price, exit_price, quantity, signal, final_exit=True):
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
    if final_exit:
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
        "StrategyGrade": analysis.get("strategy_grade", "D"),
        "RemainingQuantity": quantity, "RealizedPnL": 0.0, "AvgExitPrice": 0.0,
        "T1_Quantity": 0.0, "T2_Quantity": 0.0, "T3_Quantity": 0.0
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


def _apply_paper_partial_exit(journal, idx, row, exit_price, qty_to_close, target_flag=None, final=False):
    """Close part/all of a paper position and persist cumulative realized PnL."""
    entry = float(row["EntryPrice"])
    remaining = float(row.get("RemainingQuantity", row["Quantity"])) if pd.notna(row.get("RemainingQuantity", row["Quantity"])) else float(row["Quantity"])
    qty_to_close = min(float(qty_to_close), remaining)
    if qty_to_close <= 0:
        return False
    signal = str(row["Signal"]).upper()
    direction = 1 if ("BUY" in signal or "CALL" in signal) else -1
    pnl = (float(exit_price) - entry) * qty_to_close * direction
    # Keep the position count unchanged until the final remaining units are closed.
    will_finish = qty_to_close >= remaining
    paper_trade_exit(entry, float(exit_price), qty_to_close, signal, final_exit=will_finish)
    old_realized = float(row.get("RealizedPnL", 0) or 0) if pd.notna(row.get("RealizedPnL", 0)) else 0.0
    old_closed_qty = float(row["Quantity"]) - remaining
    new_closed_qty = old_closed_qty + qty_to_close
    old_avg = float(row.get("AvgExitPrice", 0) or 0) if pd.notna(row.get("AvgExitPrice", 0)) else 0.0
    avg_exit = ((old_avg * old_closed_qty) + (float(exit_price) * qty_to_close)) / new_closed_qty if new_closed_qty else 0.0
    new_remaining = max(0.0, remaining - qty_to_close)
    journal.at[idx, "RemainingQuantity"] = new_remaining
    journal.at[idx, "RealizedPnL"] = round(old_realized + pnl, 2)
    journal.at[idx, "AvgExitPrice"] = avg_exit
    if target_flag:
        journal.at[idx, target_flag] = True
        qty_col = target_flag.replace("_Hit", "_Quantity")
        if qty_col in journal.columns:
            journal.at[idx, qty_col] = float(row.get(qty_col, 0) or 0) + qty_to_close
    if final or new_remaining <= 0:
        journal.at[idx, "ExitPrice"] = avg_exit
        journal.at[idx, "Status"] = "CLOSED"
        journal.at[idx, "Quantity"] = float(row["Quantity"])
        total_pnl = old_realized + pnl
        journal.at[idx, "PnL"] = round(total_pnl, 2)
        journal.at[idx, "PnLPercent"] = round((total_pnl / (entry * float(row["Quantity"]))) * 100, 2) if entry else 0
        journal.at[idx, "Result"] = "WIN" if total_pnl > 0 else "LOSS" if total_pnl < 0 else "BREAKEVEN"
    save_journal(journal)
    return True


def auto_paper_trade_exit(current_price, exit_target_level=3, enabled=True, symbol_filter=None):
    """Paper-only staged exits: 1/3 at T1, 1/3 at T2, remainder at selected final target or SL."""
    try:
        if not enabled or current_price is None:
            return 0
        journal = load_journal()
        if journal.empty:
            return 0
        for col, default in [("RemainingQuantity", np.nan), ("RealizedPnL", 0.0), ("AvgExitPrice", 0.0),
                             ("T1_Quantity", 0.0), ("T2_Quantity", 0.0), ("T3_Quantity", 0.0)]:
            if col not in journal.columns:
                journal[col] = default
        open_idx = journal.index[journal["Status"].astype(str).eq("OPEN")].tolist()
        if symbol_filter:
            open_idx = [i for i in open_idx if str(journal.at[i, "Symbol"]) == str(symbol_filter)]
        closed_count = 0
        for idx in open_idx:
            row = journal.loc[idx].copy()
            total_qty = float(row["Quantity"])
            remaining_val = row.get("RemainingQuantity", np.nan)
            remaining = total_qty if pd.isna(remaining_val) else float(remaining_val)
            if remaining <= 0:
                continue
            signal = str(row["Signal"]).upper()
            is_long = "BUY" in signal or "CALL" in signal
            sl = float(row["StopLoss"]) if pd.notna(row["StopLoss"]) else None
            targets = [float(row[f"Target{i}"]) if pd.notna(row.get(f"Target{i}")) else None for i in (1,2,3)]
            sl_hit = (current_price <= sl) if is_long and sl is not None else ((current_price >= sl) if sl is not None else False)
            if sl_hit:
                if _apply_paper_partial_exit(journal, idx, row, sl, remaining, final=True):
                    closed_count += 1
                continue
            # Targets already marked are not paid twice. Close a third at each milestone.
            for level in range(1, min(int(exit_target_level), 3) + 1):
                flag = f"T{level}_Hit"
                if bool(journal.at[idx, flag]) or targets[level-1] is None:
                    continue
                hit = current_price >= targets[level-1] if is_long else current_price <= targets[level-1]
                if not hit:
                    continue
                current_remaining = float(journal.at[idx, "RemainingQuantity"]) if pd.notna(journal.at[idx, "RemainingQuantity"]) else total_qty
                # Leave at least one unit for the final target where possible.
                if level < int(exit_target_level) and current_remaining > 1:
                    qty_close = max(1, int(total_qty / 3))
                    qty_close = min(qty_close, current_remaining - 1)
                else:
                    qty_close = current_remaining
                fresh_row = journal.loc[idx].copy()
                if _apply_paper_partial_exit(journal, idx, fresh_row, targets[level-1], qty_close, flag, final=(qty_close >= current_remaining)):
                    if str(journal.at[idx, "Status"]) == "CLOSED":
                        closed_count += 1
                if str(journal.at[idx, "Status"]) == "CLOSED":
                    break
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
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS", "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS", "AXISBANK": "AXISBANK.NS", "KOTAKBANK": "KOTAKBANK.NS",
    "TCS": "TCS.NS", "INFY": "INFY.NS", "HCLTECH": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS", "TECHM": "TECHM.NS", "TATAMOTORS": "TATAMOTORS.NS",
    "MARUTI": "MARUTI.NS", "M&M": "M&M.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS", "ITC": "ITC.NS", "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS", "SUNPHARMA": "SUNPHARMA.NS"
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


# Broader market context (Yahoo symbols); these tables show latest daily closes.
global_indices = {
    "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC", "FTSE 100": "^FTSE",
    "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Shanghai": "000001.SS",
    "Brent Crude": "BZ=F", "Gold": "GC=F", "USD/INR": "INR=X"
}
indian_indices = {
    "NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA", "India VIX": "^INDIAVIX"
}
major_stocks = {
    "RELIANCE": "RELIANCE.NS", "HDFC BANK": "HDFCBANK.NS", "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS", "AXIS BANK": "AXISBANK.NS", "KOTAK BANK": "KOTAKBANK.NS",
    "TCS": "TCS.NS", "INFOSYS": "INFY.NS", "HCL TECH": "HCLTECH.NS", "WIPRO": "WIPRO.NS",
    "TECH MAHINDRA": "TECHM.NS", "TATA MOTORS": "TATAMOTORS.NS", "MARUTI": "MARUTI.NS",
    "M&M": "M&M.NS", "BAJAJ AUTO": "BAJAJ-AUTO.NS", "HERO MOTOCORP": "HEROMOTOCO.NS",
    "ITC": "ITC.NS", "L&T": "LT.NS", "BHARTI AIRTEL": "BHARTIARTL.NS", "SUN PHARMA": "SUNPHARMA.NS"
}

st.title("Elite Trading Dashboard Final Ultimate")
st.caption("Polling Primary | WebSocket Secondary | Yahoo / Angel One | NSE Website LTP fallback | Heikin-Ashi | Candlestick Patterns | OI/PCR | FII/DII | Paper Simulation Only — no broker orders")

debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
selected = st.selectbox("Select Stock / Index", list(stocks.keys()))
data_source = st.radio("Primary Data Source", ["Yahoo", "Angel"], horizontal=True, format_func=lambda x: "Yahoo Finance" if x == "Yahoo" else "Angel One")
fallback_source = st.selectbox("Angel One data unavailable hone par fallback", ["Yahoo Finance", "NSE Website (LTP only) + Yahoo candles", "No fallback"], index=0)
timeframe = st.selectbox("Select Time Frame", ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"])
auto_refresh = st.checkbox("Enable Auto Refresh", value=True)
refresh_seconds = st.selectbox("Refresh Every (seconds)", [5, 10, 15, 30, 60], index=2)
auto_exit_enabled = st.checkbox("Enable Paper Auto Exit (SL / Target)", value=st.session_state.get("auto_exit_enabled", True))
auto_exit_target_level = st.selectbox("Auto full-exit target", [1, 2, 3], index=[1, 2, 3].index(int(st.session_state.get("auto_exit_target_level", 3))), format_func=lambda x: f"Target {x}")
st.session_state["auto_exit_enabled"] = auto_exit_enabled
st.session_state["auto_exit_target_level"] = auto_exit_target_level

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
    st.caption("Polling is the PRIMARY feed. WebSocket is a SECONDARY optional feed/fallback.")
    if SmartConnect is None or pyotp is None:
        st.warning("Angel One connector packages are not installed. Yahoo Finance will continue to work; install the optional Angel dependencies from requirements-angel.txt to enable Angel One data.")
    if not SMART_WS_AVAILABLE:
        st.info("Angel One WebSocket connector is unavailable until the optional SmartAPI package is installed. Polling remains the preferred mode when Angel One is available.")
    polling_mode = st.checkbox("Enable REST LTP Polling (Primary)", value=True)
    ws_enable = st.checkbox("Enable Angel WebSocket (Secondary / Optional)", value=False)

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

if auto_refresh and st_autorefresh is not None:
    st_autorefresh(interval=refresh_seconds * 1000, key="elite_ultimate_refresh")
elif auto_refresh and st_autorefresh is None:
    st.warning("streamlit-autorefresh is not installed; automatic refresh is unavailable.")

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
    angel_credentials_ready = bool(angel_api_key and angel_client_code and angel_pin and angel_totp_secret)
    if not angel_credentials_ready:
        st.warning("Angel One credentials missing. Using selected fallback feed." if fallback_source != "No fallback" else "Please fill Angel One credentials to use this source.")
        if fallback_source in ("Yahoo Finance", "NSE Website (LTP only) + Yahoo candles"):
            data = fetch_data(ticker, yf_interval, yf_period)
            st.session_state["active_data_provider"] = "Yahoo Finance history fallback (Angel credentials missing)"
        else:
            st.stop()
    else:
        angel_interval = map_angel_interval(timeframe)
        candles = get_angel_candles(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected, angel_interval)
        data = candles_to_df(candles)

        if data.empty and fallback_source in ("Yahoo Finance", "NSE Website (LTP only) + Yahoo candles"):
            st.warning("Angel One candle data unavailable. Yahoo Finance is used for chart/history; NSE Website option is an LTP-only fallback.")
            data = fetch_data(ticker, yf_interval, yf_period)
            st.session_state["active_data_provider"] = "Yahoo Finance history fallback"
        elif data.empty:
            last_good = st.session_state.get("last_good_candles")
            data = candles_to_df(last_good)
            if not data.empty:
                st.warning("Angel API unavailable; showing last successful Angel candle cache (may be stale).")
                st.session_state["active_data_provider"] = "Angel cached candles (possibly stale)"
        else:
            st.session_state["last_good_candles"] = candles
            st.session_state["active_data_provider"] = "Angel One"

    if ws_enable:
        # Start WebSocket only as secondary feed; polling remains the preferred source.
        start_angel_websocket(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected)
        sync_ws_state_to_session()
else:
    data = fetch_data(ticker, yf_interval, yf_period)
    st.session_state["active_data_provider"] = "Yahoo Finance"

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

# Standard Japanese candlestick chart (raw OHLC) and Heikin-Ashi view are both shown.
raw_chart_data = data.tail(180)
if not raw_chart_data.empty:
    st.subheader("Japanese Candlestick / Price Action Chart")
    raw_fig = go.Figure(data=[go.Candlestick(
        x=raw_chart_data.index, open=raw_chart_data["Open"], high=raw_chart_data["High"],
        low=raw_chart_data["Low"], close=raw_chart_data["Close"], name="OHLC"
    )])
    for col, label in [("EMA20", "EMA 20"), ("EMA50", "EMA 50"), ("VWAP", "VWAP")]:
        if col in raw_chart_data:
            raw_fig.add_trace(go.Scatter(x=raw_chart_data.index, y=raw_chart_data[col], mode="lines", name=label))
    raw_fig.update_layout(height=480, xaxis_rangeslider_visible=False,
                          margin=dict(l=10, r=10, t=35, b=10),
                          title=f"{selected} — Standard OHLC ({timeframe})")
    st.plotly_chart(raw_fig, use_container_width=True)

# Heikin-Ashi candle chart (visual analysis; original OHLC remains the signal input).
ha_chart_data = convert_to_heikin_ashi(data.tail(180))
if not ha_chart_data.empty:
    st.subheader("Heikin-Ashi Candlestick Chart")
    ha_fig = go.Figure(data=[go.Candlestick(
        x=ha_chart_data.index,
        open=ha_chart_data["HA_Open"], high=ha_chart_data["HA_High"],
        low=ha_chart_data["HA_Low"], close=ha_chart_data["HA_Close"],
        name="Heikin-Ashi"
    )])
    ha_fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                         margin=dict(l=10, r=10, t=35, b=10),
                         title=f"{selected} — Heikin-Ashi ({timeframe})")
    st.plotly_chart(ha_fig, use_container_width=True)
    ha_last = ha_chart_data.iloc[-1]
    ha_state = "Bullish" if ha_last["HA_Close"] > ha_last["HA_Open"] else "Bearish" if ha_last["HA_Close"] < ha_last["HA_Open"] else "Neutral"
    st.caption(f"Latest Heikin-Ashi candle: {ha_state}. HA candles smooth price movement and can lag; use standard OHLC/LTP for execution decisions.")

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

st.subheader("Option Chain / OI / PCR Snapshot")
oc1, oc2, oc3, oc4, oc5, oc6 = st.columns(6)
with oc1: st.metric("PCR", option_metrics.get("pcr") if option_metrics.get("pcr") is not None else "N/A")
with oc2: st.metric("Call OI", f"{int(option_metrics.get('call_oi', 0)):,}")
with oc3: st.metric("Put OI", f"{int(option_metrics.get('put_oi', 0)):,}")
with oc4: st.metric("Call Day ΔOI", f"{int(option_metrics.get('call_day_change_oi', 0)):,}")
with oc5: st.metric("Put Day ΔOI", f"{int(option_metrics.get('put_day_change_oi', 0)):,}")
with oc6: st.metric("OI Bias", option_metrics.get("oi_bias", "N/A"))
oc7, oc8 = st.columns(2)
with oc7: st.metric("Call Snapshot ΔOI", option_metrics.get("call_snapshot_delta_oi") if option_metrics.get("call_snapshot_delta_oi") is not None else "Baseline")
with oc8: st.metric("Put Snapshot ΔOI", option_metrics.get("put_snapshot_delta_oi") if option_metrics.get("put_snapshot_delta_oi") is not None else "Baseline")
st.caption("NSE change-in-OI is exchange-reported day change. Snapshot ΔOI is calculated between successful app refresh snapshots; first snapshot is a baseline, not a zero-signal confirmation.")

latest_patterns = [x for x in str(data.iloc[-1].get("CandlestickPattern", "No major pattern")).split(", ") if x and x != "No major pattern"]
st.subheader("Latest Candlestick Pattern Detection")
if latest_patterns:
    st.success(" | ".join(latest_patterns))
else:
    st.info("No major candlestick pattern detected on the latest completed/current candle.")
pa1, pa2, pa3, pa4 = st.columns(4)
last_bar = data.iloc[-1]
with pa1:
    st.metric("Price Action", str(last_bar.get("PriceAction", "N/A")))
with pa2:
    st.metric("Bollinger Upper", round(float(last_bar["BB_Upper"]), 2) if pd.notna(last_bar.get("BB_Upper")) else "N/A")
with pa3:
    st.metric("Bollinger Lower", round(float(last_bar["BB_Lower"]), 2) if pd.notna(last_bar.get("BB_Lower")) else "N/A")
with pa4:
    st.metric("BB Squeeze", "Yes" if bool(last_bar.get("BB_Squeeze", False)) else "No")
with st.expander("Recent Price Action / Candlestick Details"):
    show_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "PriceAction", "CandlestickPattern",
                              "RSI", "MACD", "ADX", "ATR", "VWAP", "BB_Upper", "BB_Lower", "BB_Width_Pct"]
                 if c in data.columns]
    st.dataframe(data[show_cols].tail(30), use_container_width=True)

with st.expander("Government Announcements / Geopolitical News"):
    macro_news = get_macro_news_events()
    st.caption(f"Source: {macro_news['source']} | {macro_news['message']}")
    if macro_news["available"]:
        st.dataframe(pd.DataFrame(macro_news["articles"]), use_container_width=True, hide_index=True,
                     column_config={"URL": st.column_config.LinkColumn("Article")})
    else:
        st.warning("Macro news feed unavailable at this moment.")

with st.expander("FII / DII Activity (NSE feed)"):
    fii_dii = get_fii_dii_activity()
    st.caption(f"Source: {fii_dii['source']} | Checked: {fii_dii['as_of']} | {fii_dii['message']}")
    if fii_dii["available"]:
        st.dataframe(pd.DataFrame(fii_dii["rows"]), use_container_width=True, hide_index=True)
    else:
        st.warning("FII/DII data is not available from the NSE endpoint right now. No values are fabricated; try again later.")

news_info = {"allow_trade": True, "message": "News filter disabled", "events": []}
if news_filter_enabled:
    news_info = get_news_events(news_api_key, selected)

# Incorporate public macro headlines and NSE FII/DII net flow into the signal score.
# Missing/blocked feeds remain Neutral; the app never invents values.
macro_articles = macro_news.get("articles", []) if isinstance(macro_news, dict) else []
macro_text = " ".join(str(a.get("Headline", "")) for a in macro_articles).lower()
macro_risk_terms = ("war", "conflict", "attack", "sanction", "crisis", "escalation",
                    "border", "terror", "disruption", "emergency", "military")
news_info["macro_event_present"] = bool(macro_articles)
news_info["macro_risk_caution"] = any(term in macro_text for term in macro_risk_terms)

def _safe_flow_number(value):
    try:
        cleaned = str(value).replace(",", "").replace("₹", "").strip()
        return float(cleaned)
    except Exception:
        return None

flow_values = {}
for flow_row in (fii_dii.get("rows", []) if isinstance(fii_dii, dict) else []):
    category = str(flow_row.get("Category", "")).upper()
    net_value = _safe_flow_number(flow_row.get("Net"))
    if net_value is not None:
        if "FII" in category or "FPI" in category:
            flow_values["FII"] = net_value
        elif "DII" in category:
            flow_values["DII"] = net_value
if len(flow_values) == 2:
    combined_flow = flow_values["FII"] + flow_values["DII"]
    news_info["fii_dii_bias"] = "Bullish" if combined_flow > 0 else "Bearish" if combined_flow < 0 else "Neutral"
else:
    news_info["fii_dii_bias"] = "Neutral"

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

@st.cache_data(ttl=8, show_spinner=False)
def get_nse_website_ltp(symbol_name):
    """NSE website quote fallback for LTP only; it does not supply OHLC history."""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*", "Referer": "https://www.nseindia.com/"}
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        if symbol_name in ("NIFTY", "BANKNIFTY", "SENSEX"):
            response = session.get("https://www.nseindia.com/api/allIndices", headers=headers, timeout=8)
            response.raise_for_status()
            records = response.json().get("data", [])
            aliases = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "SENSEX": "SENSEX"}
            for row in records:
                if str(row.get("index", "")).upper() == aliases.get(symbol_name, ""):
                    value = row.get("last") or row.get("lastPrice")
                    return float(str(value).replace(",", "")) if value is not None else None
        else:
            response = session.get("https://www.nseindia.com/api/quote-equity", params={"symbol": symbol_name}, headers=headers, timeout=8)
            response.raise_for_status()
            value = response.json().get("priceInfo", {}).get("lastPrice")
            return float(str(value).replace(",", "")) if value is not None else None
    except Exception as exc:
        log_app_error("get_nse_website_ltp", exc)
    return None


live_price = None
live_price_provider = st.session_state.get("active_data_provider", data_source)
if data_source == "Angel":
    # Primary polling quote; WebSocket is deliberately secondary.
    if polling_mode:
        live_price = get_angel_ltp_polling(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected)
        if live_price is not None:
            live_price_provider = "Angel One REST Polling"
    if live_price is None and ws_enable:
        sync_ws_state_to_session()
        live_price = st.session_state.get("live_price")
        if live_price is not None:
            live_price_provider = "Angel One WebSocket (secondary)"
    if live_price is None and fallback_source == "NSE Website (LTP only) + Yahoo candles":
        live_price = get_nse_website_ltp(selected)
        if live_price is not None:
            live_price_provider = "NSE Website LTP fallback (Yahoo history candles)"
    if live_price is None and fallback_source in ("Yahoo Finance", "NSE Website (LTP only) + Yahoo candles"):
        # Yahoo supplies a recent candle close if the optional NSE website quote is blocked/unavailable.
        yahoo_fallback = fetch_data(ticker, "1m", "7d")
        if not yahoo_fallback.empty:
            live_price = float(yahoo_fallback["Close"].iloc[-1])
            live_price_provider = "Yahoo Finance fallback"

current_display_price = live_price if live_price is not None else float(analysis["close_price"])

# Auto-exit is paper-journal simulation only. It never sends broker orders.
auto_exit_enabled = st.session_state.get("auto_exit_enabled", True)
auto_exit_target_level = int(st.session_state.get("auto_exit_target_level", 3))
closed_auto = auto_paper_trade_exit(current_display_price, exit_target_level=auto_exit_target_level, enabled=auto_exit_enabled, symbol_filter=selected)

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
    st.caption(f"Active quote feed: {live_price_provider} | Polling primary: {'ON' if polling_mode else 'OFF'} | WebSocket secondary: {'ON' if ws_enable else 'OFF'}")

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

ma1, ma2, ma3, ma4, ma5 = st.columns(5)
latest_bar = data.iloc[-1]
with ma1: st.metric("SMA 20", round(float(latest_bar["SMA20"]), 2) if pd.notna(latest_bar["SMA20"]) else "N/A")
with ma2: st.metric("SMA 50", round(float(latest_bar["SMA50"]), 2) if pd.notna(latest_bar["SMA50"]) else "N/A")
with ma3: st.metric("SMA 100", round(float(latest_bar["SMA100"]), 2) if pd.notna(latest_bar["SMA100"]) else "N/A")
with ma4: st.metric("SMA 200", round(float(latest_bar["SMA200"]), 2) if pd.notna(latest_bar["SMA200"]) else "N/A")
with ma5: st.metric("ADR 14", round(float(latest_bar["ADR14"]), 2) if pd.notna(latest_bar["ADR14"]) else "N/A")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Trending Market", "Yes" if analysis["trending_market"] else "No")
with r2:
    st.metric("Sideways Market", "Yes" if analysis["sideways_market"] else "No")
with r3:
    st.metric("High Volatility", "Yes" if analysis["high_volatility_market"] else "No")
with r4:
    st.metric("Signal Quality", signal_quality)

st.subheader("Auto Exit Status (Paper Trading)")
st.metric("Auto-exit mode", "Enabled" if auto_exit_enabled else "Disabled")
st.caption(f"Stop-loss always triggers a simulated full exit. Target {auto_exit_target_level} triggers the simulated full exit. Auto-exit is scoped to the currently selected symbol and never places real broker orders.")

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
   


# =========================================================
# MARKET BREADTH + GLOBAL CONTEXT (merged from companion app)
# =========================================================
st.subheader("F&O-Style Major Stocks Scanner")
st.caption("Yahoo Finance prices may be delayed and are underlying spot prices, not guaranteed live futures quotes. Angel One polling is the primary quote feed when selected; WebSocket is secondary. Auto-exit is paper simulation only.")
with st.spinner("Loading major-stock market context..."):
    fno_scanner_df = fetch_fno_style_scanner(major_stocks)
if not fno_scanner_df.empty:
    def _market_trend_style(value):
        if value == "BULLISH": return "background-color: #163d1a; color: #7CFC00; font-weight: bold;"
        if value == "BEARISH": return "background-color: #4a1515; color: #ff6b6b; font-weight: bold;"
        if value == "NEUTRAL": return "background-color: #3a3a3a; color: #f2c94c; font-weight: bold;"
        return ""
    try:
        st.dataframe(fno_scanner_df.style.map(_market_trend_style, subset=["1m", "2m", "3m", "Direction"]), use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(fno_scanner_df, use_container_width=True, hide_index=True)
    fno_summary = summarize_fno_bias(fno_scanner_df)
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Major Stocks Bias", fno_summary["overall_bias"])
    sm2.metric("Bullish Stocks", f"{fno_summary['bullish_pct']}%")
    sm3.metric("Bearish Stocks", f"{fno_summary['bearish_pct']}%")
    sm4.metric("Strongest Gainer", fno_summary["strongest_gainer"])
    st.caption("Strongest Loser: " + fno_summary["strongest_loser"])
else:
    st.info("Major-stock scanner data unavailable right now.")

st.subheader("Global / Macro Performance — Latest Daily Close")
global_df = fetch_market_snapshot(global_indices)
if not global_df.empty:
    st.dataframe(global_df, use_container_width=True, hide_index=True)
else:
    st.info("Global market data unavailable right now.")

st.subheader("Indian Indices Performance — Latest Daily Close")
india_df = fetch_market_snapshot(indian_indices)
if not india_df.empty:
    st.dataframe(india_df, use_container_width=True, hide_index=True)
else:
    st.info("Indian index data unavailable right now.")
