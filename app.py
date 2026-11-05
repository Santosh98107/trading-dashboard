import os
import json
import math
import shutil
import hashlib
import hmac
from datetime import datetime, timedelta, time, timezone
import threading
import time as pytime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import institutional_upgrades as inst
# Visible deployment marker: confirms which source file Streamlit Cloud is running.
APP_BUILD_ID = "2026-11-04-LOSS-INTELLIGENCE-SIGNAL-V2-01"
# Optional integrations: Yahoo mode must still start if Angel SDK/TOTP or
# the auto-refresh helper is not installed.
try:
    import pyotp
except ImportError:
    pyotp = None

try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

SMART_API_AVAILABLE = SmartConnect is not None and pyotp is not None
AUTO_REFRESH_AVAILABLE = st_autorefresh is not None

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
NO_TRADE_NEAR_SR_PCT = 0.08  # legacy display threshold; engine uses an ATR-aware level buffer
GAP_FILTER_PCT = 0.75
HIGH_VOL_ATR_PCT = 1.8
SIDEWAYS_ADX_MAX = 18
TRENDING_ADX_MIN = 25
MIN_VALIDATION_SCORE_FOR_ALERT = 70
DEFAULT_INITIAL_BALANCE = 100000.0
DATA_INTEGRITY_MIN_ROWS = 30
BACKUP_RETENTION_COUNT = 30
USE_POLLING_AS_PRIMARY = False  # WebSocket is primary for Angel live LTP; REST polling is fallback.

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
    "last_tick_epoch": None,
    "generation": 0,
    "retry_exhausted": False,
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
    """Create the paper journal with backward-compatible partial-exit fields."""
    if not os.path.exists(JOURNAL_FILE):
        cols = [
            "DateTime", "TradeDate", "TradeTime", "Symbol", "Timeframe", "Signal", "SignalType", "Side", "EntryPrice", "ExitPrice",
            "Quantity", "RemainingQty", "StopLoss", "Target1", "Target2", "Target3", "TrailingStop",
            "ATR", "AutoExitEnabled", "Status", "PnL", "RealizedPnL", "PnLPercent", "Notes",
            "Result", "T1_Hit", "T2_Hit", "T3_Hit", "LastPrice", "LastUpdateTime",
            "TradeQualityScore", "ValidationScore", "StrategyGrade",
            "EMA9", "EMA21", "EMA20", "EMA50", "EMA200", "RSI", "MACD",
            "ADX", "VWAP", "Volume", "RelativeVolume", "VolumeDelta", "MarketRegime",
            "MTFTrendStatus", "TrendSlope", "SupportDistancePct", "ResistanceDistancePct",
            "DistanceEMA20Pct", "DistanceEMA50Pct", "DistanceVWAPPct", "RiskRewardRatio",
            "ConfidenceAtEntry", "HistoricalWinRateAtEntry", "SimilarTradesAtEntry",
            "LossReason", "SignalStage"
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
    """Load the journal and migrate older CSVs in memory without losing trades."""
    init_journal()
    journal = safe_read_csv(JOURNAL_FILE, pd.DataFrame())
    if journal.empty:
        for text_col in ["Symbol", "Timeframe", "Signal", "Side", "Status", "Notes", "Result", "StrategyGrade", "LastUpdateTime"]:
            if text_col in journal.columns:
                journal[text_col] = journal[text_col].astype(object)
        return journal
    changed = False
    defaults = {
        "TradeDate": "", "TradeTime": "", "SignalType": "", "Side": "", "RemainingQty": np.nan, "RealizedPnL": 0.0, "ATR": np.nan,
        "AutoExitEnabled": False, "LastPrice": np.nan, "LastUpdateTime": "",
        "EMA9": np.nan, "EMA21": np.nan, "EMA20": np.nan, "EMA50": np.nan,
        "EMA200": np.nan, "RSI": np.nan, "MACD": np.nan, "ADX": np.nan,
        "VWAP": np.nan, "Volume": np.nan, "RelativeVolume": np.nan, "VolumeDelta": np.nan,
        "MarketRegime": "Unknown", "MTFTrendStatus": "Unavailable", "TrendSlope": np.nan,
        "SupportDistancePct": np.nan, "ResistanceDistancePct": np.nan,
        "DistanceEMA20Pct": np.nan, "DistanceEMA50Pct": np.nan, "DistanceVWAPPct": np.nan,
        "RiskRewardRatio": np.nan, "ConfidenceAtEntry": np.nan,
        "HistoricalWinRateAtEntry": np.nan, "SimilarTradesAtEntry": 0,
        "LossReason": "", "SignalStage": ""
    }
    for col, default in defaults.items():
        if col not in journal.columns:
            journal[col] = default
            changed = True
    # Keep text/status fields object-typed so older CSVs with blank cells can accept later updates.
    for text_col in ["Symbol", "Timeframe", "Signal", "Side", "Status", "Notes", "Result", "StrategyGrade", "LastUpdateTime"]:
        if text_col in journal.columns:
            journal[text_col] = journal[text_col].fillna("").astype(object)
    if "Quantity" in journal.columns:
        qty = pd.to_numeric(journal["Quantity"], errors="coerce").fillna(0.0)
        rem = pd.to_numeric(journal["RemainingQty"], errors="coerce")
        missing_rem = rem.isna()
        if missing_rem.any():
            status = journal.get("Status", pd.Series("OPEN", index=journal.index)).astype(str).str.upper()
            rem.loc[missing_rem] = np.where(status.loc[missing_rem].eq("OPEN"), qty.loc[missing_rem], 0.0)
            journal["RemainingQty"] = rem
            changed = True
        if "Side" in journal.columns:
            sig = journal.get("Signal", pd.Series("", index=journal.index)).astype(str).str.upper()
            side_blank = journal["Side"].astype(str).str.strip().eq("")
            journal.loc[side_blank, "Side"] = np.where(sig.loc[side_blank].str.contains("SELL|PUT", regex=True), "SELL", "BUY")
            if side_blank.any():
                changed = True
    if changed:
        safe_write_csv(journal, JOURNAL_FILE)
    return journal



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
    return safe_write_csv(df, PAPER_STATE_FILE)


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


def update_logged_in_password(username, current_password, new_password, confirm_password):
    """Change the current user's password after verifying the existing password."""
    username = str(username or "").strip()
    current_password = str(current_password or "")
    new_password = str(new_password or "")
    confirm_password = str(confirm_password or "")

    if not username:
        return False, "No logged-in username was found. Please log in again."
    if not current_password or not new_password or not confirm_password:
        return False, "Fill all password fields."
    if len(new_password) < 8:
        return False, "New password must contain at least 8 characters."
    if new_password != confirm_password:
        return False, "New password and confirmation do not match."
    if hmac.compare_digest(new_password, current_password):
        return False, "New password must be different from the current password."

    users = safe_read_csv(USERS_FILE, pd.DataFrame(columns=["username", "password"]))
    if users.empty or not {"username", "password"}.issubset(users.columns):
        return False, "User database is unavailable or has an invalid format."

    user_mask = users["username"].astype(str).str.strip() == username
    if not user_mask.any():
        return False, "The logged-in user was not found in the user database."

    stored_hash = str(users.loc[user_mask, "password"].iloc[0]).strip()
    if not hmac.compare_digest(stored_hash, hash_password(current_password)):
        return False, "Current password is incorrect."

    users.loc[user_mask, "password"] = hash_password(new_password)
    if not safe_write_csv(users, USERS_FILE):
        return False, "Password could not be saved. Check app file permissions and error log."

    return True, "Password updated successfully. Use the new password next time you log in."


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

    # Password reset/change is available only inside an authenticated session.
    # The current password is verified before the stored hash is replaced.
    with st.sidebar.expander("Change / Reset Password", expanded=False):
        with st.form("change_logged_in_password_form", clear_on_submit=True):
            current_password = st.text_input("Current password", type="password", key="pwd_reset_current")
            new_password = st.text_input("New password (minimum 8 characters)", type="password", key="pwd_reset_new")
            confirm_password = st.text_input("Confirm new password", type="password", key="pwd_reset_confirm")
            submit_password_change = st.form_submit_button("Update Password", use_container_width=True)
        if submit_password_change:
            changed, message = update_logged_in_password(
                st.session_state.username, current_password, new_password, confirm_password
            )
            if changed:
                st.success(message)
            else:
                st.error(message)
        st.caption("For account security, the current password is required. Passwords are stored as hashes, not plain text.")

    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# =========================================================
# DATA FETCH
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_data(ticker, interval, period):
    """Yahoo OHLCV loader tolerant of yfinance MultiIndex layout changes."""
    needed = ["Open", "High", "Low", "Close", "Volume"]
    errors = []
    df = pd.DataFrame()
    try:
        try:
            df = yf.download(ticker, interval=interval, period=period,
                             progress=False, auto_adjust=False, threads=False, timeout=8,
                             multi_level_index=False)
        except TypeError:
            df = yf.download(ticker, interval=interval, period=period,
                             progress=False, auto_adjust=False, threads=False, timeout=8)
        except Exception as exc:
            errors.append(f"download: {exc}")

        # yfinance versions differ: columns may be (Price, Ticker) or (Ticker, Price).
        if isinstance(df.columns, pd.MultiIndex):
            best_level = None
            best_count = -1
            for level in range(df.columns.nlevels):
                vals = [str(v) for v in df.columns.get_level_values(level)]
                count = sum(col in vals for col in needed)
                if count > best_count:
                    best_count, best_level = count, level
            if best_level is not None:
                df.columns = df.columns.get_level_values(best_level)
        if not df.empty and all(c in df.columns for c in needed):
            df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
            df = df[needed].copy()
        else:
            # Ticker.history is a useful fallback when download returns a changed schema.
            try:
                hist = yf.Ticker(ticker).history(period=period, interval=interval,
                                                 auto_adjust=False, prepost=False, timeout=8)
                if not hist.empty and all(c in hist.columns for c in needed):
                    df = hist[needed].copy()
            except Exception as exc:
                errors.append(f"history: {exc}")

        if df.empty or not all(c in df.columns for c in needed):
            if errors:
                log_app_error("fetch_data", " | ".join(errors))
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
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


def install_smartapi_http_timeout(api):
    """Set the timeout used by SmartAPI's own requests.request call."""
    try:
        # Current smartapi-python uses self.timeout in SmartConnect._request.
        # Patching reqsession.request does not affect that code path.
        api.timeout = 10
        return True
    except Exception as exc:
        log_app_error("install_smartapi_http_timeout", exc)
        return False


def angel_login(api_key, client_code, pin, totp_secret, force_refresh=False):
    """Return a matching authenticated session; only request PIN/TOTP for fresh login."""
    try:
        if not SMART_API_AVAILABLE:
            log_app_error("angel_login", "Angel One SDK/pyotp missing. Install smartapi-python and pyotp.")
            return None, None
        api_key = str(api_key or "").strip()
        client_code = str(client_code or "").strip()
        pin = str(pin or "").strip()
        totp_secret = str(totp_secret or "").strip()
        cache_key = f"{client_code}_{api_key}"

        cached_api = st.session_state.get("angel_api")
        cached_session = st.session_state.get("angel_session")
        cached_data = (cached_session or {}).get("data") if isinstance(cached_session, dict) else None
        cache_matches = (
            cached_api is not None
            and isinstance(cached_session, dict)
            and bool(cached_data)
            and st.session_state.get("angel_cache_key") == cache_key
        )
        # Password widgets can be empty after Streamlit reruns. Reuse only the
        # authenticated session belonging to the same API key/client code.
        if cache_matches and (not force_refresh or not pin):
            return cached_api, cached_session

        if not all((api_key, client_code, pin, totp_secret)):
            return None, None

        try:
            smart_api = SmartConnect(api_key=api_key, timeout=10)
        except TypeError:
            smart_api = SmartConnect(api_key=api_key)
        install_smartapi_http_timeout(smart_api)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_code, pin, totp)

        if session and session.get("status") and session.get("data"):
            st.session_state["angel_api"] = smart_api
            st.session_state["angel_session"] = session
            st.session_state["angel_cache_key"] = cache_key
            st.session_state["angel_connected"] = True
            return smart_api, session
        return None, None
    except Exception as e:
        log_app_error("angel_login", e)
        return None, None


@st.cache_data(ttl=86400, show_spinner=False)
def get_angel_instrument_master():
    """Download and normalize Angel One's public instrument master once per day."""
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame()
        lower = {str(col).strip().lower(): col for col in frame.columns}
        def pick(*names):
            for name in names:
                if name in lower:
                    return lower[name]
            return None
        col_map = {
            "Token": pick("token", "symboltoken"),
            "TradingSymbol": pick("symbol", "tradingsymbol", "trading_symbol"),
            "Name": pick("name", "underlying"),
            "Exchange": pick("exch_seg", "exchange"),
            "InstrumentType": pick("instrumenttype", "instrument_type"),
            "Expiry": pick("expiry"),
        }
        if not all(col_map[k] for k in ("Token", "TradingSymbol", "Exchange")):
            log_app_error("get_angel_instrument_master", f"Unexpected master columns: {list(frame.columns)[:30]}")
            return pd.DataFrame()
        out = pd.DataFrame({
            key: frame[src] for key, src in col_map.items() if src is not None
        })
        for key in ("Name", "Expiry", "InstrumentType"):
            if key not in out:
                out[key] = ""
        out["Token"] = out["Token"].astype(str).str.strip()
        out["TradingSymbol"] = out["TradingSymbol"].astype(str).str.strip()
        out["Name"] = out["Name"].astype(str).str.strip().str.upper()
        out["Exchange"] = out["Exchange"].astype(str).str.strip().str.upper()
        out["Exchange"] = out["Exchange"].replace({
            "NSE_CM": "NSE", "BSE_CM": "BSE", "NSECM": "NSE", "BSECM": "BSE"
        })
        out["InstrumentType"] = out["InstrumentType"].astype(str).str.strip().str.upper()
        out["Expiry"] = out["Expiry"].astype(str).str.strip()
        out = out[(out["Token"] != "") & (out["TradingSymbol"] != "")]
        return out.reset_index(drop=True)
    except Exception as exc:
        log_app_error("get_angel_instrument_master", exc)
        return pd.DataFrame()


def resolve_angel_instrument(symbol_name):
    """Resolve exchange, trading symbol and token from master; use known safe fallbacks."""
    symbol_name = str(symbol_name or "").strip().upper()
    fallback_token = ANGEL_TOKENS.get(symbol_name)
    fallback_symbol = ANGEL_TRADING_SYMBOLS.get(symbol_name)
    fallback_exchange = ANGEL_EXCHANGES.get(symbol_name, "NSE")

    # Avoid an unnecessary public-master download for the verified built-in
    # NIFTY/BankNIFTY and core-stock mappings. Less-common instruments (including
    # SENSEX and the wider watchlist) are resolved from the live public master.
    if fallback_token and fallback_symbol:
        resolved = {
            "exchange": fallback_exchange,
            "trading_symbol": fallback_symbol,
            "token": str(fallback_token),
            "source": "Built-in verified fallback mapping",
        }
        st.session_state["angel_last_instrument"] = resolved
        return resolved

    master = get_angel_instrument_master()
    if master is not None and not master.empty:
        aliases = ANGEL_NAME_ALIASES.get(symbol_name, [symbol_name])
        aliases = [str(x).strip().upper() for x in aliases]
        exchange_preference = "BSE" if symbol_name == "SENSEX" else "NSE"
        candidates = master[master["Exchange"].isin([exchange_preference])].copy()
        if symbol_name in {"NIFTY", "BANKNIFTY", "SENSEX"}:
            candidates = candidates[
                candidates["Name"].isin(aliases)
                | candidates["TradingSymbol"].str.upper().isin(aliases)
            ]
            if "InstrumentType" in candidates:
                idx = candidates[candidates["InstrumentType"].str.contains("AMXIDX", na=False)]
                if not idx.empty:
                    candidates = idx
        else:
            target_symbols = {f"{alias}-EQ" for alias in aliases}
            candidates["_symbol_upper"] = candidates["TradingSymbol"].str.upper()
            exact = candidates[candidates["_symbol_upper"].isin(target_symbols)]
            if not exact.empty:
                candidates = exact
            else:
                candidates = candidates[candidates["Name"].isin(aliases)]
                if not candidates.empty:
                    equity = candidates[candidates["TradingSymbol"].str.upper().str.endswith("-EQ")]
                    if not equity.empty:
                        candidates = equity
        if not candidates.empty:
            row = candidates.iloc[0]
            resolved = {
                "exchange": str(row["Exchange"]).upper(),
                "trading_symbol": str(row["TradingSymbol"]),
                "token": str(row["Token"]),
                "source": "Angel instrument master",
            }
            st.session_state["angel_last_instrument"] = resolved
            return resolved

    if fallback_token and fallback_symbol:
        resolved = {
            "exchange": fallback_exchange,
            "trading_symbol": fallback_symbol,
            "token": str(fallback_token),
            "source": "Built-in fallback mapping",
        }
        st.session_state["angel_last_instrument"] = resolved
        return resolved
    st.session_state["angel_last_instrument"] = None
    return None


def get_angel_candles(api_key, client_code, pin, totp_secret, symbol_name, interval="FIVE_MINUTE"):
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        if angel is None or not session or not session.get("data"):
            return None

        instrument = resolve_angel_instrument(symbol_name)
        if not instrument:
            return {"status": False, "message": f"Angel instrument master did not resolve {symbol_name}; check master connectivity and symbol spelling"}

        ist = timezone(timedelta(hours=5, minutes=30))
        to_date = datetime.now(ist)
        # Angel One's documented per-request maximum differs by interval.
        # Use practical lookbacks that provide enough history without exceeding it.
        lookback_days = {
            "ONE_MINUTE": 20,
            "THREE_MINUTE": 45,
            "FIVE_MINUTE": 75,
            "TEN_MINUTE": 75,
            "FIFTEEN_MINUTE": 120,
            "THIRTY_MINUTE": 120,
            "ONE_HOUR": 180,
            "ONE_DAY": 365,
        }.get(interval, 20)
        from_date = to_date - timedelta(days=lookback_days)

        params = {
            "exchange": instrument["exchange"],
            "symboltoken": instrument["token"],
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        return angel.getCandleData(params)
    except Exception as e:
        log_app_error("get_angel_candles", e)
        return {"status": False, "message": str(e)}


def get_angel_candles_cached(api_key, client_code, pin, totp_secret, symbol_name,
                              interval="FIVE_MINUTE", force_refresh=False):
    """Session-scoped short cache to avoid duplicate MTF calls and broker rate bursts."""
    try:
        cache = st.session_state.get("angel_history_cache")
        if not isinstance(cache, dict):
            cache = {}
        cache_key = hashlib.sha256(
            f"{api_key}|{client_code}|{symbol_name}|{interval}".encode("utf-8")
        ).hexdigest()
        now = pytime.time()
        entry = cache.get(cache_key)
        if not force_refresh and isinstance(entry, dict):
            age = now - float(entry.get("fetched_at", 0))
            ttl = 15 if entry.get("usable") else 5
            if age <= ttl:
                return entry.get("response")

        response = get_angel_candles(
            api_key, client_code, pin, totp_secret, symbol_name, interval
        )
        payload = response.get("data") if isinstance(response, dict) else None
        usable = bool(isinstance(response, dict) and response.get("status") and payload)
        cache[cache_key] = {
            "fetched_at": now,
            "usable": usable,
            "response": response,
        }
        # Bound session cache growth when the user changes instruments/timeframes.
        if len(cache) > 80:
            cache = dict(list(cache.items())[-60:])
        st.session_state["angel_history_cache"] = cache
        return response
    except Exception as exc:
        log_app_error("get_angel_candles_cached", exc)
        return get_angel_candles(api_key, client_code, pin, totp_secret, symbol_name, interval)


def candles_to_df(candles):
    """Normalize SmartAPI candle responses without hiding broker error messages."""
    try:
        if isinstance(candles, pd.DataFrame):
            raw = candles.copy()
            if "Datetime" in raw.columns:
                raw = raw.set_index("Datetime")
            elif "timestamp" in raw.columns:
                raw = raw.set_index("timestamp")
            df = raw
        else:
            payload = candles.get("data") if isinstance(candles, dict) else candles
            if not isinstance(payload, (list, tuple)) or len(payload) == 0:
                return pd.DataFrame()
            if isinstance(payload[0], dict):
                df = pd.DataFrame(payload).rename(columns={
                    "timestamp": "Datetime", "time": "Datetime",
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"
                })
                if "Datetime" in df.columns:
                    df = df.set_index("Datetime")
            else:
                df = pd.DataFrame(payload)
                if df.shape[1] < 5:
                    return pd.DataFrame()
                cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"][:df.shape[1]]
                df.columns = cols
                df = df.set_index("Datetime")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].copy()
        df.columns = [str(c).strip().title() for c in df.columns]
        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = 0
        for col in required + ["Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)
        df["Volume"] = df["Volume"].fillna(0)
        return df[required + ["Volume"]].sort_index()
    except Exception as e:
        log_app_error("candles_to_df", e)
        return pd.DataFrame()


def summarize_angel_candle_response(response):
    """Return safe, compact broker-response diagnostics (never credentials)."""
    if response is None:
        return {"response_type": "None", "status": None, "message": "No response object returned", "bars_received": 0}
    if isinstance(response, dict):
        payload = response.get("data")
        if isinstance(payload, (list, tuple, pd.DataFrame)):
            count = len(payload)
            payload_type = type(payload).__name__
        elif isinstance(payload, dict):
            count = len(payload.get("candles", [])) if isinstance(payload.get("candles"), list) else 0
            payload_type = "dict"
        else:
            count = 0
            payload_type = type(payload).__name__ if payload is not None else "None"
        return {
            "response_type": "dict",
            "status": response.get("status"),
            "message": str(response.get("message") or response.get("error") or "")[:500],
            "error_code": response.get("errorcode") or response.get("errorCode"),
            "payload_type": payload_type,
            "bars_received": count,
            "response_keys": list(response.keys())[:20],
        }
    if isinstance(response, (list, tuple, pd.DataFrame)):
        return {"response_type": type(response).__name__, "status": None,
                "message": "", "bars_received": len(response)}
    return {"response_type": type(response).__name__, "status": None,
            "message": str(response)[:500], "bars_received": 0}


def get_angel_ltp_polling(api_key, client_code, pin, totp_secret, symbol_name):
    """Fetch Angel LTP with bounded REST timeout and retain safe diagnostic details."""
    diagnostic = {"status": "not_requested", "symbol": str(symbol_name)}
    try:
        angel, session = angel_login(api_key, client_code, pin, totp_secret)
        if angel is None or not session or not session.get("data"):
            diagnostic.update(status="not_authenticated", message="No matching authenticated Angel session")
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            return None

        instrument = resolve_angel_instrument(symbol_name)
        if not instrument:
            diagnostic.update(status="mapping_missing", message=f"Angel instrument master did not resolve {symbol_name}")
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            log_app_error("get_angel_ltp_polling", diagnostic["message"])
            return None
        token = instrument["token"]
        trading_symbol = instrument["trading_symbol"]
        exchange = instrument["exchange"]
        diagnostic.update(status="request_sent", exchange=exchange, trading_symbol=trading_symbol, token=str(token), instrument_source=instrument["source"])
        # Official smartapi-python method is ltpData(exchange, tradingsymbol, token).
        # Keep getLTPData as a compatibility fallback for wrappers that expose it.
        ltp_method = getattr(angel, "ltpData", None)
        if not callable(ltp_method):
            ltp_method = getattr(angel, "getLTPData", None)
        if not callable(ltp_method):
            diagnostic.update(status="sdk_method_missing", message="SmartAPI client exposes neither ltpData nor getLTPData")
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            log_app_error("get_angel_ltp_polling", diagnostic["message"])
            return None
        response = ltp_method(exchange, trading_symbol, str(token))
        if not isinstance(response, dict):
            diagnostic.update(status="invalid_response", response_type=type(response).__name__, message=str(response)[:300])
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            log_app_error("get_angel_ltp_polling", response)
            return None
        if response.get("status") is False:
            diagnostic.update(status="broker_error", message=str(response.get("message") or response.get("error") or "Angel quote request failed")[:400], error_code=response.get("errorcode") or response.get("errorCode"))
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            log_app_error("get_angel_ltp_polling", diagnostic)
            return None
        quote_payload = response.get("data") or {}
        quote = None
        if isinstance(quote_payload, dict):
            # Some SDK/API responses wrap the quote inside data.fetched.
            fetched = quote_payload.get("fetched")
            if isinstance(fetched, list):
                quote = next((item for item in fetched if isinstance(item, dict)), None)
            else:
                quote = quote_payload
        elif isinstance(quote_payload, list):
            quote = next((item for item in quote_payload if isinstance(item, dict)), None)

        value = None
        if isinstance(quote, dict):
            for key in ("ltp", "LTP", "last_traded_price", "lastTradedPrice"):
                if quote.get(key) not in (None, ""):
                    value = quote.get(key)
                    break
        try:
            price = float(value) if value is not None else None
        except (TypeError, ValueError):
            price = None
        if price is not None and np.isfinite(price) and price > 0:
            diagnostic.update(status="received", price=price, quote_shape=type(quote_payload).__name__)
            st.session_state["angel_last_ltp_diagnostic"] = diagnostic
            return price
        diagnostic.update(
            status="empty_quote",
            message="Successful response did not contain a usable LTP",
            response_keys=list(response.keys())[:20],
            quote_shape=type(quote_payload).__name__,
        )
        st.session_state["angel_last_ltp_diagnostic"] = diagnostic
        return None
    except Exception as e:
        diagnostic.update(status="exception_or_timeout", message=str(e)[:500])
        st.session_state["angel_last_ltp_diagnostic"] = diagnostic
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
        "ws_last_tick_epoch": state.get("last_tick_epoch"),
        "live_price": state["live_price"],
        "ws_retries": state["retries"],
        "ws_should_run": state["should_run"],
        "ws_thread_started": state["thread_started"],
        "ws_symbol": state.get("symbol"),
        "ws_retry_exhausted": state.get("retry_exhausted", False),
        "ws_log": state["log"]
    }.items():
        st.session_state[k] = v


def init_ws_state():
    defaults = {
        "ws_running": False,
        "ws_connected": False,
        "ws_error": "",
        "ws_last_tick": None,
        "ws_last_tick_epoch": None,
        "ws_thread_started": False,
        "live_price": None,
        "ws_retries": 0,
        "ws_should_run": False,
        "ws_retry_exhausted": False,
        "ws_log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    sync_ws_state_to_session()


def ws_live_quote(max_age_seconds=20, symbol_name=None):
    """Return a fresh WebSocket LTP only while its socket is connected."""
    state = ws_get_state()
    if symbol_name and str(state.get("symbol")) != str(symbol_name):
        return None
    price = state.get("live_price")
    received_at = state.get("last_tick_epoch")
    if not state.get("connected") or price is None or received_at is None:
        return None
    try:
        age = pytime.time() - float(received_at)
        price = float(price)
        if age < 0 or age > float(max_age_seconds) or not np.isfinite(price) or price <= 0:
            return None
        return price
    except Exception:
        return None


def stop_angel_websocket():
    """Invalidate the worker generation before closing the active socket."""
    try:
        state = ws_get_state()
        socket_obj = state.get("socket")
        next_generation = int(state.get("generation", 0)) + 1
        ws_set_state(
            generation=next_generation, should_run=False, connected=False,
            running=False, thread_started=False, socket=None,
            live_price=None, last_tick=None, last_tick_epoch=None,
            retry_exhausted=False, error="WebSocket stopped"
        )
        if socket_obj is not None:
            try:
                if hasattr(socket_obj, "close_connection"):
                    socket_obj.close_connection()
                elif hasattr(socket_obj, "close"):
                    socket_obj.close()
            except Exception as exc:
                ws_log(f"Socket close error: {exc}")
        sync_ws_state_to_session()
    except Exception as exc:
        ws_set_state(error=f"Stop error: {exc}")
        log_app_error("stop_angel_websocket", exc)


def extract_ltp_from_message(message):
    if not isinstance(message, dict):
        return None
    keys = ["last_traded_price", "ltp", "lastTradedPrice", "last_traded_price_int", "close"]
    for key in keys:
        if key in message and message.get(key) is not None:
            try:
                val = float(message[key])
                # SmartAPI WebSocket v2 binary last_traded_price is in paise.
                # Alternate wrapper keys (ltp/lastTradedPrice/close) are already
                # expressed in rupees and must not be divided merely because an
                # index value is greater than 10,000.
                if key in {"last_traded_price", "last_traded_price_int"} and val > 0:
                    val /= 100.0
                return val if np.isfinite(val) and val > 0 else None
            except Exception:
                return None
    return None


def start_angel_websocket(api_key, client_code, pin, totp_secret, symbol_name):
    """Start SmartWebSocketV2 as the live-LTP primary without using Streamlit state in its worker."""
    init_ws_state()
    if not SMART_WS_AVAILABLE or not SMART_API_AVAILABLE:
        ws_set_state(error="SmartAPI WebSocket/TOTP dependencies are unavailable", connected=False, running=False)
        sync_ws_state_to_session()
        return

    instrument = resolve_angel_instrument(symbol_name)
    if not instrument:
        ws_set_state(error=f"Angel instrument master did not resolve {symbol_name}", connected=False, running=False)
        sync_ws_state_to_session()
        return
    exchange_type = 1 if instrument["exchange"] == "NSE" else 3 if instrument["exchange"] == "BSE" else None
    if exchange_type is None:
        ws_set_state(error=f"WebSocket exchange mapping unsupported: {instrument['exchange']}", connected=False, running=False)
        sync_ws_state_to_session()
        return

    current = ws_get_state()
    if (current.get("thread_started") and current.get("symbol") == symbol_name
            and current.get("should_run")):
        sync_ws_state_to_session()
        return

    # Reuse the already-authenticated Streamlit session for the first socket
    # connection. This avoids generating a second Angel login just to obtain a
    # WebSocket token, which could invalidate the REST polling fallback session.
    try:
        _angel_api_for_ws = st.session_state.get("angel_api")
        _angel_session_for_ws = st.session_state.get("angel_session") or {}
        _angel_auth_data_for_ws = _angel_session_for_ws.get("data") or {}
        initial_auth_token = _angel_auth_data_for_ws.get("jwtToken") or _angel_auth_data_for_ws.get("jwt_token")
        if _angel_api_for_ws is None or not initial_auth_token:
            raise RuntimeError("No authenticated Angel API session is available for WebSocket startup")
        _feed_response_for_ws = _angel_api_for_ws.getfeedToken()
        if isinstance(_feed_response_for_ws, dict):
            initial_feed_token = (_feed_response_for_ws.get("data") or {}).get("feedToken") or _feed_response_for_ws.get("feedToken")
        else:
            initial_feed_token = _feed_response_for_ws
        if not initial_feed_token:
            raise RuntimeError("Authenticated Angel session did not return a WebSocket feed token")
    except Exception as exc:
        ws_set_state(error=f"Could not prepare authenticated WebSocket tokens: {str(exc)[:250]}", connected=False, running=False)
        log_app_error("prepare_angel_websocket_tokens", exc)
        sync_ws_state_to_session()
        return

    old_socket = current.get("socket")
    generation = int(current.get("generation", 0)) + 1
    ws_set_state(
        generation=generation, should_run=True, thread_started=True,
        symbol=symbol_name, error="Connecting to Angel One WebSocket…",
        retries=0, retry_exhausted=False, running=False, connected=False, socket=None,
        live_price=None, last_tick=None, last_tick_epoch=None, last_data_time=None
    )
    if old_socket is not None:
        try:
            old_socket.close_connection()
        except Exception:
            pass

    def generation_is_current():
        snapshot = ws_get_state()
        return snapshot.get("generation") == generation and snapshot.get("should_run")

    def create_worker_session():
        # Use the authenticated tokens captured on the Streamlit main thread.
        # The worker never reads st.session_state or creates a duplicate login.
        return initial_auth_token, initial_feed_token

    def run_socket():
        retry_count = 0
        while generation_is_current():
            sws = None
            try:
                auth_token, feed_token = create_worker_session()
                if not generation_is_current():
                    break
                sws = SmartWebSocketV2(auth_token, str(api_key).strip(), str(client_code).strip(), feed_token, max_retry_attempt=0)
                ws_set_state(socket=sws, running=True, connected=False, error="WebSocket authenticated; waiting for market ticks…")
                correlation_id = f"EL{generation:08d}"[-10:]
                token_list = [{"exchangeType": exchange_type, "tokens": [str(instrument["token"])]}]

                def on_data(wsapp, message):
                    if not generation_is_current():
                        return
                    ltp = extract_ltp_from_message(message)
                    if ltp is not None:
                        ws_set_state(
                            last_tick=message, connected=True, running=True, error="",
                            live_price=ltp, last_data_time=now_str(), last_tick_epoch=pytime.time()
                        )

                def on_open(wsapp):
                    if not generation_is_current():
                        return
                    try:
                        ws_set_state(connected=True, running=True, error="WebSocket connected; subscription sent")
                        sws.subscribe(correlation_id, 1, token_list)
                    except Exception as exc:
                        ws_set_state(error=f"WebSocket subscribe error: {exc}", connected=False)

                def on_error(wsapp, error):
                    if generation_is_current():
                        ws_set_state(error=f"WebSocket error: {error}", connected=False, running=False)

                def on_close(wsapp, *args):
                    if generation_is_current():
                        reason = " | ".join(str(item) for item in args if item is not None)
                        ws_set_state(connected=False, running=False, error=f"WebSocket closed {reason}".strip())

                sws.on_open = on_open
                sws.on_data = on_data
                sws.on_error = on_error
                sws.on_close = on_close
                sws.connect()
                if generation_is_current():
                    ws_set_state(connected=False, running=False, socket=None, error="WebSocket connection ended; retrying")
            except Exception as exc:
                if generation_is_current():
                    ws_set_state(error=f"WebSocket attempt {retry_count + 1}: {str(exc)[:350]}", connected=False, running=False, socket=None)
                log_app_error("start_angel_websocket", exc)
            finally:
                if sws is not None and not generation_is_current():
                    try:
                        sws.close_connection()
                    except Exception:
                        pass

            if not generation_is_current():
                break
            retry_count += 1
            ws_set_state(retries=retry_count)
            if retry_count >= MAX_WS_RETRIES:
                ws_set_state(
                    should_run=False, running=False, connected=False,
                    thread_started=False, socket=None, retry_exhausted=True,
                    error=f"WebSocket stopped after {MAX_WS_RETRIES} reconnect attempts; REST polling fallback remains available"
                )
                break
            pytime.sleep(WS_RETRY_DELAY_SEC)

        final_state = ws_get_state()
        if final_state.get("generation") == generation:
            ws_set_state(running=False, connected=False, thread_started=False, socket=None)

    threading.Thread(target=run_socket, name=f"AngelWS-{symbol_name}-{generation}", daemon=True).start()
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


def detect_candlestick_patterns(df):
    """Vectorized OHLC candlestick library. Patterns are heuristic confirmations, not standalone trade signals."""
    out = df.copy()
    o, h, l, c = [pd.to_numeric(out[k], errors="coerce") for k in ("Open", "High", "Low", "Close")]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    small = body <= rng * 0.30
    long_body = body >= rng * 0.60
    bull = c > o
    bear = c < o
    prev_bull, prev_bear = bull.shift(1).fillna(False), bear.shift(1).fillna(False)
    prev_body = body.shift(1)
    prev_mid = (o.shift(1) + c.shift(1)) / 2

    # Single-candle patterns
    out["Hammer"] = (lower >= body * 2.0) & (upper <= body * 0.6) & (body > 0)
    out["HangingMan"] = out["Hammer"]  # Context (preceding uptrend) is shown separately as a heuristic.
    out["InvertedHammer"] = (upper >= body * 2.0) & (lower <= body * 0.6) & (body > 0) & bear
    out["ShootingStar"] = (upper >= body * 2.0) & (lower <= body * 0.6) & (body > 0) & bear
    out["Doji"] = body <= rng * 0.10
    out["SpinningTop"] = (body <= rng * 0.30) & (upper >= body * 0.5) & (lower >= body * 0.5) & (~out["Doji"])
    out["MarubozuBull"] = bull & (upper <= rng * 0.05) & (lower <= rng * 0.05) & long_body
    out["MarubozuBear"] = bear & (upper <= rng * 0.05) & (lower <= rng * 0.05) & long_body

    # Two-candle patterns
    out["BullishEngulfing"] = bull & prev_bear & (o <= c.shift(1)) & (c >= o.shift(1)) & (body >= prev_body)
    out["BearishEngulfing"] = bear & prev_bull & (o >= c.shift(1)) & (c <= o.shift(1)) & (body >= prev_body)
    out["PiercingLine"] = bear.shift(1).fillna(False) & bull & (o < l.shift(1)) & (c > prev_mid) & (c < o.shift(1))
    out["DarkCloudCover"] = bull.shift(1).fillna(False) & bear & (o > h.shift(1)) & (c < prev_mid) & (c > o.shift(1))
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    out["TweezerBottom"] = bear.shift(1).fillna(False) & bull & ((l - prev_low).abs() <= (rng * 0.10))
    out["TweezerTop"] = bull.shift(1).fillna(False) & bear & ((h - prev_high).abs() <= (rng * 0.10))
    out["InsideBar"] = (h < prev_high) & (l > prev_low)
    out["OutsideBar"] = (h > prev_high) & (l < prev_low)

    # Three-candle patterns (middle candle small; final close confirms direction)
    o1, c1 = o.shift(2), c.shift(2)
    o2, c2 = o.shift(1), c.shift(1)
    mid_small = (c2 - o2).abs() <= (h.shift(1) - l.shift(1)).replace(0, np.nan) * 0.30
    out["MorningStar"] = (c1 < o1) & mid_small & bull & (c > (o1 + c1) / 2)
    out["EveningStar"] = (c1 > o1) & mid_small & bear & (c < (o1 + c1) / 2)
    out["ThreeWhiteSoldiers"] = bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (o < c.shift(1)) & (o.shift(1) < c.shift(2))
    out["ThreeBlackCrows"] = bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (o > c.shift(1)) & (o.shift(1) > c.shift(2))

    # Trend-context approximations to avoid presenting every long lower wick as a reversal.
    prior_up = c.shift(1) > c.shift(5)
    prior_down = c.shift(1) < c.shift(5)
    out["HangingMan"] = out["HangingMan"] & prior_up
    out["Hammer"] = out["Hammer"] & prior_down
    out["InvertedHammer"] = out["InvertedHammer"] & prior_down
    out["ShootingStar"] = out["ShootingStar"] & prior_up

    bullish_cols = ["Hammer", "InvertedHammer", "BullishEngulfing", "PiercingLine", "MorningStar", "ThreeWhiteSoldiers", "TweezerBottom", "MarubozuBull"]
    bearish_cols = ["HangingMan", "ShootingStar", "BearishEngulfing", "DarkCloudCover", "EveningStar", "ThreeBlackCrows", "TweezerTop", "MarubozuBear"]
    for col in bullish_cols + bearish_cols + ["Doji", "SpinningTop", "InsideBar", "OutsideBar"]:
        out[col] = out[col].fillna(False).astype(bool)

    names=[]; directions=[]; scores=[]
    for idx in out.index:
        found=[name for name in bullish_cols + bearish_cols + ["Doji", "SpinningTop", "InsideBar", "OutsideBar"] if bool(out.at[idx, name])]
        names.append(", ".join(found) if found else "No clear pattern")
        b=sum(bool(out.at[idx, name]) for name in bullish_cols)
        s=sum(bool(out.at[idx, name]) for name in bearish_cols)
        directions.append("Bullish" if b>s else "Bearish" if s>b else "Neutral")
        scores.append(int(np.clip((b-s)*2, -6, 6)))
    out["CandlePattern"] = names
    out["CandlePatternBias"] = directions
    out["CandlePatternScore"] = scores
    return out


def add_indicators(df):
    df = df.copy()

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
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
    df["RelativeVolume"] = (df["Volume"] / df["Volume_MA20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["VolumeDelta"] = np.sign(df["Close"].diff()).fillna(0) * df["Volume"].fillna(0)
    df["TrendSlope"] = df["EMA9"].diff(3) / 3.0
    df["DistanceEMA20Pct"] = ((df["Close"] - df["EMA20"]) / df["EMA20"].replace(0, np.nan)) * 100
    df["DistanceEMA50Pct"] = ((df["Close"] - df["EMA50"]) / df["EMA50"].replace(0, np.nan)) * 100
    df["DistanceVWAPPct"] = ((df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)) * 100 if "VWAP" in df.columns else np.nan
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
    df = detect_candlestick_patterns(df)
    return df


# =========================================================
# NIFTY TOP 30 MULTI-TIMEFRAME SCANNER
# =========================================================
NIFTY_TOP30 = {
    "RELIANCE": "RELIANCE.NS", "HDFC BANK": "HDFCBANK.NS", "ICICI BANK": "ICICIBANK.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS", "TCS": "TCS.NS", "SBI": "SBIN.NS",
    "INFOSYS": "INFY.NS", "BAJAJ FINANCE": "BAJFINANCE.NS", "LIC": "LICI.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "MARUTI": "MARUTI.NS", "HCL TECH": "HCLTECH.NS", "KOTAK BANK": "KOTAKBANK.NS",
    "SUN PHARMA": "SUNPHARMA.NS", "AXIS BANK": "AXISBANK.NS", "M&M": "M&M.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS", "TITAN": "TITAN.NS", "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS", "BEL": "BEL.NS", "ADANI PORTS": "ADANIPORTS.NS",
    "POWER GRID": "POWERGRID.NS", "TATA MOTORS": "TMPV.NS", "WIPRO": "WIPRO.NS",
    "TRENT": "TRENT.NS", "JSW STEEL": "JSWSTEEL.NS", "TATA STEEL": "TATASTEEL.NS"
}


# Global/macro and major Indian share-board. Yahoo Finance is intentionally an
# independent feed so this information remains visible even when the selected
# primary Angel/Yahoo OHLC chart feed is unavailable.
GLOBAL_MACRO_ASSETS = [
    {"name": "S&P 500", "tickers": ("^GSPC",)},
    {"name": "Hang Seng", "tickers": ("^HSI",)},
    {"name": "GIFT Nifty", "tickers": ("^NSEI",), "fallback_name": "NIFTY spot proxy (not GIFT futures)", "always_fallback_label": True},
    {"name": "India VIX", "tickers": ("^INDIAVIX",)},
    {"name": "NASDAQ", "tickers": ("^IXIC",)},
    {"name": "Shanghai Composite", "tickers": ("000001.SS", "^SSEC")},
    {"name": "USD/INR", "tickers": ("INR=X",)},
    {"name": "Dow Jones", "tickers": ("^DJI",)},
    {"name": "FTSE 100", "tickers": ("^FTSE",)},
    {"name": "Brent Crude", "tickers": ("BZ=F",)},
    {"name": "Nikkei 225", "tickers": ("^N225",)},
    {"name": "DAX", "tickers": ("^GDAXI",)},
    {"name": "Gold", "tickers": ("GC=F",)},
]

MAJOR_INDIAN_MARKET_STOCKS = [
    {"name": "Reliance", "tickers": ("RELIANCE.NS",)},
    {"name": "Axis Bank", "tickers": ("AXISBANK.NS",)},
    {"name": "HCL Tech", "tickers": ("HCLTECH.NS",)},
    {"name": "Maruti", "tickers": ("MARUTI.NS",)},
    {"name": "ITC", "tickers": ("ITC.NS",)},
    {"name": "HDFC Bank", "tickers": ("HDFCBANK.NS",)},
    {"name": "Kotak Bank", "tickers": ("KOTAKBANK.NS",)},
    {"name": "Wipro", "tickers": ("WIPRO.NS",)},
    {"name": "Mahindra & Mahindra", "tickers": ("M&M.NS",)},
    {"name": "Larsen & Toubro", "tickers": ("LT.NS",)},
    {"name": "ICICI Bank", "tickers": ("ICICIBANK.NS",)},
    {"name": "TCS", "tickers": ("TCS.NS",)},
    {"name": "Tech Mahindra", "tickers": ("TECHM.NS",)},
    {"name": "Bajaj Auto", "tickers": ("BAJAJ-AUTO.NS",)},
    {"name": "Bharti Airtel", "tickers": ("BHARTIARTL.NS",)},
]


def classify_stock_timeframe(frame):
    """Return a transparent trend classification using EMA, MACD, RSI and HA direction."""
    if frame is None or frame.empty or len(frame) < 55:
        return "N/A", 0
    try:
        x = add_indicators(frame).replace([np.inf, -np.inf], np.nan).dropna(subset=["EMA20", "EMA50", "MACD", "SignalLine", "RSI"])
        if len(x) < 2:
            return "N/A", 0
        a, b = x.iloc[-1], x.iloc[-2]
        ha = convert_to_heikin_ashi(x.tail(2))
        ha_up = bool(ha.iloc[-1]["HA_Close"] > ha.iloc[-1]["HA_Open"])
        score = 0
        score += 2 if a["EMA20"] > a["EMA50"] else -2 if a["EMA20"] < a["EMA50"] else 0
        score += 1 if a["Close"] > a["EMA20"] else -1
        score += 1 if a["MACD"] > a["SignalLine"] else -1
        score += 1 if a["MACD_Hist"] > b["MACD_Hist"] else -1
        score += 1 if a["RSI"] >= 52 else -1 if a["RSI"] <= 48 else 0
        score += 1 if ha_up else -1
        if score >= 2:
            return "Bullish", score
        if score <= -2:
            return "Bearish", score
        return "Neutral", score
    except Exception:
        return "N/A", 0


def _normalize_yahoo_frame(frame):
    """Normalize one Yahoo OHLCV frame, including either MultiIndex layout."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    needed = ["Open", "High", "Low", "Close"]
    if isinstance(df.columns, pd.MultiIndex):
        best_level = None
        best_count = -1
        for level in range(df.columns.nlevels):
            vals = [str(v).strip() for v in df.columns.get_level_values(level)]
            count = sum(col in vals for col in needed)
            if count > best_count:
                best_count, best_level = count, level
        if best_level is not None and best_count >= 4:
            df.columns = df.columns.get_level_values(best_level)
    # Case-insensitive normalization protects against provider/schema changes.
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.title()
    df = df.rename(columns=rename)
    if not all(col in df.columns for col in needed):
        return pd.DataFrame()
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    for col in needed + ["Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed)
    df["Volume"] = df["Volume"].fillna(0)
    return df[needed + ["Volume"]].loc[~df.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_yahoo_batch(tickers, interval, period):
    """Fetch many Yahoo tickers in one HTTP batch and return {ticker: OHLCV}.

    This replaces 60 serial downloads in the NIFTY-30 scanner with two batch
    requests (one intraday, one daily). Failed symbols stay empty; no values are
    fabricated or copied from another ticker.
    """
    ticker_list = tuple(str(t).strip() for t in tickers if str(t).strip())
    empty_result = {ticker: pd.DataFrame() for ticker in ticker_list}
    if not ticker_list:
        return empty_result
    try:
        try:
            raw = yf.download(
                tickers=list(ticker_list), period=period, interval=interval,
                group_by="ticker", auto_adjust=False, progress=False,
                threads=True, timeout=8, multi_level_index=True
            )
        except TypeError:
            raw = yf.download(
                tickers=list(ticker_list), period=period, interval=interval,
                group_by="ticker", auto_adjust=False, progress=False,
                threads=True, timeout=8
            )
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            log_app_error("fetch_yahoo_batch", f"Yahoo batch returned no rows for {len(ticker_list)} tickers at {interval}/{period}.")
            return empty_result

        results = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in ticker_list:
                subset = None
                for level in range(raw.columns.nlevels):
                    values = [str(v) for v in raw.columns.get_level_values(level)]
                    if ticker in values:
                        try:
                            actual = raw.columns.get_level_values(level)[values.index(ticker)]
                            subset = raw.xs(actual, axis=1, level=level, drop_level=True)
                            break
                        except Exception:
                            continue
                results[ticker] = _normalize_yahoo_frame(subset) if subset is not None else pd.DataFrame()
        elif len(ticker_list) == 1:
            results[ticker_list[0]] = _normalize_yahoo_frame(raw)
        else:
            # Some provider responses flatten a batch unexpectedly. Do not map
            # one combined series to every company; leave symbols explicitly N/A.
            log_app_error("fetch_yahoo_batch", f"Unexpected single-level columns for multi-ticker request ({interval}).")
            results = dict(empty_result)
        return results
    except Exception as exc:
        log_app_error("fetch_yahoo_batch", exc)
        return empty_result


def _market_quote_from_frames(name, ticker_used, intraday_frame, daily_frame, fallback_label=None):
    """Build one market quote using the newest 5-minute bar, else latest daily close.

    Daily % change is calculated against the last completed daily bar before the
    current quote date. Missing quotes remain NaN; values are never copied from
    other indices or companies.
    """
    empty = {
        "Name": name, "Ticker": ticker_used, "Price": np.nan,
        "ChangePct": np.nan, "Status": "No data returned", "AsOf": "N/A",
        "DisplayName": fallback_label or name,
    }
    try:
        intraday = _normalize_yahoo_frame(intraday_frame)
        daily = _normalize_yahoo_frame(daily_frame)
        intraday = intraday.dropna(subset=["Close"]) if not intraday.empty else intraday
        daily = daily.dropna(subset=["Close"]) if not daily.empty else daily

        use_intraday = not intraday.empty
        if use_intraday and not daily.empty:
            try:
                intraday_date = pd.Timestamp(intraday.index[-1]).date()
                daily_latest_date = pd.Timestamp(daily.index[-1]).date()
                # If the latest intraday bar is older than a newer EOD bar,
                # show the newer EOD value instead of a stale intraday quote.
                if daily_latest_date > intraday_date:
                    use_intraday = False
            except Exception:
                pass

        if use_intraday:
            price = float(intraday["Close"].iloc[-1])
            quote_timestamp = pd.Timestamp(intraday.index[-1])
            quote_date = quote_timestamp.date()
            previous_close = np.nan
            if not daily.empty:
                older_daily = daily[[pd.Timestamp(x).date() < quote_date for x in daily.index]]
                if not older_daily.empty:
                    previous_close = float(older_daily["Close"].iloc[-1])
                elif len(daily) >= 2:
                    previous_close = float(daily["Close"].iloc[-2])
            status = "Latest 5m bar"
        elif not daily.empty:
            price = float(daily["Close"].iloc[-1])
            quote_timestamp = pd.Timestamp(daily.index[-1])
            previous_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else np.nan
            status = "Latest daily close"
        else:
            return empty

        change_pct = np.nan
        if pd.notna(previous_close) and previous_close != 0:
            change_pct = ((price - previous_close) / previous_close) * 100.0

        try:
            as_of = quote_timestamp.strftime("%Y-%m-%d %H:%M %Z").strip()
        except Exception:
            as_of = str(quote_timestamp)
        return {
            "Name": name,
            "Ticker": ticker_used,
            "Price": round(price, 4),
            "ChangePct": round(float(change_pct), 2) if pd.notna(change_pct) else np.nan,
            "Status": status,
            "AsOf": as_of,
            "DisplayName": fallback_label or name,
        }
    except Exception as exc:
        log_app_error("market_board_quote", f"{name} ({ticker_used}): {exc}")
        return empty


def build_live_market_board_rows():
    """Fetch global/macro + major Indian quotes in two batched Yahoo requests.

    5-minute bars provide the newest available quote when the market/provider
    supplies them. Latest daily close is a clearly marked fallback outside
    market hours or where an intraday instrument is unsupported.
    """
    instruments = GLOBAL_MACRO_ASSETS + MAJOR_INDIAN_MARKET_STOCKS
    tickers = tuple(dict.fromkeys(
        ticker for item in instruments for ticker in item.get("tickers", ())
    ))
    intraday_data = fetch_yahoo_batch(tickers, "5m", "5d")
    daily_data = fetch_yahoo_batch(tickers, "1d", "5d")

    global_rows = []
    indian_rows = []
    for item in instruments:
        chosen_ticker = None
        chosen_row = None
        candidates = item.get("tickers", ())
        for candidate_index, candidate in enumerate(candidates):
            intraday = intraday_data.get(candidate, pd.DataFrame())
            daily = daily_data.get(candidate, pd.DataFrame())
            if (isinstance(intraday, pd.DataFrame) and not intraday.empty) or (isinstance(daily, pd.DataFrame) and not daily.empty):
                chosen_ticker = candidate
                label_override = None
                if item.get("fallback_name") and (candidate_index > 0 or item.get("always_fallback_label")):
                    label_override = item["fallback_name"]
                chosen_row = _market_quote_from_frames(
                    item["name"], candidate, intraday, daily, fallback_label=label_override
                )
                break
        if chosen_row is None:
            chosen_row = _market_quote_from_frames(
                item["name"], candidates[0] if candidates else "N/A",
                pd.DataFrame(), pd.DataFrame()
            )
        (global_rows if item in GLOBAL_MACRO_ASSETS else indian_rows).append(chosen_row)
    return global_rows, indian_rows


def _render_market_quote_grid(rows, columns_per_row=5):
    """Render compact market metric cards with per-instrument freshness labels."""
    if not rows:
        st.info("No instruments are configured for this market board.")
        return
    for start in range(0, len(rows), columns_per_row):
        chunk = rows[start:start + columns_per_row]
        cols = st.columns(columns_per_row)
        for col, item in zip(cols, chunk):
            with col:
                price = item.get("Price", np.nan)
                pct = item.get("ChangePct", np.nan)
                price_text = f"{float(price):,.2f}" if pd.notna(price) else "N/A"
                delta_text = f"{float(pct):+.2f}%" if pd.notna(pct) else None
                st.metric(str(item.get("DisplayName") or item.get("Name")), price_text, delta=delta_text)
                st.caption(
                    f"{item.get('Ticker', 'N/A')} · {item.get('Status', 'N/A')}\n"
                    f"As of: {item.get('AsOf', 'N/A')}"
                )


def render_live_global_india_market_board():
    """Render cached market board; network fetches are explicit or at 60s cadence.

    Previously this function fetched two large Yahoo batches on every Streamlit
    rerun before the main chart. A slow Yahoo response could make the entire app
    look frozen. Now the board never blocks initial dashboard rendering.
    """
    st.divider()
    st.header("Global / Macro Performance")
    st.caption(
        "Latest available Yahoo quote: 5-minute bars where available, otherwise latest daily close. "
        "This independent board does not block the selected Angel/Yahoo chart feed."
    )
    refresh_clicked = st.button("Refresh Global + Indian Market Data", key="refresh_global_india_market_board")
    auto_on = bool(st.session_state.get("auto_refresh_enabled", False))
    now_epoch = pytime.time()
    last_fetch_epoch = float(st.session_state.get("market_board_last_fetch_epoch", 0) or 0)
    auto_due = auto_on and (now_epoch - last_fetch_epoch >= 60)
    should_fetch = refresh_clicked or auto_due

    if should_fetch:
        try:
            fetch_yahoo_batch.clear()
            with st.spinner("Loading global indices and Indian stocks (bounded Yahoo requests)…"):
                global_rows, indian_rows = build_live_market_board_rows()
            st.session_state["market_board_cache"] = {
                "global": global_rows, "indian": indian_rows,
                "checked_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
            }
            st.session_state["market_board_last_fetch_epoch"] = now_epoch
        except Exception as exc:
            log_app_error("render_live_global_india_market_board", exc)
            st.warning(f"Market board refresh failed: {type(exc).__name__}. Main chart and scanner remain independent.")

    cached = st.session_state.get("market_board_cache")
    if not isinstance(cached, dict):
        st.info("Market overview has not been fetched yet. Press Refresh Global + Indian Market Data. The main dashboard will continue loading without waiting for this optional board.")
        global_rows, indian_rows = [], []
    else:
        global_rows = cached.get("global", [])
        indian_rows = cached.get("indian", [])
        st.caption(f"Last successful board refresh: {cached.get('checked_at', 'N/A')} | Auto refresh checks this board at most once per 60 seconds.")

    if global_rows:
        _render_market_quote_grid(global_rows, columns_per_row=5)
        unavailable_global = sum(pd.isna(row.get("Price", np.nan)) for row in global_rows)
        if unavailable_global:
            st.warning(f"{unavailable_global} global/macro instruments returned no quote. They remain N/A.")
    st.subheader("🇮🇳 Major Indian Stocks — Banks, IT, Auto and Index Leaders")
    if indian_rows:
        _render_market_quote_grid(indian_rows, columns_per_row=5)
        unavailable_indian = sum(pd.isna(row.get("Price", np.nan)) for row in indian_rows)
        if unavailable_indian:
            st.warning(f"{unavailable_indian} Indian stocks returned no quote in this Yahoo batch.")
    else:
        st.caption("Indian stock board will appear after the first manual refresh.")
    st.caption("Source: Yahoo Finance. Yahoo latency varies by instrument. GIFT Nifty proxy is labelled separately and is not represented as a futures quote.")


@st.cache_data(ttl=60, show_spinner=False)
def build_nifty_top30_scanner():
    rows = []
    intervals = {"3m": "3min", "5m": "5min", "10m": "10min", "15m": "15min"}
    symbols = tuple(NIFTY_TOP30.values())

    # Two batched requests instead of 60 sequential requests.
    intraday_data = fetch_yahoo_batch(symbols, "1m", "7d")
    daily_data = fetch_yahoo_batch(symbols, "1d", "5d")

    for company, ticker_symbol in NIFTY_TOP30.items():
        try:
            intraday = intraday_data.get(ticker_symbol, pd.DataFrame())
            daily = daily_data.get(ticker_symbol, pd.DataFrame())
            cmp = np.nan
            pct = np.nan
            status = "N/A"

            if not intraday.empty:
                cmp = float(intraday["Close"].iloc[-1])
                if len(daily) >= 2:
                    previous_close = float(daily["Close"].iloc[-2])
                    if previous_close:
                        pct = (cmp - previous_close) / previous_close * 100.0
                status = "Intraday OHLC"
            elif not daily.empty:
                # Daily close is a transparent last-known CMP fallback. Trend
                # votes remain N/A because intraday bars were not available.
                daily_close = daily["Close"].dropna()
                if len(daily_close) >= 1:
                    cmp = float(daily_close.iloc[-1])
                    if len(daily_close) >= 2 and float(daily_close.iloc[-2]) != 0:
                        pct = (cmp - float(daily_close.iloc[-2])) / float(daily_close.iloc[-2]) * 100.0
                    status = "Daily close only"

            item = {
                "Company": company,
                "CMP": round(cmp, 2) if pd.notna(cmp) else np.nan,
                "% Change": round(pct, 2) if pd.notna(pct) else np.nan,
                "3m": "N/A", "5m": "N/A", "10m": "N/A", "15m": "N/A",
                "Overall": "N/A", "Score": 0, "Data Status": status,
            }
            if not intraday.empty:
                scores = []
                for label, rule in intervals.items():
                    tf_data = resample_ohlcv(intraday, rule)
                    trend, score = classify_stock_timeframe(tf_data)
                    item[label] = trend
                    scores.append(score if trend != "N/A" else 0)
                item["Score"] = int(sum(scores))
                valid = [item[k] for k in intervals if item[k] in ("Bullish", "Bearish", "Neutral")]
                bulls = valid.count("Bullish")
                bears = valid.count("Bearish")
                item["Overall"] = "Bullish" if bulls > bears else "Bearish" if bears > bulls else "Neutral"
            rows.append(item)
        except Exception as exc:
            log_app_error("nifty_top30_scanner", f"{company}: {exc}")
            rows.append({
                "Company": company, "CMP": np.nan, "% Change": np.nan,
                "3m": "N/A", "5m": "N/A", "10m": "N/A", "15m": "N/A",
                "Overall": "N/A", "Score": 0, "Data Status": f"Error: {str(exc)[:100]}"
            })
    return pd.DataFrame(rows)


def scanner_breadth_summary(scanner_df):
    if scanner_df is None or scanner_df.empty:
        return pd.DataFrame()
    result=[]
    for tf in ["3m","5m","10m","15m"]:
        vals=scanner_df[tf].astype(str)
        bull=int((vals=="Bullish").sum()); bear=int((vals=="Bearish").sum()); neutral=int((vals=="Neutral").sum())
        covered=bull+bear+neutral
        result.append({"Timeframe":tf,"Bullish":bull,"Bearish":bear,"Neutral":neutral,
                       "Bullish %":round(100*bull/covered,1) if covered else 0.0,
                       "Bearish %":round(100*bear/covered,1) if covered else 0.0,
                       "Coverage":covered})
    all_vals=scanner_df[["3m","5m","10m","15m"]].values.ravel().tolist()
    valid=[v for v in all_vals if v in ("Bullish","Bearish","Neutral")]
    bull=valid.count("Bullish"); bear=valid.count("Bearish"); neutral=valid.count("Neutral"); total=len(valid)
    result.append({"Timeframe":"Overall (all 4 TF votes)","Bullish":bull,"Bearish":bear,"Neutral":neutral,
                   "Bullish %":round(100*bull/total,1) if total else 0.0,
                   "Bearish %":round(100*bear/total,1) if total else 0.0,"Coverage":total})
    return pd.DataFrame(result)



def blank_nifty_scanner_frame():
    """Show all 30 requested constituents before a network scan is started."""
    return pd.DataFrame([
        {
            "Company": company, "CMP": np.nan, "% Change": np.nan,
            "3m": "N/A", "5m": "N/A", "10m": "N/A", "15m": "N/A",
            "Overall": "N/A", "Score": 0, "Data Status": "Not fetched — press Run Scanner",
        }
        for company in NIFTY_TOP30
    ])


def render_nifty_scanner_result(scanner_df):
    """Render a scanner table only from real provider values; missing values stay N/A."""
    if scanner_df is None or scanner_df.empty:
        st.warning("Yahoo Finance did not return scanner rows. Check network access, provider availability and instrument symbols.")
        return
    breadth = scanner_breadth_summary(scanner_df)
    st.markdown("**NIFTY market breadth by timeframe**")
    st.dataframe(breadth, use_container_width=True, hide_index=True)
    overall_row = breadth[breadth["Timeframe"] == "Overall (all 4 TF votes)"]
    if not overall_row.empty:
        br = overall_row.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Bullish %", f"{br['Bullish %']:.1f}%", f"{int(br['Bullish'])} bullish votes")
        k2.metric("Overall Bearish %", f"{br['Bearish %']:.1f}%", f"{int(br['Bearish'])} bearish votes")
        k3.metric("Neutral Votes", int(br["Neutral"]))
        k4.metric("Total Valid TF Votes", int(br["Coverage"]))
    display_cols = ["Company", "CMP", "% Change", "3m", "5m", "10m", "15m", "Overall", "Score", "Data Status"]
    st.markdown("**Company-wise CMP, daily % change and 3m / 5m / 10m / 15m trend**")
    st.dataframe(scanner_df[display_cols].sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
    bull_tab, bear_tab, neutral_tab = st.tabs(["Bullish Companies", "Bearish Companies", "Neutral / N/A"])
    with bull_tab:
        bullish = scanner_df[scanner_df["Overall"] == "Bullish"].sort_values("Score", ascending=False)
        if bullish.empty:
            st.info("No bullish companies confirmed across the selected timeframes.")
        else:
            st.dataframe(bullish[display_cols], use_container_width=True, hide_index=True)
    with bear_tab:
        bearish = scanner_df[scanner_df["Overall"] == "Bearish"].sort_values("Score", ascending=True)
        if bearish.empty:
            st.info("No bearish companies confirmed across the selected timeframes.")
        else:
            st.dataframe(bearish[display_cols], use_container_width=True, hide_index=True)
    with neutral_tab:
        neutral = scanner_df[scanner_df["Overall"].isin(["Neutral", "N/A"])]
        if neutral.empty:
            st.info("No neutral / unavailable companies.")
        else:
            st.dataframe(neutral[display_cols], use_container_width=True, hide_index=True)
    top_bull = scanner_df[scanner_df["Overall"] == "Bullish"].sort_values("Score", ascending=False).head(5)
    top_bear = scanner_df[scanner_df["Overall"] == "Bearish"].sort_values("Score", ascending=True).head(5)
    top_b1, top_b2 = st.columns(2)
    with top_b1:
        st.markdown("**Top Bullish Companies**")
        if not top_bull.empty:
            st.dataframe(top_bull[["Company", "CMP", "% Change", "3m", "5m", "10m", "15m", "Score"]], use_container_width=True, hide_index=True)
        else:
            st.info("No top bullish candidate currently.")
    with top_b2:
        st.markdown("**Top Bearish Companies**")
        if not top_bear.empty:
            st.dataframe(top_bear[["Company", "CMP", "% Change", "3m", "5m", "10m", "15m", "Score"]], use_container_width=True, hide_index=True)
        else:
            st.info("No top bearish candidate currently.")
    if scanner_df["CMP"].notna().sum() == 0:
        st.warning("Scanner ran, but Yahoo returned no valid prices for any constituent. The table shows N/A rather than invented values.")


# =========================================================
# NEWS FILTER API — MARKETaux PRIMARY, NEWSAPI LEGACY
# =========================================================
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"

NEWS_SEARCH_ALIASES = {
    "NIFTY": '"NIFTY 50"|NIFTY|"Indian stock market"|"NSE India"|RBI',
    "BANKNIFTY": '"Bank Nifty"|"NIFTY Bank"|BANKNIFTY|"Indian banks"',
    "SENSEX": 'SENSEX|"BSE Sensex"|"Bombay Stock Exchange"|"Indian stock market"',
    "RELIANCE": '"Reliance Industries"|RELIANCE',
    "HDFCBANK": '"HDFC Bank"|HDFCBANK',
    "ICICIBANK": '"ICICI Bank"|ICICIBANK',
    "SBIN": '"State Bank of India"|SBIN|"SBI Bank"',
    "AXISBANK": '"Axis Bank"|AXISBANK',
    "KOTAKBANK": '"Kotak Mahindra Bank"|KOTAKBANK|"Kotak Bank"',
    "TCS": '"Tata Consultancy Services"|TCS',
    "INFY": '"Infosys Limited"|INFY|Infosys',
    "HCLTECH": '"HCL Technologies"|HCLTECH',
    "WIPRO": 'Wipro|WIPRO',
    "TECHM": '"Tech Mahindra"|TECHM',
    "TMPV": '"Tata Motors"|TMPV',
    "MARUTI": '"Maruti Suzuki"|MARUTI',
    "M&M": '"Mahindra and Mahindra"|"Mahindra & Mahindra"|M&M',
    "BAJAJ-AUTO": '"Bajaj Auto"|BAJAJ-AUTO',
    "HEROMOTOCO": '"Hero MotoCorp"|HEROMOTOCO',
    "ITC": '"ITC Limited"|ITC India',
    "LT": '"Larsen and Toubro"|"Larsen & Toubro"',
    "BHARTIARTL": '"Bharti Airtel"|BHARTIARTL',
    "SUNPHARMA": '"Sun Pharmaceutical"|SUNPHARMA|"Sun Pharma"',
    "TITAN": '"Titan Company"|TITAN',
    "NTPC": 'NTPC|"National Thermal Power Corporation"',
    "BEL": '"Bharat Electronics"|BEL India',
    "ONGC": 'ONGC|"Oil and Natural Gas Corporation"',
    "ADANIPORTS": '"Adani Ports"|ADANIPORTS',
    "POWERGRID": '"Power Grid Corporation"|POWERGRID',
    "TRENT": 'Trent Limited|TRENT India',
    "JSWSTEEL": '"JSW Steel"|JSWSTEEL',
    "TATASTEEL": '"Tata Steel"|TATASTEEL',
}

NEWS_ENTITY_MATCH_ALIASES = {
    "NIFTY": ("NIFTY", "NIFTY 50"),
    "BANKNIFTY": ("BANKNIFTY", "BANK NIFTY", "NIFTY BANK"),
    "SENSEX": ("SENSEX", "BSE SENSEX"),
    "RELIANCE": ("RELIANCE", "RELIANCE INDUSTRIES"),
    "HDFCBANK": ("HDFCBANK", "HDFC BANK"),
    "ICICIBANK": ("ICICIBANK", "ICICI BANK"),
    "SBIN": ("SBIN", "STATE BANK OF INDIA"),
    "AXISBANK": ("AXISBANK", "AXIS BANK"),
    "KOTAKBANK": ("KOTAKBANK", "KOTAK MAHINDRA BANK", "KOTAK BANK"),
    "TCS": ("TCS", "TATA CONSULTANCY SERVICES"),
    "INFY": ("INFY", "INFOSYS"),
    "HCLTECH": ("HCLTECH", "HCL TECHNOLOGIES"),
    "WIPRO": ("WIPRO",),
    "TECHM": ("TECHM", "TECH MAHINDRA"),
    "TMPV": ("TMPV", "TATA MOTORS"),
    "MARUTI": ("MARUTI", "MARUTI SUZUKI"),
    "M&M": ("M&M", "MAHINDRA AND MAHINDRA", "MAHINDRA & MAHINDRA"),
    "BAJAJ-AUTO": ("BAJAJ-AUTO", "BAJAJ AUTO"),
    "HEROMOTOCO": ("HEROMOTOCO", "HERO MOTOCORP"),
    "ITC": ("ITC", "ITC LIMITED"),
    "LT": ("LARSEN AND TOUBRO", "LARSEN & TOUBRO"),
    "BHARTIARTL": ("BHARTIARTL", "BHARTI AIRTEL"),
    "SUNPHARMA": ("SUNPHARMA", "SUN PHARMACEUTICAL", "SUN PHARMA"),
    "TITAN": ("TITAN", "TITAN COMPANY"),
    "NTPC": ("NTPC",),
    "BEL": ("BEL", "BHARAT ELECTRONICS"),
    "ONGC": ("ONGC", "OIL AND NATURAL GAS CORPORATION"),
    "ADANIPORTS": ("ADANIPORTS", "ADANI PORTS"),
    "POWERGRID": ("POWERGRID", "POWER GRID CORPORATION"),
    "TRENT": ("TRENT", "TRENT LIMITED"),
    "JSWSTEEL": ("JSWSTEEL", "JSW STEEL"),
    "TATASTEEL": ("TATASTEEL", "TATA STEEL"),
}

NEWS_HIGH_IMPACT_TERMS = (
    "rbi policy", "reserve bank", "monetary policy", "interest rate", "rate hike",
    "rate cut", "inflation", "quarterly results", "earnings", "profit warning",
    "revenue warning", "guidance", "sebi", "regulatory action", "investigation",
    "fraud", "default", "bankruptcy", "downgrade", "credit rating", "merger",
    "acquisition", "sanction", "tariff", "war", "geopolitical", "terror attack",
    "market crash", "trading halt", "trading suspension", "liquidity crisis",
    "government budget", "union budget", "fomc", "federal reserve", "ceasefire",
)


def _read_news_secret(secret_name):
    """Read optional deployment secret without exposing it in the UI or logs."""
    value = os.getenv(secret_name, "")
    if value:
        return str(value).strip()
    try:
        value = st.secrets.get(secret_name, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def _news_failure(provider, message):
    # Fail closed when the news filter is explicitly enabled but the provider
    # cannot be checked. This prevents silently bypassing the selected filter.
    return {
        "allow_trade": False,
        "message": str(message),
        "events": [],
        "provider": provider,
        "status": "error",
        "article_count": 0,
        "sentiment_bias": "N/A",
        "average_sentiment": None,
    }


def _news_parse_datetime(value):
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def test_news_provider_connection(api_key, provider="Marketaux"):
    """Make one uncached, minimal authenticated request to verify a news API token.

    A successful HTTP 200 with the expected response shape means the token was
    accepted even when the provider currently has zero matching articles.
    Never return or log the token or a request URL containing it.
    """
    provider_name = str(provider or "Marketaux").strip()
    provider_key = provider_name.lower()
    token = str(api_key or "").strip()
    if not token:
        return {
            "ok": False,
            "provider": provider_name,
            "http_status": None,
            "message": "Enter the API token first, then press Test API Connection.",
            "articles": None,
        }

    try:
        if provider_key == "marketaux":
            response = requests.get(
                MARKETAUX_NEWS_URL,
                params={
                    "api_token": token,
                    "search": "NIFTY",
                    "language": "en",
                    "limit": 1,
                },
                timeout=10,
            )
            expected_list_key = "data"
        elif provider_key in ("newsapi", "newsapi (legacy)"):
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "apiKey": token,
                    "q": "NIFTY",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 1,
                },
                timeout=10,
            )
            expected_list_key = "articles"
        else:
            return {
                "ok": False,
                "provider": provider_name,
                "http_status": None,
                "message": "Unsupported news provider selected.",
                "articles": None,
            }

        status_code = int(response.status_code)
        try:
            payload = response.json()
        except Exception:
            payload = None

        if status_code == 200:
            if not isinstance(payload, dict):
                return {
                    "ok": False, "provider": provider_name,
                    "http_status": status_code,
                    "message": "Provider returned HTTP 200 but the response was not valid JSON.",
                    "articles": None,
                }
            if provider_key == "marketaux":
                items = payload.get("data")
                valid_shape = isinstance(items, list)
            else:
                valid_shape = payload.get("status") == "ok" and isinstance(payload.get("articles"), list)
                items = payload.get("articles") if isinstance(payload.get("articles"), list) else None
            if not valid_shape:
                return {
                    "ok": False, "provider": provider_name,
                    "http_status": status_code,
                    "message": "Provider accepted the HTTP request but returned an unexpected response format.",
                    "articles": None,
                }
            count = len(items or [])
            return {
                "ok": True,
                "provider": provider_name,
                "http_status": status_code,
                "message": (
                    f"{provider_name} connection successful — API token accepted (HTTP 200). "
                    f"Test query returned {count} article(s). Zero articles can still mean the token is valid."
                ),
                "articles": count,
            }

        if status_code == 401:
            message = "HTTP 401 — API token is invalid or not authorized. Check the token in your provider account."
        elif status_code == 402:
            message = "HTTP 402 — provider usage limit or plan quota reached. Check your account plan/usage."
        elif status_code == 403:
            message = "HTTP 403 — this endpoint or feature is not allowed by the current provider plan."
        elif status_code == 429:
            message = "HTTP 429 — provider rate limit reached. Wait before testing again."
        elif status_code == 400:
            message = "HTTP 400 — provider rejected the test query parameters. Check the API response and plan."
        elif status_code >= 500:
            message = f"HTTP {status_code} — provider server is temporarily unavailable. Try again later."
        else:
            message = f"Provider returned HTTP {status_code}. Check the API token, plan and provider status."

        # Include only a short provider message; redact the token defensively.
        if isinstance(payload, dict):
            err = payload.get("error") or payload.get("message")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")
            else:
                detail = str(err or "")
            detail = detail.replace(token, "[redacted]").strip()[:180]
            if detail:
                message += " Provider detail: " + detail
        return {
            "ok": False,
            "provider": provider_name,
            "http_status": status_code,
            "message": message,
            "articles": None,
        }
    except requests.Timeout:
        return {
            "ok": False, "provider": provider_name, "http_status": None,
            "message": "Connection test timed out. Check hosting network access and try again.",
            "articles": None,
        }
    except requests.RequestException as exc:
        # Do not display str(exc): request exceptions may contain the full URL/token.
        return {
            "ok": False, "provider": provider_name, "http_status": None,
            "message": f"Network error during connection test ({type(exc).__name__}). Check internet access and retry.",
            "articles": None,
        }
    except Exception as exc:
        return {
            "ok": False, "provider": provider_name, "http_status": None,
            "message": f"Connection test could not be completed ({type(exc).__name__}).",
            "articles": None,
        }


@st.cache_data(ttl=300, show_spinner=False)
def get_news_events(api_key, symbol, provider="Marketaux", lookback_hours=6, article_limit=10):
    """Fetch recent financial news and derive a transparent event-risk summary.

    Marketaux is the preferred provider. NewsAPI remains available as a legacy
    option. API tokens are only request parameters and are never written to disk.
    """
    provider = str(provider or "Marketaux").strip()
    provider_key = provider.lower()
    symbol = str(symbol or "NIFTY").strip().upper()
    try:
        lookback_hours = max(1, min(int(lookback_hours), 72))
    except Exception:
        lookback_hours = 6
    try:
        article_limit = max(1, min(int(article_limit), 50))
    except Exception:
        article_limit = 10

    if not str(api_key or "").strip():
        return _news_failure(provider, f"{provider} API token is missing. Enter the key in News Filter API settings or configure it in Streamlit secrets.")

    now_utc = datetime.now(timezone.utc)
    after_utc = now_utc - timedelta(hours=lookback_hours)
    articles = []

    try:
        if provider_key == "marketaux":
            search_terms = NEWS_SEARCH_ALIASES.get(symbol, f'"{symbol}"|{symbol}')
            # Marketaux /v1/news/all accepts api_token, search, country,
            # language, published_after, group_similar and limit parameters.
            params = {
                "api_token": str(api_key).strip(),
                "search": search_terms,
                "countries": "in",
                "language": "en",
                "published_after": after_utc.strftime("%Y-%m-%dT%H:%M"),
                "group_similar": "true",
                "limit": article_limit,
            }
            response = requests.get(MARKETAUX_NEWS_URL, params=params, timeout=6)
            if response.status_code != 200:
                detail = ""
                try:
                    payload = response.json()
                    err = payload.get("error") or payload.get("errors") or payload.get("message")
                    if isinstance(err, (dict, list)):
                        detail = json.dumps(err, ensure_ascii=False)[:240]
                    elif err:
                        detail = str(err)[:240]
                except Exception:
                    detail = ""
                if detail and str(api_key).strip():
                    detail = detail.replace(str(api_key).strip(), "[redacted]")
                suffix = f": {detail}" if detail else ""
                return _news_failure("Marketaux", f"Marketaux API returned HTTP {response.status_code}{suffix}. Check the token, plan limit and API availability.")
            payload = response.json()
            if not isinstance(payload, dict):
                return _news_failure("Marketaux", "Marketaux returned an unexpected response format.")
            articles = payload.get("data") or []
            if not isinstance(articles, list):
                return _news_failure("Marketaux", "Marketaux response did not contain a valid article list.")

        elif provider_key in ("newsapi", "newsapi (legacy)"):
            params = {
                "q": symbol,
                "sortBy": "publishedAt",
                "pageSize": min(article_limit, 100),
                "language": "en",
                "from": after_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                "apiKey": str(api_key).strip(),
            }
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=6)
            if response.status_code != 200:
                try:
                    payload = response.json()
                    detail = str(payload.get("message") or payload.get("code") or "")[:240]
                except Exception:
                    detail = ""
                suffix = f": {detail}" if detail else ""
                return _news_failure("NewsAPI", f"NewsAPI returned HTTP {response.status_code}{suffix}.")
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                return _news_failure("NewsAPI", "NewsAPI returned an unsuccessful or unexpected response.")
            articles = payload.get("articles") or []
        else:
            return _news_failure(provider, "Unsupported news provider selected.")

        events = []
        sentiment_values = []
        high_impact_count = 0
        strong_negative_count = 0
        for article in articles[:article_limit]:
            title = str(article.get("title") or "Untitled news article").strip()
            description = str(article.get("description") or article.get("snippet") or "").strip()
            published = article.get("published_at") or article.get("publishedAt")
            published_dt = _news_parse_datetime(published)
            is_recent = bool(published_dt and after_utc <= published_dt <= now_utc + timedelta(minutes=5))

            if provider_key == "marketaux":
                source = str(article.get("source") or "Marketaux")
                article_url = str(article.get("url") or "")
                entity_rows = article.get("entities") or []
                entity_labels = []
                selected_scores = []
                relevant_aliases = NEWS_ENTITY_MATCH_ALIASES.get(symbol, (symbol,))
                for entity in entity_rows:
                    if not isinstance(entity, dict):
                        continue
                    entity_symbol = str(entity.get("symbol") or "").upper()
                    entity_name = str(entity.get("name") or "")
                    entity_labels.append(entity_symbol or entity_name)
                    entity_identity = f"{entity_symbol} {entity_name}".upper()
                    is_selected_entity = any(alias.upper() in entity_identity for alias in relevant_aliases)
                    raw_score = entity.get("sentiment_score")
                    try:
                        score = float(raw_score)
                        if is_selected_entity and math.isfinite(score) and -1.0 <= score <= 1.0:
                            selected_scores.append(score)
                    except (TypeError, ValueError):
                        pass
                sentiment_score = float(np.mean(selected_scores)) if selected_scores else None
            else:
                source_obj = article.get("source") or {}
                source = str(source_obj.get("name") or "NewsAPI") if isinstance(source_obj, dict) else "NewsAPI"
                article_url = str(article.get("url") or "")
                entity_labels = []
                sentiment_score = None

            combined_text = f"{title} {description}".lower()
            matched_terms = [term for term in NEWS_HIGH_IMPACT_TERMS if term in combined_text]
            negative_sentiment = sentiment_score is not None and sentiment_score <= -0.55
            high_impact = bool(is_recent and matched_terms)
            strong_negative = bool(is_recent and negative_sentiment)
            risk_flag = high_impact or strong_negative
            if high_impact:
                high_impact_count += 1
            if strong_negative:
                strong_negative_count += 1
            if is_recent and sentiment_score is not None:
                sentiment_values.append(sentiment_score)

            events.append({
                "Title": title,
                "Source": source,
                "Published (UTC)": published or "N/A",
                "Sentiment": round(sentiment_score, 3) if sentiment_score is not None else "N/A",
                "Entities": ", ".join(dict.fromkeys([x for x in entity_labels if x])) or "N/A",
                "Risk Flag": "HIGH IMPACT / NEGATIVE" if risk_flag else "Normal",
                "Risk Reason": (", ".join(matched_terms) if high_impact else ("Strong negative entity sentiment" if strong_negative else "")),
                "Article": article_url,
                "Description": description[:500],
                "_recent": is_recent,
            })

        # Keep the visible event list recent; the API already filters by time,
        # but this extra check avoids provider-side date boundary surprises.
        recent_events = [event for event in events if event.get("_recent")]
        for event in recent_events:
            event.pop("_recent", None)
        average_sentiment = float(np.mean(sentiment_values)) if sentiment_values else None
        if average_sentiment is None:
            sentiment_bias = "N/A"
        elif average_sentiment >= 0.15:
            sentiment_bias = "Positive"
        elif average_sentiment <= -0.15:
            sentiment_bias = "Negative"
        else:
            sentiment_bias = "Neutral"

        allow_trade = not any(
            str(event.get("Risk Flag", "Normal")) != "Normal" for event in recent_events
        )
        if high_impact_count or strong_negative_count:
            message = f"News risk detected: {high_impact_count} high-impact headline(s), {strong_negative_count} strongly negative sentiment item(s) within {lookback_hours}h. New signal entries are blocked by the news filter."
        elif recent_events:
            message = f"{len(recent_events)} recent relevant article(s); no configured high-impact or strong-negative risk flag detected."
        else:
            message = f"No recent relevant articles returned for {symbol} in the last {lookback_hours}h."

        return {
            "allow_trade": allow_trade,
            "message": message,
            "events": recent_events,
            "provider": "Marketaux" if provider_key == "marketaux" else "NewsAPI",
            "status": "ok",
            "article_count": len(recent_events),
            "sentiment_bias": sentiment_bias,
            "average_sentiment": round(average_sentiment, 3) if average_sentiment is not None else None,
        }
    except requests.Timeout:
        log_app_error("get_news_events", f"{provider} request timed out")
        return _news_failure(provider, f"{provider} request timed out. News filter is unavailable; new signal entries are paused.")
    except requests.RequestException as exc:
        # Avoid logging request URLs: providers receive API tokens in query params.
        log_app_error("get_news_events", f"{provider} network error: {type(exc).__name__}")
        return _news_failure(provider, f"{provider} network error ({type(exc).__name__}). News filter is unavailable; new signal entries are paused.")
    except Exception as exc:
        log_app_error("get_news_events", f"{provider} parsing error: {type(exc).__name__}")
        return _news_failure(provider, f"{provider} response could not be processed ({type(exc).__name__}). News filter is unavailable; new signal entries are paused.")


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
        session.get("https://www.nseindia.com", headers=headers, timeout=4)
        resp = session.get(url, headers=headers, timeout=4)
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
    if selected_symbol == "SENSEX":
        return None  # NSE's NIFTY option-chain PCR is not a SENSEX option-chain value.
    return "NIFTY"


def parse_option_chain_metrics(symbol="NIFTY"):
    data = get_option_chain_snapshot(symbol)
    empty = {"pcr": None, "oi_bias": "N/A", "long_unwinding": False, "short_covering": False,
             "call_oi": 0, "put_oi": 0, "call_change_oi": 0, "put_change_oi": 0}
    if not data:
        return empty
    try:
        total_ce = total_pe = ce_change_oi = pe_change_oi = 0.0
        records = data.get("records", {}).get("data", [])
        for item in records:
            ce = item.get("CE") or {}
            pe = item.get("PE") or {}
            total_ce += float(ce.get("openInterest") or 0)
            total_pe += float(pe.get("openInterest") or 0)
            ce_change_oi += float(ce.get("changeinOpenInterest") or 0)
            pe_change_oi += float(pe.get("changeinOpenInterest") or 0)
        pcr = round(total_pe / total_ce, 2) if total_ce > 0 else None
        # The old version called Put OI > Call OI "Bullish" and then scored both
        # PCR and OI, double-counting the same information. Keep OI as a descriptive
        # change-in-OI comparison and do not turn it into a directional score.
        oi_total_change = abs(ce_change_oi) + abs(pe_change_oi)
        if oi_total_change <= 0 or abs(pe_change_oi - ce_change_oi) < 0.20 * oi_total_change:
            oi_bias = "Neutral"
        elif pe_change_oi > ce_change_oi:
            oi_bias = "Put OI change higher"
        else:
            oi_bias = "Call OI change higher"
        return {"pcr": pcr, "oi_bias": oi_bias, "long_unwinding": False, "short_covering": False,
                "call_oi": int(total_ce), "put_oi": int(total_pe),
                "call_change_oi": int(ce_change_oi), "put_change_oi": int(pe_change_oi)}
    except Exception as exc:
        log_app_error("parse_option_chain_metrics", exc)
        return empty


# =========================================================
# SIGNAL ANALYSIS HELPERS
# =========================================================
def detect_gap_filter(df):
    """Detect an actual session gap, not the open-to-previous-close noise of every intraday candle."""
    neutral = {"gap_up": False, "gap_down": False, "gap_pct": 0.0, "allow_trade": True}
    try:
        if df is None or len(df) < 2 or not {"Open", "Close"}.issubset(df.columns):
            return neutral
        idx = pd.to_datetime(df.index, errors="coerce")
        work = df.loc[~idx.isna(), ["Open", "Close"]].copy()
        work.index = idx[~idx.isna()]
        if len(work) < 2:
            return neutral
        dates = pd.Series([x.date() for x in work.index], index=work.index)
        unique_dates = list(pd.unique(dates))
        if len(unique_dates) >= 2:
            current_date = unique_dates[-1]
            previous_date = unique_dates[-2]
            current_session = work.loc[dates == current_date]
            previous_session = work.loc[dates == previous_date]
            current_open = float(current_session["Open"].iloc[0])
            previous_close = float(previous_session["Close"].iloc[-1])
        else:
            # With only one intraday session in the sample, there is no reliable
            # previous-session close. Do not misclassify an ordinary candle as a gap.
            if len(work) >= 2 and (work.index[-1] - work.index[0]).total_seconds() >= 20 * 3600:
                current_open = float(work["Open"].iloc[-1])
                previous_close = float(work["Close"].iloc[-2])
            else:
                return neutral
        if not np.isfinite(previous_close) or previous_close <= 0 or not np.isfinite(current_open):
            return neutral
        gap_pct = ((current_open - previous_close) / previous_close) * 100.0
        return {
            "gap_up": gap_pct >= GAP_FILTER_PCT,
            "gap_down": gap_pct <= -GAP_FILTER_PCT,
            "gap_pct": round(float(gap_pct), 2),
            "allow_trade": abs(gap_pct) < 2.5,
        }
    except Exception as exc:
        log_app_error("detect_gap_filter", exc)
        return neutral


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

        is_breakout = bool(latest.get("Breakout", False)) if pd.notna(latest.get("Breakout", False)) else False
        is_breakdown = bool(latest.get("Breakdown", False)) if pd.notna(latest.get("Breakdown", False)) else False
        if is_breakout:
            regime = "Breakout"
        elif is_breakdown:
            regime = "Breakdown"
        elif trending and high_volatility:
            regime = "Trending High Volatility"
        elif trending:
            regime = "Trending Up" if ema20 >= ema50 else "Trending Down"
        elif sideways:
            regime = "Range Bound"
        elif high_volatility:
            regime = "High Volatility"
        elif atr_pct < 0.5:
            regime = "Low Volatility"
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
        if df is None or df.empty:
            return {"valid": False, "message": "No candle timestamp available"}
        last_ts = pd.Timestamp(df.index[-1])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert("Asia/Kolkata")
        current_t = last_ts.time()
        if current_t < time(9, 20):
            return {"valid": False, "message": "Avoid first 5-10 minutes after open"}
        if time(12, 15) <= current_t <= time(13, 15):
            return {"valid": False, "message": "Lunch low-momentum window"}
        return {"valid": True, "message": "Session timing okay"}
    except Exception:
        return {"valid": True, "message": "Session filter unavailable; not used as a blocker"}


def calculate_win_chance(backtest_win_rate, backtest_trade_count=0, backtest_is_comparable=False):
    """Return only a measured same-strategy historical rate; never a heuristic probability."""
    try:
        count = int(backtest_trade_count or 0)
        rate = float(backtest_win_rate)
        if not backtest_is_comparable or count < 30 or not np.isfinite(rate):
            return None
        return round(float(np.clip(rate, 0.0, 100.0)), 1)
    except Exception:
        return None


def analyze_signal(
    df,
    htf_bias=None,
    timeframe_label="1d",
    backtest_win_rate=0.0,
    mtf_bull_count=0,
    mtf_bear_count=0,
    mtf_available_count=0,
    option_metrics=None,
    news_info=None,
    backtest_trade_count=0,
    backtest_is_comparable=False,
):
    """Explainable multi-factor signal engine.

    Scores are directional evidence, not probabilities. MTF/volume/option data
    are treated as unavailable rather than bearish when providers do not return them.
    """
    empty = df is None or not isinstance(df, pd.DataFrame) or len(df) < 2
    if empty:
        return {
            "close_price": 0.0, "ema20": 0.0, "ema50": 0.0, "ema200": 0.0,
            "vwap": None, "rsi": 50.0, "macd": 0.0, "macd_signal": 0.0,
            "macd_hist": 0.0, "candle_pattern": "N/A", "candle_pattern_bias": "Neutral",
            "candle_pattern_score": 0, "atr": 0.0, "stoch_k": 50.0, "stoch_d": 50.0,
            "adx": 0.0, "support": 0.0, "resistance": 0.0, "volume_spike": False,
            "breakout": False, "breakdown": False, "body_strength": 0.0, "sideways": False,
            "signal_text": "NO TRADE", "signal_type": "neutral", "strength": "Low",
            "confidence": None, "win_chance": None, "win_chance_sample_size": 0,
            "stop_loss": None, "partial_exit": None, "target1": None, "target2": None,
            "target3": None, "trailing_stop": None, "rr_ratio": None, "reasons": ["At least two valid candles are required"],
            "no_trade_reason": "At least two valid candles are required", "no_trade_reasons": ["At least two valid candles are required"],
            "bullish_score": 0, "bearish_score": 0, "directional_bias": "Neutral",
            "session_message": "N/A", "supertrend_direction": 0, "ha_bullish": False,
            "ha_bearish": False, "ha_trend_up": False, "ha_trend_down": False,
            "validation_score": 0, "strategy_grade": "D", "trade_quality_score": 0,
            "market_regime": "Unknown", "trending_market": False, "sideways_market": False,
            "high_volatility_market": False, "atr_pct": 0.0, "gap_up": False, "gap_down": False,
            "gap_pct": 0.0, "mtf_bull_ok": False, "mtf_bear_ok": False, "no_trade_zone": False,
            "oi_bias": "N/A", "pcr": None, "long_unwinding": False, "short_covering": False,
            "news_message": "No analysis available", "trade_allowed": False,
            "setup_confidence_label": "Unavailable", "volume_data_available": False,
            "mtf_available_count": 0, "directional_edge": 0, "ema200_available": False,
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    def num(row, key, default=0.0):
        try:
            value = float(row.get(key, default))
            return value if np.isfinite(value) else float(default)
        except Exception:
            return float(default)

    close_price = num(latest, "Close")
    ema20 = num(latest, "EMA20", close_price)
    ema50 = num(latest, "EMA50", close_price)
    ema200 = num(latest, "EMA200", close_price)
    # EMA200 is not treated as a confirmed long-term trend until 200 real bars exist.
    ema200_available = len(df) >= 200 and pd.notna(latest.get("EMA200", np.nan)) and np.isfinite(ema200)
    prev_ema20 = num(prev, "EMA20", ema20)
    prev_ema50 = num(prev, "EMA50", ema50)
    vwap_value = num(latest, "VWAP", close_price)
    rsi = num(latest, "RSI", 50.0)
    prev_rsi = num(prev, "RSI", 50.0)
    macd = num(latest, "MACD", 0.0)
    macd_signal = num(latest, "SignalLine", 0.0)
    macd_hist = num(latest, "MACD_Hist", macd - macd_signal)
    prev_macd_hist = num(prev, "MACD_Hist", macd_hist)
    atr = max(0.0, num(latest, "ATR", 0.0))
    stoch_k = num(latest, "StochK", 50.0)
    stoch_d = num(latest, "StochD", 50.0)
    adx = max(0.0, num(latest, "ADX", 0.0))
    plus_di = num(latest, "PlusDI", 0.0)
    minus_di = num(latest, "MinusDI", 0.0)
    supertrend_dir = int(num(latest, "SuperTrendDirection", 0))
    volume_spike = bool(latest.get("Volume_Spike", False)) if pd.notna(latest.get("Volume_Spike", False)) else False
    breakout = bool(latest.get("Breakout", False)) if pd.notna(latest.get("Breakout", False)) else False
    breakdown = bool(latest.get("Breakdown", False)) if pd.notna(latest.get("Breakdown", False)) else False
    retest_bull = bool(latest.get("RetestBull", False)) if pd.notna(latest.get("RetestBull", False)) else False
    retest_bear = bool(latest.get("RetestBear", False)) if pd.notna(latest.get("RetestBear", False)) else False
    body_strength = max(0.0, min(1.0, num(latest, "BodyStrength", 0.0)))

    recent_volume = pd.to_numeric(df.get("Volume", pd.Series(0, index=df.index)).tail(20), errors="coerce").fillna(0)
    volume_data_available = bool((recent_volume > 0).sum() >= max(5, int(len(recent_volume) * 0.6)))

    ha_df = convert_to_heikin_ashi(df.tail(10))
    ha_latest = ha_df.iloc[-1]
    ha_prev = ha_df.iloc[-2]
    ha_bullish = bool(ha_latest["HA_Close"] > ha_latest["HA_Open"])
    ha_bearish = bool(ha_latest["HA_Close"] < ha_latest["HA_Open"])
    ha_trend_up = ha_bullish and bool(ha_prev["HA_Close"] > ha_prev["HA_Open"])
    ha_trend_down = ha_bearish and bool(ha_prev["HA_Close"] < ha_prev["HA_Open"])

    # Use completed prior bars for S/R so the current candle cannot define its own level.
    prior_window = df.iloc[max(0, len(df) - 21):-1]
    resistance = float(pd.to_numeric(prior_window["High"], errors="coerce").max()) if not prior_window.empty else close_price
    support = float(pd.to_numeric(prior_window["Low"], errors="coerce").min()) if not prior_window.empty else close_price
    if not np.isfinite(resistance):
        resistance = close_price
    if not np.isfinite(support):
        support = close_price
    distance_to_resistance = ((resistance - close_price) / close_price) * 100 if close_price > 0 else 0.0
    distance_to_support = ((close_price - support) / close_price) * 100 if close_price > 0 else 0.0

    bullish = 0.0
    bearish = 0.0
    reasons = []

    # Trend structure: intentionally separate, modest weights to limit correlated double-counting.
    if ema20 > ema50:
        bullish += 2; reasons.append("EMA20 above EMA50 (+2 bullish)")
    elif ema20 < ema50:
        bearish += 2; reasons.append("EMA20 below EMA50 (+2 bearish)")
    if close_price > ema20:
        bullish += 1; reasons.append("Price above EMA20 (+1 bullish)")
    elif close_price < ema20:
        bearish += 1; reasons.append("Price below EMA20 (+1 bearish)")
    if ema200_available:
        if close_price > ema200:
            bullish += 2; reasons.append("Price above mature EMA200 (+2 bullish)")
        elif close_price < ema200:
            bearish += 2; reasons.append("Price below mature EMA200 (+2 bearish)")
    else:
        reasons.append(f"EMA200 not mature ({len(df)}/200 bars); excluded from score and trend votes")
    if close_price > vwap_value:
        bullish += 1; reasons.append("Price above VWAP (+1 bullish)")
    elif close_price < vwap_value:
        bearish += 1; reasons.append("Price below VWAP (+1 bearish)")
    if supertrend_dir == 1:
        bullish += 2; reasons.append("SuperTrend bullish (+2)")
    elif supertrend_dir == -1:
        bearish += 2; reasons.append("SuperTrend bearish (+2)")

    # Momentum: RSI zones avoid the old 55/45 cliff; MACD is a separate confirmation.
    if rsi >= 55:
        bullish += 2; reasons.append("RSI bullish zone (+2)")
    elif rsi >= 51:
        bullish += 1; reasons.append("RSI mildly bullish (+1)")
    elif rsi <= 45:
        bearish += 2; reasons.append("RSI bearish zone (+2)")
    elif rsi <= 49:
        bearish += 1; reasons.append("RSI mildly bearish (+1)")
    if rsi > prev_rsi + 0.25:
        bullish += 1; reasons.append("RSI rising (+1)")
    elif rsi < prev_rsi - 0.25:
        bearish += 1; reasons.append("RSI falling (+1)")
    if macd > macd_signal:
        bullish += 2; reasons.append("MACD above signal (+2)")
    elif macd < macd_signal:
        bearish += 2; reasons.append("MACD below signal (+2)")
    if macd_hist > prev_macd_hist:
        bullish += 1; reasons.append("MACD histogram improving (+1)")
    elif macd_hist < prev_macd_hist:
        bearish += 1; reasons.append("MACD histogram weakening (+1)")
    if ema20 > prev_ema20:
        bullish += 1; reasons.append("EMA20 slope rising (+1)")
    elif ema20 < prev_ema20:
        bearish += 1; reasons.append("EMA20 slope falling (+1)")
    if adx >= 18:
        if plus_di > minus_di:
            bullish += 2; reasons.append("Directional index +DI dominant (+2)")
        elif minus_di > plus_di:
            bearish += 2; reasons.append("Directional index -DI dominant (+2)")
    else:
        reasons.append("ADX below 18: trend-strength confirmation is weak")
    if stoch_k > stoch_d and stoch_k < 80:
        bullish += 1; reasons.append("Stochastic bullish crossover (+1)")
    elif stoch_k < stoch_d and stoch_k > 20:
        bearish += 1; reasons.append("Stochastic bearish crossover (+1)")

    # Price-action confirmation. Patterns are capped and do not override trend/momentum.
    candle_pattern = str(latest.get("CandlePattern", "No clear pattern"))
    candle_bias = str(latest.get("CandlePatternBias", "Neutral"))
    try:
        candle_pattern_score = int(num(latest, "CandlePatternScore", 0))
    except Exception:
        candle_pattern_score = 0
    if candle_pattern_score > 0 or candle_bias == "Bullish":
        bullish += 1; reasons.append(f"Bullish candle pattern: {candle_pattern} (+1)")
    elif candle_pattern_score < 0 or candle_bias == "Bearish":
        bearish += 1; reasons.append(f"Bearish candle pattern: {candle_pattern} (+1)")
    if ha_trend_up:
        bullish += 1; reasons.append("Two-candle Heikin-Ashi bullish trend (+1)")
    elif ha_trend_down:
        bearish += 1; reasons.append("Two-candle Heikin-Ashi bearish trend (+1)")
    if breakout or retest_bull:
        bullish += 2; reasons.append("Bullish breakout/retest (+2)")
    if breakdown or retest_bear:
        bearish += 2; reasons.append("Bearish breakdown/retest (+2)")
    if volume_data_available and volume_spike:
        if close_price >= num(latest, "Open", close_price):
            bullish += 1; reasons.append("Above-average volume on up candle (+1)")
        else:
            bearish += 1; reasons.append("Above-average volume on down candle (+1)")
    elif not volume_data_available:
        reasons.append("Volume unavailable/non-representative for this instrument; volume is not used as a veto")

    # Higher timeframe and MTF are confirmations, not duplicated fake counts.
    mtf_available_count = max(0, int(mtf_available_count or 0))
    mtf_bull_count = max(0, int(mtf_bull_count or 0))
    mtf_bear_count = max(0, int(mtf_bear_count or 0))
    if htf_bias == "bullish":
        bullish += 2; reasons.append("Higher timeframe EMA trend bullish (+2)")
    elif htf_bias == "bearish":
        bearish += 2; reasons.append("Higher timeframe EMA trend bearish (+2)")
    if mtf_available_count > 0:
        if mtf_bull_count > mtf_bear_count:
            bullish += min(2, mtf_bull_count - mtf_bear_count); reasons.append(f"MTF majority bullish ({mtf_bull_count}/{mtf_available_count})")
        elif mtf_bear_count > mtf_bull_count:
            bearish += min(2, mtf_bear_count - mtf_bull_count); reasons.append(f"MTF majority bearish ({mtf_bear_count}/{mtf_available_count})")
        else:
            reasons.append("MTF alignment is mixed")
    else:
        reasons.append("MTF feed unavailable; no directional points or penalty applied")

    option_metrics = option_metrics if isinstance(option_metrics, dict) else {}
    pcr = option_metrics.get("pcr")
    try:
        pcr = float(pcr) if pcr is not None and np.isfinite(float(pcr)) and float(pcr) > 0 else None
    except Exception:
        pcr = None
    oi_bias = str(option_metrics.get("oi_bias", "N/A"))
    long_unwinding = bool(option_metrics.get("long_unwinding", False))
    short_covering = bool(option_metrics.get("short_covering", False))
    if pcr is not None:
        if pcr >= 1.20:
            bullish += 1; reasons.append(f"PCR supportive of bullish bias ({pcr:.2f}; +1)")
        elif pcr <= 0.80:
            bearish += 1; reasons.append(f"PCR supportive of bearish bias ({pcr:.2f}; +1)")
        else:
            reasons.append(f"PCR neutral band ({pcr:.2f})")
    else:
        reasons.append("PCR unavailable; excluded from scoring")
    # Avoid double-counting OI bias where it is derived from the same total CE/PE OI as PCR.

    regime = detect_market_regime(df)
    _news_message_for_regime = str((news_info or {}).get("message", ""))
    if (news_info or {}).get("status") == "ok" and "News risk detected" in _news_message_for_regime:
        regime["regime"] = "News Driven"
    gap_info = detect_gap_filter(df)
    session_info = session_filter(df, timeframe_label)
    news_info = news_info if isinstance(news_info, dict) else {"allow_trade": True, "message": "News filter unavailable", "events": []}
    if "allow_trade" not in news_info:
        news_info = {**news_info, "allow_trade": True}

    sideways = bool(regime.get("sideways", False))
    atr_pct = float(regime.get("atr_pct", 0.0))
    gap_allowed = bool(gap_info.get("allow_trade", True))
    session_allowed = bool(session_info.get("valid", True))
    news_allowed = bool(news_info.get("allow_trade", True))

    # Candidate direction requires both score advantage and price/trend structure.
    directional_edge = abs(bullish - bearish)
    directional_bias = "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutral"
    is_buy_candidate = bullish >= 7 and bullish - bearish >= 3
    is_sell_candidate = bearish >= 7 and bearish - bullish >= 3

    trend_vote_count = 4 if ema200_available else 3
    buy_trend_votes = sum([close_price > ema20, ema20 > ema50, (close_price > ema200) if ema200_available else False, supertrend_dir == 1])
    sell_trend_votes = sum([close_price < ema20, ema20 < ema50, (close_price < ema200) if ema200_available else False, supertrend_dir == -1])
    buy_momentum_ok = (macd > macd_signal) or (rsi >= 52) or breakout or retest_bull
    sell_momentum_ok = (macd < macd_signal) or (rsi <= 48) or breakdown or retest_bear
    adx_ok = adx >= 15 or breakout or breakdown or retest_bull or retest_bear

    # Prior-bar S/R and ATR-aware buffer: 0.5% was far too wide for NIFTY intraday.
    sr_buffer = max(0.20 * atr, close_price * 0.0005, 0.01)
    buy_near_resistance = (resistance >= close_price) and ((resistance - close_price) <= sr_buffer) and not breakout and not retest_bull
    sell_near_support = (support <= close_price) and ((close_price - support) <= sr_buffer) and not breakdown and not retest_bear
    candidate_no_trade_zone = bool((is_buy_candidate and buy_near_resistance) or (is_sell_candidate and sell_near_support))

    mtf_buy_conflict = mtf_available_count >= 2 and mtf_bear_count >= mtf_bull_count + 2
    mtf_sell_conflict = mtf_available_count >= 2 and mtf_bull_count >= mtf_bear_count + 2
    htf_buy_conflict = htf_bias == "bearish" and mtf_available_count >= 2 and mtf_bear_count > mtf_bull_count
    htf_sell_conflict = htf_bias == "bullish" and mtf_available_count >= 2 and mtf_bull_count > mtf_bear_count

    def blockers(side):
        out = []
        if side == "BUY":
            score_ok = is_buy_candidate
            votes = buy_trend_votes
            momentum_ok = buy_momentum_ok
            conflict = mtf_buy_conflict or htf_buy_conflict
            near_level = buy_near_resistance
        else:
            score_ok = is_sell_candidate
            votes = sell_trend_votes
            momentum_ok = sell_momentum_ok
            conflict = mtf_sell_conflict or htf_sell_conflict
            near_level = sell_near_support
        if not score_ok:
            out.append("Directional score is not sufficiently separated (need score >= 7 and edge >= 3)")
        if votes < 2:
            out.append(f"Trend structure incomplete ({votes}/{trend_vote_count} available confirmations; need at least 2)")
        if not momentum_ok:
            out.append("Momentum confirmation missing (MACD/RSI/breakout)")
        if not adx_ok:
            out.append(f"ADX weak ({adx:.1f}); no breakout/retest override")
        if conflict:
            out.append("Higher-timeframe/MTF direction conflicts with this setup")
        if sideways and not (breakout or breakdown or retest_bull or retest_bear) and votes < 3:
            out.append("Sideways regime without a confirmed range breakout")
        if not session_allowed:
            out.append(session_info.get("message", "Session filter blocked entry"))
        if not gap_allowed:
            out.append(f"Large session gap filter blocked entry ({gap_info.get('gap_pct', 0):.2f}%)")
        if not news_allowed:
            out.append(news_info.get("message", "News risk filter blocked entry"))
        if near_level:
            out.append("Price is too close to the prior resistance/support level")
        return out

    buy_blockers = blockers("BUY")
    sell_blockers = blockers("SELL")
    buy_allowed = is_buy_candidate and not buy_blockers
    sell_allowed = is_sell_candidate and not sell_blockers

    signal_type = "neutral"
    signal_text = "NO TRADE"
    strength = "Low"
    stop_loss = target1 = target2 = target3 = partial_exit = trailing_stop = rr_ratio = None
    selected_blockers = []
    
    selected_side = None
    
    
    # ==========================================
    # INSTITUTIONAL CONFLUENCE SCORING
    # ==========================================
    
    try:
    
        market_structure = inst.detect_market_structure(df)
        st.write("DEBUG MS:", market_structure)
        if market_structure.get("bias") == "Bullish":
    
            bullish += 15
    
            reasons.append(
                "Bullish BOS/CHOCH (+15)"
            )
    
        elif market_structure.get("bias") == "Bearish":
    
            bearish += 15
    
            reasons.append(
                "Bearish BOS/CHOCH (+15)"
            )
        # Liquidity Sweep
    
        liquidity = inst.detect_liquidity_sweep(df)
        st.write("DEBUG LS:", liquidity)
        if liquidity.get("signal") == "Bullish":
    
            bullish += 15
    
            reasons.append(
                "Bullish Liquidity Sweep (+15)"
            )
    
        elif liquidity.get("signal") == "Bearish":
    
            bearish += 15
    
            reasons.append(
                "Bearish Liquidity Sweep (+15)"
            )
    
        if liquidity.get("block_entry"):
    
            selected_blockers.append(
                f"Liquidity Sweep Active: {liquidity.get('type','Unknown')}"
            )
        # Order Blocks
    
        order_blocks = inst.detect_order_blocks(df)
        st.write("DEBUG OB:", order_blocks)
        if order_blocks.get("bias") == "Bullish":
    
            bullish += 10
    
            reasons.append(
                "Bullish Order Block (+10)"
            )
    
        elif order_blocks.get("bias") == "Bearish":
    
            bearish += 10
    
            reasons.append(
                "Bearish Order Block (+10)"
            )
        # Fair Value Gap
    
        fvg = inst.detect_fvg(df)
    
        if fvg.get("bias") == "Bullish":
    
            bullish += 10
    
            reasons.append(
                "Bullish FVG (+10)"
            )
    
        elif fvg.get("bias") == "Bearish":
    
            bearish += 10
    
            reasons.append(
                "Bearish FVG (+10)"
            )
        # Volume Profile
    
        volume_profile = inst.calculate_volume_profile(df)
    
        if volume_profile.get("bias") == "Bullish":
    
            bullish += 10
    
            reasons.append(
                "Above POC (+10)"
            )
    
        elif volume_profile.get("bias") == "Bearish":
    
            bearish += 10
    
            reasons.append(
                "Below POC (+10)"
                )
        # Relative Strength
    
        try:
    
            benchmark_df = fetch_data(
                "^NSEI",
                interval="1d",
                period="1mo"
            )
    
            rs_score = inst.calculate_relative_strength(
                df,
                benchmark_df
            )
    
            if rs_score > 1.5:
    
                bullish += 10
    
                reasons.append(
                    "Strong Relative Strength (+10)"
                )
    
            elif rs_score < -1.5:
    
                bearish += 10
    
                reasons.append(
                    "Weak Relative Strength (+10)"
                )
    
        except Exception:
    
            pass
    except Exception as e:
    
        st.write(
            f"INSTITUTIONAL ERROR: {e}"
        )
    
    
    
    
    if buy_allowed and (not sell_allowed or bullish - bearish >= bearish - bullish):
        selected_side = "BUY"
    elif sell_allowed:
        selected_side = "SELL"

    if selected_side:
        risk = max(atr, close_price * 0.001, 0.01)
        if selected_side == "BUY":
            signal_type = "buy"
            stop_loss = close_price - risk
            target1, target2, target3 = close_price + 2*risk, close_price + 3*risk, close_price + 4*risk
            trailing_stop = close_price - 0.75*risk
            strong = bullish >= 25 and (bullish - bearish) >= 6 and buy_trend_votes >= 3 and adx >= 20 and (htf_bias == "bullish" or mtf_bull_count > mtf_bear_count or breakout or volume_spike)
            signal_text = "STRONG BUY" if strong else "BUY"
            strength = "High" if strong else "Medium"
            selected_blockers = []
        else:
            signal_type = "sell"
            stop_loss = close_price + risk
            target1, target2, target3 = close_price - 2*risk, close_price - 3*risk, close_price - 4*risk
            trailing_stop = close_price + 0.75*risk
            strong = bearish >= 25 and (bearish - bullish) >= 6 and sell_trend_votes >= 3 and adx >= 20 and (htf_bias == "bearish" or mtf_bear_count > mtf_bull_count or breakdown or volume_spike)
            signal_text = "STRONG SELL" if strong else "SELL"
            strength = "High" if strong else "Medium"
        partial_exit = target1
        rr_ratio = round(abs(target1 - close_price) / abs(close_price - stop_loss), 2) if stop_loss != close_price else None
    else:
        if is_buy_candidate and (not is_sell_candidate or bullish - bearish >= bearish - bullish):
            selected_blockers = buy_blockers
        elif is_sell_candidate:
            selected_blockers = sell_blockers
        else:
            selected_blockers = ["Bullish and bearish evidence is mixed; no directional edge"]
        if not selected_blockers:
            selected_blockers = ["No valid entry setup after confirmation checks"]

    # Setup confidence is a structural score, not a probability of winning.
    if signal_type in ("buy", "sell"):
        selected_score = bullish if signal_type == "buy" else bearish
        selected_votes = buy_trend_votes if signal_type == "buy" else sell_trend_votes
        mtf_aligned = (mtf_bull_count > mtf_bear_count) if signal_type == "buy" else (mtf_bear_count > mtf_bull_count)
        confidence = int(min(92, 52 + min(18, directional_edge * 1.5) + selected_votes * 3 + (4 if mtf_aligned else 0) + (3 if htf_bias == ("bullish" if signal_type == "buy" else "bearish") else 0)))
        setup_confidence_label = "Structural setup score; not a probability"
    else:
        confidence = None
        setup_confidence_label = "No approved setup"

    # A measured historical win rate is shown only for 30+ completed trades from this exact engine.
    win_chance = calculate_win_chance(backtest_win_rate, backtest_trade_count, backtest_is_comparable)
    validation_score = 0
    chosen_score = bullish if directional_bias == "Bullish" else bearish if directional_bias == "Bearish" else 0
    chosen_votes = buy_trend_votes if directional_bias == "Bullish" else sell_trend_votes if directional_bias == "Bearish" else 0
    validation_score += min(25, int(chosen_score * 1.5))
    validation_score += min(20, chosen_votes * 5)
    validation_score += 15 if ((macd > macd_signal and directional_bias == "Bullish") or (macd < macd_signal and directional_bias == "Bearish")) else 0
    validation_score += 10 if adx >= 18 else 0
    validation_score += 10 if session_allowed and gap_allowed and news_allowed else 0
    mtf_supports_bias = ((directional_bias == "Bullish" and mtf_bull_count > mtf_bear_count) or (directional_bias == "Bearish" and mtf_bear_count > mtf_bull_count))
    validation_score += 10 if (mtf_available_count >= 2 and mtf_supports_bias) else 0
    validation_score += 5 if volume_data_available and volume_spike else 0
    validation_score += 5 if atr > 0 else 0
    validation_score = int(max(0, min(100, validation_score)))
    strategy_grade = classify_strategy_grade(validation_score) if signal_type != "neutral" else "NO SETUP"
    trade_quality_score = int(round(0.65 * validation_score + 0.35 * (confidence or 0))) if signal_type != "neutral" else 0

    no_trade_zone = candidate_no_trade_zone
    no_trade_reason = "; ".join(selected_blockers) if selected_blockers else None
    reasons.extend(selected_blockers if signal_type == "neutral" else [f"Entry approved: {signal_text}"])
    regime = detect_market_regime(df)
    gap_info = detect_gap_filter(df)
    session_info = session_filter(df, timeframe_label)
    return {
        "close_price": close_price, "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "vwap": vwap_value, "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
        "macd_hist": macd_hist, "candle_pattern": candle_pattern, "candle_pattern_bias": candle_bias,
        "candle_pattern_score": candle_pattern_score, "atr": atr, "stoch_k": stoch_k, "stoch_d": stoch_d,
        "adx": adx, "support": support, "resistance": resistance, "volume_spike": volume_spike,
        "breakout": breakout, "breakdown": breakdown, "body_strength": body_strength,
        "sideways": sideways, "signal_text": signal_text, "signal_type": signal_type,
        "strength": strength, "confidence": confidence, "win_chance": win_chance,
        "win_chance_sample_size": int(backtest_trade_count or 0) if win_chance is not None else 0,
        "setup_confidence_label": setup_confidence_label,
        "stop_loss": stop_loss, "partial_exit": partial_exit, "target1": target1,
        "target2": target2, "target3": target3, "trailing_stop": trailing_stop,
        "rr_ratio": rr_ratio, "reasons": reasons, "no_trade_reason": no_trade_reason,
        "no_trade_reasons": selected_blockers, "bullish_score": round(bullish, 1),
        "bearish_score": round(bearish, 1), "directional_bias": directional_bias,
        "directional_edge": round(directional_edge, 1), "session_message": session_info["message"],
        "supertrend_direction": supertrend_dir, "ha_bullish": ha_bullish, "ha_bearish": ha_bearish,
        "ha_trend_up": ha_trend_up, "ha_trend_down": ha_trend_down,
        "validation_score": validation_score, "strategy_grade": strategy_grade,
        "trade_quality_score": trade_quality_score, "market_regime": regime["regime"],
        "trending_market": regime["trending"], "sideways_market": regime["sideways"],
        "high_volatility_market": regime["high_volatility"], "atr_pct": regime["atr_pct"],
        "gap_up": gap_info["gap_up"], "gap_down": gap_info["gap_down"], "gap_pct": gap_info["gap_pct"],
        "mtf_bull_ok": mtf_available_count >= 2 and mtf_bull_count > mtf_bear_count,
        "mtf_bear_ok": mtf_available_count >= 2 and mtf_bear_count > mtf_bull_count,
        "no_trade_zone": no_trade_zone, "oi_bias": oi_bias, "pcr": pcr,
        "long_unwinding": long_unwinding, "short_covering": short_covering,
        "news_message": news_info.get("message", ""), "trade_allowed": signal_type in ("buy", "sell"),
        "volume_data_available": volume_data_available, "mtf_available_count": mtf_available_count,
        "ema200_available": ema200_available, "trend_vote_count": trend_vote_count,
        "buy_trend_votes": int(buy_trend_votes), "sell_trend_votes": int(sell_trend_votes),
    }


# =========================================================
# LOSS INTELLIGENCE / SIMILAR SETUPS / SIGNAL ENGINE V2
# =========================================================
LOSS_FEATURE_COLUMNS = [
    "EMA9", "EMA21", "EMA20", "EMA50", "EMA200", "RSI", "MACD", "ADX",
    "ATR", "VWAP", "Volume", "RelativeVolume", "VolumeDelta", "TrendSlope",
    "SupportDistancePct", "ResistanceDistancePct", "DistanceEMA20Pct",
    "DistanceEMA50Pct", "DistanceVWAPPct", "RiskRewardRatio"
]


def _safe_num(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def build_trade_feature_snapshot(df, analysis, mtf_bull_count=0, mtf_bear_count=0, mtf_available_count=0):
    """Capture a point-in-time feature snapshot; missing values stay NaN, never fabricated."""
    row = df.iloc[-1] if isinstance(df, pd.DataFrame) and not df.empty else pd.Series(dtype=object)
    close = _safe_num(row.get("Close"))
    support = _safe_num(analysis.get("support"))
    resistance = _safe_num(analysis.get("resistance"))
    vwap = _safe_num(row.get("VWAP", analysis.get("vwap")))
    def pct_distance(level):
        return ((close - level) / close * 100.0) if np.isfinite(close) and close > 0 and np.isfinite(level) else np.nan
    if mtf_available_count <= 0:
        mtf_status = "Unavailable"
    elif mtf_bull_count > mtf_bear_count:
        mtf_status = "Bullish alignment"
    elif mtf_bear_count > mtf_bull_count:
        mtf_status = "Bearish alignment"
    else:
        mtf_status = "Mixed"
    snap = {
        "EMA9": _safe_num(row.get("EMA9")), "EMA21": _safe_num(row.get("EMA21")),
        "EMA20": _safe_num(row.get("EMA20", analysis.get("ema20"))),
        "EMA50": _safe_num(row.get("EMA50", analysis.get("ema50"))),
        "EMA200": _safe_num(row.get("EMA200", analysis.get("ema200"))),
        "RSI": _safe_num(row.get("RSI", analysis.get("rsi"))),
        "MACD": _safe_num(row.get("MACD", analysis.get("macd"))),
        "ADX": _safe_num(row.get("ADX", analysis.get("adx"))),
        "ATR": _safe_num(row.get("ATR", analysis.get("atr"))),
        "VWAP": vwap, "Volume": _safe_num(row.get("Volume")),
        "RelativeVolume": _safe_num(row.get("RelativeVolume")),
        "VolumeDelta": _safe_num(row.get("VolumeDelta")),
        "TrendSlope": _safe_num(row.get("TrendSlope")),
        "SupportDistancePct": pct_distance(support),
        "ResistanceDistancePct": pct_distance(resistance),
        "DistanceEMA20Pct": _safe_num(row.get("DistanceEMA20Pct")),
        "DistanceEMA50Pct": _safe_num(row.get("DistanceEMA50Pct")),
        "DistanceVWAPPct": _safe_num(row.get("DistanceVWAPPct")),
        "RiskRewardRatio": _safe_num(analysis.get("rr_ratio")),
        "MarketRegime": str(analysis.get("market_regime", "Unknown")),
        "MTFTrendStatus": mtf_status,
        "ConfidenceAtEntry": _safe_num(analysis.get("confidence")),
        "HistoricalWinRateAtEntry": _safe_num(analysis.get("historical_win_rate")),
        "SimilarTradesAtEntry": int(analysis.get("similar_trades_found", 0) or 0),
        "SignalStage": str(analysis.get("signal_stage", "")),
    }
    return snap


def classify_trade_loss(snapshot, pnl, side="BUY", exit_price=None, entry_price=None):
    """Heuristic diagnosis for review; returns a probable cause, not a proven cause."""
    if _safe_num(pnl, 0.0) >= 0:
        return ""
    s = snapshot if isinstance(snapshot, dict) else {}
    rsi = _safe_num(s.get("RSI")); adx = _safe_num(s.get("ADX")); relvol = _safe_num(s.get("RelativeVolume"))
    rr = _safe_num(s.get("RiskRewardRatio")); regime = str(s.get("MarketRegime", "")).lower()
    mtf = str(s.get("MTFTrendStatus", "")).lower()
    slope = _safe_num(s.get("TrendSlope")); dist20 = _safe_num(s.get("DistanceEMA20Pct"))
    if "range" in regime or "sideways" in regime or (np.isfinite(adx) and adx < 18): return "Sideways Market / Weak Trend"
    if np.isfinite(relvol) and relvol < 0.8: return "Low Volume"
    if "conflict" in mtf or "mixed" in mtf: return "Multi-Timeframe Conflict"
    if np.isfinite(rr) and rr < 1.5: return "Poor Risk Reward"
    if str(side).upper() == "BUY" and np.isfinite(rsi) and rsi > 75: return "Overbought Entry / Trend Exhaustion"
    if str(side).upper() == "SELL" and np.isfinite(rsi) and rsi < 25: return "Oversold Entry / Trend Exhaustion"
    if np.isfinite(slope) and ((str(side).upper() == "BUY" and slope < 0) or (str(side).upper() == "SELL" and slope > 0)):
        return "Trend Exhaustion / Late Entry"
    if np.isfinite(dist20) and abs(dist20) > 1.5: return "Late Entry"
    if "breakout" in regime or "breakdown" in regime: return "Fake Breakout / Volatility Trap"
    return "Unclassified Loss - Review News, Entry Timing and Execution"


def _similarity_score(current, historical):
    """Distance over available comparable numeric features; tolerates old/incomplete journal rows."""
    pairs = []
    tolerances = {"RSI": 12, "ADX": 12, "RelativeVolume": 0.9, "RiskRewardRatio": 1.5,
                  "DistanceEMA20Pct": 0.8, "DistanceEMA50Pct": 1.2, "TrendSlope": max(abs(_safe_num(current.get("TrendSlope"), 0))*1.5, 0.05)}
    for key in ("RSI", "ADX", "RelativeVolume", "RiskRewardRatio", "DistanceEMA20Pct", "DistanceEMA50Pct", "TrendSlope"):
        a, b = _safe_num(current.get(key)), _safe_num(historical.get(key))
        if np.isfinite(a) and np.isfinite(b):
            tol = tolerances.get(key, 1.0)
            pairs.append(max(0.0, 1.0 - abs(a-b) / max(tol, 1e-6)))
    if not pairs: return 0.0
    score = float(np.mean(pairs))
    if current.get("MarketRegime") and historical.get("MarketRegime") and str(current.get("MarketRegime")) == str(historical.get("MarketRegime")):
        score = min(1.0, score + 0.08)
    if current.get("MTFTrendStatus") and historical.get("MTFTrendStatus") and str(current.get("MTFTrendStatus")) == str(historical.get("MTFTrendStatus")):
        score = min(1.0, score + 0.05)
    return score


def get_similar_trade_intelligence(snapshot, side, min_similarity=0.68):
    journal = load_journal()
    result = {"matches": pd.DataFrame(), "count": 0, "win_rate": None, "loss_rate": None,
              "avg_profit": None, "avg_loss": None, "expected_value": None, "loser_match_rate": None}
    if journal.empty or not {"Status", "Result"}.issubset(journal.columns): return result
    closed = journal[journal["Status"].astype(str).str.upper().eq("CLOSED")].copy()
    if "Side" in closed.columns: closed = closed[closed["Side"].astype(str).str.upper().eq(str(side).upper())]
    if closed.empty: return result
    scores = []
    for _, row in closed.iterrows():
        hist = {k: row.get(k) for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"]}
        score = _similarity_score(snapshot, hist)
        if score >= min_similarity:
            scores.append((row.name, score))
    if not scores: return result
    matches = closed.loc[[x[0] for x in scores]].copy()
    matches["Similarity"] = [x[1] for x in scores]
    pnl = pd.to_numeric(matches.get("PnL", pd.Series(0, index=matches.index)), errors="coerce").fillna(0)
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    count = len(matches)
    result.update({"matches": matches.sort_values("Similarity", ascending=False), "count": count,
                   "win_rate": float((pnl > 0).mean() * 100), "loss_rate": float((pnl < 0).mean() * 100),
                   "avg_profit": float(wins.mean()) if len(wins) else 0.0,
                   "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                   "expected_value": float(pnl.mean()), "loser_match_rate": float((pnl < 0).mean() * 100)})
    return result


def apply_adaptive_signal_intelligence(df, analysis, symbol, timeframe, mtf_bull_count=0, mtf_bear_count=0, mtf_available_count=0):
    """Add similar-trade learning, conservative confidence adjustment and staged signal labels."""
    side = "BUY" if analysis.get("signal_type") == "buy" else "SELL" if analysis.get("signal_type") == "sell" else "BUY"
    snapshot = build_trade_feature_snapshot(df, analysis, mtf_bull_count, mtf_bear_count, mtf_available_count)
    intelligence = get_similar_trade_intelligence(snapshot, side)
    analysis["similar_trades_found"] = intelligence["count"]
    analysis["historical_win_rate"] = intelligence["win_rate"]
    analysis["historical_loss_rate"] = intelligence["loss_rate"]
    analysis["historical_avg_profit"] = intelligence["avg_profit"]
    analysis["historical_avg_loss"] = intelligence["avg_loss"]
    analysis["historical_expected_value"] = intelligence["expected_value"]
    analysis["historical_loser_match_rate"] = intelligence["loser_match_rate"]
    analysis["feature_snapshot"] = snapshot
    ema9 = snapshot.get("EMA9"); ema21 = snapshot.get("EMA21")
    rsi = _safe_num(snapshot.get("RSI"), 50); slope = _safe_num(snapshot.get("TrendSlope"), 0)
    hist = _safe_num(df.iloc[-1].get("MACD_Hist")) if isinstance(df, pd.DataFrame) and not df.empty else 0
    prev_hist = _safe_num(df.iloc[-2].get("MACD_Hist")) if isinstance(df, pd.DataFrame) and len(df) > 1 else 0
    volume_ok = _safe_num(snapshot.get("RelativeVolume"), 0) >= 1.2
    mtf_bull = mtf_available_count >= 2 and mtf_bull_count > mtf_bear_count
    mtf_bear = mtf_available_count >= 2 and mtf_bear_count > mtf_bull_count
    ema_bull = np.isfinite(_safe_num(ema9)) and np.isfinite(_safe_num(ema21)) and ema9 > ema21
    ema_bear = np.isfinite(_safe_num(ema9)) and np.isfinite(_safe_num(ema21)) and ema9 < ema21
    if rsi > 75 and hist < prev_hist: stage = "TREND EXHAUSTION"
    elif ema_bull and hist > 0 and rsi > 50 and analysis.get("ema20", 0) > analysis.get("ema50", 0) and mtf_bull:
        stage = "STRONG BUY" if _safe_num(snapshot.get("ADX"), 0) > 20 and volume_ok else "BUY"
    elif ema_bear and hist < 0 and rsi < 50 and analysis.get("ema20", 0) < analysis.get("ema50", 0) and mtf_bear:
        stage = "STRONG SELL" if _safe_num(snapshot.get("ADX"), 0) > 20 and volume_ok else "SELL"
    elif ema_bull and hist > 0 and rsi > 50: stage = "EARLY BUY WATCH"
    elif ema_bear and hist < 0 and rsi < 50: stage = "EARLY SELL WATCH"
    elif slope > 0 and hist >= prev_hist and rsi > 48: stage = "TREND FORMING UP"
    elif slope < 0 and hist <= prev_hist and rsi < 52: stage = "TREND FORMING DOWN"
    else: stage = "NO CLEAR STAGE"
    analysis["signal_stage"] = stage
    # Confidence remains a score, not a calibrated probability. Historical evidence is blended only with 30+ matches.
    base = _safe_num(analysis.get("confidence"), 50)
    if intelligence["count"] >= 30 and intelligence["win_rate"] is not None:
        hist_component = float(intelligence["win_rate"])
        adjusted = 0.65 * base + 0.35 * hist_component
    else:
        adjusted = base
    risk_penalty = 0
    if intelligence["count"] >= 10 and intelligence["loser_match_rate"] is not None and intelligence["loser_match_rate"] >= 65:
        risk_penalty = 12
        analysis["repeat_loss_warning"] = True
    else:
        analysis["repeat_loss_warning"] = False
    if "RANGE" in str(analysis.get("market_regime", "")).upper() or analysis.get("sideways_market"):
        risk_penalty += 5
    if not volume_ok and _safe_num(snapshot.get("Volume"), 0) > 0:
        risk_penalty += 3
    rr_value = _safe_num(snapshot.get("RiskRewardRatio"), np.nan)
    adx_value = _safe_num(snapshot.get("ADX"), np.nan)
    confidence_bonus = 0
    if np.isfinite(rr_value):
        confidence_bonus += 3 if rr_value >= 2.0 else (-5 if rr_value < 1.5 else 0)
    if np.isfinite(adx_value):
        confidence_bonus += 2 if adx_value >= 25 else (-4 if adx_value < 18 else 0)
    if volume_ok:
        confidence_bonus += 3
    if analysis.get("trending_market"):
        confidence_bonus += 2
    if analysis.get("high_volatility_market"):
        risk_penalty += 3
    analysis["confidence"] = int(max(0, min(100, round(adjusted - risk_penalty + confidence_bonus)))) if analysis.get("signal_type") in ("buy", "sell") else None
    if analysis.get("confidence") is not None:
        analysis["trade_quality_score"] = int(round(0.65 * _safe_num(analysis.get("validation_score"), 0) + 0.35 * analysis["confidence"]))
    analysis["adaptive_confidence_label"] = "Historical blend active" if intelligence["count"] >= 30 else "Indicator-led; historical sample insufficient"
    analysis["repeat_loss_penalty"] = risk_penalty
    analysis["similar_trade_block_suggested"] = bool(intelligence["count"] >= 10 and (intelligence["loser_match_rate"] or 0) >= 75)
    return analysis


def train_trade_success_model(min_trades=30):
    """Evaluate journal readiness immediately; fit a research-only model after 30 usable trades."""
    journal = load_journal()
    if journal.empty:
        return {"ok": False, "trained": False, "message": "No completed journal trades yet.", "trades": 0}
    closed = journal[journal.get("Status", pd.Series("", index=journal.index)).astype(str).str.upper().eq("CLOSED")].copy()
    total_closed = len(closed)
    if total_closed == 0:
        return {"ok": False, "trained": False, "message": "No completed journal trades yet.", "trades": 0}

    features = [c for c in ["EMA9", "EMA21", "EMA20", "EMA50", "EMA200", "RSI", "MACD", "ADX", "ATR", "VWAP", "Volume", "RelativeVolume", "VolumeDelta", "TrendSlope", "SupportDistancePct", "ResistanceDistancePct", "DistanceEMA20Pct", "DistanceEMA50Pct", "DistanceVWAPPct", "RiskRewardRatio", "TradeQualityScore", "ValidationScore"] if c in closed.columns]
    if len(features) < 6 or "PnL" not in closed.columns:
        return {"ok": False, "trained": False, "message": "Dataset evaluation complete: insufficient captured numeric feature columns or PnL field. Ensure new closed trades save setup snapshots.", "trades": total_closed, "features_found": len(features)}

    X = closed[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_raw = pd.to_numeric(closed["PnL"], errors="coerce")
    valid = X.notna().sum(axis=1) >= 6
    valid &= y_raw.notna()
    X, y = X.loc[valid], (y_raw.loc[valid] > 0).astype(int)
    wins = int((y == 1).sum()); losses = int((y == 0).sum())
    coverage = float(valid.mean() * 100) if len(valid) else 0.0
    readiness = {"trades": int(len(X)), "total_closed": int(total_closed), "wins": wins,
                 "losses": losses, "win_rate": float(wins / len(y) * 100) if len(y) else None,
                 "feature_count": len(features), "features": features, "coverage_pct": coverage}
    if len(X) < int(min_trades):
        return {"ok": True, "trained": False, "readiness": readiness,
                "message": f"Dataset evaluation completed. {len(X)}/{min_trades} usable closed trades collected; model training remains locked until {min_trades}. Add more completed trades with captured setup metrics."}
    if y.nunique() < 2 or min(wins, losses) < 5:
        return {"ok": True, "trained": False, "readiness": readiness,
                "message": "Dataset evaluation completed, but both win and loss classes need at least 5 examples before a meaningful model evaluation."}

    X = X.fillna(X.median()).fillna(0)
    # Preserve chronological order: older trades train the model, newer trades test it.
    split_at = int(len(X) * 0.70)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {"ok": True, "trained": False, "readiness": readiness,
                "message": "Dataset is large enough, but the chronological train/test windows do not both contain wins and losses. Collect more varied trades before fitting."}
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
        model_name = "Random Forest"
        model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=2,
                                       class_weight="balanced", random_state=42)
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.04,
                                  subsample=0.85, colsample_bytree=0.85,
                                  eval_metric="logloss", random_state=42)
            model_name = "XGBoost"
        except Exception:
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=180, max_depth=4, learning_rate=0.04,
                                       class_weight="balanced", verbosity=-1, random_state=42)
                model_name = "LightGBM"
            except Exception:
                pass
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        try: auc = float(roc_auc_score(y_test, probabilities))
        except Exception: auc = None
        return {"ok": True, "trained": True, "model": model_name, "trades": int(len(X)),
                "train_trades": int(len(X_train)), "test_trades": int(len(X_test)),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "auc": auc, "features": features, "readiness": readiness,
                "warning": "Research-only chronological holdout evaluation. This model is not connected to live signals and is not a profitability guarantee. Validate with more out-of-sample trades and realistic fees/slippage."}
    except Exception as exc:
        return {"ok": False, "trained": False, "readiness": readiness,
                "message": f"ML dependencies or training unavailable: {type(exc).__name__}: {exc}"}


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
            "Confidence": analysis.get("confidence") if analysis.get("confidence") is not None else np.nan,
            "WinChance": analysis.get("win_chance") if analysis.get("win_chance") is not None else np.nan,
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


def paper_trade_entry(price, quantity, side="BUY"):
    """Reserve virtual cash/margin for either a paper long or paper short."""
    try:
        price = float(price)
        quantity = int(quantity)
        side = str(side).upper()
        if not np.isfinite(price) or price <= 0 or quantity <= 0 or side not in ("BUY", "SELL"):
            return False
        state = load_paper_state()
        bal = float(state.loc[0, "Balance"])
        used = float(state.loc[0, "UsedMargin"])
        reserved = price * quantity
        # This simulator reserves full notional for both long and short positions.
        if bal + 1e-9 < reserved:
            return False
        state.loc[0, "Balance"] = bal - reserved
        state.loc[0, "UsedMargin"] = used + reserved
        state.loc[0, "OpenPositions"] = int(state.loc[0, "OpenPositions"]) + 1
        return save_paper_state(state)
    except Exception as exc:
        log_app_error("paper_trade_entry", exc)
        return False



def paper_trade_exit(entry_price, exit_price, quantity, signal, final_close=True):
    """Release reserved notional and credit/debit simulated P&L."""
    state = load_paper_state()
    bal = float(state.loc[0, "Balance"])
    used = float(state.loc[0, "UsedMargin"])
    entry_price = float(entry_price)
    exit_price = float(exit_price)
    quantity = float(quantity)
    reserved = entry_price * quantity
    is_long = "BUY" in str(signal).upper() or "CALL" in str(signal).upper()
    pnl = (exit_price - entry_price) * quantity if is_long else (entry_price - exit_price) * quantity
    state.loc[0, "Balance"] = bal + reserved + pnl
    state.loc[0, "UsedMargin"] = max(0.0, used - reserved)
    state.loc[0, "OpenPositions"] = max(0, int(state.loc[0, "OpenPositions"]) - (1 if final_close and quantity > 0 else 0))
    save_paper_state(state)
    return float(pnl)



def get_suggested_paper_levels(analysis, side, entry_price):
    """Return side-correct suggested levels; ATR fallback is explicitly a simulation plan."""
    side = str(side).upper()
    entry = float(entry_price)
    atr = float(analysis.get("atr") or 0.0)
    risk = min(max(atr if np.isfinite(atr) else 0.0, entry * 0.003, 0.01), entry * 0.20)
    same_signal = (side == "BUY" and analysis.get("signal_type") == "buy") or (side == "SELL" and analysis.get("signal_type") == "sell")
    if same_signal:
        sl = analysis.get("stop_loss")
        t1, t2, t3 = analysis.get("target1"), analysis.get("target2"), analysis.get("target3")
        try:
            sl, t1, t2, t3 = float(sl), float(t1), float(t2), float(t3)
            valid = (sl < entry < t1 < t2 < t3) if side == "BUY" else (t3 < t2 < t1 < entry < sl)
            if not valid:
                raise ValueError("Signal levels are not on the correct side")
        except Exception:
            sl = t1 = t2 = t3 = None
    else:
        sl = t1 = t2 = t3 = None
    if sl is None:
        if side == "BUY":
            sl, t1, t2, t3 = entry - risk, entry + 2 * risk, entry + 3 * risk, entry + 4 * risk
        else:
            sl, t1, t2, t3 = entry + risk, entry - 2 * risk, entry - 3 * risk, entry - 4 * risk
    trailing = sl
    return {
        "side": side, "entry": entry, "stop_loss": float(sl),
        "target1": float(t1), "target2": float(t2), "target3": float(t3),
        "trailing_stop": float(trailing), "atr": float(atr if atr > 0 else risk)
    }


def add_trade_to_journal(symbol, timeframe, signal, entry_price, quantity, stop_loss, target1, target2, target3,
                         trailing_stop, analysis, notes="", auto_exit=True, atr=None):
    journal = load_journal()
    side = "SELL" if "SELL" in str(signal).upper() else "BUY"
    new_row = {
        "DateTime": now_str(), "TradeDate": datetime.now().strftime("%Y-%m-%d"),
        "TradeTime": datetime.now().strftime("%H:%M:%S"), "Symbol": symbol, "Timeframe": timeframe,
        "Signal": side, "SignalType": str(analysis.get("signal_text", side)), "Side": side,
        "EntryPrice": float(entry_price), "ExitPrice": np.nan,
        "Quantity": int(quantity), "RemainingQty": int(quantity), "StopLoss": float(stop_loss),
        "Target1": float(target1), "Target2": float(target2), "Target3": float(target3),
        "TrailingStop": float(trailing_stop), "ATR": float(atr if atr is not None else analysis.get("atr", 0.0)),
        "AutoExitEnabled": bool(auto_exit), "Status": "OPEN", "PnL": 0.0, "RealizedPnL": 0.0,
        "PnLPercent": 0.0, "Notes": notes, "Result": "", "T1_Hit": False, "T2_Hit": False,
        "T3_Hit": False, "LastPrice": float(entry_price), "LastUpdateTime": now_str(),
        "TradeQualityScore": analysis.get("trade_quality_score", 0),
        "ValidationScore": analysis.get("validation_score", 0),
        "StrategyGrade": analysis.get("strategy_grade", "D"),
        **(analysis.get("feature_snapshot", {}) if isinstance(analysis.get("feature_snapshot"), dict) else {}),
        "LossReason": ""
    }
    journal = pd.concat([journal, pd.DataFrame([new_row])], ignore_index=True)
    save_journal(journal)
    return True



def close_trade_in_journal(index_id, exit_price, result=None, t1_hit=None, t2_hit=None, t3_hit=None):
    """Legacy compatibility closer; closes the position's currently remaining quantity."""
    journal = load_journal()
    if index_id not in journal.index:
        return False
    row = journal.loc[index_id]
    entry = float(row["EntryPrice"])
    qty = float(row.get("RemainingQty", row["Quantity"]))
    side = str(row.get("Side", row["Signal"])).upper()
    if qty <= 0:
        return False
    pnl = paper_trade_exit(entry, float(exit_price), qty, side)
    prior = float(row.get("RealizedPnL", 0.0) or 0.0)
    total = prior + pnl
    journal.at[index_id, "RemainingQty"] = 0
    journal.at[index_id, "RealizedPnL"] = total
    journal.at[index_id, "ExitPrice"] = float(exit_price)
    journal.at[index_id, "Status"] = "CLOSED"
    journal.at[index_id, "PnL"] = round(total, 2)
    journal.at[index_id, "PnLPercent"] = round((total / (entry * float(row["Quantity"])) * 100) if entry else 0, 2)
    journal.at[index_id, "Result"] = result or ("WIN" if total > 0 else "LOSS" if total < 0 else "BREAKEVEN")
    journal.at[index_id, "LastPrice"] = float(exit_price)
    journal.at[index_id, "LastUpdateTime"] = now_str()
    for col, value in (("T1_Hit", t1_hit), ("T2_Hit", t2_hit), ("T3_Hit", t3_hit)):
        if value is not None:
            journal.at[index_id, col] = bool(value)
    if str(journal.at[index_id, "Result"]).upper() == "LOSS":
        loss_snapshot = {k: journal.at[index_id, k] for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"] if k in journal.columns}
        journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total, side, float(exit_price), entry)
    save_journal(journal)
    return True



def _paper_bool(value, default=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    try:
        return bool(value)
    except Exception:
        return default



def render_paper_exit_mode_selector():
    """Offer an explicit Manual or Auto Exit mode; Manual is the safe default."""
    valid_modes = {"Manual Exit", "Auto Exit (Enabled)"}
    if st.session_state.get("paper_exit_mode") not in valid_modes:
        st.session_state["paper_exit_mode"] = "Manual Exit"
    mode = st.radio(
        "Exit Management Mode",
        options=["Manual Exit", "Auto Exit (Enabled)"],
        horizontal=True,
        key="paper_exit_mode",
        help=(
            "Manual Exit: positions remain open until you press CLOSE. "
            "Auto Exit: the selected symbol is monitored on Streamlit reruns/auto-refresh "
            "and may scale out at targets or close at stop/trailing stop."
        ),
    )
    enabled = mode == "Auto Exit (Enabled)"
    if enabled:
        st.success("AUTO EXIT ENABLED — keep Auto Refresh ON and keep the position's symbol selected.")
    else:
        st.info("MANUAL EXIT MODE — positions close only when you press CLOSE.")
    return enabled


def sync_open_position_auto_exit(enabled, journal=None):
    """Apply the selected Manual/Auto mode to every currently open paper journal row."""
    if journal is None:
        journal = load_journal()
    if journal.empty:
        return journal
    if "AutoExitEnabled" not in journal.columns:
        journal["AutoExitEnabled"] = False
    status = journal.get("Status", pd.Series("", index=journal.index)).astype(str).str.upper()
    open_mask = status.eq("OPEN")
    if not open_mask.any():
        return journal
    current_flags = journal.loc[open_mask, "AutoExitEnabled"].map(lambda value: _paper_bool(value, False))
    if not current_flags.eq(bool(enabled)).all():
        journal.loc[open_mask, "AutoExitEnabled"] = bool(enabled)
        save_journal(journal)
    return journal


def _paper_side_pnl(entry, exit_price, quantity, side):
    return (exit_price - entry) * quantity if str(side).upper() == "BUY" else (entry - exit_price) * quantity


def close_paper_position(index_id, exit_price, reason="Manual Close"):
    """Close all remaining units of one paper position at the supplied simulated price."""
    journal = load_journal()
    if journal.empty or index_id not in journal.index:
        return False, "Paper position was not found."
    row = journal.loc[index_id]
    if str(row.get("Status", "")).upper() != "OPEN":
        return False, "This paper position is already closed."
    try:
        entry = float(row["EntryPrice"])
        qty = int(float(row.get("RemainingQty", row["Quantity"])))
        exit_price = float(exit_price)
        side = str(row.get("Side", row["Signal"])).upper()
        if qty <= 0 or exit_price <= 0 or not np.isfinite(exit_price):
            return False, "Invalid quantity or exit price."
        pnl = paper_trade_exit(entry, exit_price, qty, side)
        prior = float(row.get("RealizedPnL", 0.0) or 0.0)
        total = prior + pnl
        journal.at[index_id, "RemainingQty"] = 0
        journal.at[index_id, "RealizedPnL"] = total
        journal.at[index_id, "ExitPrice"] = exit_price
        journal.at[index_id, "LastPrice"] = exit_price
        journal.at[index_id, "LastUpdateTime"] = now_str()
        journal.at[index_id, "Status"] = "CLOSED"
        journal.at[index_id, "PnL"] = round(total, 2)
        initial_notional = entry * float(row.get("Quantity", qty))
        journal.at[index_id, "PnLPercent"] = round((total / initial_notional * 100) if initial_notional else 0.0, 2)
        journal.at[index_id, "Result"] = "WIN" if total > 0 else "LOSS" if total < 0 else "BREAKEVEN"
        journal.at[index_id, "Notes"] = (str(row.get("Notes", "")) + f" | {reason}").strip(" |")
        if total < 0:
            loss_snapshot = {k: row.get(k) for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"]}
            journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total, side, exit_price, entry)
        save_journal(journal)
        return True, f"Paper position closed. Total realized P&L: ₹{total:,.2f}"
    except Exception as exc:
        log_app_error("close_paper_position", exc)
        return False, f"Could not close paper position: {exc}"


def _paper_partial_exit(index_id, exit_price, quantity, target_flag=None):
    """Execute a virtual partial exit and update remaining quantity/realized P&L."""
    journal = load_journal()
    if journal.empty or index_id not in journal.index:
        return False
    row = journal.loc[index_id]
    if str(row.get("Status", "")).upper() != "OPEN":
        return False
    entry = float(row["EntryPrice"])
    side = str(row.get("Side", row["Signal"])).upper()
    remaining = int(float(row.get("RemainingQty", row["Quantity"])))
    q = min(remaining, max(0, int(quantity)))
    if q <= 0:
        return False
    pnl = paper_trade_exit(entry, float(exit_price), q, side, final_close=(q >= remaining))
    total_realized = float(row.get("RealizedPnL", 0.0) or 0.0) + pnl
    new_remaining = remaining - q
    journal.at[index_id, "RemainingQty"] = new_remaining
    journal.at[index_id, "RealizedPnL"] = total_realized
    journal.at[index_id, "PnL"] = round(total_realized, 2)
    journal.at[index_id, "LastPrice"] = float(exit_price)
    journal.at[index_id, "LastUpdateTime"] = now_str()
    if target_flag:
        journal.at[index_id, target_flag] = True
    if new_remaining <= 0:
        journal.at[index_id, "RemainingQty"] = 0
        journal.at[index_id, "ExitPrice"] = float(exit_price)
        journal.at[index_id, "Status"] = "CLOSED"
        journal.at[index_id, "Result"] = "WIN" if total_realized > 0 else "LOSS" if total_realized < 0 else "BREAKEVEN"
        initial_notional = entry * float(row.get("Quantity", q))
        journal.at[index_id, "PnLPercent"] = round((total_realized / initial_notional * 100) if initial_notional else 0.0, 2)
        if total_realized < 0:
            loss_snapshot = {k: row.get(k) for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"]}
            journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total_realized, side, float(exit_price), entry)
    save_journal(journal)
    return True


def auto_paper_trade_exit(current_price, symbol=None, atr=None):
    """Auto-manage only the currently selected symbol; scale out at T1/T2 and close at T3/SL."""
    try:
        price = float(current_price)
        if not np.isfinite(price) or price <= 0:
            return {"closed": 0, "partial": 0, "updated": 0}
        journal = load_journal()
        if journal.empty:
            return {"closed": 0, "partial": 0, "updated": 0}
        open_rows = journal[journal["Status"].astype(str).str.upper().eq("OPEN")].copy()
        closed_count = partial_count = updated_count = 0
        for idx, original in open_rows.iterrows():
            if symbol and str(original.get("Symbol", "")) != str(symbol):
                continue
            if not _paper_bool(original.get("AutoExitEnabled", False), False):
                continue
            # Refresh the row after any partial operation.
            journal = load_journal()
            if idx not in journal.index or str(journal.at[idx, "Status"]).upper() != "OPEN":
                continue
            row = journal.loc[idx]
            side = str(row.get("Side", row.get("Signal", "BUY"))).upper()
            is_long = side == "BUY"
            entry = float(row["EntryPrice"])
            initial_qty = int(float(row.get("Quantity", 1)))
            rem_qty = int(float(row.get("RemainingQty", initial_qty)))
            if rem_qty <= 0:
                continue
            row_atr = row.get("ATR", np.nan)
            try:
                atr_value = float(row_atr) if pd.notna(row_atr) and float(row_atr) > 0 else float(atr or 0.0)
            except Exception:
                atr_value = float(atr or 0.0)
            trail_gap = max(0.5 * atr_value, entry * 0.0005, 0.01)
            old_trail = row.get("TrailingStop", row.get("StopLoss", np.nan))
            try:
                old_trail = float(old_trail) if pd.notna(old_trail) else float(row["StopLoss"])
            except Exception:
                old_trail = float(row["StopLoss"])
            new_trail = max(old_trail, price - trail_gap) if is_long else min(old_trail, price + trail_gap)
            journal.at[idx, "TrailingStop"] = new_trail
            journal.at[idx, "LastPrice"] = price
            journal.at[idx, "LastUpdateTime"] = now_str()
            save_journal(journal)

            stop_loss = float(row["StopLoss"])
            effective_stop = max(stop_loss, new_trail) if is_long else min(stop_loss, new_trail)
            stop_hit = price <= effective_stop if is_long else price >= effective_stop
            if stop_hit:
                ok, _ = close_paper_position(idx, price, "Auto Exit — Stop Loss / Trailing Stop")
                if ok:
                    closed_count += 1
                continue

            target_specs = [
                ("T1_Hit", "Target1", 0.34),
                ("T2_Hit", "Target2", 0.33),
                ("T3_Hit", "Target3", 1.00),
            ]
            for flag, target_col, fraction in target_specs:
                journal = load_journal()
                if idx not in journal.index or str(journal.at[idx, "Status"]).upper() != "OPEN":
                    break
                row_now = journal.loc[idx]
                if _paper_bool(row_now.get(flag, False), False):
                    continue
                target = row_now.get(target_col, np.nan)
                if pd.isna(target):
                    continue
                target = float(target)
                target_hit = price >= target if is_long else price <= target
                if not target_hit:
                    continue
                current_remaining = int(float(row_now.get("RemainingQty", row_now["Quantity"])))
                if fraction >= 1.0:
                    q_exit = current_remaining
                else:
                    q_exit = min(current_remaining, max(1, int(round(initial_qty * fraction))))
                if _paper_partial_exit(idx, target, q_exit, target_flag=flag):
                    partial_count += 1
                    # After T1 protect the remainder around breakeven; after T2 protect T1.
                    journal = load_journal()
                    if idx in journal.index and str(journal.at[idx, "Status"]).upper() == "OPEN":
                        protect_level = entry if flag == "T1_Hit" else float(row_now.get("Target1", entry))
                        existing = float(journal.at[idx, "TrailingStop"])
                        tightened = max(existing, protect_level) if is_long else min(existing, protect_level)
                        journal.at[idx, "TrailingStop"] = tightened
                        save_journal(journal)
            # Count a fully closed position once; partial_count tracks scale-outs.
            final_journal = load_journal()
            if idx in final_journal.index and str(final_journal.at[idx, "Status"]).upper() == "CLOSED":
                closed_count += 1
            updated_count += 1
        return {"closed": closed_count, "partial": partial_count, "updated": updated_count}
    except Exception as exc:
        log_app_error("auto_paper_trade_exit", exc)
        return {"closed": 0, "partial": 0, "updated": 0}



# =========================================================
# BACKTEST / WALK FORWARD / SHARPE
# =========================================================

def render_paper_trading_no_data_controls(selected, timeframe, current_price=None):
    """Keep virtual paper controls usable when the provider has no valid OHLC quote."""
    st.subheader("Paper Trading — Manual / Auto Exit | Market Feed Unavailable")
    st.caption("Virtual orders only; no broker orders are sent. Signal scores/PCR need provider data. Auto Exit can use a valid Angel LTP when available; otherwise it waits for the market feed. Enter a manual price only when you know the current traded price.")
    state = load_paper_state()
    journal = load_journal()
    auto_enabled = render_paper_exit_mode_selector()
    journal = sync_open_position_auto_exit(auto_enabled, journal)

    # In the no-OHLC branch, a valid Angel LTP can still manage the selected
    # symbol. If no live quote exists, Auto Exit stays armed and waits safely.
    if auto_enabled:
        try:
            valid_price = float(current_price) if current_price is not None else 0.0
        except Exception:
            valid_price = 0.0
        if np.isfinite(valid_price) and valid_price > 0:
            auto_result = auto_paper_trade_exit(valid_price, symbol=selected, atr=None)
            if auto_result.get("closed", 0) or auto_result.get("partial", 0):
                st.success(
                    f"Auto Exit: {auto_result.get('closed', 0)} position(s) closed; "
                    f"{auto_result.get('partial', 0)} target scale-out(s) completed."
                )
        else:
            st.warning("Auto Exit is ENABLED but waiting for a valid selected-symbol market price. It will manage this symbol on the next refresh after a price is available.")

    # Re-read the account after any automatic partial/full exit before showing balances.
    state = load_paper_state()
    journal = load_journal()
    balance = float(state.loc[0, "Balance"])
    used = float(state.loc[0, "UsedMargin"])
    open_rows = journal[journal.get("Status", pd.Series(dtype=str)).astype(str).str.upper().eq("OPEN")].copy() if not journal.empty else pd.DataFrame()
    realized = 0.0
    if not journal.empty:
        if "RealizedPnL" in journal.columns:
            realized += pd.to_numeric(journal["RealizedPnL"], errors="coerce").fillna(0).sum()
        if "PnL" in journal.columns:
            closed_mask = journal.get("Status", pd.Series("", index=journal.index)).astype(str).str.upper().eq("CLOSED")
            zero_realized = pd.to_numeric(journal.get("RealizedPnL", pd.Series(0, index=journal.index)), errors="coerce").fillna(0).eq(0)
            realized += pd.to_numeric(journal.loc[closed_mask & zero_realized, "PnL"], errors="coerce").fillna(0).sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Paper Cash", f"₹{balance:,.2f}")
    with c2:
        st.metric("Used Virtual Margin", f"₹{used:,.2f}")
    with c3:
        st.metric("Realized Paper P&L", f"₹{realized:,.2f}")
    with c4:
        st.metric("Open Positions", int(len(open_rows)))
    st.write("Bullish Score: N/A  |  Bearish Score: N/A  |  PCR: N/A — valid candles/option-chain data are unavailable.")
    with st.expander("Open a manual paper position", expanded=True):
        entry_price = st.number_input("Manual paper entry price (₹)", min_value=0.0, value=0.0, step=0.05, key="paper_no_data_entry_price")
        atr_value = st.number_input("ATR estimate (optional; zero uses a small percentage fallback)", min_value=0.0, value=0.0, step=0.05, key="paper_no_data_atr")
        qty = st.number_input("Quantity", min_value=1, max_value=1000000, value=1, step=1, key="paper_no_data_qty")
        if entry_price > 0:
            buy_plan = get_suggested_paper_levels({"atr": atr_value, "signal_type": "neutral"}, "BUY", entry_price)
            sell_plan = get_suggested_paper_levels({"atr": atr_value, "signal_type": "neutral"}, "SELL", entry_price)
            st.write({"BUY": {"Entry": round(buy_plan["entry"], 2), "SL": round(buy_plan["stop_loss"], 2), "T1": round(buy_plan["target1"], 2), "T2": round(buy_plan["target2"], 2), "T3": round(buy_plan["target3"], 2), "Trailing SL": round(buy_plan["trailing_stop"], 2)}, "SELL": {"Entry": round(sell_plan["entry"], 2), "SL": round(sell_plan["stop_loss"], 2), "T1": round(sell_plan["target1"], 2), "T2": round(sell_plan["target2"], 2), "T3": round(sell_plan["target3"], 2), "Trailing SL": round(sell_plan["trailing_stop"], 2)}})
        b1, b2 = st.columns(2)
        with b1:
            buy_manual = st.button("BUY / LONG", key="paper_no_data_buy", use_container_width=True)
        with b2:
            sell_manual = st.button("SELL / SHORT", key="paper_no_data_sell", use_container_width=True)
        if buy_manual or sell_manual:
            side = "BUY" if buy_manual else "SELL"
            no_data_risk_ok, no_data_risk_message = risk_check_before_trade()
            if entry_price <= 0:
                st.error("Enter a valid manual entry price first.")
            elif not no_data_risk_ok:
                st.error(f"Paper order blocked by risk controls: {no_data_risk_message}")
            elif entry_price * int(qty) > balance:
                st.error("Insufficient virtual cash for the requested quantity.")
            else:
                analysis_stub = {"atr": atr_value, "signal_type": "neutral", "trade_quality_score": 0, "validation_score": 0, "strategy_grade": "N/A"}
                plan = get_suggested_paper_levels(analysis_stub, side, entry_price)
                if paper_trade_entry(entry_price, int(qty), side=side):
                    add_trade_to_journal(selected, timeframe, side, entry_price, int(qty), plan["stop_loss"], plan["target1"], plan["target2"], plan["target3"], plan["trailing_stop"], analysis_stub, notes="Manual price paper entry while market feed unavailable", auto_exit=auto_enabled, atr=plan["atr"])
                    st.success(f"Manual paper {side} position opened. No real order was sent.")
                    st.rerun()
                else:
                    st.error("Paper account could not reserve the required virtual cash.")
    st.markdown("**Open paper positions — manual close**")
    if open_rows.empty:
        st.info("No open paper positions.")
    else:
        for idx, pos in open_rows.iterrows():
            entry = float(pos["EntryPrice"])
            try:
                last = float(pos.get("LastPrice", entry))
                if not np.isfinite(last) or last <= 0:
                    last = entry
            except Exception:
                last = entry
            with st.container(border=True):
                st.write(f"**{pos.get('Symbol', 'N/A')} — {pos.get('Side', pos.get('Signal', 'N/A'))}** · Remaining {int(float(pos.get('RemainingQty', pos.get('Quantity', 0))))} · Entry ₹{entry:,.2f}")
                exit_price = st.number_input("Manual exit price (₹)", min_value=0.01, value=float(last), step=0.05, key=f"paper_no_data_close_price_{idx}")
                if st.button("CLOSE PAPER POSITION", key=f"paper_no_data_close_{idx}"):
                    closed, message = close_paper_position(idx, exit_price, "Manual close without feed")
                    if closed:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    st.markdown("**Paper journal**")
    if journal.empty:
        st.caption("No paper trades recorded.")
    else:
        cols = [c for c in ["DateTime", "Symbol", "Side", "EntryPrice", "Quantity", "RemainingQty", "ExitPrice", "Status", "RealizedPnL", "PnL", "Result"] if c in journal.columns]
        st.dataframe(journal[cols].tail(100), use_container_width=True, hide_index=True)


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


def fallback_ema_backtest(df, symbol="", timeframe_label="", cost_bps=5.0):
    """Transparent fallback when the strict signal engine produces no trades.

    Uses closed-bar EMA trend/pullback signals with ATR-based exits. This is
    explicitly labelled as a fallback in the trade log; it never fabricates data.
    """
    if df is None or df.empty or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    if not {"EMA20", "EMA50", "RSI", "ATR"}.issubset(work.columns):
        work = add_indicators(work)
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[~work.index.isna()].replace([np.inf, -np.inf], np.nan)
    needed = ["Open", "High", "Low", "Close", "EMA20", "EMA50", "RSI", "ATR"]
    work = work.dropna(subset=needed)
    if len(work) < 35:
        return pd.DataFrame()

    trades = []
    i = 1
    while i < len(work) - 2:
        prev, cur = work.iloc[i - 1], work.iloc[i]
        bullish_cross = prev["EMA20"] <= prev["EMA50"] and cur["EMA20"] > cur["EMA50"]
        bearish_cross = prev["EMA20"] >= prev["EMA50"] and cur["EMA20"] < cur["EMA50"]
        bullish_pullback = (cur["EMA20"] > cur["EMA50"] and prev["Close"] < prev["EMA20"] and cur["Close"] >= cur["EMA20"] and 45 <= cur["RSI"] <= 70)
        bearish_pullback = (cur["EMA20"] < cur["EMA50"] and prev["Close"] > prev["EMA20"] and cur["Close"] <= cur["EMA20"] and 30 <= cur["RSI"] <= 55)
        side = "BUY" if (bullish_cross or bullish_pullback) else ("SELL" if (bearish_cross or bearish_pullback) else None)
        if side is None:
            i += 1
            continue

        entry_i = i + 1
        entry = float(work.iloc[entry_i]["Open"])
        atr = float(cur["ATR"])
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        risk = 1.5 * atr
        target = 2.0 * atr
        sl = entry - risk if side == "BUY" else entry + risk
        tp = entry + target if side == "BUY" else entry - target
        future = work.iloc[entry_i:min(entry_i + 11, len(work))]
        if future.empty:
            break
        exit_price, result, exit_i = None, "TIME EXIT", entry_i
        for j, (_, bar) in enumerate(future.iterrows()):
            exit_i = entry_i + j
            if side == "BUY":
                if float(bar["Low"]) <= sl:
                    exit_price, result = sl, "LOSS"
                    break
                if float(bar["High"]) >= tp:
                    exit_price, result = tp, "WIN"
                    break
            else:
                if float(bar["High"]) >= sl:
                    exit_price, result = sl, "LOSS"
                    break
                if float(bar["Low"]) <= tp:
                    exit_price, result = tp, "WIN"
                    break
        if exit_price is None:
            exit_price = float(future.iloc[-1]["Close"])
            result = "WIN" if ((exit_price > entry) if side == "BUY" else (exit_price < entry)) else "LOSS"
        gross = (exit_price - entry) if side == "BUY" else (entry - exit_price)
        cost = (abs(entry) + abs(exit_price)) * (float(cost_bps) / 10000.0)
        pnl = round(gross - cost, 2)
        trades.append({
            "Date": pd.to_datetime(work.index[i]).strftime("%d-%m-%Y"),
            "Time": pd.to_datetime(work.index[i]).strftime("%H:%M:%S"),
            "Symbol": symbol, "Timeframe": timeframe_label,
            "Signal": side, "Entry": round(entry, 2), "SL": round(sl, 2),
            "T1": round(tp, 2), "T2": np.nan, "T3": np.nan,
            "Exit": round(exit_price, 2), "Result": result, "PnL": pnl,
            "T1_Hit": result == "WIN", "T2_Hit": False, "T3_Hit": False,
            "Method": "EMA trend/pullback fallback"
        })
        i = max(i + 1, exit_i + 1)
    return pd.DataFrame(trades)


def simple_backtest(df, symbol="", timeframe_label="", option_metrics=None, news_info=None):
    """Causal same-timeframe backtest with next-bar fills and staged ATR exits.

    Current-day PCR/option-chain values and fabricated MTF alignment are never
    passed into historical candles. Stop is checked before targets when OHLC
    cannot reveal intrabar order; this is the conservative assumption.
    """
    if df is None or df.empty or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return 0, 0, 0, 0.0, pd.DataFrame()
    work = df.copy()
    required_cols = [
        "Open", "High", "Low", "Close", "EMA20", "EMA50", "EMA200", "VWAP", "RSI",
        "MACD", "SignalLine", "ATR", "ADX", "SuperTrendDirection", "Volume_Spike"
    ]
    if any(col not in work.columns for col in required_cols):
        try:
            work = add_indicators(work)
        except Exception as exc:
            log_app_error("backtest_indicator_rebuild", exc)
            return 0, 0, 0, 0.0, pd.DataFrame()
    if any(col not in work.columns for col in required_cols):
        return 0, 0, 0, 0.0, pd.DataFrame()

    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[~work.index.isna()].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["Open", "High", "Low", "Close", "EMA20", "EMA50", "EMA200", "RSI", "MACD", "SignalLine", "ATR", "ADX"])
    if len(work) < 70:
        return 0, 0, 0, 0.0, pd.DataFrame()

    trades = []
    # EMA50 and the signal structure need a meaningful warm-up window.
    for i in range(55, len(work) - 10):
        # analyze_signal only needs the current/prior bar, recent 20-bar levels,
        # recent HA candles and already-computed indicators. A rolling 250-bar
        # view avoids repeatedly copying an ever-growing history (O(n^2)).
        mini_df = work.iloc[max(0, i - 249):i + 1]
        latest = mini_df.iloc[-1]
        # Do not inject the current option chain or synthetic 4/4 MTF alignment
        # into historical observations; those values were not known then.
        historical_news = news_info if isinstance(news_info, dict) else {"allow_trade": True, "message": "No historical news replay"}
        analysis = analyze_signal(
            mini_df,
            htf_bias=None,
            timeframe_label=timeframe_label if timeframe_label else "1d",
            backtest_win_rate=0.0,
            mtf_bull_count=0,
            mtf_bear_count=0,
            mtf_available_count=0,
            option_metrics=None,
            news_info=historical_news,
            backtest_trade_count=0,
            backtest_is_comparable=False,
        )
        if analysis["signal_text"] not in ("STRONG BUY", "STRONG SELL"):
            continue

        future = work.iloc[i + 1:i + 11]
        if future.empty:
            continue
        # A completed-bar signal is filled at the next candle OPEN.
        entry = float(future.iloc[0]["Open"])
        atr_value = float(analysis.get("atr") or 0.0)
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
            continue
        risk = max(atr_value, entry * 0.001, 0.01)
        side = "BUY" if analysis["signal_type"] == "buy" else "SELL"
        is_long = side == "BUY"
        sl = entry - risk if is_long else entry + risk
        t1 = entry + 2*risk if is_long else entry - 2*risk
        t2 = entry + 3*risk if is_long else entry - 3*risk
        t3 = entry + 4*risk if is_long else entry - 4*risk
        trail = sl
        trail_gap = max(0.5 * atr_value, entry * 0.0005, 0.01)
        remaining = 1.0
        realized_points = 0.0
        t1_hit = t2_hit = t3_hit = False
        exit_reason = "TIME EXIT"
        exit_time = future.index[-1]
        final_exit_price = float(future.iloc[-1]["Close"])

        for bar_time, bar in future.iterrows():
            op = float(bar["Open"]); hi = float(bar["High"]); lo = float(bar["Low"]); cl = float(bar["Close"])
            effective_stop = max(sl, trail) if is_long else min(sl, trail)
            stop_hit = (lo <= effective_stop) if is_long else (hi >= effective_stop)
            if stop_hit:
                # If a candle gaps through the stop, assume the worse OPEN fill.
                fill = min(op, effective_stop) if is_long else max(op, effective_stop)
                realized_points += ((fill - entry) if is_long else (entry - fill)) * remaining
                final_exit_price = float(fill)
                remaining = 0.0
                exit_reason = "STOP / TRAILING STOP"
                exit_time = bar_time
                break

            # Process targets in sequence. If several are crossed in one candle,
            # stop has priority; the scale-outs are applied at their target prices.
            target_sequence = [
                ("T1", t1, 0.34),
                ("T2", t2, 0.33),
                ("T3", t3, 1.00),
            ]
            for label, target, fraction in target_sequence:
                flag_already_hit = {"T1": t1_hit, "T2": t2_hit, "T3": t3_hit}[label]
                target_hit = (hi >= target) if is_long else (lo <= target)
                if flag_already_hit or not target_hit or remaining <= 0:
                    continue
                qty_exit = remaining if label == "T3" else min(remaining, fraction)
                realized_points += ((target - entry) if is_long else (entry - target)) * qty_exit
                remaining -= qty_exit
                if label == "T1":
                    t1_hit = True
                    trail = max(trail, entry) if is_long else min(trail, entry)
                elif label == "T2":
                    t2_hit = True
                    trail = max(trail, t1) if is_long else min(trail, t1)
                else:
                    t3_hit = True
                    final_exit_price = float(target)
                    exit_reason = "T3 FINAL EXIT"
                    exit_time = bar_time
                if remaining <= 1e-9:
                    remaining = 0.0
                    break
            if remaining <= 0:
                break

            # Close-based dynamic trailing is applied for the NEXT bar, avoiding
            # using a bar's final close to stop out earlier inside that same bar.
            dynamic_trail = cl - trail_gap if is_long else cl + trail_gap
            trail = max(trail, dynamic_trail) if is_long else min(trail, dynamic_trail)

        if remaining > 0:
            last_bar = future.iloc[-1]
            last_close = float(last_bar["Close"])
            realized_points += ((last_close - entry) if is_long else (entry - last_close)) * remaining
            final_exit_price = last_close
            exit_time = future.index[-1]
            exit_reason = "TIME EXIT"
            remaining = 0.0

        result = "WIN" if realized_points > 0 else "LOSS" if realized_points < 0 else "BREAKEVEN"
        trades.append({
            "Date": pd.to_datetime(work.index[i]).strftime("%d-%m-%Y"),
            "Time": pd.to_datetime(work.index[i]).strftime("%H:%M:%S"),
            "Symbol": symbol, "Timeframe": timeframe_label,
            "Signal": side, "Entry": round(entry, 2), "SL": round(sl, 2),
            "T1": round(t1, 2), "T2": round(t2, 2), "T3": round(t3, 2),
            "Exit": round(float(final_exit_price), 2),
            "Result": result, "PnL": round(realized_points, 2),
            "T1_Hit": t1_hit, "T2_Hit": t2_hit, "T3_Hit": t3_hit,
            "ExitReason": exit_reason, "ExitModel": "Next-open fill; 34/33/remainder scale-out; conservative stop-first OHLC"
        })

    trades_df = pd.DataFrame(trades)
    # If the composite engine has no completed sample, expose a separate,
    # explicitly labelled EMA fallback. The UI does not call that its win chance.
    if trades_df.empty:
        trades_df = fallback_ema_backtest(work, symbol, timeframe_label)
    total = len(trades_df)
    wins = int((trades_df["Result"] == "WIN").sum()) if total else 0
    losses = int((trades_df["Result"] == "LOSS").sum()) if total else 0
    win_rate = (wins / total * 100.0) if total else 0.0
    return wins, losses, total, round(win_rate, 2), trades_df


def walk_forward_validation(df, symbol="", timeframe_label="", option_metrics=None, news_info=None):
    """Chronological walk-forward report; fallback EMA folds are labelled and excluded from composite averages."""
    try:
        if df.empty or len(df) < 150:
            return {"folds": 0, "validated_folds": 0, "avg_win_rate": None, "avg_profit_factor": None, "avg_expectancy": None, "table": pd.DataFrame()}
        # Walk-forward is an interactive report, not a full-history batch job.
        # Bound its workload so long 1m histories cannot freeze the Streamlit run.
        df = df.tail(1200).copy()

        fold_results = []
        total_len = len(df)
        train_size = int(total_len * 0.6)
        test_size = int(total_len * 0.2)
        start = 0
        fold_no = 1
        while (start + train_size + test_size) <= total_len:
            test_start = start + train_size
            test_end = test_start + test_size
            # Include 55 prior candles only as indicator/warm-up context. The
            # evaluator starts at index 55, so its first eligible signal is the
            # first candle in this fold's test window, not in the warm-up window.
            warmup_start = max(0, test_start - 55)
            test_with_warmup = df.iloc[warmup_start:test_end].copy()
            wins, losses, total, win_rate, trades_df = simple_backtest(
                test_with_warmup, symbol=symbol, timeframe_label=timeframe_label,
                option_metrics=None,
                news_info={"allow_trade": True, "message": "Historical news not replayed", "events": []},
            )
            stats = compute_trade_statistics(trades_df)
            is_fallback = not trades_df.empty and "Method" in trades_df.columns
            fold_results.append({
                "Fold": fold_no,
                "TestStart": str(test_with_warmup.index[55].date()) if len(test_with_warmup) > 55 else "N/A",
                "TestEnd": str(test_with_warmup.index[-1].date()) if not test_with_warmup.empty else "N/A",
                "Trades": total,
                "Wins": wins,
                "Losses": losses,
                "WinRate": win_rate if total else np.nan,
                "ProfitFactor": stats["profit_factor"] if total else np.nan,
                "Expectancy": stats["expectancy"] if total else np.nan,
                "Method": "EMA fallback — not comparable" if is_fallback else "Composite signal engine" if total else "No completed trades",
            })
            fold_no += 1
            start += test_size

        result_df = pd.DataFrame(fold_results)
        valid = result_df[result_df["Method"] == "Composite signal engine"].copy()
        if valid.empty:
            return {"folds": len(result_df), "validated_folds": 0, "avg_win_rate": None,
                    "avg_profit_factor": None, "avg_expectancy": None, "table": result_df}
        return {
            "folds": len(result_df),
            "validated_folds": len(valid),
            "avg_win_rate": round(float(valid["WinRate"].mean()), 2),
            "avg_profit_factor": round(float(valid["ProfitFactor"].replace(999.0, 5.0).mean()), 2),
            "avg_expectancy": round(float(valid["Expectancy"].mean()), 2),
            "table": result_df,
        }
    except Exception as exc:
        log_app_error("walk_forward_validation", exc)
        return {"folds": 0, "validated_folds": 0, "avg_win_rate": None,
                "avg_profit_factor": None, "avg_expectancy": None, "table": pd.DataFrame()}


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
    candles = get_angel_candles_cached(api_key, client_code, pin, totp_secret, selected, interval)
    data = candles_to_df(candles)
    if data.empty:
        return data
    if tf == "3m":
        data = resample_ohlcv(data, "3min")
    elif tf == "10m":
        data = resample_ohlcv(data, "10min")
    return data


def get_mtf_setup_metrics(data_source, selected, ticker, tf, angel_creds):
    """Return real timeframe-specific EMA/RSI/ADX/relative-volume metrics."""
    result = {"timeframe": tf, "available": False, "ema9": np.nan, "ema21": np.nan,
              "ema50": np.nan, "rsi": np.nan, "adx": np.nan, "relative_volume": np.nan,
              "trend": "Unavailable"}
    try:
        frame = get_data_for_timeframe(data_source, selected, ticker, tf, angel_creds)
        if frame is None or frame.empty:
            return result
        frame = add_indicators(frame).replace([np.inf, -np.inf], np.nan)
        if len(frame) < 25:
            return result
        row = frame.iloc[-1]
        for out_key, col in (("ema9", "EMA9"), ("ema21", "EMA21"), ("ema50", "EMA50"),
                             ("rsi", "RSI"), ("adx", "ADX"), ("relative_volume", "RelativeVolume")):
            result[out_key] = _safe_num(row.get(col))
        result["available"] = all(np.isfinite(result[k]) for k in ("ema9", "ema21", "ema50", "rsi", "adx"))
        if result["available"]:
            result["trend"] = "Bullish" if result["ema9"] > result["ema21"] else "Bearish" if result["ema9"] < result["ema21"] else "Neutral"
        return result
    except Exception:
        return result


def apply_tiered_early_alerts(analysis, metrics_by_tf):
    """Apply user-defined 3m early, 3m+5m confirmed, and 3m+5m+15m strong stages."""
    m3, m5, m15 = (metrics_by_tf.get(tf, {}) for tf in ("3m", "5m", "15m"))
    valid = lambda m: bool(m.get("available"))
    def bullish(m): return valid(m) and m["ema9"] > m["ema21"]
    def bearish(m): return valid(m) and m["ema9"] < m["ema21"]
    def stacked_bull(m): return bullish(m) and m["ema21"] > m["ema50"]
    def stacked_bear(m): return bearish(m) and m["ema21"] < m["ema50"]
    r3 = _safe_num(m3.get("rsi")); a3 = _safe_num(m3.get("adx")); v3 = _safe_num(m3.get("relative_volume"))
    r5 = _safe_num(m5.get("rsi")); a5 = _safe_num(m5.get("adx")); v5 = _safe_num(m5.get("relative_volume"))
    # Early alert is deliberately a watch state, not an entry approval.
    buy_early = bullish(m3) and np.isfinite(r3) and r3 > 50
    buy_confirmed = bullish(m3) and bullish(m5) and r3 > 55 and a3 > 20 and np.isfinite(v3) and v3 > 1.5
    buy_strong = (stacked_bull(m3) and stacked_bull(m5) and stacked_bull(m15)
                  and r3 > 60 and a3 > 25 and np.isfinite(v3) and v3 > 1.8)
    sell_early = bearish(m3) and np.isfinite(r3) and r3 < 50
    sell_confirmed = bearish(m3) and bearish(m5) and r3 < 45 and a3 > 20 and np.isfinite(v3) and v3 > 1.5
    sell_strong = (stacked_bear(m3) and stacked_bear(m5) and stacked_bear(m15)
                   and r3 < 40 and a3 > 25 and np.isfinite(v3) and v3 > 1.8)
    if buy_strong: stage, direction, tier = "STRONG ENTRY – BUY", "BUY", 3
    elif buy_confirmed: stage, direction, tier = "CONFIRMED ENTRY – BUY", "BUY", 2
    elif buy_early: stage, direction, tier = "EARLY ALERT – TREND START (BUY WATCH)", "BUY", 1
    elif sell_strong: stage, direction, tier = "STRONG ENTRY – SELL", "SELL", 3
    elif sell_confirmed: stage, direction, tier = "CONFIRMED ENTRY – SELL", "SELL", 2
    elif sell_early: stage, direction, tier = "EARLY ALERT – TREND START (SELL WATCH)", "SELL", 1
    else: stage, direction, tier = "NO TIERED ALERT", "NEUTRAL", 0
    analysis["signal_stage"] = stage
    analysis["tiered_alert_direction"] = direction
    analysis["tiered_alert_level"] = tier
    analysis["tiered_alert_metrics"] = metrics_by_tf
    analysis["tiered_alert_rules"] = {
        "3m_trend_start": "EMA9 > EMA21 and RSI > 50",
        "confirmed_buy": "3m + 5m EMA9 > EMA21; 3m RSI > 55; 3m ADX > 20; 3m Relative Volume > 1.5x Volume MA20",
        "strong_buy": "3m + 5m + 15m EMA9 > EMA21 > EMA50; 3m RSI > 60; 3m ADX > 25; 3m Relative Volume > 1.8x Volume MA20",
        "confirmed_sell": "Mirrored bearish conditions; 3m RSI < 45",
        "strong_sell": "Mirrored bearish EMA stack; 3m RSI < 40"
    }
    return analysis


def get_mtf_signal(data_source, selected, ticker, tf, angel_creds):
    data = get_data_for_timeframe(data_source, selected, ticker, tf, angel_creds)
    if data.empty:
        return "⚪ N/A"
    data = add_indicators(data).replace([np.inf, -np.inf], np.nan).ffill()
    if len(data) < 55:
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
    "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HCLTECH": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS",
    "TECHM": "TECHM.NS",
    "TMPV": "TMPV.NS",
    "MARUTI": "MARUTI.NS",
    "M&M": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "NTPC": "NTPC.NS",
    "BEL": "BEL.NS",
    "ONGC": "ONGC.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "POWERGRID": "POWERGRID.NS",
    "TRENT": "TRENT.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "TATASTEEL": "TATASTEEL.NS"
}

ANGEL_TOKENS = {
    "SBIN": "3045", "RELIANCE": "2885", "INFY": "1594", "TCS": "11536",
    "NIFTY": "99926000", "BANKNIFTY": "99926009", "HDFCBANK": "1333",
    "ICICIBANK": "4963", "ITC": "1660"
}

# SmartAPI quote endpoint requires exchange + trading symbol + symbol token.
ANGEL_TRADING_SYMBOLS = {
    "SBIN": "SBIN-EQ", "RELIANCE": "RELIANCE-EQ", "INFY": "INFY-EQ",
    "TCS": "TCS-EQ", "NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
    "HDFCBANK": "HDFCBANK-EQ", "ICICIBANK": "ICICIBANK-EQ", "ITC": "ITC-EQ"
}
ANGEL_EXCHANGES = {key: "NSE" for key in ANGEL_TOKENS}
ANGEL_EXCHANGES["SENSEX"] = "BSE"
ANGEL_NAME_ALIASES = {
    "NIFTY": ["NIFTY", "NIFTY 50"],
    "BANKNIFTY": ["BANKNIFTY", "NIFTY BANK"],
    "SENSEX": ["SENSEX", "BSE SENSEX", "S&P BSE SENSEX"],
    "TMPV": ["TMPV", "TATA MOTORS"],
    "M&M": ["M&M", "M AND M", "MAHINDRA AND MAHINDRA"],
    "BAJAJ-AUTO": ["BAJAJ-AUTO", "BAJAJ AUTO"],
    "HEROMOTOCO": ["HEROMOTOCO", "HERO MOTOCORP"],
    "LT": ["LT", "LARSEN"],
    "BHARTIARTL": ["BHARTIARTL", "BHARTI AIRTEL"],
    "ADANIPORTS": ["ADANIPORTS", "ADANI PORTS"],
    "POWERGRID": ["POWERGRID", "POWER GRID"],
    "JSWSTEEL": ["JSWSTEEL", "JSW STEEL"],
    "TATASTEEL": ["TATASTEEL", "TATA STEEL"],
}

st.title("Elite Trading Dashboard Final Ultimate")
st.caption("WebSocket Primary | REST Polling Fallback | Walk Forward | Auto Exit | News Filter | Dynamic Option Chain")
st.sidebar.caption(f"App build: {APP_BUILD_ID}")

selected = st.selectbox("Select Stock / Index", list(stocks.keys()))
data_source = st.radio("Data Source", ["Yahoo", "Angel"], horizontal=True)
timeframe = st.selectbox("Select Time Frame", ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"])

st.subheader("Market Data Refresh Controls")
refresh_col1, refresh_col2, refresh_col3 = st.columns([1.25, 1.0, 1.0])
with refresh_col1:
    refresh_market_now = st.button(
        "🔄 Manual Refresh Now", key="refresh_primary_market_data",
        type="primary", use_container_width=True,
        help="Clear market-data caches and request fresh data from the selected provider."
    )
with refresh_col2:
    auto_refresh = st.toggle(
        "Enable Auto Refresh", value=False, key="auto_refresh_enabled",
        help="Turn automatic page refresh ON or OFF. Manual refresh remains available either way."
    )
with refresh_col3:
    refresh_seconds = st.selectbox(
        "Auto interval (seconds)", [5, 10, 15, 30, 60], index=1,
        key="auto_refresh_interval_seconds", disabled=not auto_refresh
    )

if refresh_market_now:
    # Flush every market-data cache used by the board, scanner, chart and context panels.
    for _cached_fn in (
        fetch_data, fetch_yahoo_batch, build_nifty_top30_scanner,
        get_news_events, get_option_chain_snapshot
    ):
        try:
            _cached_fn.clear()
        except Exception as _cache_clear_error:
            log_app_error("manual_refresh_cache_clear", _cache_clear_error)
    st.session_state["angel_history_cache"] = {}
    st.session_state["last_manual_refresh_at"] = now_str()
    st.session_state["force_market_refresh"] = True
else:
    st.session_state["force_market_refresh"] = False

if st.session_state.get("last_manual_refresh_at"):
    st.caption("Last manual refresh: " + str(st.session_state["last_manual_refresh_at"]))
if auto_refresh and AUTO_REFRESH_AVAILABLE:
    st.caption(f"Auto refresh is ON — every {refresh_seconds} seconds. Manual refresh is still available.")
elif auto_refresh and not AUTO_REFRESH_AVAILABLE:
    st.warning("Auto refresh is enabled in the UI, but streamlit-autorefresh is unavailable. Reinstall requirements.txt and redeploy.")
else:
    st.caption("Auto refresh is OFF. Use Manual Refresh Now whenever you want fresh data.")

# Keep this above the OHLC no-data guard: the global/macro and Indian share
# boards must continue to refresh even if the selected primary chart feed fails.
if auto_refresh and AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds * 1000, key="elite_live_market_board_refresh")

with st.expander("Telegram Alert Settings"):
    telegram_enabled = st.checkbox("Enable Telegram Alerts", value=False)
    telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID")
    auto_send_telegram = st.checkbox("Auto-send only High Quality New Signals", value=False)

with st.expander("News Filter API"):
    news_provider = st.selectbox(
        "News Provider", ["Marketaux", "NewsAPI (legacy)"], index=0,
        key="news_provider_selection",
        help="Marketaux is the primary financial-news provider. NewsAPI is retained as a legacy fallback option."
    )
    st.caption(
        "There is no separate provider login popup inside this dashboard. "
        "Marketaux authenticates with the API token from your Marketaux account dashboard."
    )
    if news_provider == "Marketaux":
        if "marketaux_api_token_input" not in st.session_state:
            st.session_state["marketaux_api_token_input"] = (
                st.session_state.get("marketaux_api_token_saved") or _read_news_secret("MARKETAUX_API_TOKEN")
            )
        marketaux_api_token = st.text_input(
            "Marketaux API Token", type="password", key="marketaux_api_token_input",
            help="Paste the API token from your Marketaux account dashboard. It stays in this app session only."
        )
        news_api_key = str(
            marketaux_api_token or st.session_state.get("marketaux_api_token_saved") or ""
        ).strip()
        if marketaux_api_token:
            st.session_state["marketaux_api_token_saved"] = news_api_key
    else:
        if "newsapi_legacy_key_input" not in st.session_state:
            st.session_state["newsapi_legacy_key_input"] = (
                st.session_state.get("newsapi_legacy_key_saved") or _read_news_secret("NEWSAPI_API_KEY")
            )
        newsapi_legacy_key = st.text_input(
            "NewsAPI Key (legacy provider)", type="password", key="newsapi_legacy_key_input"
        )
        news_api_key = str(
            newsapi_legacy_key or st.session_state.get("newsapi_legacy_key_saved") or ""
        ).strip()
        if newsapi_legacy_key:
            st.session_state["newsapi_legacy_key_saved"] = news_api_key

    # Bind the visible test result to a hash of the token, so changing the token
    # automatically invalidates an old green success indicator without exposing it.
    _news_token_fingerprint = hashlib.sha256(news_api_key.encode("utf-8")).hexdigest() if news_api_key else ""
    _stored_news_test_fingerprint = st.session_state.get("news_api_test_token_fingerprint", "")
    if _stored_news_test_fingerprint and _stored_news_test_fingerprint != _news_token_fingerprint:
        st.session_state.pop("news_api_test_result", None)
        st.session_state.pop("news_api_test_token_fingerprint", None)

    if st.button(
        f"Test {news_provider} API Connection",
        key="test_news_api_connection_button",
        use_container_width=True,
        disabled=not bool(news_api_key),
    ):
        with st.spinner(f"Checking {news_provider} API token…"):
            _test_result = test_news_provider_connection(news_api_key, news_provider)
        st.session_state["news_api_test_result"] = _test_result
        st.session_state["news_api_test_token_fingerprint"] = _news_token_fingerprint

    _visible_news_test = st.session_state.get("news_api_test_result")
    if _visible_news_test and st.session_state.get("news_api_test_token_fingerprint") == _news_token_fingerprint:
        if _visible_news_test.get("ok"):
            st.success(_visible_news_test.get("message", "News API connection successful."))
        else:
            st.error(_visible_news_test.get("message", "News API connection failed."))
    else:
        st.info("Enter your API token and press Test API Connection. The result will show whether the provider accepted the token.")

    news_filter_enabled = st.checkbox(
        "Enable News Filter", value=False, key="news_filter_enabled"
    )
    news_lookback_hours = st.selectbox(
        "News risk lookback (hours)", [3, 6, 12, 24], index=1,
        key="news_lookback_hours"
    )
    news_article_limit = st.selectbox(
        "Maximum articles", [5, 10, 15, 20], index=1,
        key="news_article_limit"
    )
    if news_filter_enabled:
        if news_api_key:
            st.caption(f"{news_provider} news filter is enabled. Token is masked and is not saved to a CSV file.")
        else:
            st.warning(f"Enter your {news_provider} API key. The news filter will pause new signal entries until the provider can be checked.")

with st.expander("Economic News Keyword Filter", expanded=False):
    st.caption("Filters fetched news headlines using the economic event keywords listed below. This is a keyword-based news filter, not an official economic calendar.")
    economic_keyword_text = st.text_input(
        "Economic event keywords (comma-separated)",
        value=st.session_state.get("economic_keyword_text", "RBI, CPI, GDP, NFP, FOMC"),
        key="economic_keyword_text",
        help="Edit the plain-text list. Matching is case-insensitive against available headline, source and entity text."
    )
    economic_calendar_filter_enabled = st.checkbox(
        "Enable economic keyword filter", value=True, key="economic_calendar_filter_enabled"
    )
    economic_keywords = [
        item.strip().casefold() for item in str(economic_keyword_text).split(",") if item.strip()
    ]
    st.caption("Active keywords: " + (", ".join(item.upper() for item in economic_keywords) if economic_keywords else "None"))
    st.markdown("**Important:** This filter only searches fetched news headlines and related text for matching keywords. It does not confirm event dates, release times, or whether an announcement is scheduled. Verify official schedules separately.")

with st.expander("Angel One API Settings"):
    # Submit the four credentials together. A Streamlit form prevents partial
    # reruns while a password field is being edited and avoids stale widget reads.
    with st.form("angel_connect_form", clear_on_submit=False):
        angel_api_key = st.text_input(
            "Angel API Key", type="password", key="angel_api_key_input"
        )
        angel_client_code = st.text_input(
            "Angel Client Code", key="angel_client_code_input"
        )
        angel_pin = st.text_input(
            "Angel PIN", type="password", key="angel_pin_input"
        )
        angel_totp_secret = st.text_input(
            "Angel TOTP Secret", type="password", key="angel_totp_secret_input"
        )
        connect_angel = st.form_submit_button("Test Angel Login", use_container_width=True)

    ws_enable = st.checkbox(
        "Use Angel WebSocket as Primary Live Feed", value=True,
        key="angel_ws_primary_enable_v17",
        help="WebSocket is used first for live LTP. If it is disconnected or stale, REST LTP polling is used as fallback. Historical OHLC candles still come from the broker REST candle API."
    )
    polling_mode = st.checkbox(
        "Enable Angel LTP Polling Fallback", value=True,
        key="angel_ltp_polling_fallback_v17",
        help="Automatically use Angel REST LTP if the WebSocket is not connected or its last tick is stale."
    )
    st.caption("WebSocket is primary for live LTP. REST polling is the automatic fallback. Historical OHLC candles continue to come from the broker REST Candle API. Turn the dashboard Auto Refresh ON to repaint incoming WebSocket ticks on screen; manual refresh remains available.")

    if connect_angel:
        # Use the values submitted by the form directly; do not depend on a
        # subsequent rerun to populate password-widget session state.
        _submitted_api = str(angel_api_key or "").strip()
        _submitted_client = str(angel_client_code or "").strip()
        _submitted_pin = str(angel_pin or "").strip()
        _submitted_totp = str(angel_totp_secret or "").strip()
        _submitted_missing = []
        if not _submitted_api:
            _submitted_missing.append("API Key")
        if not _submitted_client:
            _submitted_missing.append("Client Code")
        if not _submitted_pin:
            _submitted_missing.append("PIN")
        if not _submitted_totp:
            _submitted_missing.append("TOTP Secret")

        if _submitted_missing:
            st.session_state["angel_credentials_login_ok"] = False
            st.error("Angel login was not attempted. Missing: " + ", ".join(_submitted_missing))
        else:
            # Keep the credentials only in this Streamlit server session.
            # They are never written to CSV, logs, or displayed in diagnostics.
            st.session_state["angel_saved_api_key"] = _submitted_api
            st.session_state["angel_saved_client_code"] = _submitted_client
            st.session_state["angel_saved_pin"] = _submitted_pin
            st.session_state["angel_saved_totp_secret"] = _submitted_totp
            angel, session = angel_login(
                _submitted_api, _submitted_client, _submitted_pin,
                _submitted_totp, force_refresh=True
            )
            if angel and session:
                st.session_state["angel_credentials_login_ok"] = True
                st.success("Angel One Login Successful. Authenticated session is ready.")
            else:
                st.session_state["angel_credentials_login_ok"] = False
                st.error("Angel login failed. Check API key, Client Code, PIN and TOTP Secret, and review the app error log.")

    ws_action_col1, ws_action_col2 = st.columns(2)
    with ws_action_col1:
        if st.button("Stop WebSocket"):
            st.session_state["ws_manual_pause_requested"] = True
            stop_angel_websocket()
            sync_ws_state_to_session()
            st.warning("WebSocket stopped. REST LTP polling fallback will remain available.")
    with ws_action_col2:
        if st.button("Reconnect / Resume WebSocket"):
            st.session_state["ws_manual_pause_requested"] = False
            ws_set_state(retry_exhausted=False, retries=0)
            st.success("WebSocket reconnect requested. It will start on this rerun.")

# Resolve credentials from the visible widgets first, then the independent
# server-session copy saved during Test Angel Login. This avoids losing them
# when Streamlit cleans up widget state during reruns/auto-refresh.
angel_api_key = str(
    st.session_state.get("angel_api_key_input")
    or st.session_state.get("angel_saved_api_key")
    or angel_api_key or ""
).strip()
angel_client_code = str(
    st.session_state.get("angel_client_code_input")
    or st.session_state.get("angel_saved_client_code")
    or angel_client_code or ""
).strip()
angel_pin = str(
    st.session_state.get("angel_pin_input")
    or st.session_state.get("angel_saved_pin")
    or angel_pin or ""
).strip()
angel_totp_secret = str(
    st.session_state.get("angel_totp_secret_input")
    or st.session_state.get("angel_saved_totp_secret")
    or angel_totp_secret or ""
).strip()

# Mirror any currently entered widget values into the session copy as well.
# Nothing is written to disk; these values live only in the Streamlit session.
if all((angel_api_key, angel_client_code, angel_pin, angel_totp_secret)):
    st.session_state["angel_saved_api_key"] = angel_api_key
    st.session_state["angel_saved_client_code"] = angel_client_code
    st.session_state["angel_saved_pin"] = angel_pin
    st.session_state["angel_saved_totp_secret"] = angel_totp_secret

# This board intentionally renders before the selected OHLC pipeline. It remains
# visible if Angel/Yahoo chart candles are unavailable later in the script.
render_live_global_india_market_board()


ticker = stocks[selected]

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
yahoo_primary_rows = 0
yahoo_fallback_rows = 0
active_data_source = data_source
angel_candles_response = None
angel_candle_diagnostics = {"status": "not_requested"}
angel_ltp_value = None
angel_ltp_status = "not_requested"
angel_ltp_source = "Not requested"
angel_session_ready = False

if "last_good_candles_by_key" not in st.session_state:
    st.session_state["last_good_candles_by_key"] = {}

# A successful login stores the authenticated SmartAPI object/session.  Use
# this same session for both candle history and LTP polling.
_angel_cache_key = f"{angel_client_code}_{angel_api_key}"
_angel_cached_session = st.session_state.get("angel_session")
_angel_session_ready = bool(
    st.session_state.get("angel_api") is not None
    and isinstance(_angel_cached_session, dict)
    and _angel_cached_session.get("data")
    and st.session_state.get("angel_cache_key") == _angel_cache_key
)
# Distinguish missing inputs from an authenticated-session cache. After a Cloud
# redeploy/restart, in-memory SmartAPI sessions are reset and the user must log in
# again in this running app session. A prior successful login from an older build
# is not proof that this new session is authenticated.
_angel_missing_fields = []
if not str(angel_api_key).strip():
    _angel_missing_fields.append("API Key")
if not str(angel_client_code).strip():
    _angel_missing_fields.append("Client Code")
if not _angel_session_ready:
    if not str(angel_pin).strip():
        _angel_missing_fields.append("PIN")
    if not str(angel_totp_secret).strip():
        _angel_missing_fields.append("TOTP Secret")
_angel_credentials_ready = bool(
    str(angel_api_key).strip()
    and str(angel_client_code).strip()
    and (_angel_session_ready or (str(angel_pin).strip() and str(angel_totp_secret).strip()))
)

st.info(
    f"Market data request: {data_source} | {selected} | {timeframe}. "
    "Angel live LTP priority: WebSocket primary with REST polling fallback. Historical OHLC remains REST candle data. "
    "Network requests use bounded timeouts; optional market-board and walk-forward jobs no longer block initial page load."
)

if data_source != "Angel":
    _ws_snapshot = ws_get_state()
    if _ws_snapshot.get("should_run"):
        stop_angel_websocket()

if data_source == "Angel":
    if not _angel_credentials_ready:
        _ws_snapshot = ws_get_state()
        if _ws_snapshot.get("should_run"):
            stop_angel_websocket()
        missing_text = ", ".join(_angel_missing_fields) or "authenticated Angel session"
        angel_ltp_status = "not_requested_missing_credentials"
        angel_candle_diagnostics = {
            "status": "not_requested",
            "reason": f"Required fields/session missing: {missing_text}",
        }
        st.error(
            f"Angel One data requests were NOT sent. Missing: {missing_text}. "
            "Re-enter credentials and click Test Angel Login in this deployed app session."
        )
    else:
        # WebSocket is the primary live-LTP source. REST polling is requested
        # only when the socket is not connected, has no tick yet, or its tick is stale.
        _ws_manual_paused = bool(st.session_state.get("ws_manual_pause_requested", False))
        _ws_snapshot = ws_get_state()
        _ws_retry_blocked_for_same_symbol = bool(
            _ws_snapshot.get("retry_exhausted", False)
            and str(_ws_snapshot.get("symbol")) == str(selected)
        )
        if ws_enable and not _ws_manual_paused and not _ws_retry_blocked_for_same_symbol:
            start_angel_websocket(angel_api_key, angel_client_code, angel_pin, angel_totp_secret, selected)
        elif (not ws_enable or _ws_manual_paused) and _ws_snapshot.get("should_run"):
            stop_angel_websocket()

        _ws_quote = ws_live_quote(
            max_age_seconds=max(20, int(refresh_seconds) * 2 if auto_refresh else 20),
            symbol_name=selected
        ) if ws_enable else None
        if _ws_quote is not None:
            angel_ltp_value = _ws_quote
            angel_ltp_status = "received_websocket_primary"
            angel_ltp_source = "Angel WebSocket (Primary)"
        elif polling_mode:
            with st.spinner(f"Angel One: WebSocket unavailable/stale; requesting REST LTP fallback for {selected}…"):
                angel_ltp_value = get_angel_ltp_polling(
                    angel_api_key, angel_client_code, angel_pin,
                    angel_totp_secret, selected
                )
            angel_ltp_status = "received_polling_fallback" if angel_ltp_value is not None else "polling_fallback_no_quote"
            angel_ltp_source = "Angel REST Polling (Fallback)" if angel_ltp_value is not None else "No live quote"
        else:
            if ws_enable and st.session_state.get("ws_manual_pause_requested", False):
                angel_ltp_status = "websocket_manually_paused"
                angel_ltp_source = "WebSocket paused; no polling quote"
            elif ws_enable and ws_get_state().get("retry_exhausted", False):
                angel_ltp_status = "websocket_retry_limit_reached"
                angel_ltp_source = "WebSocket reconnect required"
            else:
                angel_ltp_status = "websocket_waiting_no_polling_fallback" if ws_enable else "all_live_quote_sources_disabled"
                angel_ltp_source = "Waiting for WebSocket tick" if ws_enable else "Live quote disabled"

        angel_interval = map_angel_interval(timeframe)
        with st.spinner(f"Angel One: requesting {angel_interval} historical candles for {selected}…"):
            angel_candles_response = get_angel_candles_cached(
                angel_api_key, angel_client_code, angel_pin,
                angel_totp_secret, selected, angel_interval,
                force_refresh=bool(refresh_market_now or st.session_state.get("force_market_refresh", False))
            )
        angel_candle_diagnostics = summarize_angel_candle_response(angel_candles_response)
        data = candles_to_df(angel_candles_response)

        angel_history_key = f"{selected}|{angel_interval}"
        if not data.empty:
            last_good_map = st.session_state.get("last_good_candles_by_key", {})
            last_good_map[angel_history_key] = angel_candles_response
            st.session_state["last_good_candles_by_key"] = last_good_map
        else:
            last_good_map = st.session_state.get("last_good_candles_by_key", {})
            last_good = last_good_map.get(angel_history_key)
            data = candles_to_df(last_good)
            if not data.empty:
                st.warning(f"Angel One returned no usable new candles for {selected} / {angel_interval}; showing only the last successful history for this same instrument/timeframe.")

        # Transparent secondary OHLC fallback: keep Angel as the selected broker
        # for LTP, but use Yahoo history if Angel's candle endpoint is empty.
        if data.empty:
            with st.spinner(f"Angel candles unavailable; trying Yahoo Finance fallback for {selected}…"):
                yahoo_fallback = fetch_data(ticker, yf_interval, yf_period)
            yahoo_fallback_rows = int(len(yahoo_fallback))
            if not yahoo_fallback.empty:
                data = yahoo_fallback
                active_data_source = "Yahoo"
                st.warning("Angel One candle history is unavailable. Using Yahoo Finance OHLC as a clearly-labelled fallback; Angel LTP polling remains separate.")

else:
    with st.spinner(f"Yahoo Finance: loading {ticker} ({yf_interval})…"):
        data = fetch_data(ticker, yf_interval, yf_period)
    yahoo_primary_rows = int(len(data))
    if data.empty and _angel_credentials_ready:
        # Optional failover if Yahoo is temporarily unavailable but the user has
        # already connected Angel One in this same session.
        angel_interval = map_angel_interval(timeframe)
        angel_candles_response = get_angel_candles_cached(
            angel_api_key, angel_client_code, angel_pin,
            angel_totp_secret, selected, angel_interval,
            force_refresh=bool(refresh_market_now or st.session_state.get("force_market_refresh", False))
        )
        angel_candle_diagnostics = summarize_angel_candle_response(angel_candles_response)
        angel_fallback = candles_to_df(angel_candles_response)
        if not angel_fallback.empty:
            data = angel_fallback
            active_data_source = "Angel"
            st.warning("Yahoo Finance returned no OHLC rows. Using connected Angel One candle history as a clearly-labelled fallback.")

if data.empty:
    if data_source == "Angel" and (polling_mode or ws_enable):
        if angel_ltp_value is not None:
            st.metric(f"Angel One Latest Traded Price — {angel_ltp_source}", f"{angel_ltp_value:,.2f}")
        elif angel_ltp_status == "not_requested_missing_credentials":
            st.warning(
                "Angel LTP polling was skipped because the current app session is not authenticated. "
                f"Missing: {', '.join(_angel_missing_fields) or 'Angel session'}."
            )
        elif angel_ltp_status == "no_quote":
            st.warning("Angel LTP request was attempted but returned no usable quote. Check the broker response, exchange/symbol mapping and token.")
        else:
            st.caption(f"Angel LTP polling status: {angel_ltp_status}")
    st.warning("No usable OHLC candles were returned by the selected source or its available fallback. The chart and indicator calculations are paused rather than displaying invented candles.")
    st.subheader("Market Data Status")
    st.info(
        f"{data_source} did not return usable historical candles for {selected} ({timeframe}). "
        "Charts and indicator calculations are paused until a valid provider response is available. "
        "The market overview, paper journal and manual paper controls remain available."
    )
    st.subheader("Market Dashboard — awaiting valid OHLC candles")
    st.info("Angel login and quote polling are separate from historical candles. The dashboard will not invent OHLC bars or indicator values when both sources return no usable history.")
    st.subheader("Major Stocks Scanner — NIFTY Top 30")
    st.caption("Yahoo Finance scanner: 30 constituents across 3m / 5m / 10m / 15m. It runs separately from the primary candle feed.")
    st.info("The scanner is deliberately not auto-started on a failed main feed. This prevents slow bulk requests from making the whole page appear blank.")
    if st.button("Run / refresh NIFTY Top 30 scanner", key="run_scanner_no_main_data"):
        fetch_yahoo_batch.clear()
        build_nifty_top30_scanner.clear()
        try:
            with st.spinner("Fetching NIFTY Top 30 in two Yahoo Finance batch requests…"):
                st.session_state["nifty30_scanner_result"] = build_nifty_top30_scanner()
            st.session_state["nifty30_scanner_last_run"] = now_str()
        except Exception as scanner_error:
            st.error(f"Scanner error: {scanner_error}")
            log_app_error("nifty_top30_scanner_ui", scanner_error)
    scanner_result = st.session_state.get("nifty30_scanner_result")
    if isinstance(scanner_result, pd.DataFrame) and not scanner_result.empty:
        st.caption("Last scanner run: " + str(st.session_state.get("nifty30_scanner_last_run", "unknown")))
        render_nifty_scanner_result(scanner_result)
    else:
        st.dataframe(blank_nifty_scanner_frame(), use_container_width=True, hide_index=True)
        st.caption("Press Run / refresh NIFTY Top 30 scanner to retrieve market data. Prices and trends are not fabricated.")
    st.subheader("Charts and analytics")
    st.info("Heikin-Ashi candles, MACD, RSI, VWAP, ADX, MTF, signal scoring and backtest are paused until a real OHLC history response is received.")
    st.subheader("Saved Trade Journal")
    try:
        saved_journal = load_journal()
        st.dataframe(saved_journal.tail(100), use_container_width=True, hide_index=True)
    except Exception as journal_error:
        st.error(f"Journal error: {journal_error}")
    st.info("Live ticks may be unavailable outside market hours. Historical candles and live quotes depend on the selected provider and instrument permissions.")
    render_paper_trading_no_data_controls(
        selected, timeframe,
        current_price=(angel_ltp_value if data_source == "Angel" else None)
    )
    if auto_refresh and AUTO_REFRESH_AVAILABLE:
        st.caption("Auto-refresh remains active for the Global/Macro and Major Indian Stocks market board. The chart waits for valid OHLC candles.")
    elif auto_refresh and not AUTO_REFRESH_AVAILABLE:
        st.warning("Auto-refresh is enabled but streamlit-autorefresh is not installed. Install requirements.txt and redeploy.")
    st.stop()

if active_data_source != data_source:
    st.info(f"Selected source: {data_source} | Actual OHLC candle source: {active_data_source}. The data source is labelled separately so fallback data is not mistaken for broker data.")

if timeframe == "3m":
    data = resample_ohlcv(data, "3min")
elif timeframe == "10m":
    data = resample_ohlcv(data, "10min")

ok, msg = data_integrity_check(data, ["Open", "High", "Low", "Close", "Volume"], min_rows=2)
if not ok:
    st.error(f"Data integrity failed: {msg}")
    st.stop()

data = add_indicators(data).replace([np.inf, -np.inf], np.nan).ffill()
# Keep interactive backtests bounded. Indicator columns are already computed on
# the full chart frame; 1,200 recent bars provide a responsive intraday sample.
MAX_BACKTEST_BARS = 1200
backtest_data = data.tail(MAX_BACKTEST_BARS).copy()

if len(data) < 35:
    st.warning(f"Only {len(data)} candles available. Live price / chart can display, but backtest and some long-period indicators need more history.")

htf_interval, _ = get_higher_timeframe(timeframe)
angel_creds = (angel_api_key, angel_client_code, angel_pin, angel_totp_secret)
htf_data = get_data_for_timeframe(active_data_source, selected, ticker, htf_interval, angel_creds)

if not htf_data.empty:
    htf_data = add_indicators(htf_data).replace([np.inf, -np.inf], np.nan).ffill()
    if len(htf_data) >= 55:
        htf_latest = htf_data.iloc[-1]
        htf_bias = "bullish" if htf_latest["EMA20"] > htf_latest["EMA50"] else "bearish"
    else:
        htf_bias = None
else:
    htf_bias = None

tiered_tf_metrics = {
    tf: get_mtf_setup_metrics(active_data_source, selected, ticker, tf, angel_creds)
    for tf in ("3m", "5m", "15m")
}

mtf_table = pd.DataFrame({
    "Timeframe": ["1m", "3m", "5m", "15m"],
    "Trend": [
        get_mtf_signal(active_data_source, selected, ticker, "1m", angel_creds),
        get_mtf_signal(active_data_source, selected, ticker, "3m", angel_creds),
        get_mtf_signal(active_data_source, selected, ticker, "5m", angel_creds),
        get_mtf_signal(active_data_source, selected, ticker, "15m", angel_creds)
    ]
})

bull_count = sum("Bullish" in sig for sig in mtf_table["Trend"])
bear_count = sum("Bearish" in sig for sig in mtf_table["Trend"])
mtf_available_count = sum("N/A" not in str(sig) for sig in mtf_table["Trend"])

dynamic_oc_symbol = get_dynamic_option_chain_symbol(selected)
option_metrics = parse_option_chain_metrics(dynamic_oc_symbol) if dynamic_oc_symbol else {
    "pcr": None, "oi_bias": "N/A", "long_unwinding": False,
    "short_covering": False, "call_oi": 0, "put_oi": 0
}

news_info = {
    "allow_trade": True, "message": "News filter disabled", "events": [],
    "provider": news_provider, "status": "disabled", "article_count": 0,
    "sentiment_bias": "N/A", "average_sentiment": None
}
if news_filter_enabled:
    with st.spinner(f"Fetching {news_provider} financial news for {selected}…"):
        news_info = get_news_events(
            news_api_key, selected, provider=news_provider,
            lookback_hours=news_lookback_hours, article_limit=news_article_limit
        )

# The current live Marketaux/NewsAPI snapshot must not be injected into past
# candles; doing so would create look-ahead bias in historical backtest results.
historical_news_context = {
    "allow_trade": True,
    "message": "Live news is not replayed in historical tests; no look-ahead news filter applied.",
    "events": [],
    "status": "historical_not_replayed",
}
# Cache the interactive backtest by the actual recent candle content. Streamlit
# auto-refresh can rerun the page every few seconds even when no new candle exists;
# do not recompute the same 1,200-bar simulation on every browser repaint.
try:
    _bt_hash = int(pd.util.hash_pandas_object(backtest_data, index=True).sum()) if not backtest_data.empty else 0
except Exception:
    _bt_hash = len(backtest_data)
_backtest_cache_key = f"{selected}|{timeframe}|{len(backtest_data)}|{_bt_hash}"
_backtest_cache = st.session_state.get("backtest_result_cache")
if isinstance(_backtest_cache, dict) and _backtest_cache.get("key") == _backtest_cache_key:
    wins, losses, total_trades, backtest_win_rate, backtest_trades_df = _backtest_cache["result"]
else:
    with st.spinner("Calculating recent-history backtest…"):
        _backtest_result = simple_backtest(
            backtest_data,
            symbol=selected,
            timeframe_label=timeframe,
            option_metrics=option_metrics,
            news_info=historical_news_context
        )
    wins, losses, total_trades, backtest_win_rate, backtest_trades_df = _backtest_result
    st.session_state["backtest_result_cache"] = {
        "key": _backtest_cache_key,
        "result": _backtest_result,
        "calculated_at": now_str(),
    }

backtest_is_comparable = bool(
    total_trades >= 30 and
    not backtest_trades_df.empty and
    "Method" not in backtest_trades_df.columns
)
analysis = analyze_signal(
    data,
    htf_bias=htf_bias,
    timeframe_label=timeframe,
    backtest_win_rate=backtest_win_rate,
    mtf_bull_count=bull_count,
    mtf_bear_count=bear_count,
    mtf_available_count=mtf_available_count,
    option_metrics=option_metrics,
    news_info=news_info,
    backtest_trade_count=total_trades,
    backtest_is_comparable=backtest_is_comparable,
)
analysis = apply_adaptive_signal_intelligence(
    data, analysis, selected, timeframe,
    mtf_bull_count=bull_count, mtf_bear_count=bear_count,
    mtf_available_count=mtf_available_count
)
analysis = apply_tiered_early_alerts(analysis, tiered_tf_metrics)

stats = compute_trade_statistics(backtest_trades_df, initial_capital=DEFAULT_INITIAL_BALANCE)
equity_curve_df, max_drawdown = compute_equity_curve_from_trades(backtest_trades_df, initial_capital=DEFAULT_INITIAL_BALANCE)

with st.expander("🧠 Loss Intelligence & Similar Historical Setups", expanded=True):
    intel = get_similar_trade_intelligence(analysis.get("feature_snapshot", {}), "BUY" if analysis.get("signal_type") == "buy" else "SELL")
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Signal Stage", analysis.get("signal_stage", "N/A"))
    i2.metric("Similar Trades", intel.get("count", 0))
    i3.metric("Historical Win Rate", f"{intel['win_rate']:.1f}%" if intel.get("win_rate") is not None else "N/A")
    i4.metric("Historical Loss Rate", f"{intel['loss_rate']:.1f}%" if intel.get("loss_rate") is not None else "N/A")
    i5.metric("Expected PnL / Trade", f"₹{intel['expected_value']:,.2f}" if intel.get("expected_value") is not None else "N/A")
    st.markdown("**Tiered early-alert rules (3m / 5m / 15m)**")
    st.caption("Early alert = 3m EMA9 > EMA21 and RSI > 50. Confirmed BUY = 3m + 5m EMA9 > EMA21, 3m RSI > 55, ADX > 20 and relative volume > 1.5x. Strong BUY = 3m + 5m + 15m EMA9 > EMA21 > EMA50, 3m RSI > 60, ADX > 25 and relative volume > 1.8x. SELL alerts use mirrored bearish conditions.")
    tf_cols = st.columns(3)
    for col, tf in zip(tf_cols, ("3m", "5m", "15m")):
        m = analysis.get("tiered_alert_metrics", {}).get(tf, {})
        if m.get("available"):
            col.metric(f"{tf} Trend", m.get("trend", "N/A"))
            col.caption(f"EMA9 {m['ema9']:.2f} | EMA21 {m['ema21']:.2f} | EMA50 {m['ema50']:.2f}")
            col.caption(f"RSI {m['rsi']:.1f} | ADX {m['adx']:.1f} | Rel.Vol {m['relative_volume']:.2f}x" if np.isfinite(m.get("relative_volume", np.nan)) else f"RSI {m['rsi']:.1f} | ADX {m['adx']:.1f} | Rel.Vol N/A")
        else:
            col.metric(f"{tf} Trend", "N/A")
            col.caption("Insufficient or unavailable timeframe data; alert tier not confirmed.")
    st.caption(f"Adaptive confidence: {analysis.get('confidence', 'N/A')} / 100 | {analysis.get('adaptive_confidence_label', '')} | Similarity matching is heuristic and only uses stored, available fields.")
    block_repeat_losers = st.toggle("Block signals matching repeated historical losing setups", value=False, key="block_repeat_loser_setups")
    repeat_loss_block_active = bool(block_repeat_losers and analysis.get("similar_trade_block_suggested"))
    if analysis.get("repeat_loss_warning"):
        st.warning(f"High-risk repeat-loss pattern detected. Historical loser match rate: {analysis.get('historical_loser_match_rate', 0):.1f}%. Confidence penalty: {analysis.get('repeat_loss_penalty', 0)} points.")
    if repeat_loss_block_active:
        analysis["trade_allowed"] = False
        st.error("Signal blocked by your optional repeat-loss prevention setting. Review the setup before any paper entry.")
    if intel.get("count", 0) > 0:
        st.write(f"Average winning trade: ₹{intel['avg_profit']:,.2f} | Average losing trade: ₹{intel['avg_loss']:,.2f} | Expected value: ₹{intel['expected_value']:,.2f}")
    else:
        st.info("No sufficiently similar completed trades found yet. Historical learning will improve as your journal collects closed trades.")
    st.markdown("**Captured live setup metrics**")
    snap_view = analysis.get("feature_snapshot", {})
    metric_cols = st.columns(5)
    for col, key in zip(metric_cols, ["EMA9", "EMA21", "RSI", "ADX", "RelativeVolume"]):
        value = snap_view.get(key)
        col.metric(key, f"{value:.2f}" if isinstance(value, (int, float)) and np.isfinite(value) else "N/A")
    st.caption("Loss labels are probable diagnostic categories, not proof of causation. News reversal and execution/slippage causes require reliable event and fill data.")
    _ml_journal = load_journal()
    closed_count_ml = int((_ml_journal.get("Status", pd.Series(dtype=str)).astype(str).str.upper() == "CLOSED").sum()) if not _ml_journal.empty else 0
    st.caption(f"Optional ML layer: {closed_count_ml}/30 completed trades collected. Dataset evaluation is available now; model fitting unlocks after 30 usable trades with at least 5 wins and 5 losses.")
    if st.button("Evaluate Dataset / Train ML Model", key="train_loss_intelligence_ml"):
        with st.spinner("Evaluating historical trade dataset…"):
            ml_result = train_trade_success_model(30)
        if ml_result.get("ok") and not ml_result.get("trained"):
            readiness = ml_result.get("readiness", {})
            st.info(ml_result.get("message", "Dataset evaluation completed."))
            if readiness:
                st.write(f"Usable trades: {readiness.get('trades', 0)} | Wins: {readiness.get('wins', 0)} | Losses: {readiness.get('losses', 0)} | Feature coverage: {readiness.get('coverage_pct', 0):.1f}% | Numeric features: {readiness.get('feature_count', 0)}")
        elif ml_result.get("ok") and ml_result.get("trained"):
            auc_text = f" | AUC: {ml_result['auc']:.3f}" if ml_result.get("auc") is not None else ""
            st.success(f"{ml_result['model']} chronological holdout evaluation | Accuracy: {ml_result['accuracy']:.1%} | Balanced accuracy: {ml_result['balanced_accuracy']:.1%}{auc_text} | Train: {ml_result['train_trades']} | Test: {ml_result['test_trades']}")
            st.caption(ml_result.get("warning", ""))
        else:
            st.warning(ml_result.get("message", "ML training is not available."))

if backtest_is_comparable:
    health_score, health_label = strategy_health_monitor(
        analysis["validation_score"],
        backtest_win_rate,
        stats["profit_factor"] if stats["profit_factor"] != 999.0 else 5.0,
        max_drawdown
    )
else:
    health_score, health_label = None, "Insufficient comparable sample"

best_hour, worst_hour, hour_perf_df = best_worst_time_of_day(backtest_trades_df)
# Best/worst setup labels are based on completed journal regimes, not just BUY vs SELL.
_setup_journal = load_journal()
if (not _setup_journal.empty and "MarketRegime" in _setup_journal.columns
        and "PnL" in _setup_journal.columns and "Status" in _setup_journal.columns):
    _setup_closed = _setup_journal[_setup_journal["Status"].astype(str).str.upper().eq("CLOSED")].copy()
    _setup_closed["PnL"] = pd.to_numeric(_setup_closed["PnL"], errors="coerce")
    _setup_closed = _setup_closed.dropna(subset=["PnL"])
    _setup_perf = _setup_closed.groupby("MarketRegime", dropna=False)["PnL"].agg(["count", "mean", "sum"])
    _setup_perf = _setup_perf[_setup_perf["count"] >= 2].sort_values("mean", ascending=False)
    best_setup_label = str(_setup_perf.index[0]) if not _setup_perf.empty else "Insufficient journal sample"
    worst_setup_label = str(_setup_perf.index[-1]) if not _setup_perf.empty else "Insufficient journal sample"
else:
    _setup_perf = pd.DataFrame()
    best_setup_label = worst_setup_label = "Insufficient journal sample"

if bull_count >= 3:
    mtf_grade = "Strong Bullish Alignment"
elif bear_count >= 3:
    mtf_grade = "Strong Bearish Alignment"
else:
    mtf_grade = "Mixed Timeframes"

mtf_confirmation = "Mixed / unavailable"
if analysis["signal_type"] == "buy":
    mtf_confirmation = "Bullish Alignment" if analysis.get("mtf_bull_ok") or htf_bias == "bullish" else "No bullish confirmation"
elif analysis["signal_type"] == "sell":
    mtf_confirmation = "Bearish Alignment" if analysis.get("mtf_bear_ok") or htf_bias == "bearish" else "No bearish confirmation"
else:
    mtf_confirmation = "No approved directional setup"

quality_score = float(analysis.get("trade_quality_score", 0) or 0)
if not analysis.get("trade_allowed", False):
    signal_quality = "No Valid Setup"
elif quality_score >= 85:
    signal_quality = "Excellent"
elif quality_score >= 70:
    signal_quality = "Strong"
elif quality_score >= 55:
    signal_quality = "Moderate"
else:
    signal_quality = "Weak"

live_price = None
live_price_source = "OHLC candle close"
if data_source == "Angel":
    _fresh_ws_quote = ws_live_quote(
        max_age_seconds=max(20, int(refresh_seconds) * 2 if auto_refresh else 20),
        symbol_name=selected
    ) if ws_enable else None
    if _fresh_ws_quote is not None:
        live_price = _fresh_ws_quote
        live_price_source = "Angel WebSocket (Primary)"
        angel_ltp_source = live_price_source
    elif angel_ltp_value is not None:
        live_price = float(angel_ltp_value)
        live_price_source = angel_ltp_source or "Angel REST Polling (Fallback)"
    sync_ws_state_to_session()

current_display_price = live_price if live_price is not None and live_price > 0 else analysis["close_price"]

closed_auto = 0

option_suggestion = suggest_option_strike(selected, analysis["close_price"], analysis["signal_type"])
expiry_seconds = get_signal_expiry_remaining(selected, timeframe, analysis["signal_text"])

_confidence_text = f"{analysis['confidence']}%" if analysis.get("confidence") is not None else "N/A (no approved setup)"
_win_rate_text = (f"{analysis['win_chance']:.1f}% from {analysis.get('win_chance_sample_size', 0)} comparable trades"
                  if analysis.get("win_chance") is not None else "N/A (requires 30+ comparable trades)")
alert_text = (
    f"{selected} | {timeframe} | {analysis['signal_text']} | "
    f"Grade: {analysis['strategy_grade']} | Validation: {analysis['validation_score']} | "
    f"Quality: {analysis['trade_quality_score']} | Setup Confidence: {_confidence_text} | "
    f"Backtest Win Rate: {_win_rate_text} | Price: {round(current_display_price, 2)} | "
    f"RR: {analysis['rr_ratio'] if analysis['rr_ratio'] is not None else 'N/A'} | OC: {dynamic_oc_symbol} | {option_suggestion['text']}"
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

st.subheader("Japanese Heikin-Ashi Price Action Chart")
chart_ha = convert_to_heikin_ashi(data.tail(180))
if not chart_ha.empty:
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=chart_ha.index, open=chart_ha["HA_Open"], high=chart_ha["HA_High"],
        low=chart_ha["HA_Low"], close=chart_ha["HA_Close"], name="Heikin-Ashi"
    ))
    if "EMA20" in chart_ha.columns:
        price_fig.add_trace(go.Scatter(x=chart_ha.index, y=chart_ha["EMA20"], mode="lines", name="EMA 20"))
    if "EMA50" in chart_ha.columns:
        price_fig.add_trace(go.Scatter(x=chart_ha.index, y=chart_ha["EMA50"], mode="lines", name="EMA 50"))
    if "VWAP" in chart_ha.columns:
        price_fig.add_trace(go.Scatter(x=chart_ha.index, y=chart_ha["VWAP"], mode="lines", name="VWAP"))
    price_fig.update_layout(height=520, template="plotly_dark", xaxis_rangeslider_visible=False,
                            title=f"{selected} — Japanese Heikin-Ashi Price Action | {timeframe}",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(price_fig, use_container_width=True)

macd_fig = go.Figure()
macd_fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="MACD Histogram"))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD"))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["SignalLine"], mode="lines", name="Signal Line"))
macd_fig.update_layout(height=260, template="plotly_dark", title="MACD Momentum")
st.plotly_chart(macd_fig, use_container_width=True)

latest_pattern_row = data.iloc[-1]
st.info(f"Latest candle pattern: {latest_pattern_row.get('CandlePattern', 'N/A')} | Bias: {latest_pattern_row.get('CandlePatternBias', 'N/A')} | Pattern score: {latest_pattern_row.get('CandlePatternScore', 0)}")
with st.expander("Candlestick Pattern Library — latest detected patterns"):
    pattern_cols = ["Hammer", "HangingMan", "InvertedHammer", "ShootingStar", "Doji", "SpinningTop", "MarubozuBull", "MarubozuBear", "BullishEngulfing", "BearishEngulfing", "PiercingLine", "DarkCloudCover", "MorningStar", "EveningStar", "ThreeWhiteSoldiers", "ThreeBlackCrows", "TweezerBottom", "TweezerTop", "InsideBar", "OutsideBar"]
    active_patterns = [name for name in pattern_cols if bool(latest_pattern_row.get(name, False))]
    st.write(", ".join(active_patterns) if active_patterns else "No recognized pattern on the latest completed candle.")

st.subheader("Live Signal Analytics")

if data_source == "Angel":
    p1, p2, p3, p4 = st.columns(4)
    _ws_status_snapshot = ws_get_state()
    _ws_is_effectively_primary = bool(
        ws_enable
        and not st.session_state.get("ws_manual_pause_requested", False)
        and not (_ws_status_snapshot.get("retry_exhausted") and str(_ws_status_snapshot.get("symbol")) == str(selected))
    )
    with p1:
        st.metric("Live Feed Priority", "WebSocket Primary" if _ws_is_effectively_primary else ("REST Polling Primary" if polling_mode else "Live LTP Disabled"))
    with p2:
        if _ws_status_snapshot.get("connected"):
            _ws_status_text = "Connected"
        elif _ws_status_snapshot.get("retry_exhausted") and str(_ws_status_snapshot.get("symbol")) == str(selected):
            _ws_status_text = "Retry limit reached"
        elif st.session_state.get("ws_manual_pause_requested", False):
            _ws_status_text = "Paused manually"
        else:
            _ws_status_text = "Connecting / Offline"
        st.metric("WebSocket Status", _ws_status_text)
    with p3:
        st.metric("Live Price Source", live_price_source)
    with p4:
        st.metric("Live Price", round(live_price, 2) if live_price is not None else "-")
    if ws_enable and not _ws_status_snapshot.get("connected"):
        st.caption("WebSocket is the primary live-LTP source. REST polling is the configured fallback while the socket connects or reconnects.")
    elif ws_enable:
        st.caption("WebSocket is connected. Historical candles are still loaded through Angel REST Candle API; WebSocket ticks update the live LTP display.")

if analysis["signal_type"] == "buy":
    st.success(alert_text)
elif analysis["signal_type"] == "sell":
    st.error(alert_text)
else:
    st.warning(alert_text)

with st.expander("Signal scoring breakdown and entry blockers", expanded=(analysis.get("signal_type") == "neutral")):
    st.write(f"Directional bias: **{analysis.get('directional_bias', 'Neutral')}** | Bullish score: **{analysis.get('bullish_score', 0)}** | Bearish score: **{analysis.get('bearish_score', 0)}** | Edge: **{analysis.get('directional_edge', 0)}**")
    st.write(f"Trend structure: **BUY {analysis.get('buy_trend_votes', 0)}/{analysis.get('trend_vote_count', 3)}** | **SELL {analysis.get('sell_trend_votes', 0)}/{analysis.get('trend_vote_count', 3)}**")
    st.write(f"MTF data available: **{analysis.get('mtf_available_count', 0)}/4** | Volume data reliable: **{'Yes' if analysis.get('volume_data_available') else 'No'}** | PCR: **{analysis.get('pcr') if analysis.get('pcr') is not None else 'N/A'}**")
    if analysis.get("no_trade_reasons"):
        st.markdown("**Why no trade:**")
        for _reason in analysis["no_trade_reasons"]:
            st.write(f"• {_reason}")
    else:
        st.markdown("**Why this setup qualified:**")
        for _reason in analysis.get("reasons", [])[-8:]:
            st.write(f"• {_reason}")

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
    st.metric("Health Score", health_score if health_score is not None else "N/A")
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
    st.metric("Backtest Win Rate", f"{analysis['win_chance']:.1f}%" if analysis.get("win_chance") is not None else "N/A")
with top4:
    st.metric("Setup Confidence", f"{analysis['confidence']}%" if analysis.get("confidence") is not None else "N/A")
st.caption("Setup confidence is a structural score, not a predicted probability. Backtest win rate is shown only after 30+ comparable same-engine trades.")

a1, a2, a3, a4, a5, a6 = st.columns(6)
with a1:
    st.metric("ADX", round(analysis["adx"], 2))
with a2:
    _ema200_position = ("Above / bullish" if analysis["close_price"] > analysis["ema200"] else "Below / bearish" if analysis["close_price"] < analysis["ema200"] else "At EMA200") if analysis.get("ema200_available") else "Not mature"
    st.metric("Price vs EMA200", _ema200_position)
with a3:
    st.metric("SuperTrend", "Bullish" if analysis["supertrend_direction"] == 1 else "Bearish" if analysis["supertrend_direction"] == -1 else "Neutral")
with a4:
    st.metric("RR Ratio", analysis["rr_ratio"] if analysis["rr_ratio"] else "-")
with a5:
    st.metric("Dynamic OC", dynamic_oc_symbol)
with a6:
    if not news_filter_enabled:
        _news_metric_value = "OFF"
    elif news_info.get("status") == "error":
        _news_metric_value = "UNAVAILABLE"
    else:
        _news_metric_value = "BLOCKED" if not news_info.get("allow_trade", False) else "CLEAR"
    st.metric("News Filter", _news_metric_value)

b1, b2, b3, b4, b5, b6 = st.columns(6)
with b1:
    st.metric("VWAP", round(float(analysis["vwap"]), 2) if analysis["vwap"] is not None else "N/A")
with b2:
    st.metric("EMA20", round(analysis["ema20"], 2))
with b3:
    st.metric("EMA50", round(analysis["ema50"], 2))
with b4:
    st.metric("EMA200", round(analysis["ema200"], 2) if analysis.get("ema200_available") else "N/A — need 200 bars")
with b5:
    st.metric("RSI", round(analysis["rsi"], 2))
with b6:
    st.metric("ATR", round(analysis["atr"], 2))

mac1, mac2, mac3, mac4 = st.columns(4)
with mac1:
    st.metric("MACD", round(float(analysis["macd"]), 4))
with mac2:
    st.metric("MACD Signal", round(float(analysis["macd_signal"]), 4))
with mac3:
    st.metric("MACD Histogram", round(float(analysis["macd_hist"]), 4))
with mac4:
    st.metric("Candle Pattern Score", analysis["candle_pattern_score"])

score1, score2, score3, score4 = st.columns(4)
with score1:
    st.metric("Bullish Score", analysis["bullish_score"])
with score2:
    st.metric("Bearish Score", analysis["bearish_score"])
with score3:
    st.metric("PCR", analysis["pcr"] if analysis["pcr"] is not None else "N/A")
with score4:
    st.metric("OI Bias", analysis["oi_bias"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Support", round(analysis["support"], 2))
with c2:
    st.metric("Resistance", round(analysis["resistance"], 2))
with c3:
    st.metric("Gap %", analysis["gap_pct"])
with c4:
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

st.subheader("Paper Trading — BUY / SELL / CLOSE | Virtual Orders Only")
st.caption("No broker orders are sent. Suggested levels are analytical estimates, not guaranteed outcomes. Short positions reserve full notional as virtual margin.")

paper_state = load_paper_state()
paper_balance = float(paper_state.loc[0, "Balance"])
used_margin = float(paper_state.loc[0, "UsedMargin"])
paper_journal = load_journal()
open_paper = paper_journal[paper_journal.get("Status", pd.Series(dtype=str)).astype(str).str.upper().eq("OPEN")].copy() if not paper_journal.empty else pd.DataFrame()
realized_pnl = 0.0
if not paper_journal.empty:
    if "RealizedPnL" in paper_journal.columns:
        realized_pnl += pd.to_numeric(paper_journal["RealizedPnL"], errors="coerce").fillna(0).sum()
    if "PnL" in paper_journal.columns:
        legacy_closed = paper_journal.get("Status", pd.Series("", index=paper_journal.index)).astype(str).str.upper().eq("CLOSED")
        legacy_zero_realized = pd.to_numeric(paper_journal.get("RealizedPnL", 0), errors="coerce").fillna(0).eq(0)
        realized_pnl += pd.to_numeric(paper_journal.loc[legacy_closed & legacy_zero_realized, "PnL"], errors="coerce").fillna(0).sum()

unrealized_selected = 0.0
if not open_paper.empty:
    for _, pos in open_paper.iterrows():
        if str(pos.get("Symbol", "")) != str(selected):
            continue
        qty_now = float(pos.get("RemainingQty", pos.get("Quantity", 0)) or 0)
        side_now = str(pos.get("Side", pos.get("Signal", "BUY"))).upper()
        unrealized_selected += _paper_side_pnl(float(pos["EntryPrice"]), float(current_display_price), qty_now, side_now)

paper_auto_exit_enabled = render_paper_exit_mode_selector()

# The selected mode immediately arms or disarms every open virtual position.
paper_journal = sync_open_position_auto_exit(paper_auto_exit_enabled, paper_journal)

if paper_auto_exit_enabled:
    auto_result = auto_paper_trade_exit(current_display_price, symbol=selected, atr=analysis.get("atr"))
    if auto_result.get("closed", 0) or auto_result.get("partial", 0):
        st.success(
            f"Auto-exit manager: {auto_result.get('closed', 0)} position(s) fully closed; "
            f"{auto_result.get('partial', 0)} target scale-out(s) executed."
        )
    # Refresh balances and open-position rows after any automatic partial/full exit.
    paper_state = load_paper_state()
    paper_balance = float(paper_state.loc[0, "Balance"])
    used_margin = float(paper_state.loc[0, "UsedMargin"])
    paper_journal = load_journal()
    open_paper = paper_journal[paper_journal.get("Status", pd.Series(dtype=str)).astype(str).str.upper().eq("OPEN")].copy() if not paper_journal.empty else pd.DataFrame()
    realized_pnl = 0.0
    if not paper_journal.empty:
        realized_pnl += pd.to_numeric(paper_journal.get("RealizedPnL", 0), errors="coerce").fillna(0).sum()
        legacy_closed = paper_journal.get("Status", pd.Series("", index=paper_journal.index)).astype(str).str.upper().eq("CLOSED")
        legacy_zero_realized = pd.to_numeric(paper_journal.get("RealizedPnL", 0), errors="coerce").fillna(0).eq(0)
        if "PnL" in paper_journal.columns:
            realized_pnl += pd.to_numeric(paper_journal.loc[legacy_closed & legacy_zero_realized, "PnL"], errors="coerce").fillna(0).sum()
    unrealized_selected = 0.0
    if not open_paper.empty:
        for _, pos in open_paper.iterrows():
            if str(pos.get("Symbol", "")) == str(selected):
                qty_now = float(pos.get("RemainingQty", pos.get("Quantity", 0)) or 0)
                side_now = str(pos.get("Side", pos.get("Signal", "BUY"))).upper()
                unrealized_selected += _paper_side_pnl(float(pos["EntryPrice"]), float(current_display_price), qty_now, side_now)

# Dashboard's suggested directional plan is selected from the stronger score. Both BUY and SELL can be placed manually.
plan_side = "BUY" if int(analysis.get("bullish_score", 0)) >= int(analysis.get("bearish_score", 0)) else "SELL"
paper_plan = get_suggested_paper_levels(analysis, plan_side, current_display_price)
recommended_qty_risk = calculate_position_size(current_display_price, paper_plan["stop_loss"], paper_balance)
max_affordable_qty = int(max(0.0, paper_balance) // max(float(current_display_price), 0.01))
recommended_qty = max(1, min(int(recommended_qty_risk), max(1, max_affordable_qty)))

pm1, pm2, pm3, pm4 = st.columns(4)
with pm1:
    st.metric("Paper Cash", f"₹{paper_balance:,.2f}")
with pm2:
    st.metric("Used Virtual Margin", f"₹{used_margin:,.2f}")
with pm3:
    st.metric("Realized Paper P&L", f"₹{realized_pnl:,.2f}")
with pm4:
    st.metric("Open Paper Positions", int(len(open_paper)))
pm5, pm6, pm7 = st.columns(3)
with pm5:
    st.metric("Selected Symbol Unrealized P&L", f"₹{unrealized_selected:,.2f}")
with pm6:
    st.metric("Bullish / Bearish Score", f"{analysis['bullish_score']} / {analysis['bearish_score']}")
with pm7:
    st.metric("PCR", analysis["pcr"] if analysis["pcr"] is not None else "N/A")

st.markdown(f"**Suggested plan direction:** {plan_side} / {'LONG' if plan_side == 'BUY' else 'SHORT'}  ·  **Current signal:** {analysis['signal_text']}")
st.caption("When the signal is NO TRADE, the levels below are an ATR-based paper-planning reference only; they do not convert the signal into a live recommendation.")
pl1, pl2, pl3, pl4, pl5, pl6 = st.columns(6)
with pl1:
    st.metric("Suggested Entry", round(paper_plan["entry"], 2))
with pl2:
    st.metric("Suggested Stop Loss", round(paper_plan["stop_loss"], 2))
with pl3:
    st.metric("Target 1", round(paper_plan["target1"], 2))
with pl4:
    st.metric("Target 2", round(paper_plan["target2"], 2))
with pl5:
    st.metric("Target 3", round(paper_plan["target3"], 2))
with pl6:
    st.metric("Initial Trailing Stop", round(paper_plan["trailing_stop"], 2))

st.markdown("**Place a virtual paper order**")
order_qty = st.number_input(
    "Quantity", min_value=1, max_value=1000000, value=int(recommended_qty), step=1,
    key=f"paper_order_qty_{selected}_{timeframe}"
)
order_cols = st.columns(2)
with order_cols[0]:
    buy_clicked = st.button("BUY / LONG", type="primary", use_container_width=True, key="paper_buy_order")
with order_cols[1]:
    sell_clicked = st.button("SELL / SHORT", type="secondary", use_container_width=True, key="paper_sell_order")

if buy_clicked or sell_clicked:
    order_side = "BUY" if buy_clicked else "SELL"
    order_plan = get_suggested_paper_levels(analysis, order_side, current_display_price)
    order_qty_int = int(order_qty)
    required_margin = float(current_display_price) * order_qty_int
    if repeat_loss_block_active:
        st.error("Paper order blocked: this setup matches a repeated historical losing pattern. Turn off the optional block only if you have reviewed the risk.")
    elif not risk_ok:
        st.error(f"Paper order blocked by risk controls: {risk_msg}")
    elif required_margin > paper_balance:
        st.error(f"Insufficient virtual cash. Required margin ₹{required_margin:,.2f}; available ₹{paper_balance:,.2f}.")
    elif not (order_plan["stop_loss"] > 0 and order_plan["target1"] > 0 and order_plan["target2"] > 0 and order_plan["target3"] > 0):
        st.error("The suggested price levels are invalid. Check the current market price and ATR data.")
    else:
        entered = paper_trade_entry(current_display_price, order_qty_int, side=order_side)
        if entered:
            add_trade_to_journal(
                selected, timeframe, order_side, current_display_price, order_qty_int,
                order_plan["stop_loss"], order_plan["target1"], order_plan["target2"],
                order_plan["target3"], order_plan["trailing_stop"], analysis,
                notes=f"Manual paper {order_side} entry; virtual only",
                auto_exit=paper_auto_exit_enabled, atr=order_plan["atr"]
            )
            st.success(f"Paper {order_side} order opened: {selected} × {order_qty_int} at ₹{current_display_price:,.2f}. No real order was sent.")
            st.rerun()
        else:
            st.error("Insufficient virtual cash or paper-state write failed.")

st.markdown("**Open paper positions**")
st.caption("Auto Exit mode: the currently selected symbol is checked on each Streamlit rerun/auto-refresh for stop loss, trailing stop and T1/T2/T3. Keep Auto Refresh ON and keep that symbol selected. Manual Exit mode disables automatic exits; the CLOSE control remains available for every open position. This is a paper simulator, not a background execution service.")
if open_paper.empty:
    st.info("No open paper positions. Use BUY / LONG or SELL / SHORT to open a virtual position.")
else:
    display_cols = [c for c in ["Symbol", "Side", "Timeframe", "EntryPrice", "RemainingQty", "StopLoss", "TrailingStop", "Target1", "Target2", "Target3", "T1_Hit", "T2_Hit", "T3_Hit", "LastPrice", "AutoExitEnabled"] if c in open_paper.columns]
    st.dataframe(open_paper[display_cols], use_container_width=True, hide_index=True)
    for pos_idx, pos in open_paper.iterrows():
        pos_symbol = str(pos.get("Symbol", ""))
        pos_side = str(pos.get("Side", pos.get("Signal", "BUY"))).upper()
        pos_entry = float(pos.get("EntryPrice", 0.0))
        pos_remaining = int(float(pos.get("RemainingQty", pos.get("Quantity", 0)) or 0))
        if pos_symbol == str(selected):
            default_exit_price = float(current_display_price)
            close_help = "Current selected-symbol market price will be used."
        else:
            try:
                default_exit_price = float(pos.get("LastPrice", pos_entry))
                if not np.isfinite(default_exit_price) or default_exit_price <= 0:
                    default_exit_price = pos_entry
            except Exception:
                default_exit_price = pos_entry
            close_help = "This symbol is not selected. Verify or edit the last exit price before closing."
        with st.container(border=True):
            close_cols = st.columns([2, 1, 1, 1, 1])
            with close_cols[0]:
                st.markdown(f"**{pos_symbol} — {pos_side}**")
                st.caption(f"Remaining: {pos_remaining} · Entry: ₹{pos_entry:,.2f}")
            with close_cols[1]:
                st.metric("Last Price", round(default_exit_price, 2))
            with close_cols[2]:
                st.metric("Trailing SL", round(float(pos.get("TrailingStop", pos.get("StopLoss", 0)) or 0), 2))
            with close_cols[3]:
                st.metric("Unrealized P&L", f"₹{_paper_side_pnl(pos_entry, default_exit_price, pos_remaining, pos_side):,.2f}")
            with close_cols[4]:
                exit_price_input = st.number_input(
                    "Exit Price", min_value=0.01, value=float(default_exit_price), step=0.05,
                    key=f"paper_manual_exit_price_{pos_idx}", help=close_help
                )
                if st.button("CLOSE", key=f"paper_close_position_{pos_idx}", use_container_width=True):
                    closed, close_message = close_paper_position(pos_idx, exit_price_input, "Manual Close")
                    if closed:
                        st.success(close_message)
                        st.rerun()
                    else:
                        st.error(close_message)

st.markdown("**Paper trade history**")
if paper_journal.empty:
    st.caption("No paper trades have been recorded yet.")
else:
    history_cols = [c for c in ["DateTime", "Symbol", "Side", "EntryPrice", "Quantity", "RemainingQty", "ExitPrice", "Status", "RealizedPnL", "PnL", "Result", "LossReason", "Notes"] if c in paper_journal.columns]
    st.dataframe(paper_journal[history_cols].tail(100), use_container_width=True, hide_index=True)

with st.expander("Why this signal? — Explainable Signal Factors", expanded=False):
    ex1, ex2, ex3, ex4 = st.columns(4)
    ex1.metric("Bullish Score", analysis.get("bullish_score", 0))
    ex2.metric("Bearish Score", analysis.get("bearish_score", 0))
    ex3.metric("Historical Win %", f"{analysis['historical_win_rate']:.1f}%" if analysis.get("historical_win_rate") is not None else "N/A")
    ex4.metric("Historical Loss %", f"{analysis['historical_loss_rate']:.1f}%" if analysis.get("historical_loss_rate") is not None else "N/A")
    st.markdown("**Bullish factors**")
    for reason in analysis.get("reasons", []):
        if any(token in str(reason).lower() for token in ["bull", "macd above", "ema", "breakout", "vwap", "volume", "higher timeframe", "trend"]):
            st.write("-", reason)
    st.markdown("**Bearish / conflicting factors**")
    for reason in analysis.get("no_trade_reasons", []):
        st.write("-", reason)
    st.markdown("**Risk factors**")
    for risk_text in [analysis.get("news_message", ""), analysis.get("session_message", ""), analysis.get("market_regime", ""), "Repeat-loss warning active" if analysis.get("repeat_loss_warning") else "No repeated-loss warning"]:
        if risk_text:
            st.write("-", risk_text)
    st.caption(f"Adaptive confidence score: {analysis.get('confidence', 'N/A')} / 100. This is a ranking score, not a calibrated probability or guarantee.")

if news_filter_enabled:
    st.subheader("News Filter Events")
    st.caption(
        f"Provider: {news_info.get('provider', news_provider)} | "
        f"Articles: {news_info.get('article_count', 0)} | "
        f"Sentiment: {news_info.get('sentiment_bias', 'N/A')} "
        f"({news_info.get('average_sentiment') if news_info.get('average_sentiment') is not None else 'N/A'})"
    )
    if news_info.get("status") == "error":
        st.warning(news_info.get("message", "News provider unavailable. New signal entries are paused."))
    elif news_info.get("events"):
        _news_df = pd.DataFrame(news_info["events"])
        if economic_calendar_filter_enabled and economic_keywords:
            _economic_search_cols = [c for c in ["Title", "Source", "Entities", "Risk Reason"] if c in _news_df.columns]
            if _economic_search_cols:
                _economic_text = _news_df[_economic_search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.casefold()
                _economic_mask = _economic_text.apply(lambda value: any(keyword in value for keyword in economic_keywords))
                _economic_matches = _news_df.loc[_economic_mask].copy()
            else:
                _economic_matches = _news_df.iloc[0:0].copy()
            st.markdown("**Economic keyword matches**")
            st.caption(f"Matched {_economic_matches.shape[0]} of {_news_df.shape[0]} fetched news items using: {', '.join(item.upper() for item in economic_keywords)}")
            if not _economic_matches.empty:
                _news_df = _economic_matches
            else:
                st.info("No fetched headlines matched the economic keyword list. This does not mean no scheduled event is upcoming.")
                _news_df = _news_df.iloc[0:0].copy()
        _news_columns = [c for c in ["Title", "Source", "Published (UTC)", "Sentiment", "Entities", "Risk Flag", "Risk Reason", "Article"] if c in _news_df.columns]
        if not _news_df.empty:
            st.dataframe(
                _news_df[_news_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Article": st.column_config.LinkColumn("Article", display_text="Open article")
                }
            )
        st.caption(news_info.get("message", "News events loaded."))
    else:
        st.info(news_info.get("message", "No recent relevant news."))

st.subheader("Major Stocks Scanner — NIFTY Top 30 | CMP & % Change")
st.caption("30 major NIFTY constituents scanned on 3m, 5m, 10m and 15m. Yahoo Finance is independent of the primary Angel/Yahoo candle selection.")
if st.button("Run / refresh NIFTY Top 30 scanner", key="run_scanner_with_main_data"):
    fetch_yahoo_batch.clear()
    build_nifty_top30_scanner.clear()
    try:
        with st.spinner("Scanning NIFTY Top 30 across 3m / 5m / 10m / 15m…"):
            st.session_state["nifty30_scanner_result"] = build_nifty_top30_scanner()
        st.session_state["nifty30_scanner_last_run"] = now_str()
    except Exception as scanner_error:
        st.error(f"Scanner error: {scanner_error}")
        log_app_error("nifty_top30_scanner_ui", scanner_error)
scanner_result = st.session_state.get("nifty30_scanner_result")
if isinstance(scanner_result, pd.DataFrame) and not scanner_result.empty:
    st.caption("Last scanner run: " + str(st.session_state.get("nifty30_scanner_last_run", "unknown")))
    render_nifty_scanner_result(scanner_result)
else:
    st.dataframe(blank_nifty_scanner_frame(), use_container_width=True, hide_index=True)
    st.info("The 30-stock scanner is ready. Press Run / refresh NIFTY Top 30 scanner to fetch provider data.")

st.subheader("Backtest Analytics")
st.caption(f"Backtest input: {len(backtest_data):,} candles | {backtest_data.index.min()} to {backtest_data.index.max()} | {selected} | {timeframe}")
if backtest_trades_df.empty:
    st.warning("No completed backtest trades were generated. Check the selected data source, date/history range and candle count. At least 35 valid candles are required; the fallback also needs valid EMA/RSI/ATR values.")
else:
    method_note = "EMA trend/pullback fallback" if ("Method" in backtest_trades_df.columns and backtest_trades_df["Method"].eq("EMA trend/pullback fallback").any()) else "Composite signal engine"
    st.caption(f"Trades generated: {len(backtest_trades_df)} | Engine: {method_note}. Results are historical simulations, not guaranteed future performance.")
    if not backtest_is_comparable:
        st.info("This sample is not eligible for the dashboard's historical win-rate estimate. It is a fallback or has fewer than 30 comparable composite-engine trades.")
ba1, ba2, ba3, ba4, ba5, ba6 = st.columns(6)
with ba1:
    st.metric("Backtest Wins", wins)
with ba2:
    st.metric("Backtest Losses", losses)
with ba3:
    st.metric("Total Trades", total_trades)
with ba4:
    st.metric("Backtest Win Rate", f"{backtest_win_rate:.1f}%" if backtest_is_comparable else "N/A")
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
setup_cols = st.columns(2)
setup_cols[0].metric("Best Performing Setup", best_setup_label)
setup_cols[1].metric("Worst Performing Setup", worst_setup_label)

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
wf_ctrl1, wf_ctrl2, wf_ctrl3 = st.columns([1.35, 1.0, 1.0])
with wf_ctrl1:
    wf_manual_refresh = st.button(
        "Run / Refresh Walk Forward Now", key="walk_forward_manual_refresh_v17",
        type="primary", use_container_width=True,
        help="Run the chronological walk-forward validation immediately using the current selected symbol, timeframe and available candle history."
    )
with wf_ctrl2:
    wf_auto_refresh = st.toggle(
        "Auto Refresh Walk Forward", value=False,
        key="walk_forward_auto_refresh_enabled_v17",
        help="Re-run the walk-forward report at the selected interval. This is independent from market-data auto refresh."
    )
with wf_ctrl3:
    wf_refresh_seconds = st.selectbox(
        "Walk-forward interval", [60, 180, 300, 600, 900], index=2,
        format_func=lambda x: f"{x // 60} min",
        key="walk_forward_auto_refresh_interval_v17",
        disabled=not wf_auto_refresh
    )

wf_auto_tick = None
if wf_auto_refresh and AUTO_REFRESH_AVAILABLE:
    wf_auto_tick = st_autorefresh(
        interval=int(wf_refresh_seconds) * 1000,
        key="walk_forward_independent_auto_refresh_v17"
    )
elif wf_auto_refresh and not AUTO_REFRESH_AVAILABLE:
    st.warning("Walk-forward auto refresh needs streamlit-autorefresh. Manual Run / Refresh remains available.")

# Store the latest report in session state. Global market auto-refresh does not
# repeatedly recompute this expensive report unless its own auto-refresh is ON.
try:
    _wf_tail = backtest_data.tail(500)
    _wf_content_hash = int(pd.util.hash_pandas_object(_wf_tail, index=True).sum()) if not _wf_tail.empty else 0
    _wf_data_key = f"{selected}|{timeframe}|{len(backtest_data)}|{_wf_content_hash}"
except Exception:
    _wf_data_key = f"{selected}|{timeframe}|{len(backtest_data)}"

_wf_cached = st.session_state.get("walk_forward_cached_result")
_wf_cached_key = st.session_state.get("walk_forward_cached_data_key")
_wf_previous_auto = bool(st.session_state.get("walk_forward_auto_was_enabled", False))
_wf_previous_tick = st.session_state.get("walk_forward_last_auto_tick")
_wf_auto_tick_changed = bool(
    wf_auto_refresh and wf_auto_tick is not None
    and (_wf_previous_tick is None or wf_auto_tick != _wf_previous_tick)
)
_wf_auto_just_enabled = bool(wf_auto_refresh and not _wf_previous_auto)
_wf_should_run = bool(
    wf_manual_refresh
    or (wf_auto_refresh and (_wf_auto_tick_changed or _wf_auto_just_enabled))
)

if _wf_should_run:
    with st.spinner("Running chronological walk-forward validation…"):
        walk_forward_result = walk_forward_validation(
            backtest_data, symbol=selected, timeframe_label=timeframe,
            option_metrics=None, news_info=historical_news_context
        )
    st.session_state["walk_forward_cached_result"] = walk_forward_result
    st.session_state["walk_forward_cached_data_key"] = _wf_data_key
    st.session_state["walk_forward_last_run_at"] = now_str()
    st.session_state["walk_forward_last_auto_tick"] = wf_auto_tick
    st.session_state["walk_forward_auto_was_enabled"] = wf_auto_refresh
else:
    st.session_state["walk_forward_auto_was_enabled"] = wf_auto_refresh
    if wf_auto_tick is not None:
        st.session_state["walk_forward_last_auto_tick"] = wf_auto_tick
    if _wf_cached_key == _wf_data_key and _wf_cached is not None:
        walk_forward_result = _wf_cached
    else:
        walk_forward_result = {
            "folds": 0, "validated_folds": 0, "avg_win_rate": None,
            "avg_profit_factor": None, "avg_expectancy": None,
            "table": pd.DataFrame()
        }
        if _wf_cached is None:
            st.info("Walk-forward has not run yet. Press Run / Refresh Walk Forward Now, or enable its independent Auto Refresh. It will not run automatically during initial page load.")
        if _wf_cached is not None:
            st.info("Market history has changed since the last walk-forward run. Press Run / Refresh Walk Forward Now, or enable its Auto Refresh option. The displayed metrics are not silently carried over from the previous dataset.")

if st.session_state.get("walk_forward_last_run_at"):
    st.caption("Last walk-forward calculation: " + str(st.session_state["walk_forward_last_run_at"]))
if wf_auto_refresh and AUTO_REFRESH_AVAILABLE:
    st.caption(f"Walk-forward auto refresh is ON — every {wf_refresh_seconds // 60} minute(s).")
else:
    st.caption("Walk-forward auto refresh is OFF. Use Run / Refresh Walk Forward Now whenever you want to recalculate.")

wf1, wf2, wf3, wf4 = st.columns(4)
with wf1:
    st.metric("Folds", walk_forward_result["folds"])
with wf2:
    st.metric("Validated Folds", walk_forward_result.get("validated_folds", 0))
with wf3:
    st.metric("Avg Win Rate", f"{walk_forward_result['avg_win_rate']:.2f}%" if walk_forward_result.get("avg_win_rate") is not None else "N/A")
with wf4:
    st.metric("Avg Profit Factor", f"{walk_forward_result['avg_profit_factor']:.2f}" if walk_forward_result.get("avg_profit_factor") is not None else "N/A")

if walk_forward_result.get("avg_expectancy") is not None:
    st.caption(f"Average validated-fold expectancy: {walk_forward_result['avg_expectancy']:.2f} points per unit")
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

if auto_refresh and not AUTO_REFRESH_AVAILABLE:
    st.caption("Auto-refresh helper is not installed; use the manual browser refresh or install streamlit-autorefresh.")

# =========================================================
# EDUCATIONAL MARKET LEARNING LAB (isolated; descriptive only)
# =========================================================


# =========================================================
# EDUCATIONAL MARKET LEARNING LAB (isolated; descriptive only)
# =========================================================
# This section is intentionally read-only. It does not write to the existing
# signal engine, paper-trading state, journal, broker APIs, alerts, or orders.

def _edu_ohlcv_frame(frame):
    """Return a clean OHLCV copy without mutating the app's market dataframe."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).strip().title() for c in out.columns]
    aliases = {"Adj Close": "Close", "Datetime": "Date", "Timestamp": "Date"}
    out = out.rename(columns=aliases)
    required = ["Open", "High", "Low", "Close"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    for c in required + (["Volume"] if "Volume" in out.columns else []):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required)
    out = out[(out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) &
              (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))]
    return out.tail(1500)


def _edu_pivots(frame, left=3, right=3):
    """Find simple local swing points for classroom market-structure study."""
    if len(frame) < left + right + 3:
        return []
    highs = frame["High"].to_numpy(dtype=float)
    lows = frame["Low"].to_numpy(dtype=float)
    pivots = []
    for i in range(left, len(frame) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            pivots.append((i, "H", float(highs[i])))
        if lows[i] == min(lows[i-left:i+right+1]):
            pivots.append((i, "L", float(lows[i])))
    return sorted(pivots)


def _edu_structure(frame):
    pivots = _edu_pivots(frame)
    prior = {"H": None, "L": None}
    rows = []
    for idx, kind, price in pivots:
        label = "Swing High" if kind == "H" else "Swing Low"
        previous = prior[kind]
        if previous is not None:
            if kind == "H":
                label = "Higher High (HH)" if price > previous[1] else "Lower High (LH)" if price < previous[1] else "Equal High"
            else:
                label = "Higher Low (HL)" if price > previous[1] else "Lower Low (LL)" if price < previous[1] else "Equal Low"
        rows.append({"Row": idx, "Date": str(frame.index[idx]), "Type": kind, "Price": price, "Observation": label})
        prior[kind] = (idx, price)
    # Mark close-cross events against the most recent confirmed swing levels.
    # BOS/CHOCH are educational labels only; no direction/action is emitted.
    swing_highs = [(i, p) for i, k, p in pivots if k == "H"]
    swing_lows = [(i, p) for i, k, p in pivots if k == "L"]
    prior_break_direction = 0
    closes = frame["Close"].to_numpy(dtype=float)
    for i in range(1, len(frame)):
        hlevels = [(j, p) for j, p in swing_highs if j < i]
        llevels = [(j, p) for j, p in swing_lows if j < i]
        if hlevels and closes[i-1] <= hlevels[-1][1] < closes[i]:
            direction = 1
            event = "CHOCH observation" if prior_break_direction == -1 else "BOS observation"
            rows.append({"Row": i, "Date": str(frame.index[i]), "Type": "Close break", "Price": float(closes[i]), "Observation": event + " — above prior swing high"})
            prior_break_direction = direction
        elif llevels and closes[i-1] >= llevels[-1][1] > closes[i]:
            direction = -1
            event = "CHOCH observation" if prior_break_direction == 1 else "BOS observation"
            rows.append({"Row": i, "Date": str(frame.index[i]), "Type": "Close break", "Price": float(closes[i]), "Observation": event + " — below prior swing low"})
            prior_break_direction = direction
    return pd.DataFrame(rows).sort_values("Row").reset_index(drop=True) if rows else pd.DataFrame()


def _edu_volume_profile(frame, bins=30):
    if frame.empty or "Volume" not in frame.columns:
        return pd.DataFrame(), None
    lo, hi = float(frame["Low"].min()), float(frame["High"].max())
    if not np.isfinite(lo + hi) or hi <= lo:
        return pd.DataFrame(), None
    edges = np.linspace(lo, hi, bins + 1)
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    vol, _ = np.histogram(typical, bins=edges, weights=frame["Volume"].fillna(0).clip(lower=0))
    centers = (edges[:-1] + edges[1:]) / 2
    result = pd.DataFrame({"Price Band Midpoint": centers, "Observed Volume": vol})
    poc = float(centers[int(np.argmax(vol))]) if len(vol) and vol.sum() > 0 else None
    return result, poc


def _edu_oi_classification(frame):
    needed = {"Change in Price", "Change in OI"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["Change in Price"] = pd.to_numeric(out["Change in Price"], errors="coerce")
    out["Change in OI"] = pd.to_numeric(out["Change in OI"], errors="coerce")
    def classify(r):
        p, o = r["Change in Price"], r["Change in OI"]
        if pd.isna(p) or pd.isna(o): return "Insufficient data"
        if p > 0 and o > 0: return "Long buildup pattern"
        if p < 0 and o > 0: return "Short buildup pattern"
        if p > 0 and o < 0: return "Short covering pattern"
        if p < 0 and o < 0: return "Long unwinding pattern"
        return "Mixed / unchanged"
    out["Descriptive OI Category"] = out.apply(classify, axis=1)
    return out


with st.expander("📚 Educational Market Learning Lab — descriptive analysis only", expanded=False):
    st.caption("Read-only learning tools. Outputs are historical observations, not trade instructions, forecasts, or order controls. This section is isolated from the existing signal and paper-trading logic.")
    edu_tabs = st.tabs(["Market Structure", "Liquidity & Sweeps", "Sector / Relative Strength", "OI Patterns", "Order Blocks & FVG", "Volume Profile", "AI Trainer"])
    edu_frame = _edu_ohlcv_frame(data if "data" in globals() else pd.DataFrame())

    with edu_tabs[0]:
        st.markdown("**Swing structure: HH / HL / LH / LL, BOS and CHOCH study**")
        if edu_frame.empty:
            st.info("Valid OHLC history is not available for this symbol/timeframe.")
        else:
            structure = _edu_structure(edu_frame)
            st.dataframe(structure.tail(100), use_container_width=True, hide_index=True)
            if len(structure) >= 4:
                st.caption("BOS/CHOCH are context-dependent concepts. Review swing sequence and prior structure manually; this lab does not label directional trade signals.")
            else:
                st.info("More historical candles are needed to identify a useful swing sequence.")

    with edu_tabs[1]:
        st.markdown("**Liquidity areas and possible sweep observations**")
        if edu_frame.empty:
            st.info("Valid OHLC history is not available.")
        else:
            piv = _edu_pivots(edu_frame)
            levels = [x[2] for x in piv]
            observations = []
            for i in range(1, len(edu_frame)):
                row = edu_frame.iloc[i]
                prior_levels = [x[2] for x in piv if x[0] < i]
                if prior_levels:
                    nearby_high = max((v for v in prior_levels if v >= row["Close"]), default=None)
                    nearby_low = min((v for v in prior_levels if v <= row["Close"]), default=None)
                    if nearby_high is not None and row["High"] > nearby_high and row["Close"] < nearby_high:
                        observations.append({"Date": str(edu_frame.index[i]), "Observation": "Upper-level probe and close back below", "Reference level": nearby_high})
                    if nearby_low is not None and row["Low"] < nearby_low and row["Close"] > nearby_low:
                        observations.append({"Date": str(edu_frame.index[i]), "Observation": "Lower-level probe and close back above", "Reference level": nearby_low})
            st.dataframe(pd.DataFrame(observations).tail(100), use_container_width=True, hide_index=True)
            st.caption("A wick or level breach alone cannot establish intent, stop hunting, or a confirmed false breakout.")

    with edu_tabs[2]:
        st.markdown("**Sector comparison and relative-strength worksheet**")
        st.caption("Upload a CSV with columns: Symbol, Sector, Close. Optional: BenchmarkClose. Values are compared descriptively over the uploaded rows.")
        sector_file = st.file_uploader("Upload sector comparison CSV", type=["csv"], key="edu_sector_csv")
        if sector_file is not None:
            try:
                sf = pd.read_csv(sector_file)
                required = {"Symbol", "Sector", "Close"}
                if required.issubset(sf.columns):
                    sf["Close"] = pd.to_numeric(sf["Close"], errors="coerce")
                    if "BenchmarkClose" in sf.columns:
                        sf["BenchmarkClose"] = pd.to_numeric(sf["BenchmarkClose"], errors="coerce")
                        sf["Relative Ratio"] = sf["Close"] / sf["BenchmarkClose"].replace(0, np.nan)
                    st.dataframe(sf, use_container_width=True, hide_index=True)
                    sector_summary = sf.groupby("Sector", dropna=True).agg(Constituents=("Symbol", "count"), Mean_Close=("Close", "mean")).reset_index()
                    st.markdown("**Uploaded sector coverage**")
                    st.dataframe(sector_summary, use_container_width=True, hide_index=True)
                else:
                    st.warning("CSV must contain Symbol, Sector and Close columns.")
            except Exception as e:
                st.error(f"Could not read sector CSV: {e}")
        else:
            st.info("Upload a historical sector CSV to view sector coverage and relative-value columns. No sector ranking or trade recommendation is generated.")

    with edu_tabs[3]:
        st.markdown("**Open-interest buildup categories**")
        st.caption("Upload rows with Change in Price and Change in OI; add Symbol/Expiry/Strike columns if available.")
        oi_file = st.file_uploader("Upload option-chain OI CSV", type=["csv"], key="edu_oi_csv")
        if oi_file is not None:
            try:
                oi_raw = pd.read_csv(oi_file)
                oi_result = _edu_oi_classification(oi_raw)
                if oi_result.empty:
                    st.warning("CSV needs Change in Price and Change in OI columns.")
                else:
                    st.dataframe(oi_result, use_container_width=True, hide_index=True)
                    st.caption("These four labels are conventional descriptive interpretations; they do not establish participant intent or predict price direction.")
            except Exception as e:
                st.error(f"Could not read OI CSV: {e}")
        else:
            st.info("No OI file uploaded.")

    with edu_tabs[4]:
        st.markdown("**Order Block and Fair Value Gap (FVG) visual annotations**")
        if len(edu_frame) < 3:
            st.info("At least three valid OHLC rows are required.")
        else:
            zones = []
            for i in range(1, len(edu_frame)):
                prev, cur = edu_frame.iloc[i-1], edu_frame.iloc[i]
                if prev["Close"] < prev["Open"] and cur["Close"] > cur["Open"] and cur["Close"] > prev["High"]:
                    zones.append({"Date": str(edu_frame.index[i-1]), "Annotation": "Bullish-candle displacement context", "Zone Low": float(prev["Low"]), "Zone High": float(prev["Open"])})
                elif prev["Close"] > prev["Open"] and cur["Close"] < cur["Open"] and cur["Close"] < prev["Low"]:
                    zones.append({"Date": str(edu_frame.index[i-1]), "Annotation": "Bearish-candle displacement context", "Zone Low": float(prev["Open"]), "Zone High": float(prev["High"])})
            gaps = []
            for i in range(2, len(edu_frame)):
                a, c = edu_frame.iloc[i-2], edu_frame.iloc[i]
                if c["Low"] > a["High"]:
                    gaps.append({"Date": str(edu_frame.index[i]), "Annotation": "Upward three-candle gap", "Zone Low": float(a["High"]), "Zone High": float(c["Low"])})
                elif c["High"] < a["Low"]:
                    gaps.append({"Date": str(edu_frame.index[i]), "Annotation": "Downward three-candle gap", "Zone Low": float(c["High"]), "Zone High": float(a["Low"])})
            left, right = st.columns(2)
            with left:
                st.markdown("**Candidate candle zones**")
                st.dataframe(pd.DataFrame(zones).tail(75), use_container_width=True, hide_index=True)
            with right:
                st.markdown("**Three-candle gaps**")
                st.dataframe(pd.DataFrame(gaps).tail(75), use_container_width=True, hide_index=True)
            st.caption("These are mechanical chart annotations, not institutionally verified order blocks or actionable retest levels.")

    with edu_tabs[5]:
        st.markdown("**Volume Profile: POC / value-area approximation / HVN-LVN**")
        profile, poc = _edu_volume_profile(edu_frame)
        if profile.empty:
            st.info("Volume data is unavailable; profile cannot be calculated.")
        else:
            total_volume = float(profile["Observed Volume"].sum())
            profile["Volume Share %"] = (profile["Observed Volume"] / total_volume * 100) if total_volume else 0.0
            profile["Node"] = "Normal"
            if len(profile) >= 3:
                q75, q25 = profile["Observed Volume"].quantile([0.75, 0.25])
                profile.loc[profile["Observed Volume"] >= q75, "Node"] = "High-volume node (HVN)"
                profile.loc[profile["Observed Volume"] <= q25, "Node"] = "Low-volume node (LVN)"
            # Approximate 70% value area by expanding from the POC bin.
            if total_volume > 0:
                vol_values = profile["Observed Volume"].to_numpy(dtype=float)
                center = int(np.argmax(vol_values)); lo_i = hi_i = center; accumulated = vol_values[center]
                while accumulated < total_volume * 0.70 and (lo_i > 0 or hi_i < len(vol_values)-1):
                    left_v = vol_values[lo_i-1] if lo_i > 0 else -1
                    right_v = vol_values[hi_i+1] if hi_i < len(vol_values)-1 else -1
                    if right_v > left_v and hi_i < len(vol_values)-1:
                        hi_i += 1; accumulated += vol_values[hi_i]
                    elif lo_i > 0:
                        lo_i -= 1; accumulated += vol_values[lo_i]
                    else:
                        break
                val_approx = float(np.linspace(float(edu_frame["Low"].min()), float(edu_frame["High"].max()), 31)[lo_i])
                vah_approx = float(np.linspace(float(edu_frame["Low"].min()), float(edu_frame["High"].max()), 31)[hi_i+1])
            else:
                val_approx = vah_approx = None
            p1, p2, p3 = st.columns(3)
            p1.metric("POC (approx.)", f"{poc:,.2f}" if poc is not None else "N/A")
            p2.metric("VAL (approx.)", f"{val_approx:,.2f}" if val_approx is not None else "N/A")
            p3.metric("VAH (approx.)", f"{vah_approx:,.2f}" if vah_approx is not None else "N/A")
            st.dataframe(profile.sort_values("Price Band Midpoint"), use_container_width=True, hide_index=True)
            st.caption("POC/VAH/VAL are histogram approximations. HVN/LVN labels use relative volume quartiles; binning choices affect all values.")

    with edu_tabs[6]:
        st.markdown("**AI Trainer — unsupervised historical candle-pattern grouping**")
        st.caption("The trainer groups historical candle/volume feature shapes for learning. It does not train outcome labels, predict returns, estimate win probability, or generate trading decisions.")
        if len(edu_frame) < 40:
            st.info("At least 40 valid OHLC rows are needed for a useful demonstration.")
        else:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                feature_frame = pd.DataFrame(index=edu_frame.index)
                candle_range = (edu_frame["High"] - edu_frame["Low"]).replace(0, np.nan)
                feature_frame["Body / Range"] = (edu_frame["Close"] - edu_frame["Open"]) / candle_range
                feature_frame["Upper Wick / Range"] = (edu_frame["High"] - edu_frame[["Open", "Close"]].max(axis=1)) / candle_range
                feature_frame["Lower Wick / Range"] = (edu_frame[["Open", "Close"]].min(axis=1) - edu_frame["Low"]) / candle_range
                feature_frame["Range %"] = candle_range / edu_frame["Close"].replace(0, np.nan) * 100
                if "Volume" in edu_frame.columns:
                    vol_mean = edu_frame["Volume"].rolling(20, min_periods=5).mean()
                    feature_frame["Volume / 20-bar mean"] = edu_frame["Volume"] / vol_mean.replace(0, np.nan)
                feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
                max_clusters = min(6, max(2, len(feature_frame) // 15))
                cluster_count = st.slider("Pattern groups", 2, max_clusters, min(3, max_clusters), key="edu_ai_clusters")
                if len(feature_frame) >= cluster_count * 5:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(feature_frame)
                    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
                    labels = model.fit_predict(scaled)
                    result = feature_frame.copy()
                    result["Pattern Group"] = labels.astype(int)
                    st.metric("Historical rows grouped", f"{len(result):,}")
                    st.dataframe(result.tail(100), use_container_width=True)
                    st.markdown("**Group feature averages**")
                    st.dataframe(result.groupby("Pattern Group").mean(numeric_only=True).round(4), use_container_width=True)
                    st.caption("Groups are descriptive clusters, not good/bad labels. Re-running on different history or settings can change group assignments.")
                else:
                    st.info("Not enough clean feature rows for the selected number of groups.")
            except ImportError:
                st.warning("Install scikit-learn to enable the educational AI Trainer: pip install scikit-learn")
            except Exception as e:
                st.error(f"AI Trainer could not complete: {e}")
            

            # Aapki existing script jahan stream render karti hai:
            inst.render_institutional_dashboard(data, selected, fetch_data("^NSEI", "1d", "30d"), load_journal())
            # Institutional Upgrades Dashboard Integration
            st.divider()
            st.header("Institutional Upgrades & Advanced Analytics Suite")
            
            # Benchmark data fetch karke dashboard render karein
            nifty_bm = fetch_data("^NSEI", "1d", "30d")
            inst.render_institutional_dashboard(data, selected, nifty_bm, load_journal())
