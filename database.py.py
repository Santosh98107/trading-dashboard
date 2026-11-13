"""database.py - file paths, IST time helpers, password hashing, backups, CSV safety and the
journal / paper-state / risk-state load & save functions.  Split out of app.py; behaviour is unchanged."""
import os
import json
import math
import shutil
import hashlib
import hmac
import gzip
import threading
import time as pytime
from datetime import datetime, timedelta, time, timezone

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# FILES / CONFIG
# =========================================================
# Set ELITE_DATA_DIR to a PERSISTENT volume in production. Streamlit Community Cloud's
# local disk is wiped on every restart/redeploy, which would erase the journal and account.
DATA_DIR = os.environ.get("ELITE_DATA_DIR", "elite_data")
STORAGE_IS_CONFIGURED = "ELITE_DATA_DIR" in os.environ
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

DEFAULT_INITIAL_BALANCE = 100000.0
DATA_INTEGRITY_MIN_ROWS = 30
BACKUP_RETENTION_COUNT = 30

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def ist_now():
    """Naive datetime in Indian Standard Time (servers such as Streamlit Cloud run in UTC)."""
    return datetime.now(IST_OFFSET).replace(tzinfo=None)


def now_str():
    return ist_now().strftime("%d-%m-%Y %H:%M:%S")


_PBKDF2_ITERATIONS = 240_000


def hash_password(password: str, salt=None) -> str:
    """Salted PBKDF2-HMAC-SHA256 -> 'pbkdf2$iterations$salt_hex$hash_hex'."""
    salt_bytes = bytes.fromhex(salt) if salt else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode(), salt_bytes, _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt_bytes.hex()}${digest.hex()}"


def _legacy_hash(password: str) -> str:
    return hashlib.sha256(str(password).encode()).hexdigest()


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Accepts the new PBKDF2 format and legacy unsalted SHA-256."""
    stored = str(stored or "").strip()
    try:
        if stored.startswith("pbkdf2$"):
            _, iters, salt_hex, hash_hex = stored.split("$")
            digest = hashlib.pbkdf2_hmac("sha256", str(password).encode(), bytes.fromhex(salt_hex), int(iters))
            return hmac.compare_digest(digest.hex(), hash_hex)
        return hmac.compare_digest(_legacy_hash(password), stored)
    except Exception:
        return False


def needs_password_upgrade(stored: str) -> bool:
    return not str(stored or "").strip().startswith("pbkdf2$")


DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
ALWAYS_ALLOW_DEFAULT_LOGIN = True   # admin / admin123 always works, no conditions


def _initial_admin_password():
    """Default admin password is admin123 (change it after first login).
    ADMIN_PASSWORD in st.secrets / environment overrides the default if you prefer."""
    configured = ""
    try:
        configured = str(st.secrets.get("ADMIN_PASSWORD", "") or "")
    except Exception:
        configured = ""
    configured = configured or os.environ.get("ADMIN_PASSWORD", "")
    return (configured or DEFAULT_ADMIN_PASSWORD), False


def _reset_key():
    """Optional RESET_KEY (st.secrets / environment) that authorises resetting a forgotten password."""
    try:
        key = str(st.secrets.get("RESET_KEY", "") or "")
    except Exception:
        key = ""
    return key or os.environ.get("RESET_KEY", "")


def reset_password_with_key(username, reset_key, new_password, confirm_password):
    """Reset a (forgotten) password from the login screen. Requires the configured RESET_KEY."""
    username = str(username or "").strip()
    configured = _reset_key()
    if not configured:
        return False, ("Reset key is not configured. Set RESET_KEY in secrets/environment, or delete "
                       f"'{USERS_FILE}' to restore {DEFAULT_ADMIN_USERNAME} / {DEFAULT_ADMIN_PASSWORD}.")
    if not hmac.compare_digest(str(reset_key or ""), configured):
        return False, "Reset key is incorrect."
    if not username:
        return False, "Enter the username to reset."
    if len(str(new_password or "")) < 6:
        return False, "New password must contain at least 6 characters."
    if new_password != confirm_password:
        return False, "New password and confirmation do not match."
    users = safe_read_csv(USERS_FILE, pd.DataFrame(columns=["username", "password"]))
    if users.empty or not {"username", "password"}.issubset(users.columns):
        users = pd.DataFrame(columns=["username", "password"])
    mask = users["username"].astype(str).str.strip() == username
    if mask.any():
        users.loc[mask, "password"] = hash_password(new_password)
    elif username == DEFAULT_ADMIN_USERNAME:
        users = pd.concat([users, pd.DataFrame([{"username": username, "password": hash_password(new_password)}])], ignore_index=True)
    else:
        return False, "User not found."
    return (True, "Password reset. You can log in now.") if safe_write_csv(users, USERS_FILE) else (False, "Could not save the new password.")


_BACKUP_MIN_INTERVAL_SEC = 600
_LAST_BACKUP_AT = {}


def backup_file(path):
    try:
        if os.path.exists(path):
            now_ts = pytime.time()
            if now_ts - _LAST_BACKUP_AT.get(path, 0) < _BACKUP_MIN_INTERVAL_SEC:
                return   # auto-refresh rewrites files every few seconds; do not flood the backup history
            _LAST_BACKUP_AT[path] = now_ts
            ts = ist_now().strftime("%Y%m%d_%H%M%S")
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
            old = pd.concat([old, row], ignore_index=True).tail(500)   # keep the log bounded
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
        # Atomic save: write a temp file, then swap it in. A crash or restart in the middle of a save can
        # no longer leave a half-written (corrupt) journal / users / paper-state file.
        tmp_path = f"{path}.tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        log_app_error("safe_write_csv", f"{path} -> {e}")
        try:
            if os.path.exists(f"{path}.tmp"):
                os.remove(f"{path}.tmp")
        except Exception:
            pass
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
FIRST_RUN_PASSWORD_FILE = os.path.join(DATA_DIR, "FIRST_RUN_ADMIN_PASSWORD.txt")


def _create_admin_user():
    password, _ = _initial_admin_password()
    safe_write_csv(pd.DataFrame([{"username": DEFAULT_ADMIN_USERNAME, "password": hash_password(password)}]), USERS_FILE)


def init_users():
    if not os.path.exists(USERS_FILE):
        _create_admin_user()
        return
    try:
        users = pd.read_csv(USERS_FILE)
        if "username" not in users.columns or "password" not in users.columns:
            raise ValueError("Invalid users file structure")
    except Exception:
        _create_admin_user()


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
            "LossReason", "SignalStage", "ExitTime"
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
            "CapitalProtectionPct": 20.0,
            "MaxOpenPositions": 3
        }])
        safe_write_csv(df, RISK_STATE_FILE)


def init_weekly_risk():
    if not os.path.exists(WEEKLY_RISK_FILE):
        df = pd.DataFrame([{
            "WeekKey": ist_now().strftime("%G-W%V"),
            "WeeklyLossLimit": 5000.0,
            "WeeklyPnL": 0.0,
            "WeeklyLock": False
        }])
        safe_write_csv(df, WEEKLY_RISK_FILE)


def init_monthly_risk():
    if not os.path.exists(MONTHLY_RISK_FILE):
        df = pd.DataFrame([{
            "MonthKey": ist_now().strftime("%Y-%m"),
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
        for text_col in ["Symbol", "Timeframe", "Signal", "Side", "Status", "Notes", "Result", "StrategyGrade", "LastUpdateTime", "ExitTime", "LossReason", "SignalStage"]:
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
        "LossReason": "", "SignalStage": "", "ExitTime": ""
    }
    for col, default in defaults.items():
        if col not in journal.columns:
            journal[col] = default
            changed = True
    # Keep text/status fields object-typed so older CSVs with blank cells can accept later updates.
    for text_col in ["Symbol", "Timeframe", "Signal", "Side", "Status", "Notes", "Result", "StrategyGrade", "LastUpdateTime", "ExitTime", "LossReason", "SignalStage"]:
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
    df = safe_read_csv(RISK_STATE_FILE)
    if not df.empty and "MaxOpenPositions" not in df.columns:
        df["MaxOpenPositions"] = 3
    return df


def _write_csv_if_changed(df, path):
    """Skip the disk write when the file already holds identical content.

    The risk-lock files were rewritten (and backed up) on every Streamlit rerun, i.e. every few
    seconds with auto-refresh, which slowed the page and wore the disk for no reason."""
    try:
        if os.path.exists(path):
            old = pd.read_csv(path)
            if (list(old.columns) == list(df.columns) and len(old) == len(df)
                    and old.astype(str).reset_index(drop=True).equals(df.astype(str).reset_index(drop=True))):
                return True
    except Exception:
        pass
    return safe_write_csv(df, path)


def save_risk_state(df):
    _write_csv_if_changed(df, RISK_STATE_FILE)


def load_weekly_risk():
    init_weekly_risk()
    df = safe_read_csv(WEEKLY_RISK_FILE)
    current_key = ist_now().strftime("%G-W%V")
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
    _write_csv_if_changed(df, WEEKLY_RISK_FILE)


def load_monthly_risk():
    init_monthly_risk()
    df = safe_read_csv(MONTHLY_RISK_FILE)
    current_key = ist_now().strftime("%Y-%m")
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
    _write_csv_if_changed(df, MONTHLY_RISK_FILE)


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
    if len(new_password) < 6:
        return False, "New password must contain at least 6 characters."
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
    if not verify_password(current_password, stored_hash):
        return False, "Current password is incorrect."

    users.loc[user_mask, "password"] = hash_password(new_password)
    if not safe_write_csv(users, USERS_FILE):
        return False, "Password could not be saved. Check app file permissions and error log."

    return True, "Password updated successfully. Use the new password next time you log in."

