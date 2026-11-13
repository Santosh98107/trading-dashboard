"""database.py - file paths, IST time helpers, password hashing, backups, CSV safety and the
journal / paper-state / risk-state load & save functions.  Split out of app.py; behaviour is unchanged."""
import os
import json
import math
import shutil
import hashlib
import hmac
import gzip
import io
import re
import sqlite3
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
DB_FILE = os.path.join(DATA_DIR, "elite.db")   # SQLite database: journal, users, paper/risk state, summaries, audit, error log

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
        return False, ("Reset key is not configured. Set RESET_KEY in secrets/environment, or log in with the "
                       f"default {DEFAULT_ADMIN_USERNAME} / {DEFAULT_ADMIN_PASSWORD} and use Change Password.")
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


# =========================================================
# SQLITE STORAGE
# Every former CSV data file (journal, users, paper / risk state, summaries, signal audit, error log,
# AI Lab history) is now a table in one SQLite file (DB_FILE). The function names safe_read_csv /
# safe_write_csv / backup_file are kept so the rest of the app is unchanged.
#   * writes run inside one transaction (all-or-nothing), WAL mode lets readers and a writer coexist
#   * values are stored as TEXT and read back through pandas' own CSV type inference, so numbers, dates and
#     True/False columns come back exactly as they did from the CSV files
#   * an existing CSV is imported into its table automatically the first time it is needed and is left on disk
# =========================================================
_DB_LOCK = threading.RLock()


def _table_name(path):
    """Table for a data-file path, or None when the path is outside DATA_DIR (then plain CSV is used)."""
    try:
        full = os.path.abspath(str(path))
        if os.path.dirname(full) != os.path.abspath(DATA_DIR) or not full.lower().endswith(".csv"):
            return None
        return re.sub(r"\W", "_", os.path.splitext(os.path.basename(full))[0])
    except Exception:
        return None


def _db_connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_FILE, timeout=20, isolation_level=None)   # autocommit; we open transactions explicitly
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=20000")
    except Exception:
        pass
    return con


def _q(name):
    return '"' + str(name).replace('"', '""') + '"'


def _table_exists(con, table):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _frame_to_text_rows(df):
    """DataFrame -> (columns, rows of str/None) using exactly the text pandas would write to a CSV."""
    text = pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=False)
    cols = [str(c) for c in text.columns]
    rows = [tuple((v if v != "" else None) for v in r) for r in text.itertuples(index=False, name=None)]
    return cols, rows


def _store_table(con, table, df):
    cols, rows = _frame_to_text_rows(df)
    if not cols:
        cols = ["_empty"]
        rows = []
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(f"DROP TABLE IF EXISTS {_q(table)}")
        con.execute(f"CREATE TABLE {_q(table)} (" + ", ".join(f"{_q(c)} TEXT" for c in cols) + ")")
        if rows:
            con.executemany(f"INSERT INTO {_q(table)} VALUES (" + ",".join("?" * len(cols)) + ")", rows)
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise


def _load_table(con, table):
    cur = con.execute(f"SELECT * FROM {_q(table)} ORDER BY rowid")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if cols == ["_empty"]:
        return pd.DataFrame()
    text = pd.DataFrame(rows, columns=cols, dtype=object)
    return pd.read_csv(io.StringIO(text.to_csv(index=False)))   # same type inference as reading the old CSV


def _migrate_csv_if_needed(con, path, table, log=True):
    """One-time import of a legacy CSV into its table. The CSV file itself is left untouched."""
    if _table_exists(con, table) or not os.path.exists(path):
        return
    try:
        _store_table(con, table, pd.read_csv(path))
    except Exception as e:
        if log:
            log_app_error("csv_to_sqlite_migration", f"{path} -> {e}")


def db_exists(path):
    table = _table_name(path)
    if table is None:
        return os.path.exists(path)
    with _DB_LOCK:
        con = _db_connect()
        try:
            _migrate_csv_if_needed(con, path, table)
            return _table_exists(con, table)
        finally:
            con.close()


def db_read(path):
    table = _table_name(path)
    if table is None:
        return pd.read_csv(path)
    with _DB_LOCK:
        con = _db_connect()
        try:
            _migrate_csv_if_needed(con, path, table)
            if not _table_exists(con, table):
                raise FileNotFoundError(f"table {table} does not exist")
            return _load_table(con, table)
        finally:
            con.close()


def db_write(df, path):
    table = _table_name(path)
    if table is None:
        tmp_path = f"{path}.tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        return
    with _DB_LOCK:
        con = _db_connect()
        try:
            _store_table(con, table, df)
        finally:
            con.close()


def backup_file(path):
    """Throttled snapshot of the whole SQLite database (SQLite's online-backup API: safe while the app is running)."""
    try:
        if not os.path.exists(DB_FILE):
            return
        now_ts = pytime.time()
        if now_ts - _LAST_BACKUP_AT.get("db", 0) < _BACKUP_MIN_INTERVAL_SEC:
            return   # auto-refresh saves every few seconds; do not flood the backup history
        _LAST_BACKUP_AT["db"] = now_ts
        ts = ist_now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(BACKUP_DIR, f"{ts}_elite.db")
        src = _db_connect()
        try:
            dst = sqlite3.connect(dest)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
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
    """Append one row to the error-log table (kept to the latest 500 rows)."""
    try:
        with _DB_LOCK:
            con = _db_connect()
            try:
                table = _table_name(ERROR_LOG_FILE)
                _migrate_csv_if_needed(con, ERROR_LOG_FILE, table, log=False)
                con.execute(f"CREATE TABLE IF NOT EXISTS {_q(table)} (DateTime TEXT, Source TEXT, Message TEXT)")
                con.execute(f"INSERT INTO {_q(table)} VALUES (?,?,?)", (now_str(), str(source), str(message)))
                con.execute(f"DELETE FROM {_q(table)} WHERE rowid <= (SELECT MAX(rowid) FROM {_q(table)}) - 500")
            finally:
                con.close()
    except Exception:
        pass


def safe_read_csv(path, default_df=None):
    """Read a data table (name kept from the CSV days). Returns default_df / empty frame when it does not exist."""
    try:
        if db_exists(path):
            return db_read(path)
        return default_df.copy() if default_df is not None else pd.DataFrame()
    except Exception as e:
        log_app_error("safe_read_csv", f"{path} -> {e}")
        return default_df.copy() if default_df is not None else pd.DataFrame()


def safe_write_csv(df, path):
    """Save a data table atomically (single SQLite transaction)."""
    try:
        backup_file(path)
        db_write(df, path)
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
FIRST_RUN_PASSWORD_FILE = os.path.join(DATA_DIR, "FIRST_RUN_ADMIN_PASSWORD.txt")


def _create_admin_user():
    password, _ = _initial_admin_password()
    safe_write_csv(pd.DataFrame([{"username": DEFAULT_ADMIN_USERNAME, "password": hash_password(password)}]), USERS_FILE)


def init_users():
    if not db_exists(USERS_FILE):
        _create_admin_user()
        return
    try:
        users = db_read(USERS_FILE)
        if "username" not in users.columns or "password" not in users.columns:
            raise ValueError("Invalid users file structure")
    except Exception:
        _create_admin_user()


def init_journal():
    """Create the paper journal with backward-compatible partial-exit fields."""
    if not db_exists(JOURNAL_FILE):
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
    if not db_exists(PAPER_STATE_FILE):
        df = pd.DataFrame([{
            "Balance": DEFAULT_INITIAL_BALANCE,
            "UsedMargin": 0.0,
            "OpenPositions": 0,
            "InitialBalance": DEFAULT_INITIAL_BALANCE
        }])
        safe_write_csv(df, PAPER_STATE_FILE)


def init_daily_summary():
    if not db_exists(DAILY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "Date", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), DAILY_SUMMARY_FILE)


def init_weekly_summary():
    if not db_exists(WEEKLY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "WeekKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), WEEKLY_SUMMARY_FILE)


def init_monthly_summary():
    if not db_exists(MONTHLY_SUMMARY_FILE):
        safe_write_csv(pd.DataFrame(columns=[
            "MonthKey", "TotalTrades", "Wins", "Losses", "RealizedPnL", "WinRate"
        ]), MONTHLY_SUMMARY_FILE)


def init_risk_state():
    if not db_exists(RISK_STATE_FILE):
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
    if not db_exists(WEEKLY_RISK_FILE):
        df = pd.DataFrame([{
            "WeekKey": ist_now().strftime("%G-W%V"),
            "WeeklyLossLimit": 5000.0,
            "WeeklyPnL": 0.0,
            "WeeklyLock": False
        }])
        safe_write_csv(df, WEEKLY_RISK_FILE)


def init_monthly_risk():
    if not db_exists(MONTHLY_RISK_FILE):
        df = pd.DataFrame([{
            "MonthKey": ist_now().strftime("%Y-%m"),
            "MonthlyLossLimit": 15000.0,
            "MonthlyPnL": 0.0,
            "MonthlyLock": False
        }])
        safe_write_csv(df, MONTHLY_RISK_FILE)


def init_signal_audit():
    if not db_exists(SIGNAL_AUDIT_FILE):
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
        if db_exists(path):
            old = db_read(path)
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



# =========================================================
# OPTION VOLUME HISTORY (live Put / Call volume flow)
# NSE reports volume as a running total for the day. Saving a snapshot at every NSE update lets the
# dashboard show how much Call and Put volume was added in the last few minutes (the live flow).
# =========================================================
def record_option_volume(symbol, om):
    """Save one snapshot of day-so-far Call / Put volume. Duplicates (same NSE timestamp) and stale
    snapshots are ignored."""
    try:
        if not isinstance(om, dict) or om.get("chain_stale") or om.get("call_volume") is None:
            return
        if not (om.get("call_volume") or om.get("put_volume")):
            return
        chain_time = str(om.get("chain_time") or "")
        with _DB_LOCK:
            con = _db_connect()
            try:
                con.execute('CREATE TABLE IF NOT EXISTS option_volume_history (Symbol TEXT, ChainTime TEXT, RecordedEpoch REAL, '
                            'RecordedAt TEXT, Expiry TEXT, CallVolume REAL, PutVolume REAL, CallOI REAL, PutOI REAL)')
                last = con.execute("SELECT ChainTime, Expiry FROM option_volume_history WHERE Symbol=? ORDER BY rowid DESC LIMIT 1", (symbol,)).fetchone()
                if last and chain_time and last[0] == chain_time and last[1] == str(om.get("chain_expiry")):
                    return
                con.execute("INSERT INTO option_volume_history VALUES (?,?,?,?,?,?,?,?,?)",
                            (symbol, chain_time, pytime.time(), now_str(), str(om.get("chain_expiry")),
                             float(om.get("call_volume") or 0), float(om.get("put_volume") or 0),
                             float(om.get("call_oi") or 0), float(om.get("put_oi") or 0)))
                con.execute("DELETE FROM option_volume_history WHERE RecordedEpoch < ?", (pytime.time() - 3 * 86400,))
            finally:
                con.close()
    except Exception as e:
        log_app_error("record_option_volume", e)


def load_option_volume(symbol, hours=9):
    """Snapshots of the last `hours` hours as a DataFrame: Time (IST), Expiry, Call, Put."""
    try:
        with _DB_LOCK:
            con = _db_connect()
            try:
                if not _table_exists(con, "option_volume_history"):
                    return pd.DataFrame(columns=["Epoch", "Time", "Expiry", "Call", "Put"])
                rows = con.execute("SELECT RecordedEpoch, RecordedAt, Expiry, CallVolume, PutVolume FROM option_volume_history "
                                   "WHERE Symbol=? AND RecordedEpoch >= ? ORDER BY RecordedEpoch",
                                   (symbol, pytime.time() - hours * 3600)).fetchall()
            finally:
                con.close()
        df = pd.DataFrame(rows, columns=["Epoch", "Time", "Expiry", "Call", "Put"])
        df["Time"] = pd.to_datetime(df["Time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        return df
    except Exception as e:
        log_app_error("load_option_volume", e)
        return pd.DataFrame(columns=["Epoch", "Time", "Expiry", "Call", "Put"])


def option_volume_live_flow(symbol, window_minutes=15):
    """Call / Put volume ADDED over roughly the last `window_minutes` (latest total minus an earlier total).
    Returns None until two snapshots at least 2 minutes apart exist."""
    df = load_option_volume(symbol)
    if df.empty or len(df) < 2:
        return None
    latest = df.iloc[-1]
    df = df[(df["Expiry"] == latest["Expiry"]) & (df["Time"].dt.date == latest["Time"].date())]
    if len(df) < 2:
        return None
    target = float(latest["Epoch"]) - window_minutes * 60
    older = df[df["Epoch"] <= target]
    base = older.iloc[-1] if not older.empty else df.iloc[0]
    minutes = (float(latest["Epoch"]) - float(base["Epoch"])) / 60.0
    if minutes < 2:
        return None
    d_call, d_put = float(latest["Call"] - base["Call"]), float(latest["Put"] - base["Put"])
    if d_call < 0 or d_put < 0:      # volume counters restart each day; ignore a reset
        return None
    total = d_call + d_put
    live_pcr = round(d_put / d_call, 2) if d_call > 0 else None
    if total <= 0:
        dominant = "No new volume"
    elif d_call >= 1.15 * d_put:
        dominant = "Call volume rising faster"
    elif d_put >= 1.15 * d_call:
        dominant = "Put volume rising faster"
    else:
        dominant = "Balanced"
    return {"minutes": round(minutes, 1), "d_call": int(d_call), "d_put": int(d_put), "live_pcr": live_pcr,
            "dominant": dominant, "history": df[["Time", "Call", "Put"]]}
