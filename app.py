import os
from datetime import datetime, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from SmartApi import SmartConnect
import pyotp
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Elite Trading Dashboard Ultimate", layout="wide")

JOURNAL_FILE = "trade_journal.csv"
SIGNAL_STATE_FILE = "last_signal_state.txt"
PAPER_STATE_FILE = "paper_trade_state.csv"
USERS_FILE = "users.csv"
SESSION_SUMMARY_FILE = "daily_summary.csv"

def init_users():
    if not os.path.exists(USERS_FILE):

        pd.DataFrame([
            {
                "username": "admin",
                "password": "admin123"
            }
        ]).to_csv(USERS_FILE, index=False)

    else:

        try:

            users = pd.read_csv(USERS_FILE)

            if "username" not in users.columns:
                raise Exception()

            if "password" not in users.columns:
                raise Exception()

        except Exception:

            pd.DataFrame([
                {
                    "username": "admin",
                    "password": "admin123"
                }
            ]).to_csv(USERS_FILE, index=False)

def login_panel():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:

        st.sidebar.subheader("🔐 Login")

        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input(
            "Password",
            type="password"
        )

        
        

        if st.sidebar.button("Login"):

            if (
                username.strip() == "admin"
                and
                password.strip() == "admin123"
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.sidebar.error("Invalid credentials")

        st.stop()

    st.sidebar.success(
        f"Logged in as: {st.session_state.username}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

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
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl = (entry - exit_price) * qty
            pnl_pct = ((entry - exit_price) / entry) * 100

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
def resample_to_3m(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    return df.resample("3min").agg({
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

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

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


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    
    df["TradeDate"] = pd.to_datetime(df.index).date
    
    df["VWAP"] = (
        (tp * df["Volume"])
        .groupby(df["TradeDate"])
        .cumsum()
        /
        df["Volume"]
        .groupby(df["TradeDate"])
        .cumsum()
    )
    
    df.drop(columns=["TradeDate"], inplace=True)
 
    df["VWAP"] = df["VWAP"].replace([np.inf, -np.inf], np.nan)
    df["VWAP"] = df["VWAP"].fillna(df["Close"])
    

    df["RSI"] = compute_rsi(df["Close"], 14)

    df["RSI"] = df["RSI"].replace([np.inf, -np.inf], np.nan)
    df["RSI"] = df["RSI"].fillna(50)
    df["RSI"] = df["RSI"].clip(lower=1, upper=99)

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
        &
        (df["Close"] > df["Open"])
        &
        (
            df["Close"]
            >
            (
                (df["Open"].shift(1) + df["Close"].shift(1))
                / 2
            )
        )
    )

    df["DarkCloudCover"] = (
        (df["Close"].shift(1) > df["Open"].shift(1))
        &
        (df["Close"] < df["Open"])
        &
        (
            df["Close"]
            <
            (
                (df["Open"].shift(1) + df["Close"].shift(1))
                / 2
            )
        )
    )
    df["TweezerBottom"] = (
        abs(df["Low"] - df["Low"].shift(1))
        <= (df["ATR"] * 0.1)
    )
    # Bullish Harami
    df["BullishHarami"] = (
        (df["Close"].shift(1) < df["Open"].shift(1))
        &
        (df["Close"] > df["Open"])
        &
        (df["Open"] > df["Close"].shift(1))
        &
        (df["Close"] < df["Open"].shift(1))
    )

    # Bearish Harami
    df["BearishHarami"] = (
        (df["Close"].shift(1) > df["Open"].shift(1))
        &
        (df["Close"] < df["Open"])
        &
        (df["Open"] < df["Close"].shift(1))
        &
        (df["Close"] > df["Open"].shift(1))
    )
    df["Doji"] = candle_body <= (candle_range * 0.1)
    df["TweezerTop"] = (
        abs(df["High"] - df["High"].shift(1))
        <= (df["ATR"] * 0.1)
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
        &
        (df["Close"].shift(1) > df["Open"].shift(1))
        &
        (df["Close"].shift(2) > df["Open"].shift(2))
        &
        (df["Close"] > df["Close"].shift(1))
        &
        (df["Close"].shift(1) > df["Close"].shift(2))
    )

    df["ThreeBlackCrows"] = (
        (df["Close"] < df["Open"])
        &
        (df["Close"].shift(1) < df["Open"].shift(1))
        &
        (df["Close"].shift(2) < df["Open"].shift(2))
        &
        (df["Close"] < df["Close"].shift(1))
        &
        (df["Close"].shift(1) < df["Close"].shift(2))
    )
    # Morning Star
    df["MorningStar"] = (
        (df["Close"].shift(2) < df["Open"].shift(2))
        &
        (abs(df["Close"].shift(1) - df["Open"].shift(1))
         < candle_body.shift(1))
        &
        (df["Close"] > df["Open"])
    )

    # Evening Star
    df["EveningStar"] = (
        (df["Close"].shift(2) > df["Open"].shift(2))
        &
        (abs(df["Close"].shift(1) - df["Open"].shift(1))
         < candle_body.shift(1))
        &
        (df["Close"] < df["Open"])
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
    
    
def get_mtf_signal(ticker, tf, period):

    if tf == "1m":
        data = fetch_data(ticker, "1m", "7d")

    elif tf == "3m":
        data = fetch_data(ticker, "1m", "7d")

        if not data.empty:
            data = resample_to_3m(data)

    elif tf == "5m":
        data = fetch_data(ticker, "5m", "30d")

    elif tf == "15m":
        data = fetch_data(ticker, "15m", "30d")

    else:
        return "⚪ N/A"

    if data.empty:
        return "⚪ N/A"

    data = add_indicators(data)

    data = data.bfill()
    data = data.ffill()
    if len(data) < 5:
        return "⚪ N/A"

    latest = data.iloc[-1]

    if latest["EMA20"] > latest["EMA50"]:
        return "🟢 Bullish"

    elif latest["EMA20"] < latest["EMA50"]:
        return "🔴 Bearish"

    return "⚪ Neutral"
    
    
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
        return "60m", "90d"
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

def angel_login(
    api_key,
    client_code,
    pin,
    totp_secret
):
    try:
        if "angel_api" in st.session_state:

            return (
                st.session_state["angel_api"],
                st.session_state["angel_session"]
            )
        smart_api = SmartConnect(
            api_key=api_key
        )

        totp = pyotp.TOTP(
            totp_secret
        ).now()

        session = smart_api.generateSession(
            client_code,
            pin,
            totp
        )

        if session:
        
            st.session_state["angel_api"] = smart_api
            st.session_state["angel_session"] = session
        
            return smart_api, session
        return None, None

    except Exception as e:

        st.error(
            f"Angel Login Failed: {e}"
        )

        return None, None
def get_angel_profile(
    api_key,
    client_code,
    pin,
    totp_secret
):
    try:

        angel, session = angel_login(
            api_key,
            client_code,
            pin,
            totp_secret
        )

        if angel is None:
            return None

        profile = angel.getProfile(
            session["data"]["refreshToken"]
        )

        return profile

    except Exception as e:
    
        return {
            "status": False,
            "message": str(e)
        }
def get_angel_candles(
    api_key,
    client_code,
    pin,
    totp_secret,
    symbol_name
):
    try:

        angel, session = angel_login(
            api_key,
            client_code,
            pin,
            totp_secret
        )

        if angel is None:
            return None
        from datetime import timedelta

        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        # st.write("Using Token:", ANGEL_TOKENS.get(symbol_name, "3045"))
        params = {
            "exchange": "NSE",
            "symboltoken": ANGEL_TOKENS.get(symbol_name, "3045"),
            "interval": "FIVE_MINUTE",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        candles = angel.getCandleData(params)

        return candles

    except Exception as e:

        return str(e)        
def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
 
        payload = {
            "chat_id": chat_id,
            "text": message
        }
 
        response = requests.post(
            url,
            data=payload,
            timeout=10
        )
 
        return response.status_code == 200
 
    except Exception:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def analyze_signal(df, htf_bias=None, timeframe_label="1d", backtest_win_rate=50):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close_price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema20_slope = ema20 - float(prev["EMA20"])
    ema50_slope = ema50 - float(prev["EMA50"])
    vwap = float(latest["VWAP"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["SignalLine"])
    atr = float(latest["ATR"])
    stoch_k = float(latest["StochK"])
    stoch_d = float(latest["StochD"])
    adx = float(latest["ADX"])
    support = float(latest["PivotLow"])
    resistance = float(latest["PivotHigh"])
    distance_to_resistance = (
        (resistance - close_price) / close_price
    ) * 100

    distance_to_support = (
        (close_price - support) / close_price
    ) * 100
    trend_strength = abs(ema20 - ema50) / close_price * 100
    volume_spike = bool(latest["Volume_Spike"])
    breakout = bool(latest["Breakout"])
    breakdown = bool(latest["Breakdown"])
    body_strength = float(latest["BodyStrength"])
    hammer = bool(latest.get("Hammer", False))
    shooting_star = bool(latest.get("ShootingStar", False))
    bull_maru = bool(latest.get("BullishMarubozu", False))
    bear_maru = bool(latest.get("BearishMarubozu", False))
    doji = bool(latest.get("Doji", False))
    bull_engulf = bool(latest.get("BullishEngulfing", False))
    bear_engulf = bool(latest.get("BearishEngulfing", False))
    retest_bull = bool(latest.get("RetestBull", False))
    retest_bear = bool(latest.get("RetestBear", False))
    plus_di = float(latest["PlusDI"])
    minus_di = float(latest["MinusDI"])

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
        bullish += 2
        reasons.append("Bullish Marubozu detected")
    
    if shooting_star:
        bearish += 2
        reasons.append("Shooting Star detected")
    
    if bear_maru:
        bearish += 2
        reasons.append("Bearish Marubozu detected")    
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
    sideways = bool(
        adx < 18
        and trend_strength < 0.3
    )
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

    if not sideways and session_info["valid"]:
        if (
            bullish >= 8
            and (bullish - bearish) >= 3
            and rsi > 55
            and macd > macd_signal
            and adx > 25
        ):
            signal_text = "STRONG BUY"
            signal_type = "buy"
            strength = "High"
            stop_loss = close_price - atr
            partial_exit = close_price + (0.8 * atr)
            target1 = close_price + atr
            target2 = close_price + (2 * atr)
            trailing_stop = close_price - (0.5 * atr)

        elif (
            bearish >= 8
            and (bearish - bullish) >= 3
            and rsi < 45
            and macd < macd_signal
            and adx > 25
        ):
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

    confidence = int((max(bullish, bearish) / 15) * 100)
    confidence = min(confidence, 100)

    win_chance = calculate_win_chance(
        bullish, bearish, adx, volume_spike, breakout, breakdown,
        htf_bias, sideways, rr_ratio, backtest_win_rate
    )

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


def simple_backtest(df):
    results = []
    df = df.copy().dropna()
    
    
    
    
    for i in range(30, len(df) - 10):
        sub = df.iloc[:i + 1]
        if len(sub) < 30:
            continue

        analysis = analyze_signal(sub, htf_bias=None, timeframe_label="1d", backtest_win_rate=50)
        future = df.iloc[i + 1:i + 10]

        if future.empty:
            continue

        future_high = float(future["High"].max())
        future_low = float(future["Low"].min())

        if analysis["signal_type"] == "buy" and analysis["stop_loss"] and analysis["target1"]:
            if future_high >= analysis["target1"]:
                results.append(1)
            elif future_low <= analysis["stop_loss"]:
                results.append(0)

        elif analysis["signal_type"] == "sell" and analysis["stop_loss"] and analysis["target1"]:
            if future_low <= analysis["target1"]:
                results.append(1)
            elif future_high >= analysis["stop_loss"]:
                results.append(0)

    total = len(results)
    wins = sum(results) if total > 0 else 0
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0
    return wins, losses, total, win_rate


login_panel()
init_journal()
init_paper_state()
init_daily_summary()

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
    "TCS": "28973",
    "NIFTY": "99926000",
    "BANKNIFTY": "26009"
}
st.title("📈 Elite Trading Dashboard Ultimate")
st.caption("Journal + Entry/Exit + Paper Trading + Auto Telegram + Option Suggestion + Daily Summary")

selected = st.selectbox("Select Stock / Index", list(stocks.keys()))
data_source = st.radio(
    "Data Source",
    ["Yahoo", "Angel"],
    horizontal=True
)
st.write("Selected Symbol:", selected)
timeframe = st.selectbox("Select Time Frame", ["1m","3m", "5m", "10m", "15m", "30m", "1h", "1d"])
auto_refresh = st.checkbox("Enable Auto Refresh", value=True)
refresh_seconds = st.selectbox("Refresh Every (seconds)", [10, 20, 30, 60], index=2)

with st.expander("Telegram Alert Settings"):
    telegram_enabled = st.checkbox("Enable Telegram Alerts", value=False)
    telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID")
    auto_send_telegram = st.checkbox("Auto-send on New Signal", value=False)
with st.expander("Angel One API Settings"):
    search_symbol = st.text_input(
        "Search Angel Symbol",
        value="NIFTY"
    )
    angel_enabled = st.checkbox("Enable Angel One API", value=False)

    angel_api_key = st.text_input("Angel API Key", type="password")
    angel_client_code = st.text_input("Angel Client Code")
    angel_pin = st.text_input("Angel PIN", type="password")
    angel_totp_secret = st.text_input("Angel TOTP Secret", type="password")
if angel_enabled:

    if st.button("Test Angel Login"):

        angel, session = angel_login(
            angel_api_key,
            angel_client_code,
            angel_pin,
            angel_totp_secret
        )

        if angel:
            st.success("✅ Angel One Login Successful")
            st.success("✅ Angel Session Active")

            st.write(session)
candles = None
if "last_good_candles" not in st.session_state:
    st.session_state["last_good_candles"] = None
if data_source == "Angel":

    candles = get_angel_candles(
        angel_api_key,
        angel_client_code,
        angel_pin,
        angel_totp_secret,
        selected
    )
    
        
if data_source == "Angel":

    st.subheader("Angel Candle Test")

    st.write(
        "Angel Token:",
        ANGEL_TOKENS.get(selected, "3045")
    )

    st.write("Selected Symbol:", selected)

    if isinstance(candles, dict) and candles.get("data"):
        st.session_state["last_good_candles"] = candles
        df = pd.DataFrame(
            candles["data"],
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        st.dataframe(df.tail(10))            
# profile = get_angel_profile(
#     angel_api_key,
#     angel_client_code,
#     angel_pin,
#     angel_totp_secret
# )

# st.write(profile)            
if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, key="ultimate_refresh")

if timeframe == "1m":
    yf_interval = "1m"
    yf_period = "7d"
elif timeframe == "3m":
    yf_interval = "1m"
    yf_period = "7d"
elif timeframe == "5m":
    yf_interval = "5m"
    yf_period = "30d"
elif timeframe == "10m":
    yf_interval = "5m"
    yf_period = "30d"
elif timeframe == "15m":
    yf_interval = "15m"
    yf_period = "30d"
elif timeframe == "30m":
    yf_interval = "30m"
    yf_period = "60d"
elif timeframe == "1h":
    yf_interval = "60m"
    yf_period = "90d"
else:
    yf_interval = "1d"
    yf_period = "1y"
st.write("Current Data Source:", data_source)

ticker = stocks[selected]

if data_source == "Angel":

    if isinstance(candles, dict) and candles.get("data"):

        data = pd.DataFrame(
            candles["data"],
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        data["Datetime"] = pd.to_datetime(
            data["Datetime"]
        )

        data = data.set_index(
            "Datetime"
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    else:

        data = pd.DataFrame()

else:

    data = fetch_data(
        ticker,
        yf_interval,
        yf_period
    )
st.write("Rows Loaded:", len(data))
st.write("Data Columns:", list(data.columns))
st.write("Data Source Selected:", data_source)
 
st.dataframe(data.tail())
 
if data.empty:
    st.error("No data found")
    st.stop()
if timeframe == "3m":
    data = resample_to_3m(data)
elif timeframe == "10m":
    data = resample_to_10m(data)

data = add_indicators(data)



 
# Only drop rows where critical indicators are missing
required_cols = [
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "SignalLine",
    "ATR"
]
 
 # data = data.dropna(subset=required_cols).copy()
 
data = data.bfill()
data = data.ffill()
 

 
if len(data) < 30:
    st.warning(f"Only {len(data)} rows available after indicator calculation.")
    st.stop()

htf_interval, htf_period = get_higher_timeframe(timeframe)
if data_source == "Angel":

    htf_data = data.copy()

else:

    htf_data = fetch_data(
        ticker,
        htf_interval,
        htf_period
    )

if not htf_data.empty:
    htf_data = add_indicators(htf_data)
    htf_data = htf_data.bfill()
    htf_data = htf_data.ffill()
    if len(htf_data) > 20:
        htf_latest = htf_data.iloc[-1]
        htf_bias = "bullish" if htf_latest["EMA20"] > htf_latest["EMA50"] else "bearish"
    else:
        htf_bias = None
else:
    htf_bias = None

wins, losses, total_trades, backtest_win_rate = simple_backtest(data)
analysis = analyze_signal(
    data,
    htf_bias=htf_bias,
    timeframe_label=timeframe,
    backtest_win_rate=backtest_win_rate
)
# VWAP Debug
latest = data.iloc[-1]

st.subheader("VWAP Debug")

st.write("Close Price:", round(float(latest["Close"]), 2))
st.write("VWAP:", round(float(latest["VWAP"]), 2))

st.dataframe(
    data[["Close", "High", "Low", "Volume", "VWAP"]].tail(10),
    use_container_width=True
)
st.write("Volume Unique Values:")
st.write(data["Volume"].tail(20).unique())
st.write("First Row Time :", data.index[0])
st.write("Last Row Time :", data.index[-1])
mtf_table = pd.DataFrame({
    "Timeframe": ["1m", "3m", "5m", "15m"],
    "Trend": [
        get_mtf_signal(ticker, "1m", "7d"),
        get_mtf_signal(ticker, "3m", "7d"),
        get_mtf_signal(ticker, "5m", "30d"),
        get_mtf_signal(ticker, "15m", "30d")
    ]
})
bull_count = 0
bear_count = 0

for sig in mtf_table["Trend"]:
    if "Bullish" in sig:
        bull_count += 1
    elif "Bearish" in sig:
        bear_count += 1

if bull_count >= 3:
    mtf_grade = "🔥 Strong Bullish Alignment"

elif bear_count >= 3:
    mtf_grade = "🔥 Strong Bearish Alignment"

else:
    mtf_grade = "⚠️ Mixed Timeframes"
mtf_confirmation = "➖ Neutral"

if analysis["signal_type"] == "buy" and htf_bias == "bullish":
    mtf_confirmation = "✅ Bullish Alignment"

elif analysis["signal_type"] == "sell" and htf_bias == "bearish":
    mtf_confirmation = "✅ Bearish Alignment"

elif analysis["signal_type"] in ["buy", "sell"]:
    mtf_confirmation = "⚠️ Timeframe Conflict"
quality_score = (
    analysis["win_chance"] +
    analysis["confidence"]
) / 2

if quality_score >= 85:
    signal_quality = "🔥 Excellent"
elif quality_score >= 70:
    signal_quality = "✅ Strong"
elif quality_score >= 55:
    signal_quality = "⚠️ Moderate"
else:
    signal_quality = "❌ Weak"
latest = data.iloc[-1]

patterns = []
if latest["InvertedHammer"]:
    patterns.append("🟢 Inverted Hammer")

if latest["HangingMan"]:
    patterns.append("🔴 Hanging Man")

if latest["DragonflyDoji"]:
    patterns.append("🟢 Dragonfly Doji")

if latest["GravestoneDoji"]:
    patterns.append("🔴 Gravestone Doji")
if latest["ThreeWhiteSoldiers"]:
    patterns.append("🟢 Three White Soldiers")

if latest["ThreeBlackCrows"]:
    patterns.append("🔴 Three Black Crows")
if latest["Hammer"]:
    patterns.append("🟢 Hammer")

if latest["ShootingStar"]:
    patterns.append("🔴 Shooting Star")

if latest["BullishMarubozu"]:
    patterns.append("🟢 Bullish Marubozu")

if latest["BearishMarubozu"]:
    patterns.append("🔴 Bearish Marubozu")

if latest["Doji"]:
    patterns.append("🟡 Doji")
if latest["BullishEngulfing"]:
    patterns.append("🟢 Bullish Engulfing")

if latest["BearishEngulfing"]:
    patterns.append("🔴 Bearish Engulfing")
if latest["PiercingPattern"]:
    patterns.append("🟢 Piercing Pattern")

if latest["DarkCloudCover"]:
    patterns.append("🔴 Dark Cloud Cover")
if latest["TweezerBottom"]:
    patterns.append("🟢 Tweezer Bottom")

if latest["TweezerTop"]:
    patterns.append("🔴 Tweezer Top")
if latest["MorningStar"]:
    patterns.append("🟢 Morning Star")

if latest["EveningStar"]:
    patterns.append("🔴 Evening Star")
if latest["BullishHarami"]:
    patterns.append("🟢 Bullish Harami")

if latest["BearishHarami"]:
    patterns.append("🔴 Bearish Harami")
if bool(latest.get("RetestBull", False)):
    patterns.append("🚀 Bullish Retest")

if bool(latest.get("RetestBear", False)):
    patterns.append("🔻 Bearish Retest")
option_suggestion = suggest_option_strike(selected, analysis["close_price"], analysis["signal_type"])

alert_text = (
    f"{selected} | {timeframe} | {analysis['signal_text']} | "
    f"Strength: {analysis['strength']} | Confidence: {analysis['confidence']}% | "
    f"Win Chance: {analysis['win_chance']}% | Price: {round(analysis['close_price'], 2)} | "
    f"{option_suggestion['text']}"
)

current_signal_key = f"{selected}|{timeframe}|{analysis['signal_text']}|{round(analysis['close_price'], 2)}"

last_signal_key = ""
if os.path.exists(SIGNAL_STATE_FILE):
    with open(SIGNAL_STATE_FILE, "r") as f:
        last_signal_key = f.read().strip()

if telegram_enabled and telegram_bot_token and telegram_chat_id:
    if st.button("Send Telegram Alert Now"):
        sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_text)
        if sent:
            st.success("Telegram alert sent successfully.")
        else:
            st.error("Failed to send Telegram alert.")

    if auto_send_telegram and current_signal_key != last_signal_key and analysis["signal_type"] != "neutral":
        sent = send_telegram_alert(telegram_bot_token, telegram_chat_id, alert_text)
        if sent:
            with open(SIGNAL_STATE_FILE, "w") as f:
                f.write(current_signal_key)

st.subheader("🚨 Live Signal Analytics")
st.subheader("📊 Multi Timeframe Alignment")
st.dataframe(mtf_table, use_container_width=True)
col1, col2 = st.columns(2)

with col1:
    st.metric("Bullish TF", bull_count)

with col2:
    st.metric("Bearish TF", bear_count)
if "Strong Bullish" in mtf_grade:
    st.success(mtf_grade)

elif "Strong Bearish" in mtf_grade:
    st.error(mtf_grade)

else:
    st.warning(mtf_grade)
st.subheader("🕯 Candlestick Patterns")

if patterns:
    st.success(" | ".join(patterns))
else:
    st.info("No major candlestick pattern detected")
if analysis["signal_type"] == "buy":
    st.success(alert_text)
elif analysis["signal_type"] == "sell":
    st.error(alert_text)
else:
    st.warning(alert_text)
if analysis["signal_type"] == "buy" and bear_count >= 3:
    st.error("⚠️ BUY signal hai lekin MTF Alignment Bearish hai")

if analysis["signal_type"] == "sell" and bull_count >= 3:
    st.error("⚠️ SELL signal hai lekin MTF Alignment Bullish hai")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

with m1:
    st.metric("Current Price", round(analysis["close_price"], 2))
    st.metric("VWAP", round(analysis["vwap"], 2))

with m2:
    st.metric("EMA20", round(analysis["ema20"], 2))
    st.metric("EMA50", round(analysis["ema50"], 2))

with m3:
    st.metric("RSI", round(analysis["rsi"], 2))
    st.metric("ATR", round(analysis["atr"], 2))

with m4:
    st.metric("MACD", round(analysis["macd"], 2))
    st.metric("ADX", round(analysis["adx"], 2))

with m5:
    st.metric("Win Chance %", analysis["win_chance"])
    st.metric("Confidence %", analysis["confidence"])

with m6:
    st.metric("Bullish Score", analysis["bullish_score"])
    st.metric("Bearish Score", analysis["bearish_score"])
with m7:
    st.metric("Signal Quality", signal_quality)

a1, a2, a3, a4, a5,a6 = st.columns(6)

with a1:
    st.metric("Support", round(analysis["support"], 2))
with a2:
    st.metric("Resistance", round(analysis["resistance"], 2))
with a3:
    st.metric("Backtest Wins", wins)
with a4:
    st.metric("Backtest Losses", losses)
with a5:
    st.metric("Backtest Win Rate", f"{backtest_win_rate:.1f}%")
with a6:
    st.metric("MTF Status", mtf_confirmation)
st.subheader("🎯 Option CALL/PUT Suggestion")
if option_suggestion["type"] == "CALL":
    st.success(option_suggestion["text"])
elif option_suggestion["type"] == "PUT":
    st.error(option_suggestion["text"])
else:
    st.warning(option_suggestion["text"])

if analysis["stop_loss"] is not None:
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.metric("Stop Loss", round(analysis["stop_loss"], 2))
    with t2:
        st.metric("Partial Exit", round(analysis["partial_exit"], 2) if analysis["partial_exit"] else "-")
    with t3:
        st.metric("Target 1", round(analysis["target1"], 2) if analysis["target1"] else "-")
    with t4:
        st.metric("Target 2", round(analysis["target2"], 2) if analysis["target2"] else "-")
    with t5:
        st.metric("Trailing Stop", round(analysis["trailing_stop"], 2) if analysis["trailing_stop"] else "-")

with st.expander("Why this signal?"):
    for reason in analysis["reasons"]:
        st.write("-", reason)

st.subheader("💼 Paper Trading")
paper_state = load_paper_state()

p1, p2, p3 = st.columns(3)
with p1:
    st.metric("Paper Balance", round(float(paper_state.loc[0, "Balance"]), 2))
with p2:
    st.metric("Used Margin", round(float(paper_state.loc[0, "UsedMargin"]), 2))
with p3:
    st.metric("Open Positions", int(paper_state.loc[0, "OpenPositions"]))

paper_qty = st.number_input("Paper Trade Quantity", min_value=1, value=1, step=1)

if st.button("Take Paper Trade"):
    if analysis["signal_type"] in ["buy", "sell"]:
        success = paper_trade_entry(analysis["close_price"], paper_qty)
        if success:
            add_trade_to_journal(
                symbol=selected,
                timeframe=timeframe,
                signal=analysis["signal_text"],
                entry_price=analysis["close_price"],
                quantity=paper_qty,
                stop_loss=analysis["stop_loss"],
                target1=analysis["target1"],
                target2=analysis["target2"],
                trailing_stop=analysis["trailing_stop"],
                notes="Paper Trade"
            )
            st.success("Paper trade entered and journal updated.")
            st.rerun()
        else:
            st.error("Insufficient paper balance.")
    else:
        st.warning("No valid signal for paper trade.")

st.subheader("Candlestick Chart")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candles"
))

fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], mode="lines", name="EMA20", line=dict(color="orange", width=1.5)))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], mode="lines", name="EMA50", line=dict(color="blue", width=1.5)))
fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP", line=dict(color="purple", width=1.3, dash="dot")))
fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="gray", width=1, dash="dash")))
fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower", line=dict(color="gray", width=1, dash="dash")))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["BuyMarker"],
    mode="markers",
    name="Buy Signal",
    marker=dict(symbol="triangle-up", color="green", size=12)
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["SellMarker"],
    mode="markers",
    name="Sell Signal",
    marker=dict(symbol="triangle-down", color="red", size=12)
))

fig.add_hline(y=analysis["support"], line_dash="dot", line_color="green", annotation_text="Support")
fig.add_hline(y=analysis["resistance"], line_dash="dot", line_color="red", annotation_text="Resistance")

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    title=f"{selected} Candlestick Chart ({timeframe})"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("MACD Chart")
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD", line=dict(color="cyan")))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["SignalLine"], mode="lines", name="Signal", line=dict(color="orange")))
macd_fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram"))
macd_fig.update_layout(height=300, template="plotly_dark", title="MACD")
st.plotly_chart(macd_fig, use_container_width=True)

st.subheader("RSI / Stochastic / ADX Chart")
multi_fig = go.Figure()
multi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color="yellow")))
multi_fig.add_trace(go.Scatter(x=data.index, y=data["StochK"], mode="lines", name="Stoch K", line=dict(color="lime")))
multi_fig.add_trace(go.Scatter(x=data.index, y=data["StochD"], mode="lines", name="Stoch D", line=dict(color="red")))
multi_fig.add_trace(go.Scatter(x=data.index, y=data["ADX"], mode="lines", name="ADX", line=dict(color="white")))
multi_fig.update_layout(height=350, template="plotly_dark", title="Momentum + Trend Strength")
st.plotly_chart(multi_fig, use_container_width=True)

st.subheader("📝 Trade Journal")

col_j1, col_j2 = st.columns(2)
with col_j1:
    qty = st.number_input("Manual Journal Quantity", min_value=1, value=1, step=1, key="journal_qty")
    note_text = st.text_input("Trade Notes", value="", key="journal_note")
    if st.button("Add Current Signal to Journal"):
        if analysis["signal_type"] in ["buy", "sell"]:
            add_trade_to_journal(
                symbol=selected,
                timeframe=timeframe,
                signal=analysis["signal_text"],
                entry_price=analysis["close_price"],
                quantity=qty,
                stop_loss=analysis["stop_loss"],
                target1=analysis["target1"],
                target2=analysis["target2"],
                trailing_stop=analysis["trailing_stop"],
                notes=note_text
            )
            st.success("Trade added to journal.")
            st.rerun()
        else:
            st.warning("No valid trade signal to add.")

journal_df = load_journal()
st.dataframe(journal_df, use_container_width=True)

st.subheader("📌 Entry / Exit History")

open_trades = journal_df[journal_df["Status"] == "OPEN"]

if not open_trades.empty:
    st.write("Open Trades")
    st.dataframe(open_trades, use_container_width=True)

    open_indices = open_trades.index.tolist()
    selected_trade_index = st.selectbox("Select Open Trade Row to Exit", open_indices)

    exit_price_input = st.number_input("Exit Price", min_value=0.0, value=float(analysis["close_price"]), step=0.1)

    if st.button("Close Selected Trade"):
        row = journal_df.loc[selected_trade_index]
        pnl = paper_trade_exit(
            entry_price=float(row["EntryPrice"]),
            exit_price=exit_price_input,
            quantity=float(row["Quantity"]),
            signal=str(row["Signal"])
        )
        close_trade_in_journal(selected_trade_index, exit_price_input)
        update_daily_summary()
        st.success(f"Trade closed. Paper PnL: {round(pnl, 2)}")
        st.rerun()
else:
    st.info("No open trades available.")

closed_trades = journal_df[journal_df["Status"] == "CLOSED"]
if not closed_trades.empty:
    st.write("Closed Trades")
    st.dataframe(closed_trades, use_container_width=True)

    total_pnl = pd.to_numeric(closed_trades["PnL"], errors="coerce").fillna(0).sum()
    avg_pnl_pct = pd.to_numeric(closed_trades["PnLPercent"], errors="coerce").fillna(0).mean()

    p1, p2 = st.columns(2)
    with p1:
        st.metric("Total Realized PnL", round(total_pnl, 2))
    with p2:
        st.metric("Average PnL %", round(avg_pnl_pct, 2))

st.subheader("📊 Daily Summary Report")
init_daily_summary()
summary_df = pd.read_csv(SESSION_SUMMARY_FILE)
st.dataframe(summary_df, use_container_width=True)

st.subheader("Downloads")

market_csv = data.to_csv().encode("utf-8")
st.download_button(
    label="📥 Download Market Data CSV",
    data=market_csv,
    file_name=f"{selected}_{timeframe}_market_data.csv",
    mime="text/csv"
)

journal_csv = journal_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Trade Journal CSV",
    data=journal_csv,
    file_name="trade_journal.csv",
    mime="text/csv"
)

summary_csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Daily Summary CSV",
    data=summary_csv,
    file_name="daily_summary.csv",
    mime="text/csv"
)
