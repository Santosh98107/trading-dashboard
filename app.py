import os, hashlib, time, requests
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Production signal gates requested for this build.
MANDATORY_ADX = 20
MANDATORY_MTF_ALIGN_COUNT = 2
STRICT_TRADE_ONLY_STRONG = False
DISABLE_BUY_BIAS_TRADING = False
DISABLE_SELL_BIAS_TRADING = False

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(
    page_title='Elite Trading Dashboard Pro',
    page_icon='📈',
    layout='wide'
)

def sha256(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def login():
    if st.session_state.get('authenticated'):
        return True

    st.title('🔐 Trading Intelligence Terminal')
    st.subheader('Application Login')

    user_expected = os.getenv('APP_USERNAME', 'admin')
    hash_expected = os.getenv(
        'APP_PASSWORD_SHA256',
        sha256('admin123')
    )

    with st.form('login_form'):
        user = st.text_input('Username')
        password = st.text_input(
            'Password',
            type='password'
        )
        submit = st.form_submit_button('Login')

    if submit:
        if (
            user == user_expected
            and sha256(password) == hash_expected
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error('Username or password is incorrect.')

    st.caption(
        'Testing default: admin / admin123. '
        'Change before public deployment.'
    )
    return False


if not login():
    st.stop()

st.title('📈 Elite Trading Dashboard Pro')

st.caption(
    'Technical analysis • Global markets • PCR/OI context '
    '• Backtesting • Paper trading'
)

st.warning(
    'Research and paper-trading tool only. '
    'Yahoo Finance data may be delayed or incomplete. '
    'No live broker orders are placed. '
    'Scores are not guaranteed win probabilities.'
)

for k, v in {
    'paper_trades': [],
    'paper_position': None,
    'angel_candles': None,
    'signal_audit': [],
    'weekly_lock': False,
    'monthly_lock': False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


SYMBOLS = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'NIFTY IT': '^CNXIT',
    'NIFTY AUTO': '^CNXAUTO',
    'RELIANCE': 'RELIANCE.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'SBI': 'SBIN.NS',
    'AXIS BANK': 'AXISBANK.NS',
    'KOTAK BANK': 'KOTAKBANK.NS',
    'TCS': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'HCLTECH': 'HCLTECH.NS',
    'WIPRO': 'WIPRO.NS',
    'TECH MAHINDRA': 'TECHM.NS',
    'MARUTI': 'MARUTI.NS',
    'M&M': 'M&M.NS',
    'TATA MOTORS': 'TMPV.NS',
    'BAJAJ AUTO': 'BAJAJ-AUTO.NS',
    'EICHER MOTORS': 'EICHERMOT.NS',
    'RELIANCE ENERGY CONTEXT': 'RELIANCE.NS'
}

GLOBAL = {
    'US S&P 500': '^GSPC',
    'US Nasdaq': '^IXIC',
    'US Dow Jones': '^DJI',
    'Japan Nikkei': '^N225',
    'Hong Kong Hang Seng': '^HSI',
    'China Shanghai': '000001.SS',
    'South Korea KOSPI': '^KS11',
    'Brent Crude': 'BZ=F',
    'USD/INR': 'INR=X',
    'India VIX': '^INDIAVIX'
}

ANGEL_MASTER_URL = (
    "https://margincalculator.angelone.in/"
    "OpenAPI_File/files/OpenAPIScripMaster.json"
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_angel_master():
    """Download Angel One's public instrument master and cache it for one hour."""

    response = requests.get(
        ANGEL_MASTER_URL,
        timeout=25
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list) or not payload:
        raise RuntimeError(
            "Angel One instrument master returned an unexpected format."
        )

    frame = pd.DataFrame(payload)

    required = {"symbol", "token", "exch_seg"}

    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Instrument master is missing symbol/token/exchange fields."
        )

    for col in (
        "symbol",
        "name",
        "exch_seg",
        "instrumenttype",
        "token"
    ):
        if col not in frame.columns:
            frame[col] = ""

        frame[col] = frame[col].fillna("").astype(str)

    frame["_sym_norm"] = (
        frame["symbol"]
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    frame["_name_norm"] = (
        frame["name"]
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    frame["_exchange_norm"] = (
        frame["exch_seg"].str.upper()
    )

    return frame


ANGEL_ALIASES = {
    "NIFTY 50": ["NIFTY 50", "NIFTY50"],
    "BANK NIFTY": [
        "NIFTY BANK",
        "BANKNIFTY",
        "NIFTYBANK"
    ],
    "SENSEX": ["SENSEX"],
    "NIFTY IT": ["NIFTY IT", "NIFTYIT"],
    "NIFTY AUTO": ["NIFTY AUTO", "NIFTYAUTO"],
    "RELIANCE": ["RELIANCE-EQ", "RELIANCE"],
    "HDFC BANK": ["HDFCBANK-EQ", "HDFCBANK"],
    "ICICI BANK": ["ICICIBANK-EQ", "ICICIBANK"],
    "SBI": ["SBIN-EQ", "SBIN"],
    "AXIS BANK": ["AXISBANK-EQ", "AXISBANK"],
    "KOTAK BANK": ["KOTAKBANK-EQ", "KOTAKBANK"],
    "TCS": ["TCS-EQ", "TCS"],
    "INFOSYS": ["INFY-EQ", "INFY"],
    "HCLTECH": ["HCLTECH-EQ", "HCLTECH"],
    "WIPRO": ["WIPRO-EQ", "WIPRO"],
    "TECH MAHINDRA": ["TECHM-EQ", "TECHM"],
    "MARUTI": ["MARUTI-EQ", "MARUTI"],
    "M&M": ["M&M-EQ", "MM-EQ", "M&M"],
    "TATA MOTORS": [
        "TMPV-EQ",
        "TATAMOTORS-EQ",
        "TMPV",
        "TATAMOTORS"
    ],
    "BAJAJ AUTO": [
        "BAJAJ-AUTO-EQ",
        "BAJAJAUTO-EQ",
        "BAJAJ-AUTO"
    ],
    "EICHER MOTORS": [
        "EICHERMOT-EQ",
        "EICHERMOT"
    ],
}
def resolve_angel_instrument(master, display_name, custom_query=""):
    """Resolve a selected display name to its exchange, trading symbol and token."""
    if master is None or master.empty:
        raise RuntimeError("Angel One instrument master is empty.")

    if display_name == "Custom Yahoo ticker":
        query = custom_query.strip().upper()
        if not query:
            raise RuntimeError(
                "Enter an instrument name/symbol to search; token is resolved automatically."
            )
        candidates = [query, query + "-EQ"]
        preferred_exchange = "NSE"
    else:
        candidates = ANGEL_ALIASES.get(display_name, [display_name])
        preferred_exchange = "BSE" if display_name == "SENSEX" else "NSE"

    norm_candidates = [
        "".join(ch for ch in c.upper() if ch.isalnum())
        for c in candidates
    ]

    subset = master[
        master["_exchange_norm"].isin([
            preferred_exchange,
            "BSE" if preferred_exchange == "NSE" else "NSE"
        ])
    ].copy()

    for candidate in norm_candidates:
        hits = subset[subset["_sym_norm"] == candidate]

        if not hits.empty:
            hits = hits.copy()
            hits["_exchange_rank"] = (
                hits["_exchange_norm"] != preferred_exchange
            ).astype(int)

            hits["_derivative_rank"] = (
                hits["_sym_norm"].str.contains(
                    r"FUT|CE|PE",
                    regex=True
                )
            ).astype(int)

            hits = hits.sort_values([
                "_exchange_rank",
                "_derivative_rank"
            ])

            row = hits.iloc[0]

            return {
                "exchange": row["exch_seg"].upper(),
                "symbol": row["symbol"],
                "token": row["token"],
                "name": row["name"] or row["symbol"]
            }

    for candidate in norm_candidates:
        hits = subset[
            (subset["_name_norm"] == candidate)
            | (subset["_sym_norm"] == candidate)
        ]

        if not hits.empty:
            hits = hits.copy()
            hits["_exchange_rank"] = (
                hits["_exchange_norm"] != preferred_exchange
            ).astype(int)

            row = hits.sort_values("_exchange_rank").iloc[0]

            return {
                "exchange": row["exch_seg"].upper(),
                "symbol": row["symbol"],
                "token": row["token"],
                "name": row["name"] or row["symbol"]
            }

    raise RuntimeError(
        f"'{display_name if display_name != 'Custom Yahoo ticker' else custom_query}' "
        "was not found in Angel One's instrument master. "
        "Try another listed instrument or refresh the master."
    )


def fetch_angel_candles(api, instrument, interval, lookback_days):
    now = datetime.now()
    start = now - pd.Timedelta(days=int(lookback_days))

    response = api.getCandleData({
        "exchange": instrument["exchange"],
        "symboltoken": str(instrument["token"]),
        "interval": interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    })

    if (
        not response
        or response.get("status") is False
        or not response.get("data")
    ):
        message = (
            (response or {}).get("message")
            or (response or {}).get("errorcode")
            or "No candles returned"
        )

        raise RuntimeError(
            f"Angel One candle request failed: {message}. "
            "Check API permissions, market hours and selected interval."
        )

    cdf = pd.DataFrame(
        response["data"],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    cdf["timestamp"] = pd.to_datetime(
        cdf["timestamp"],
        errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        cdf[col] = pd.to_numeric(cdf[col], errors="coerce")

    return (
        cdf.dropna(
            subset=["timestamp", "open", "high", "low", "close"]
        )
        .set_index("timestamp")
        .sort_index()
    )


def norm(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()

    x = raw.copy()

    if isinstance(x
    def patterns(d):
    """Identify common Japanese candlestick patterns."""
    if len(d) < 3:
        return ['Insufficient candles']

    a, b, c = d.iloc[-1], d.iloc[-2], d.iloc[-3]

    body = abs(a.close - a.open)
    rng = max(a.high - a.low, 1e-9)
    upper = a.high - max(a.open, a.close)
    lower = min(a.open, a.close) - a.low
    out = []

    # Two-candle patterns
    if a.close > a.open and b.close < b.open and a.close >= b.open and a.open <= b.close:
        out.append('Bullish Engulfing')

    if a.close < a.open and b.close > b.open and a.open >= b.close and a.close <= b.open:
        out.append('Bearish Engulfing')

    if b.close < b.open and a.close > a.open and a.open <= b.close and a.close >= (b.open+b.close)/2:
        out.append('Piercing Pattern')

    if b.close > b.open and a.close < a.open and a.open >= b.close and a.close <= (b.open+b.close)/2:
        out.append('Dark Cloud Cover')

    if b.close < b.open and a.close > a.open and a.open >= b.close and a.close <= b.open:
        out.append('Bullish Harami')

    if b.close > b.open and a.close < a.open and a.open <= b.close and a.close >= b.open:
        out.append('Bearish Harami')

    # Single-candle patterns
    if body/rng <= .1:
        out.append('Doji')

    if .1 < body/rng <= .3 and upper > body and lower > body:
        out.append('Spinning Top')

    if body/rng >= .9 and a.close > a.open:
        out.append('Bullish Marubozu')

    if body/rng >= .9 and a.close < a.open:
        out.append('Bearish Marubozu')

    if lower >= 2*max(body, 1e-9) and upper <= max(body, 1e-9)*1.2:
        out.append('Hammer' if a.close > a.open else 'Hanging Man')

    if upper >= 2*max(body, 1e-9) and lower <= max(body, 1e-9)*1.2:
        out.append('Inverted Hammer' if a.close > a.open else 'Shooting Star')

    tol = max(rng*.10, 1e-8)

    if abs(a.low-b.low) <= tol and b.close < b.open and a.close > a.open:
        out.append('Tweezer Bottom (possible)')

    if abs(a.high-b.high) <= tol and b.close > b.open and a.close < a.open:
        out.append('Tweezer Top (possible)')

    # Three-candle patterns
    if c.close < c.open and abs(b.close-b.open) < abs(c.close-c.open)*.6 and a.close > a.open and a.close > (c.open+c.close)/2:
        out.append('Morning Star (possible)')

    if c.close > c.open and abs(b.close-b.open) < abs(c.close-c.open)*.6 and a.close < a.open and a.close < (c.open+c.close)/2:
        out.append('Evening Star (possible)')

    if c.close < c.open and b.close > b.open and a.close > a.open and b.close > c.close and a.close > b.close:
        out.append('Three White Soldiers (possible)')

    if c.close > c.open and b.close < b.open and a.close < a.open and b.close < c.close and a.close < b.close:
        out.append('Three Black Crows (possible)')

    if a.high < b.high and a.low > b.low:
        out.append('Inside Bar')

    if a.high > b.high and a.low < b.low:
        out.append('Outside Bar')

    return out or ['No major pattern detected']


def safe(v):
    try:
        return float(v) if pd.notna(v) and np.isfinite(float(v)) else np.nan
    except Exception:
        return np.nan


def fmt(v, n=2):
    return 'N/A' if not np.isfinite(safe(v)) else f'{safe(v):,.{n}f}'


def get_signal(r):
    bull = bear = 0
    reasons = []

    if r.close > r.ema20:
        bull += 1
        reasons.append('Price > EMA20')
    elif r.close < r.ema20:
        bear += 1
        reasons.append('Price < EMA20')

    if r.ema20 > r.ema50:
        bull += 1
    elif r.ema20 < r.ema50:
        bear += 1

    if pd.notna(r.get('ema200', np.nan)):
        if r.close > r.ema200:
            bull += 1
            reasons.append('Price above EMA200')
        elif r.close < r.ema200:
            bear += 1
            reasons.append('Price below EMA200')

    if pd.notna(r.get('supertrend_direction', np.nan)):
        if r.supertrend_direction > 0:
            bull += 1
            reasons.append('SuperTrend bullish')
        elif r.supertrend_direction < 0:
            bear += 1
            reasons.append('SuperTrend bearish')

    if pd.notna(r.get('ha_close', np.nan)) and pd.notna(r.get('ha_open', np.nan)):
        if r.ha_close > r.ha_open:
            bull += 1
            reasons.append('Heikin-Ashi bullish')
        elif r.ha_close < r.ha_open:
            bear += 1
            reasons.append('Heikin-Ashi bearish')

    if pd.notna(r.vwap):
        if r.close > r.vwap:
            bull += 1
        elif r.close < r.vwap:
            bear += 1

    if pd.notna(r.rsi):
        if 52 <= r.rsi <= 70:
            bull += 1
        elif 30 <= r.rsi <= 48:
            bear += 1

    if pd.notna(r.adx) and r.adx >= MANDATORY_ADX:
        if r.di_plus > r.di_minus:
            bull += 1
        elif r.di_minus > r.di_plus:
            bear += 1

    if pd.notna(r.vol_ratio) and r.vol_ratio >= 1.2:
        if r.close > r.open:
            bull += 1
        elif r.close < r.open:
            bear += 1

    if pd.notna(r.prior_high20) and r.close > r.prior_high20:
        bull += 2
        reasons.append('20-candle breakout')

    if pd.notna(r.prior_low20) and r.close < r.prior_low20:
        bear += 2
        reasons.append('20-candle breakdown')

    if r.get('bull_pattern_score', 0) > 0:
        bull += 1
        reasons.append('Bullish candle pattern')

    if r.get('bear_pattern_score', 0) > 0:
        bear += 1
        reasons.append('Bearish candle pattern')

    if pd.notna(r.get('macd', np.nan)) and pd.notna(r.get('macd_signal', np.nan)):
        if r.macd > r.macd_signal and r.macd_hist > 0:
            bull += 1
            reasons.append('MACD bullish')
        elif r.macd < r.macd_signal and r.macd_hist < 0:
            bear += 1
            reasons.append('MACD bearish')

    if pd.notna(r.get('bb_mid', np.nan)):
        if r.close > r.bb_mid and r.close < r.bb_upper:
            bull += 1
            reasons.append('Bollinger midline bullish')
        elif r.close < r.bb_mid and r.close > r.bb_lower:
            bear += 1
            reasons.append('Bollinger midline bearish')
        elif r.close >= r.bb_upper and r.close > r.open:
            bull += 1
            reasons.append('Upper-band momentum')
        elif r.close <= r.bb_lower and r.close < r.open:
            bear += 1
            reasons.append('Lower-band momentum')

    sig = (
        'BUY' if bull >= 4 and bull >= bear+2
        else 'SELL' if bear >= 4 and bear >= bull+2
        else 'WAIT'
    )

    return sig, bull, bear, reasons


def pct(s):
    if len(s) < 2:
        return np.nan

    p = safe(s.iloc[-2])
    q = safe(s.iloc[-1])

    return (q/p-1)*100 if np.isfinite(p) and p else np.nan


def daily_info(ticker):
    d = yahoo(ticker, '2y', '1d')

    if d.empty:
        return d, {}

    d = indicators(d)
    prev = d.iloc[-2] if len(d) > 1 else d.iloc[-1]
    last = d.iloc[-1]

    lev = {
        'Previous Close': safe(prev.close),
        'Previous High': safe(prev.high),
        'Previous Low': safe(prev.low)
    }

    for n in [20, 50, 100, 200]:
        lev[f'{n} DMA'] = safe(last[f'sma{n}'])

    return d, lev


def mtf(ticker):
    out = []
    cache = {}

    for label, interval in [
        ('1m', '1m'),
        ('3m', '1m'),
        ('5m', '5m'),
        ('15m', '15m')
    ]:
        if interval not in cache:
            cache[interval] = yahoo(ticker, '5d', interval)

        x = cache[interval].copy()

        if label == '3m' and not x.empty:
            x = resample(x, '3min')

        if len(x) < 30:
            out.append({
                'Timeframe': label,
                'Trend': 'N/A',
                'Signal': 'N/A',
                'RSI': np.nan,
                'ADX': np.nan,
                'Last candle': 'N/A'
            })
            continue

        z = indicators(x)
        r = z.iloc[-1]
        sig, _, _, _ = get_signal(r)

        out.append({
            'Timeframe': label,
            'Trend': 'Bullish' if r.ema20 > r.ema50 else 'Bearish' if r.ema20 < r.ema50 else 'Sideways',
            'Signal': sig,
            'RSI': safe(r.rsi),
            'ADX': safe(r.adx),
            'Last candle': str(z.index[-1])
        })

    return pd.DataFrame(out)


# Market dashboards and scanner watchlists
global_indices = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Gift Nifty": "NIFTY=F",
    "USD/INR": "INR=X",
    "Brent Crude": "BZ=F",
    "Gold": "GC=F",
    "India VIX": "^INDIAVIX"
}

indian_indices = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO"
}

major_stocks = {
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Tata Motors": "TMPV.NS",
    "Maruti": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Eicher Motors": "EICHERMOT.NS"
}


def fetch_market_snapshot(mapping, period="5d", interval="1d"):
    rows = []

    for name, symbol in mapping.items():
        frame = yahoo(symbol, period, interval)

        if frame.empty:
            rows.append({
                "Instrument": name,
                "Ticker": symbol,
                "CMP": np.nan,
                "Change %": np.nan,
                "Trend": "Unavailable",
                "Last Update": "N/A"
            })
            continue

        cmp = safe(frame.close.iloc[-1])
        prev = safe(frame.close.iloc[-2]) if len(frame) > 1 else np.nan

        change = (
            (cmp/prev-1)*100
            if np.isfinite(cmp) and np.isfinite(prev) and prev
            else np.nan
        )

        rows.append({
            "Instrument": name,
            "Ticker": symbol,
            "CMP": cmp,
            "Change %": change,
            "Trend": (
                "Bullish" if np.isfinite(change) and change > 0
                else "Bearish" if np.isfinite(change) and change < 0
                else "Flat" if np.isfinite(change)
                else "Unavailable"
            ),
            "Last Update": str(frame.index[-1])
        })

    return pd.DataFrame(rows)


def get_intraday_trend_label(frame):
    if frame is None or frame.empty or len(frame) < 25:
        return "N/A"

    z = indicators(norm(frame))
    r = z.iloc[-1]

    if safe(r.get('ema20')) > safe(r.get('ema50')) and safe(r.get('close')) > safe(r.get('vwap')):
        return "Bullish"

    if safe(r.get('ema20')) < safe(r.get('ema50')) and safe(r.get('close')) < safe(r.get('vwap')):
        return "Bearish"

    return "Sideways / Mixed"


def safe_fetch_intraday_trend(ticker, period="5d", interval="5m"):
    try:
        frame = yahoo(ticker, period, interval)
        return get_intraday_trend_label(frame)
    except Exception:
        return "Unavailable"


def summarize_fno_bias(scanner):
    if scanner is None or scanner.empty or "Change %" not in scanner:
        return {
            "Bias": "Unavailable",
            "Bullish": 0,
            "Bearish": 0,
            "Coverage": 0
        }

    values = pd.to_numeric(
        scanner["Change %"],
        errors="coerce"
    ).dropna()

    bull = int((values > 0).sum())
    bear = int((values < 0).sum())

    bias = (
        "Bullish" if bull > bear
        else "Bearish" if bear > bull
        else "Mixed"
    )

    return {
        "Bias": bias,
        "Bullish": bull,
        "Bearish": bear,
        "Coverage": int(len(values))
    }


def fetch_fno_style_scanner(mapping=None):
    mapping = mapping or major_stocks
    rows = []

    for name, symbol in mapping.items():
        frame = yahoo(symbol, "5d", "5m")

        if frame.empty:
            frame = yahoo(symbol, "5d", "1d")

        if frame.empty:
            rows.append({
                "Instrument": name,
                "CMP": np.nan,
                "Change %": np.nan,
                "Intraday Trend": "Unavailable",
                "Scanner Bias": "N/A"
            })
            continue

        cmp = safe(frame.close.iloc[-1])
        prev = safe(frame.close.iloc[-2]) if len(frame) > 1 else np.nan

        change = (
            (cmp/prev-1)*100
            if np.isfinite(cmp) and np.isfinite(prev) and prev
            else np.nan
        )

        trend = get_intraday_trend_label(frame)

        rows.append({
            "Instrument": name,
            "CMP": cmp,
            "Change %": change,
            "Intraday Trend": trend,
            "Scanner Bias": (
                "BULLISH" if trend == "Bullish" and np.isfinite(change) and change > 0
                else "BEARISH" if trend == "Bearish" and np.isfinite(change) and change < 0
                else "WAIT"
            )
        })

    return pd.DataFrame(rows).sort_values(
        "Change %",
        ascending=False,
        na_position="last"
    )
    # ============================================================
# PART 4: PCR, OI BIAS, NEWS CONTEXT, VALIDATION & STRATEGY
# ============================================================

def calculate_pcr(option_chain):
    """
    Calculate Put-Call Ratio from an option-chain DataFrame.

    Expected columns:
    CE_OI, PE_OI, CE_OI_CHANGE, PE_OI_CHANGE
    """

    result = {
        "PCR": np.nan,
        "PCR Change": np.nan,
        "Call OI": np.nan,
        "Put OI": np.nan,
        "Call OI Change": np.nan,
        "Put OI Change": np.nan,
        "OI Bias": "Unavailable"
    }

    if option_chain is None or option_chain.empty:
        return result

    df = option_chain.copy()
    df.columns = [str(c).upper().replace(" ", "_") for c in df.columns]

    required = ["CE_OI", "PE_OI"]

    if not all(c in df.columns for c in required):
        return result

    for col in [
        "CE_OI",
        "PE_OI",
        "CE_OI_CHANGE",
        "PE_OI_CHANGE"
    ]:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

    call_oi = df["CE_OI"].sum(min_count=1)
    put_oi = df["PE_OI"].sum(min_count=1)

    call_change = df["CE_OI_CHANGE"].sum(min_count=1)
    put_change = df["PE_OI_CHANGE"].sum(min_count=1)

    pcr = (
        put_oi / call_oi
        if pd.notna(call_oi) and call_oi > 0
        else np.nan
    )

    pcr_change = (
        put_change / call_change
        if pd.notna(call_change) and call_change > 0
        else np.nan
    )

    if pd.notna(put_change) and pd.notna(call_change):
        if put_change > 0 and call_change < 0:
            bias = "Bullish"
        elif call_change > 0 and put_change < 0:
            bias = "Bearish"
        elif put_change > call_change:
            bias = "Positive OI Bias"
        elif call_change > put_change:
            bias = "Negative OI Bias"
        else:
            bias = "Mixed"
    else:
        bias = "Change data unavailable"

    result.update({
        "PCR": pcr,
        "PCR Change": pcr_change,
        "Call OI": call_oi,
        "Put OI": put_oi,
        "Call OI Change": call_change,
        "Put OI Change": put_change,
        "OI Bias": bias
    })

    return result


def summarize_oi_bias(call_oi_change, put_oi_change):
    call_change = safe(call_oi_change)
    put_change = safe(put_oi_change)

    if not np.isfinite(call_change) or not np.isfinite(put_change):
        return "Unavailable"

    if put_change > 0 and call_change < 0:
        return "Bullish"

    if call_change > 0 and put_change < 0:
        return "Bearish"

    if put_change > call_change:
        return "Positive OI Bias"

    if call_change > put_change:
        return "Negative OI Bias"

    return "Mixed"


def get_option_chain_snapshot(api, exchange, symbol_token, strike_count=10):
    """
    Fetch an option-chain snapshot using an available broker API method.

    This function deliberately does not invent
    # ============================================================
# PART 5: PAPER TRADING, CAPITAL PROTECTION, JOURNAL ANALYTICS
# ============================================================

INITIAL_CAPITAL = 100000.0


def initialize_paper_account():
    defaults = {
        "paper_cash": INITIAL_CAPITAL,
        "paper_realized": 0.0,
        "paper_position": None,
        "paper_trades": [],
        "trade_journal": [],
        "weekly_lock": False,
        "monthly_lock": False,
        "capital_lock": False,
        "peak_equity": INITIAL_CAPITAL,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def paper_equity(current_price=None):
    cash = safe(st.session_state.get("paper_cash", INITIAL_CAPITAL))

    if not np.isfinite(cash):
        cash = INITIAL_CAPITAL

    position = st.session_state.get("paper_position")

    if not position:
        return cash

    price = safe(current_price)

    if not np.isfinite(price):
        price = safe(position.get("entry", 0))

    quantity = safe(position.get("quantity", 0))
    entry = safe(position.get("entry", 0))

    if not all(np.isfinite(v) for v in [price, quantity, entry]):
        return cash

    unrealized = (
        (price - entry) * quantity
        if position.get("side") == "BUY"
        else (entry - price) * quantity
    )

    return cash + unrealized


def capital_protection(current_equity):
    """
    Paper-account protection checks.
    Does not connect to or control a broker account.
    """

    equity = safe(current_equity)

    if not np.isfinite(equity) or equity <= 0:
        st.session_state.capital_lock = True
        return {
            "locked": True,
            "reason": "Invalid paper equity",
            "drawdown_pct": np.nan
        }

    peak = max(
        safe(st.session_state.get("peak_equity", INITIAL_CAPITAL)),
        equity
    )

    st.session_state.peak_equity = peak

    drawdown = (
        (peak - equity) / peak * 100
        if peak > 0
        else 0
    )

    locked = drawdown >= 10

    if locked:
        st.session_state.capital_lock = True

    return {
        "locked": locked,
        "reason": (
            "Paper capital drawdown limit reached"
            if locked
            else "Within paper drawdown limit"
        ),
        "drawdown_pct": drawdown
    }


def period
# ============================================================
# PART 6: DATA SOURCE, ANGEL ONE LOGIN, LIVE DATA & MTF
# ============================================================

# Optional Angel One dependencies
try:
    from SmartApi import SmartConnect
    ANGEL_SDK_AVAILABLE = True
except ImportError:
    SmartConnect = None
    ANGEL_SDK_AVAILABLE = False

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    pyotp = None
    PYOTP_AVAILABLE = False


def angel_login(api_key, client_code, pin, totp_secret):
    """Create an authenticated Angel One session."""

    if not ANGEL_SDK_AVAILABLE:
        raise RuntimeError(
            "SmartAPI SDK missing. Install smartapi-python."
        )

    if not PYOTP_AVAILABLE:
        raise RuntimeError(
            "pyotp missing. Install pyotp."
        )

    if not all([api_key, client_code, pin, totp_secret]):
        raise RuntimeError(
            "Enter API key, client code, PIN and TOTP secret."
        )

    api = SmartConnect(api_key=api_key)

    otp = pyotp.TOTP(
        totp_secret.replace(" ", "")
    ).now()

    session = api.generateSession(
        client_code,
        pin,
        otp
    )

    if not isinstance(session, dict):
        raise RuntimeError(
            "Unexpected Angel One login response."
        )

    if session.get("status") is False:
        raise RuntimeError(
            session.get("message", "Angel One login failed.")
        )

    data = session.get("data") or {}

    if not data.get("jwtToken"):
        raise RuntimeError(
            "Angel One did not return a JWT token."
        )

    return api, session


def angel_status_check(api):
    if api is None:
        return "Not connected"

    return "Connected session available"


def load_selected_data(
    source,
    ticker,
    selected_name,
    interval,
    period,
    angel_api=None,
    angel_master=None,
    custom_query=""
):
    """
    Load candles from the selected data source.

    Yahoo data and Angel One data are kept separate.
    No fabricated price fallback is used.
    

    if source == "Yahoo Finance":
        return yahoo(ticker, period, interval), {
            "source": "Yahoo Finance",
            "instrument": ticker,
            "token": "Not applicable"
        }

    if angel_api is None:
        raise RuntimeError(
            "Angel One is selected but no authenticated session exists."
        )

    if angel_master is None or angel_master.empty:
        raise RuntimeError(
            "Angel One instrument master is unavailable."
        )

    instrument =
    # ============================================================
# PART 7: FINAL DASHBOARD, CHARTS, PAPER TRADING & REPORTS
# ============================================================

st.divider()

st.title("📈 TRADING INTELLIGENCE TERMINAL")
st.caption(
    "Technical analysis and educational paper-trading dashboard. "
    "Signals are estimates, not guaranteed outcomes."
)


# ------------------------------------------------------------
# 1. CURRENT MARKET OVERVIEW
# ------------------------------------------------------------

st.header("1. Current Market Overview")

latest_price = safe(r.get("close"))
previous_price = safe(
    df["close"].iloc[-2] if len(df) > 1 else latest_price
)

change = latest_price - previous_price

change_pct = (
    (change / previous_price) * 100
    if previous_price else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Selected Instrument",
    selected_name
)

c2.metric(
    "Latest Price",
    fmt(latest_price),
    f"{change:+.2f} ({change_pct:+.2f}%)"
)

c3.metric(
    "Timeframe",
    interval
)

c4.metric(
    "Data Source",
    data_info.get("source", source)
)


# ------------------------------------------------------------
# 2. CURRENT SIGNAL
# ------------------------------------------------------------

st.header("2. Current Signal Analysis")

s1, s2, s3, s4 = st.columns(4)

s1.metric("Current Signal", signal)
s2.metric("Bullish Score", f"{bull_score}/100")
s3.metric("Bearish Score", f"{bear_score}/100")
s4.metric("Signal Strength", f"{confidence:.1f}/100")

s5, s6, s7, s8 = st.columns(4)

s5.metric("Validation Score", f"{validation_score:.1f}%")
s6.metric("Strategy Grade", str(grade))
s7.metric("Trade Quality", f"{quality:.1f}/100")
s8.metric("System Health", f"{health:.1f}/100")

st.info(
    "Signal strength is a technical scoring measure. "
    "It is not a verified probability of winning a trade."
)

st.subheader("Signal Confirmation Factors")

factor_data = [
    ("EMA 20", fmt(r.get("ema20"))),
    ("EMA 50", fmt(r.get("ema50"))),
    ("EMA 200", fmt(r.get("ema200"))),
    ("RSI", fmt(r.get("rsi"))),
    ("ADX", fmt(r.get("adx"))),
    ("ATR", fmt(r.get("atr"))),
    ("VWAP", fmt(r.get("vwap"))),
    ("MACD", fmt(r.get("macd"))),
    ("SuperTrend", fmt(r.get("supertrend"))),
    ("Volume Ratio", fmt(r.get("vol_ratio"))),
    ("Stochastic", fmt(r.get("stoch_k"))),
    ("MTF Bias", mtf_bias),
]

st.dataframe(
    pd.DataFrame(
        factor_data,
        columns=["Indicator", "Current Value"]
    ),
    use_container_width=True,
    hide_index=True
)

st.subheader("Signal Reasons")

if signal_reasons:
    for reason in signal_reasons:
        st.write("•", reason)
else:
    st.write("No additional confirmation reasons available.")


# ------------------------------------------------------------
# 3. MULTI-TIMEFRAME ANALYSIS
# ------------------------------------------------------------

st.header("3. Multi-Timeframe Analysis")

if mtf_frame is not None and not mtf_frame.empty:

    st.dataframe(
        mtf_frame,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Multi-timeframe information is not available."
    )

st.caption(
    "MTF confirmation is based on available candle history. "
    "Missing timeframe data is not treated as confirmation."
)


# ------------------------------------------------------------
# 4. CANDLESTICK AND HEIKIN ASHI CHART
# ------------------------------------------------------------

st.header("4. Candlestick & Heikin Ashi Chart")

chart_mode = st.radio(
    "Chart Display",
    [
        "Japanese Candlestick",
        "Heikin Ashi",
        "Both"
    ],
    horizontal=True
)

chart_data = df.tail(250).copy()

def make_candle_chart(frame, title, heikin=False):

    fig = go.Figure()

    if heikin:
        fig.add_trace(
            go.Candlestick(
                x=frame.index,
                open=frame["ha_open"],
                high=frame["ha_high"],
                low=frame["ha_low"],
                close=frame["ha_close"],
                name="Heikin Ashi"
            )
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=frame.index,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="Candles"
            )
        )

    if "ema20" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["ema20"],
                name="EMA 20",
                line=dict(width=1)
            )
        )

    if "ema50" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["ema50"],
                name="EMA 50",
                line=dict(width=1)
            )
        )

    if "ema200" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["ema200"],
                name="EMA 200",
                line=dict(width=1)
            )
        )

    if "vwap" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["vwap"],
                name="VWAP",
                line=dict(width=1, dash="dot")
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h")
    )

    return fig


if chart_mode == "Japanese Candlestick":

    st.plotly_chart(
        make_candle_chart(
            chart_data,
            "Japanese Candlestick Chart"
        ),
        use_container_width=True
    )

elif chart_mode == "Heikin Ashi":

    st.plotly_chart(
        make_candle_chart(
            chart_data,
            "Heikin Ashi Chart",
            heikin=True
        ),
        use_container_width=True
    )

else:

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            make_candle_chart(
                chart_data,
                "Japanese Candlestick"
            ),
            use_container_width=True
        )

    with right:
        st.plotly_chart(
            make_candle_chart(
                chart_data,
                "Heikin Ashi",
                heikin=True
            ),
            use_container_width=True
        )


# ------------------------------------------------------------
# 5. CANDLESTICK PATTERN SCANNER
# ------------------------------------------------------------

st.header("5. Candlestick Pattern Scanner")

if patterns_found is not None and not patterns_found.empty:

    st.dataframe(
        patterns_found.tail(30),
        use_container_width=True
    )

else:
    st.info(
        "No recognized candlestick pattern in the available candles."
    )


# ------------------------------------------------------------
# 6. VOLUME AND MOMENTUM
# ------------------------------------------------------------

st.header("6. Volume & Momentum Analysis")

volume_fig = go.Figure()

volume_fig.add_trace(
    go.Bar(
        x=chart_data.index,
        y=chart_data["volume"],
        name="Volume"
    )
)

volume_fig.update_layout(
    title="Trading Volume",
    height=300,
    template="plotly_dark"
)

st.plotly_chart(
    volume_fig,
    use_container_width=True
)

momentum_columns = [
    col for col in [
        "rsi",
        "adx",
        "stoch_k"
    ]
    if col in chart_data.columns
]

if momentum_columns:

    momentum_fig = go.Figure()

    for col in momentum_columns:
        momentum_fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[col],
                name=col.upper()
            )
        )

    momentum_fig.update_layout(
        title="Momentum Indicators",
        height=350,
        template="plotly_dark"
    )

    st.plotly_chart(
        momentum_fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# 7. SUPPORT, RESISTANCE AND RISK LEVELS
# ------------------------------------------------------------

st.header("7. Support, Resistance & Risk Levels")

recent = df.tail(50)

support = safe(recent["low"].min())
resistance = safe(recent["high"].max())

a, b, c = st.columns(3)

a.metric("Recent Support", fmt(support))
b.metric("Recent Resistance", fmt(resistance))
c.metric("Current ATR", fmt(r.atr))

st.subheader("Illustrative Risk Levels")

risk_table = pd.DataFrame(
    [
        ("Entry Reference", risk_levels.get("entry")),
        ("Stop Loss", risk_levels.get("stop_loss")),
        ("Target", risk_levels.get("target")),
        ("Risk / Reward", f"1:{int(rr_choice)}"),
    ],
    columns=["Level", "Value"]
)

st.dataframe(
    risk_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "These are analytical reference levels only. "
    "They are not instructions to place a real order."
)


# ------------------------------------------------------------
# 8. PCR AND OPEN INTEREST
# ------------------------------------------------------------

st.header("8. PCR & Open Interest Analysis")

st.info(
    "PCR and OI require valid option-chain data. "
    "They must not be inferred from ordinary candle volume."
)

with st.expander("Enter option-chain data for analysis"):

    st.write(
        "Paste or enter aggregated option-chain values "
        "when you have a verified data source."
    )

    ce_oi = st.number_input(
        "Call Open Interest",
        min_value=0.0,
        value=0.0
    )

    pe_oi = st.number_input(
        "Put Open Interest",
        min_value=0.0,
        value=0.0
    )

    ce_change = st.number_input(
        "Call OI Change",
        value=0.0
    )

    pe_change = st.number_input(
        "Put OI Change",
        value=0.0
    )

    if st.button("Calculate PCR and OI Bias"):

        option_input = pd.DataFrame(
            [{
                "CE_OI": ce_oi,
                "PE_OI": pe_oi,
                "CE_OI_CHANGE": ce_change,
                "PE_OI_CHANGE": pe_change
            }]
        )

        pcr_result = calculate_pcr(option_input)

        st.session_state.pcr_result = pcr_result

        st.json(pcr_result)

pcr_result = st.session_state.get(
    "pcr_result",
    {}
)

if pcr_result:
    st.write("Latest calculated PCR/OI result:")
    st.json(pcr_result)
else:
    st.warning(
        "PCR/OI is not currently available from a verified live option chain."
    )


# ------------------------------------------------------------
# 9. WALK FORWARD VALIDATION AND BACKTEST
# ------------------------------------------------------------

st.header("9. Historical Strategy Evaluation")

try:

    backtest = walk_forward_validation(df)

    if isinstance(backtest, dict):

        st.json(backtest)

        equity_values = backtest.get(
            "equity_curve",
            backtest.get("equity", [])
        )

        if isinstance(equity_values, (list, tuple, np.ndarray, pd.Series)):

            equity_fig = go.Figure()

            equity_fig.add_trace(
                go.Scatter(
                    y=list(equity_values),
                    mode="lines",
                    name="Strategy Equity"
                )
            )

            equity_fig.update_layout(
                title="Walk Forward Equity Curve",
                height=350,
                template="plotly_dark"
            )

            st.plotly_chart(
                equity_fig,
                use_container_width=True
            )

    elif isinstance(backtest, pd.DataFrame):

        st.dataframe(
            backtest,
            use_container_width=True
        )

    else:
        st.write(backtest)

except Exception as exc:

    st.warning(
        f"Historical evaluation unavailable: {exc}"
    )


try:

    sharpe = calculate_sharpe(df)
    drawdown = max_drawdown(df)

    x, y = st.columns(2)

    x.metric(
        "Sharpe Ratio",
        fmt(sharpe)
    )

    y.metric(
        "Maximum Drawdown",
        fmt(drawdown)
    )

except Exception as exc:

    st.caption(
        f"Risk statistics unavailable: {exc}"
    )


# ------------------------------------------------------------
# 10. PAPER TRADING
# ------------------------------------------------------------

st.header("10. Paper Trading Simulator")

st.warning(
    "Paper trading only. This dashboard does not place real broker orders."
)

paper_cash_value = st.session_state.get(
    "paper_cash",
    100000.0
)

paper_current_equity = paper_equity(latest_price)

p1, p2, p3 = st.columns(3)

p1.metric(
    "Paper Cash",
    fmt(paper_cash_value)
)

p2.metric(
    "Paper Equity",
    fmt(paper_current_equity)
)

p3.metric(
    "Realized P/L",
    fmt(st.session_state.get("paper_realized", 0))
)

st.write(
    "Capital Protection:",
    "LOCKED" if capital_status else "ACTIVE"
)

st.write(
    "Weekly / Monthly Lock:",
    "LOCKED" if period_status else "ACTIVE"
)

paper_quantity = st.number_input(
    "Paper Quantity",
    min_value=1,
    max_value=100000,
    value=1,
    step=1
)

position = st.session_state.get(
    "paper_position"
)

if position is None:

    st.write(
        f"Paper reference price: {latest_price:.2f}"
    )

    buy_col, sell_col = st.columns(2)

    with buy_col:

        if st.button(
            "🟢 Open Paper BUY",
            use_container_width=True
        ):

            if paper_entry_allowed(
                paper_current_equity
            ):

                try:
                    open_paper_position(
                        "BUY",
                        latest_price,
                        paper_quantity,
                        safe(r.atr)
                    )

                    st.success(
                        "Paper BUY position opened."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Paper BUY failed: {exc}"
                    )

            else:

                st.error(
                    "Paper entry blocked by capital protection."
                )

    with sell_col:

        if st.button(
            "🔴 Open Paper SELL",
            use_container_width=True
        ):

            if paper_entry_allowed(
                paper_current_equity
            ):

                try:
                    open_paper_position(
                        "SELL",
                        latest_price,
                        paper_quantity,
                        safe(r.atr)
                    )

                    st.success(
                        "Paper SELL position opened."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Paper SELL failed: {exc}"
                    )

            else:

                st.error(
                    "Paper entry blocked by capital protection."
                )

else:

    st.subheader("Open Paper Position")

    st.json(position)

    try:

        direction = position.get(
            "direction",
            position.get("side", "BUY")
        )

        entry_price = safe(
            position.get("entry_price")
        )

        if direction == "BUY":
            unrealized = (
                latest_price - entry_price
            ) * safe(position.get("quantity", 1))
        else:
            unrealized = (
                entry_price - latest_price
            ) * safe(position.get("quantity", 1))

        st.metric(
            "Unrealized Paper P/L",
            fmt(unrealized)
        )

    except Exception:
        pass

    if st.button(
        "Close Paper Position",
        use_container_width=True
    ):

        try:

            close_paper_position(
                latest_price,
                "Manual paper close"
            )

            st.success(
                "Paper position closed."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Unable to close paper position: {exc}"
            )


# ------------------------------------------------------------
# 11. TRADE JOURNAL AND ANALYTICS
# ------------------------------------------------------------

st.header("11. Paper Trade Journal")

try:

    journal = st.session_state.get(
        "trade_journal",
        []
    )

    if isinstance(journal, pd.DataFrame):
        journal_df = journal.copy()
    else:
        journal_df = pd.DataFrame(journal)

    if not journal_df.empty:

        st.dataframe(
            journal_df,
            use_container_width=True
        )

        csv_data = journal_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Paper Trade Journal CSV",
            data=csv_data,
            file_name="paper_trade_journal.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No completed paper trades available yet."
        )

except Exception as exc:

    st.warning(
        f"Journal display issue: {exc}"
    )


try:

    journal_stats = journal_analytics()

    if isinstance(journal_stats, dict):
        st.subheader("Journal Statistics")
        st.json(journal_stats)

except Exception as exc:

    st.caption(
        f"Journal analytics unavailable: {exc}"
    )


# ------------------------------------------------------------
# 12. SIGNAL AUDIT
# ------------------------------------------------------------

st.header("12. Signal Audit Trail")

try:

    audit_rows = st.session_state.get(
        "signal_audit",
        []
    )

    audit_df = pd.DataFrame(audit_rows)

    if not audit_df.empty:

        st.dataframe(
            audit_df.tail(100),
            use_container_width=True
        )

        audit_csv = audit_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Signal Audit CSV",
            data=audit_csv,
            file_name="signal_audit.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No signal audit records available."
        )

except Exception as exc:

    st.warning(
        f"Signal audit display issue: {exc}"
    )


# ------------------------------------------------------------
# 13. MARKET WATCHLIST
# ------------------------------------------------------------

st.header("13. Market Watchlist")

with st.expander("Global Markets"):

    try:

        global_data = fetch_market_snapshot(
            global_indices
        )

        if isinstance(global_data, pd.DataFrame):
            st.dataframe(
                global_data,
                use_container_width=True
            )
        else:
            st.write(global_data)

    except Exception as exc:

        st.warning(
            f"Global market snapshot unavailable: {exc}"
        )


with st.expander("Indian Indices"):

    try:

        indian_data = fetch_market_snapshot(
            indian_indices
        )

        if isinstance(indian_data, pd.DataFrame):
            st.dataframe(
                indian_data,
                use_container_width=True
            )
        else:
            st.write(indian_data)

    except Exception as exc:

        st.warning(
            f"Indian indices unavailable: {exc}"
        )


with st.expander("Major Indian Stocks"):

    try:

        stock_data = fetch_market_snapshot(
            major_stocks
        )

        if isinstance(stock_data, pd.DataFrame):
            st.dataframe(
                stock_data,
                use_container_width=True
            )
        else:
            st.write(stock_data)

    except Exception as exc:

        st.warning(
            f"Major stocks unavailable: {exc}"
        )


# ------------------------------------------------------------
# 14. FINAL SYSTEM STATUS
# ------------------------------------------------------------

st.divider()

st.header("14. System Status")

status_data = [
    ("Selected Data Source", source),
    ("Angel One SDK", "Available" if ANGEL_SDK_AVAILABLE else "Not installed"),
    ("TOTP Support", "Available" if PYOTP_AVAILABLE else "Not installed"),
    ("Angel One Session", angel_status_check(angel_api)),
    ("Candle Count", len(data)),
    ("Selected Timeframe", interval),
    ("Auto Refresh", "ON" if auto_refresh else "OFF"),
    ("MTF Bias", mtf_bias),
    ("Paper Position", "OPEN" if st.session_state.get("paper_position") else "CLOSED"),
]

st.dataframe(
    pd.DataFrame(
        status_data,
        columns=["Component", "Status"]
    ),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Important: REST API refresh is periodic polling, not tick-by-tick "
    "WebSocket streaming. Data availability depends on broker permissions, "
    "API limits, exchange hours and the selected instrument."
)

st.caption(
    "PCR/OI, FII/DII, news and geopolitical information are not assumed "
    "to be live unless connected to a verified data source."
)

st.success(
    "Trading Intelligence Terminal dashboard execution completed."
)

st.caption(
    "Educational analysis and paper simulation only. "
    "No signal guarantees a profitable outcome."
)
