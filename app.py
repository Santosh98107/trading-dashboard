import os, hashlib, time, requests, threading, uuid
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

st.set_page_config(page_title='Trading Intelligence Terminal', page_icon='📈', layout='wide')

def sha256(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def login():
    if st.session_state.get('authenticated'): return True
    st.title('🔐 Trading Intelligence Terminal')
    st.subheader('Application Login')
    user_expected = os.getenv('APP_USERNAME', 'admin')
    hash_expected = os.getenv('APP_PASSWORD_SHA256', sha256('admin123'))
    with st.form('login_form'):
        user = st.text_input('Username')
        password = st.text_input('Password', type='password')
        submit = st.form_submit_button('Login')
    if submit:
        if user == user_expected and sha256(password) == hash_expected:
            st.session_state.authenticated = True
            st.rerun()
        else: st.error('Username or password is incorrect.')
    st.caption('Testing default: admin / admin123. Change before public deployment.')
    return False
if not login(): st.stop()

st.title('📈 Trading Intelligence Terminal')
st.caption('Technical analysis • Global markets • PCR/OI context • Backtesting • Paper trading')
st.warning('Research and paper-trading tool only. Yahoo Finance data may be delayed or incomplete. No live broker orders are placed. Scores are not guaranteed win probabilities.')

for k,v in {'paper_trades': [], 'paper_position': None, 'angel_candles': None, 'signal_audit': [], 'weekly_lock': False, 'monthly_lock': False}.items():
    if k not in st.session_state: st.session_state[k] = v

SYMBOLS = {
 'NIFTY 50':'^NSEI','BANK NIFTY':'^NSEBANK','SENSEX':'^BSESN','NIFTY IT':'^CNXIT','NIFTY AUTO':'^CNXAUTO',
 'RELIANCE':'RELIANCE.NS','HDFC BANK':'HDFCBANK.NS','ICICI BANK':'ICICIBANK.NS','SBI':'SBIN.NS','AXIS BANK':'AXISBANK.NS','KOTAK BANK':'KOTAKBANK.NS',
 'TCS':'TCS.NS','INFOSYS':'INFY.NS','HCLTECH':'HCLTECH.NS','WIPRO':'WIPRO.NS','TECH MAHINDRA':'TECHM.NS',
 'MARUTI':'MARUTI.NS','M&M':'M&M.NS','TATA MOTORS':'TMPV.NS','BAJAJ AUTO':'BAJAJ-AUTO.NS','EICHER MOTORS':'EICHERMOT.NS','RELIANCE ENERGY CONTEXT':'RELIANCE.NS'
}
GLOBAL = {'US S&P 500':'^GSPC','US Nasdaq':'^IXIC','US Dow Jones':'^DJI','Japan Nikkei':'^N225','Hong Kong Hang Seng':'^HSI','China Shanghai':'000001.SS','South Korea KOSPI':'^KS11','Brent Crude':'BZ=F','USD/INR':'INR=X','India VIX':'^INDIAVIX'}

ANGEL_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

@st.cache_data(ttl=3600, show_spinner=False)
def load_angel_master():
    """Download Angel One's public instrument master and cache it for one hour."""
    response = requests.get(ANGEL_MASTER_URL, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Angel One instrument master returned an unexpected format.")
    frame = pd.DataFrame(payload)
    required = {"symbol", "token", "exch_seg"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Instrument master is missing symbol/token/exchange fields.")
    for col in ("symbol", "name", "exch_seg", "instrumenttype", "token"):
        if col not in frame.columns: frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str)
    frame["_sym_norm"] = frame["symbol"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    frame["_name_norm"] = frame["name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    frame["_exchange_norm"] = frame["exch_seg"].str.upper()
    return frame

ANGEL_ALIASES = {
    "NIFTY 50": ["NIFTY 50", "NIFTY50"],
    "BANK NIFTY": ["NIFTY BANK", "BANKNIFTY", "NIFTYBANK"],
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
    "TATA MOTORS": ["TMPV-EQ", "TATAMOTORS-EQ", "TMPV", "TATAMOTORS"],
    "BAJAJ AUTO": ["BAJAJ-AUTO-EQ", "BAJAJAUTO-EQ", "BAJAJ-AUTO"],
    "EICHER MOTORS": ["EICHERMOT-EQ", "EICHERMOT"],
}

def resolve_angel_instrument(master, display_name, custom_query=""):
    """Resolve a selected display name to its exchange, trading symbol and token."""
    if master is None or master.empty: raise RuntimeError("Angel One instrument master is empty.")
    if display_name == "Custom Yahoo ticker":
        query = custom_query.strip().upper()
        if not query: raise RuntimeError("Enter an instrument name/symbol to search; token is resolved automatically.")
        candidates = [query, query + "-EQ"]
        preferred_exchange = "NSE"
    else:
        candidates = ANGEL_ALIASES.get(display_name, [display_name])
        preferred_exchange = "BSE" if display_name == "SENSEX" else "NSE"
    norm_candidates = ["".join(ch for ch in c.upper() if ch.isalnum()) for c in candidates]
    subset = master[master["_exchange_norm"].isin([preferred_exchange, "BSE" if preferred_exchange == "NSE" else "NSE"])].copy()
    # Exact symbol match first; stock entries are preferred over derivative contracts.
    for candidate in norm_candidates:
        hits = subset[subset["_sym_norm"] == candidate]
        if not hits.empty:
            hits = hits.copy()
            hits["_exchange_rank"] = (hits["_exchange_norm"] != preferred_exchange).astype(int)
            hits["_derivative_rank"] = hits["_sym_norm"].str.contains(r"FUT|CE|PE", regex=True).astype(int)
            hits = hits.sort_values(["_exchange_rank", "_derivative_rank"])
            row = hits.iloc[0]
            return {"exchange": row["exch_seg"].upper(), "symbol": row["symbol"], "token": row["token"], "name": row["name"] or row["symbol"]}
    # Index symbols sometimes use the display name in the master instead of the symbol field.
    for candidate in norm_candidates:
        hits = subset[(subset["_name_norm"] == candidate) | (subset["_sym_norm"] == candidate)]
        if not hits.empty:
            hits = hits.copy(); hits["_exchange_rank"] = (hits["_exchange_norm"] != preferred_exchange).astype(int)
            row = hits.sort_values("_exchange_rank").iloc[0]
            return {"exchange": row["exch_seg"].upper(), "symbol": row["symbol"], "token": row["token"], "name": row["name"] or row["symbol"]}
    raise RuntimeError(f"'{display_name if display_name != 'Custom Yahoo ticker' else custom_query}' was not found in Angel One's instrument master. Try another listed instrument or refresh the master.")

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
    if not response or response.get("status") is False or not response.get("data"):
        message = (response or {}).get("message") or (response or {}).get("errorcode") or "No candles returned"
        raise RuntimeError(f"Angel One candle request failed: {message}. Check API permissions, market hours and selected interval.")
    cdf = pd.DataFrame(response["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]: cdf[col] = pd.to_numeric(cdf[col], errors="coerce")
    return cdf.dropna(subset=["timestamp", "open", "high", "low", "close"]).set_index("timestamp").sort_index()

class AngelWebSocketFeed:
    """Background Angel One LTP feed. It only reads market data; it never places orders."""
    def __init__(self, api_key, client_code, auth_token, feed_token, instrument):
        self.api_key = api_key
        self.client_code = client_code
        self.auth_token = auth_token
        self.feed_token = feed_token
        self.instrument = dict(instrument)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.connected = False
        self.ws = None
        self.thread = None
        self.correlation_id = 't-' + uuid.uuid4().hex[:12]

    def _on_open(self, wsapp):
        try:
            exchange_type = 3 if self.instrument.get('exchange') == 'BSE' else 1
            token_list = [{'exchangeType': exchange_type, 'tokens': [str(self.instrument['token'])]}]
            wsapp.subscribe(self.correlation_id, 1, token_list)  # 1 = LTP mode
            self.connected = True
            self.error = None
        except Exception as exc:
            self.error = f'Subscription failed: {exc}'

    def _on_data(self, wsapp, message):
        try:
            if not isinstance(message, dict):
                return
            raw_ltp = message.get('last_traded_price')
            if raw_ltp is None:
                return
            ltp = float(raw_ltp) / 100.0  # SmartAPI sends prices in paise
            ts = message.get('exchange_timestamp') or message.get('last_traded_timestamp')
            tick_time = pd.to_datetime(ts, unit='ms', errors='coerce') if ts is not None else pd.Timestamp.now()
            tick = {'ltp': ltp, 'timestamp': tick_time, 'raw': message}
            with self.lock:
                self.latest = tick
        except Exception as exc:
            self.error = f'Tick parse error: {exc}'

    def _on_error(self, wsapp, error):
        self.error = str(error)
        self.connected = False

    def _on_close(self, wsapp):
        self.connected = False

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
            self.ws = SmartWebSocketV2(self.auth_token, self.api_key, self.client_code, self.feed_token)
            self.ws.on_open = self._on_open
            self.ws.on_data = self._on_data
            self.ws.on_error = self._on_error
            self.ws.on_close = self._on_close
            self.thread = threading.Thread(target=self.ws.connect, daemon=True, name='AngelOneMarketDataWS')
            self.thread.start()
        except Exception as exc:
            self.error = f'WebSocket start failed: {exc}'

    def latest_tick(self):
        with self.lock:
            return dict(self.latest) if self.latest else None

    def stop(self):
        try:
            if self.ws:
                self.ws.close_connection()
        except Exception:
            pass
        self.connected = False

def norm(raw):
    if raw is None or raw.empty: return pd.DataFrame()
    x=raw.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).lower().replace(' ','') for c in x.columns]
    if not all(c in x.columns for c in ['open','high','low','close']): return pd.DataFrame()
    if 'volume' not in x: x['volume']=0.0
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['open','high','low','close']).loc[lambda z: ~z.index.duplicated(keep='last')].sort_index()
@st.cache_data(ttl=15,show_spinner=False)
def yahoo(ticker,period,interval):
    try: return norm(yf.download(ticker,period=period,interval=interval,auto_adjust=False,progress=False,threads=False))
    except Exception: return pd.DataFrame()
def resample(x,rule):
    if x.empty:return x
    return x.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open','high','low','close'])
def indicators(x):
    d=x.copy().sort_index()
    d['ema20']=d.close.ewm(span=20,adjust=False).mean(); d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    # MACD and Bollinger Bands are calculated on the same candle series used for signal scoring
    d['macd']=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
    d['macd_signal']=d.macd.ewm(span=9,adjust=False).mean(); d['macd_hist']=d.macd-d.macd_signal
    d['bb_mid']=d.close.rolling(20,min_periods=20).mean(); d['bb_std']=d.close.rolling(20,min_periods=20).std(ddof=0)
    d['bb_upper']=d.bb_mid+2*d.bb_std; d['bb_lower']=d.bb_mid-2*d.bb_std
    for n in [20,50,100,200]: d[f'sma{n}']=d.close.rolling(n,min_periods=n).mean()
    d['ema200']=d.close.ewm(span=200,adjust=False).mean()
    # Heikin-Ashi candles (display/analysis columns; raw OHLC remains unchanged).
    d['ha_close']=(d.open+d.high+d.low+d.close)/4
    ha_open=[]
    for i, value in enumerate(d['ha_close'].to_numpy()):
        ha_open.append((d.open.iloc[i]+d.close.iloc[i])/2 if i==0 else (ha_open[i-1]+d['ha_close'].iloc[i-1])/2)
    d['ha_open']=ha_open
    d['ha_high']=pd.concat([d.high,d.ha_open,d.ha_close],axis=1).max(axis=1)
    d['ha_low']=pd.concat([d.low,d.ha_open,d.ha_close],axis=1).min(axis=1)
    # True range is calculated before SuperTrend so the indicator is available
    # on the first pass (the previous draft referenced it before assignment).
    pc=d.close.shift()
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    # Wilder-style SuperTrend with bands seeded at the first valid ATR candle.
    st_mult, st_period = 3.0, 10
    st_atr=tr.ewm(alpha=1/st_period,adjust=False,min_periods=st_period).mean()
    hl2=(d.high+d.low)/2
    upper=hl2+st_mult*st_atr; lower=hl2-st_mult*st_atr
    final_upper=pd.Series(np.nan,index=d.index,dtype=float)
    final_lower=pd.Series(np.nan,index=d.index,dtype=float)
    st_dir=pd.Series(1,index=d.index,dtype=int)
    valid_atr=np.flatnonzero(st_atr.notna().to_numpy())
    if len(valid_atr):
        first=int(valid_atr[0])
        final_upper.iloc[first]=upper.iloc[first]
        final_lower.iloc[first]=lower.iloc[first]
        for i in range(first+1,len(d)):
            final_upper.iloc[i]=upper.iloc[i] if upper.iloc[i]<final_upper.iloc[i-1] or d.close.iloc[i-1]>final_upper.iloc[i-1] else final_upper.iloc[i-1]
            final_lower.iloc[i]=lower.iloc[i] if lower.iloc[i]>final_lower.iloc[i-1] or d.close.iloc[i-1]<final_lower.iloc[i-1] else final_lower.iloc[i-1]
            if st_dir.iloc[i-1] < 0 and d.close.iloc[i] > final_upper.iloc[i-1]: st_dir.iloc[i]=1
            elif st_dir.iloc[i-1] > 0 and d.close.iloc[i] < final_lower.iloc[i-1]: st_dir.iloc[i]=-1
            else: st_dir.iloc[i]=st_dir.iloc[i-1]
    d['supertrend_direction']=st_dir
    d['supertrend']=np.where(st_dir>0,final_lower,final_upper)
    delta=d.close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan); d['rsi']=100-100/(1+rs); d.loc[(loss==0)&(gain>0),'rsi']=100; d.loc[(gain==0)&(loss>0),'rsi']=0
    d['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    up=d.high.diff(); down=-d.low.diff(); pdm=pd.Series(np.where((up>down)&(up>0),up,0),index=d.index); mdm=pd.Series(np.where((down>up)&(down>0),down,0),index=d.index)
    d['di_plus']=100*pdm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/d.atr.replace(0,np.nan); d['di_minus']=100*mdm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/d.atr.replace(0,np.nan)
    d['dx']=100*(d.di_plus-d.di_minus).abs()/(d.di_plus+d.di_minus).replace(0,np.nan); d['adx']=d.dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    lo=d.low.rolling(14).min(); hi=d.high.rolling(14).max(); d['stoch_k']=100*(d.close-lo)/(hi-lo).replace(0,np.nan); d['stoch_d']=d.stoch_k.rolling(3).mean()
    typical=(d.high+d.low+d.close)/3; sess=pd.Series(pd.DatetimeIndex(d.index).date,index=d.index); pv=typical*d.volume
    d['vwap']=pv.groupby(sess).cumsum()/d.volume.groupby(sess).cumsum().replace(0,np.nan)
    d['vol_avg20']=d.volume.rolling(20,min_periods=5).mean(); d['vol_ratio']=d.volume/d.vol_avg20.replace(0,np.nan)
    d['prior_high20']=d.high.rolling(20,min_periods=10).max().shift(1); d['prior_low20']=d.low.rolling(20,min_periods=10).min().shift(1)
    # Simplified, vectorised candlestick confirmation features for scoring
    po=d.open.shift(1); pc=d.close.shift(1); body=(d.close-d.open).abs(); rng=(d.high-d.low).replace(0,np.nan)
    prev_body=(pc-po).abs()
    d['bull_pattern_score']=0
    d['bear_pattern_score']=0
    bull_eng=(d.close>d.open)&(pc<po)&(d.close>=po)&(d.open<=pc)
    bear_eng=(d.close<d.open)&(pc>po)&(d.open>=pc)&(d.close<=po)
    lower=d[['open','close']].min(axis=1)-d.low; upper=d.high-d[['open','close']].max(axis=1)
    hammer=(lower>=2*body.clip(lower=1e-9))&(upper<=body.clip(lower=1e-9)*1.2)&(d.close>d.open)
    shooting=(upper>=2*body.clip(lower=1e-9))&(lower<=body.clip(lower=1e-9)*1.2)&(d.close<d.open)
    inverted_hammer=(upper>=2*body.clip(lower=1e-9))&(lower<=body.clip(lower=1e-9)*1.2)&(d.close>d.open)
    hanging_man=(lower>=2*body.clip(lower=1e-9))&(upper<=body.clip(lower=1e-9)*1.2)&(d.close<d.open)
    bull_marubozu=(body/rng>=.9)&(d.close>d.open)
    bear_marubozu=(body/rng>=.9)&(d.close<d.open)
    tolerance=(d.atr.fillna(d.close.abs()*.001)*.10).clip(lower=1e-8)
    tweezer_bottom=(d.low-d.low.shift()).abs()<=tolerance
    tweezer_bottom=tweezer_bottom&(pc<po)&(d.close>d.open)
    tweezer_top=(d.high-d.high.shift()).abs()<=tolerance
    tweezer_top=tweezer_top&(pc>po)&(d.close<d.open)
    doji=(body/rng<=.1)
    # Single-candle and two-candle pattern scores. These scores feed get_signal().
    bull_harami=(pc<po)&(d.close>d.open)&(d.open>=pc)&(d.close<=po)
    bear_harami=(pc>po)&(d.close<d.open)&(d.open<=pc)&(d.close>=po)
    midpoint_prev=(po+pc)/2
    piercing=(pc<po)&(d.close>d.open)&(d.open<pc)&(d.close>midpoint_prev)&(d.close<po)
    dark_cloud=(pc>po)&(d.close<d.open)&(d.open>pc)&(d.close<midpoint_prev)&(d.close>po)
    d['bull_pattern_score'] = (bull_eng.astype(int) + hammer.astype(int) + inverted_hammer.astype(int) + bull_marubozu.astype(int) + tweezer_bottom.astype(int) + bull_harami.astype(int) + piercing.astype(int))
    d['bear_pattern_score'] = (bear_eng.astype(int) + shooting.astype(int) + hanging_man.astype(int) + bear_marubozu.astype(int) + tweezer_top.astype(int) + bear_harami.astype(int) + dark_cloud.astype(int))
    # Three-candle reversal patterns; simplified rules for screening, not guarantees.
    po2=d.open.shift(2); pc2=d.close.shift(2); body2=(pc2-po2).abs()
    morning=(pc2<po2)&(prev_body<body2*.6)&(d.close>d.open)&(d.close>(po2+pc2)/2)
    evening=(pc2>po2)&(prev_body<body2*.6)&(d.close<d.open)&(d.close<(po2+pc2)/2)
    bull_soldiers=(d.close>d.open)&(pc>po)&(pc2>po2)&(d.close>pc)&(pc>pc2)
    bear_crows=(d.close<d.open)&(pc<po)&(pc2<po2)&(d.close<pc)&(pc<pc2)
    d.loc[morning|bull_soldiers,'bull_pattern_score'] += 1
    d.loc[evening|bear_crows,'bear_pattern_score'] += 1
    d.loc[doji,'bull_pattern_score']=0; d.loc[doji,'bear_pattern_score']=0
    return d
def patterns(d):
    if len(d)<3:return ['Insufficient candles']
    a,b,c=d.iloc[-1],d.iloc[-2],d.iloc[-3]
    body=abs(a.close-a.open); rng=max(a.high-a.low,1e-9)
    upper=a.high-max(a.open,a.close); lower=min(a.open,a.close)-a.low; out=[]
    # Two-candle patterns
    if a.close>a.open and b.close<b.open and a.close>=b.open and a.open<=b.close: out.append('Bullish Engulfing')
    if a.close<a.open and b.close>b.open and a.open>=b.close and a.close<=b.open: out.append('Bearish Engulfing')
    if b.close<b.open and a.close>a.open and a.open<=b.close and a.close>=(b.open+b.close)/2: out.append('Piercing Pattern')
    if b.close>b.open and a.close<a.open and a.open>=b.close and a.close<=(b.open+b.close)/2: out.append('Dark Cloud Cover')
    if b.close<b.open and a.close>a.open and a.open>=b.close and a.close<=b.open: out.append('Bullish Harami')
    if b.close>b.open and a.close<a.open and a.open<=b.close and a.close>=b.open: out.append('Bearish Harami')
    # Single-candle patterns
    if body/rng<=.1: out.append('Doji')
    if .1<body/rng<=.3 and upper>body and lower>body: out.append('Spinning Top')
    if body/rng>=.9 and a.close>a.open: out.append('Bullish Marubozu')
    if body/rng>=.9 and a.close<a.open: out.append('Bearish Marubozu')
    if lower>=2*max(body,1e-9) and upper<=max(body,1e-9)*1.2:
        out.append('Hammer' if a.close>a.open else 'Hanging Man')
    if upper>=2*max(body,1e-9) and lower<=max(body,1e-9)*1.2:
        out.append('Inverted Hammer' if a.close>a.open else 'Shooting Star')
    tol=max(rng*.10,1e-8)
    if abs(a.low-b.low)<=tol and b.close<b.open and a.close>a.open: out.append('Tweezer Bottom (possible)')
    if abs(a.high-b.high)<=tol and b.close>b.open and a.close<a.open: out.append('Tweezer Top (possible)')
    # Three-candle patterns
    if c.close<c.open and abs(b.close-b.open)<abs(c.close-c.open)*.6 and a.close>a.open and a.close>(c.open+c.close)/2: out.append('Morning Star (possible)')
    if c.close>c.open and abs(b.close-b.open)<abs(c.close-c.open)*.6 and a.close<a.open and a.close<(c.open+c.close)/2: out.append('Evening Star (possible)')
    if c.close<c.open and b.close>b.open and a.close>a.open and b.close>c.close and a.close>b.close: out.append('Three White Soldiers (possible)')
    if c.close>c.open and b.close<b.open and a.close<a.open and b.close<c.close and a.close<b.close: out.append('Three Black Crows (possible)')
    if a.high<b.high and a.low>b.low: out.append('Inside Bar')
    if a.high>b.high and a.low<b.low: out.append('Outside Bar')
    return out or ['No major pattern detected']
def safe(v):
    try: return float(v) if pd.notna(v) and np.isfinite(float(v)) else np.nan
    except Exception:return np.nan
def fmt(v,n=2): return 'N/A' if not np.isfinite(safe(v)) else f'{safe(v):,.{n}f}'
def get_signal(r):
    bull=bear=0; reasons=[]
    if r.close>r.ema20: bull+=1; reasons.append('Price > EMA20')
    elif r.close<r.ema20: bear+=1; reasons.append('Price < EMA20')
    if r.ema20>r.ema50: bull+=1
    elif r.ema20<r.ema50: bear+=1
    # Longer trend, SuperTrend and Heikin-Ashi are explicit signal confirmations.
    if pd.notna(r.get('ema200', np.nan)):
        if r.close > r.ema200: bull+=1; reasons.append('Price above EMA200')
        elif r.close < r.ema200: bear+=1; reasons.append('Price below EMA200')
    if pd.notna(r.get('supertrend_direction', np.nan)):
        if r.supertrend_direction > 0: bull+=1; reasons.append('SuperTrend bullish')
        elif r.supertrend_direction < 0: bear+=1; reasons.append('SuperTrend bearish')
    if pd.notna(r.get('ha_close', np.nan)) and pd.notna(r.get('ha_open', np.nan)):
        if r.ha_close > r.ha_open: bull+=1; reasons.append('Heikin-Ashi bullish')
        elif r.ha_close < r.ha_open: bear+=1; reasons.append('Heikin-Ashi bearish')
    if pd.notna(r.vwap):
        if r.close>r.vwap: bull+=1
        elif r.close<r.vwap: bear+=1
    if pd.notna(r.rsi):
        if 52<=r.rsi<=70: bull+=1
        elif 30<=r.rsi<=48: bear+=1
    if pd.notna(r.adx) and r.adx>=MANDATORY_ADX:
        if r.di_plus>r.di_minus: bull+=1
        elif r.di_minus>r.di_plus: bear+=1
    if pd.notna(r.vol_ratio) and r.vol_ratio>=1.2:
        if r.close>r.open: bull+=1
        elif r.close<r.open: bear+=1
    if pd.notna(r.prior_high20) and r.close>r.prior_high20: bull+=2; reasons.append('20-candle breakout')
    if pd.notna(r.prior_low20) and r.close<r.prior_low20: bear+=2; reasons.append('20-candle breakdown')
    if 'bull_pattern_score' in r.index and r.bull_pattern_score>0: bull+=1; reasons.append('Bullish candle pattern')
    if 'bear_pattern_score' in r.index and r.bear_pattern_score>0: bear+=1; reasons.append('Bearish candle pattern')
    # EMA, MACD and Bollinger confirmation all contribute to the combined signal.
    if pd.notna(r.get('macd',np.nan)) and pd.notna(r.get('macd_signal',np.nan)):
        if r.macd>r.macd_signal and r.macd_hist>0: bull+=1; reasons.append('MACD bullish')
        elif r.macd<r.macd_signal and r.macd_hist<0: bear+=1; reasons.append('MACD bearish')
    if pd.notna(r.get('bb_mid',np.nan)) and pd.notna(r.get('bb_upper',np.nan)) and pd.notna(r.get('bb_lower',np.nan)):
        if r.close>r.bb_mid and r.close<r.bb_upper: bull+=1; reasons.append('Bollinger midline bullish')
        elif r.close<r.bb_mid and r.close>r.bb_lower: bear+=1; reasons.append('Bollinger midline bearish')
        elif r.close>=r.bb_upper and r.close>r.open: bull+=1; reasons.append('Upper-band momentum')
        elif r.close<=r.bb_lower and r.close<r.open: bear+=1; reasons.append('Lower-band momentum')
    sig='BUY' if bull>=4 and bull>=bear+2 else 'SELL' if bear>=4 and bear>=bull+2 else 'WAIT'
    return sig,bull,bear,reasons
def pct(s):
    if len(s)<2:return np.nan
    p=safe(s.iloc[-2]); q=safe(s.iloc[-1]); return (q/p-1)*100 if np.isfinite(p) and p else np.nan
def _daily_levels(d):
    if d is None or d.empty:return pd.DataFrame(),{}
    d=indicators(norm(d)); prev=d.iloc[-2] if len(d)>1 else d.iloc[-1]; last=d.iloc[-1]
    lev={'Previous Close':safe(prev.close),'Previous High':safe(prev.high),'Previous Low':safe(prev.low)}
    for n in [20,50,100,200]:lev[f'{n} DMA']=safe(last.get(f'sma{n}',np.nan))
    return d,lev

def daily_info(ticker):
    # Yahoo-only daily history helper; used only when Yahoo Finance is the selected provider.
    return _daily_levels(yahoo(ticker,'2y','1d'))

def daily_info_angel(api, instrument):
    # Keep daily support/resistance and DMA calculations on the same selected Angel One source.
    return _daily_levels(fetch_angel_candles(api,instrument,'ONE_DAY',365))

def mtf(ticker):
    out=[]; cache={}
    for label,iv in [('1m','1m'),('3m','1m'),('5m','5m'),('15m','15m')]:
        if iv not in cache:cache[iv]=yahoo(ticker,'5d',iv)
        x=cache[iv].copy()
        if label=='3m' and not x.empty:x=resample(x,'3min')
        if len(x)<30:
            out.append({'Timeframe':label,'Trend':'N/A','Signal':'N/A','RSI':np.nan,'ADX':np.nan,'Last candle':'N/A'}); continue
        z=indicators(x); r=z.iloc[-1]; sig,_,_,_=get_signal(r)
        out.append({'Timeframe':label,'Trend':'Bullish' if r.ema20>r.ema50 else 'Bearish' if r.ema20<r.ema50 else 'Sideways','Signal':sig,'RSI':safe(r.rsi),'ADX':safe(r.adx),'Last candle':str(z.index[-1])})
    return pd.DataFrame(out)

def mtf_angel(api, instrument):
    """Build MTF rows from Angel One REST candles only; no silent Yahoo mixing."""
    out=[]; cache={}
    angel_intervals={'1m':'ONE_MINUTE','3m':'THREE_MINUTE','5m':'FIVE_MINUTE','15m':'FIFTEEN_MINUTE'}
    for label in ('1m','3m','5m','15m'):
        try:
            iv=angel_intervals[label]
            if iv not in cache:
                cache[iv]=fetch_angel_candles(api,instrument,iv,5)
            x=cache[iv]
            if len(x)<30:
                out.append({'Timeframe':label,'Trend':'N/A','Signal':'N/A','RSI':np.nan,'ADX':np.nan,'Last candle':'N/A'}); continue
            z=indicators(x); r=z.iloc[-1]; sig,_,_,_=get_signal(r)
            out.append({'Timeframe':label,'Trend':'Bullish' if r.ema20>r.ema50 else 'Bearish' if r.ema20<r.ema50 else 'Sideways','Signal':sig,'RSI':safe(r.rsi),'ADX':safe(r.adx),'Last candle':str(z.index[-1])})
        except Exception as exc:
            out.append({'Timeframe':label,'Trend':'Unavailable','Signal':'N/A','RSI':np.nan,'ADX':np.nan,'Last candle':str(exc)[:100]})
    return pd.DataFrame(out)
# Market context watchlists copied/adapted from the supplied app.py.
global_indices = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
    "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Shanghai Composite": "000001.SS",
    "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Gift Nifty": "NIFTY=F",
    "USD/INR": "INR=X", "Brent Crude": "BZ=F", "Gold": "GC=F", "India VIX": "^INDIAVIX",
}
indian_indices = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO",
}
major_stocks = {
    "Reliance": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS", "Axis Bank": "AXISBANK.NS", "Kotak Bank": "KOTAKBANK.NS",
    "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS", "Tech Mahindra": "TECHM.NS", "Tata Motors": "TMPV.NS",
    "Maruti": "MARUTI.NS", "Mahindra & Mahindra": "M&M.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS", "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Eicher Motors": "EICHERMOT.NS",
}

def fetch_market_snapshot(mapping, period="5d", interval="1d"):
    rows=[]
    for name, symbol in mapping.items():
        frame=yahoo(symbol,period,interval)
        if frame.empty:
            rows.append({"Instrument":name,"Ticker":symbol,"CMP":np.nan,"Change %":np.nan,"Trend":"Unavailable","Last Update":"N/A"})
            continue
        cmp=safe(frame.close.iloc[-1]); prev=safe(frame.close.iloc[-2]) if len(frame)>1 else np.nan
        change=(cmp/prev-1)*100 if np.isfinite(cmp) and np.isfinite(prev) and prev else np.nan
        rows.append({"Instrument":name,"Ticker":symbol,"CMP":cmp,"Change %":change,
                     "Trend":"Bullish" if np.isfinite(change) and change>0 else "Bearish" if np.isfinite(change) and change<0 else "Flat" if np.isfinite(change) else "Unavailable",
                     "Last Update":str(frame.index[-1])})
    return pd.DataFrame(rows)

def get_intraday_trend_label(frame):
    if frame is None or frame.empty or len(frame)<25: return "N/A"
    z=indicators(norm(frame)); r=z.iloc[-1]
    if safe(r.get('ema20'))>safe(r.get('ema50')) and safe(r.get('close'))>safe(r.get('vwap')): return "Bullish"
    if safe(r.get('ema20'))<safe(r.get('ema50')) and safe(r.get('close'))<safe(r.get('vwap')): return "Bearish"
    return "Sideways / Mixed"

def safe_fetch_intraday_trend(ticker, period="5d", interval="5m"):
    try:
        frame=yahoo(ticker,period,interval)
        return get_intraday_trend_label(frame)
    except Exception:
        return "Unavailable"

def summarize_fno_bias(scanner):
    # This is an underlying-stock momentum proxy, not an exchange option-chain OI feed.
    if scanner is None or scanner.empty or "Change %" not in scanner: return {"Bias":"Unavailable","Bullish":0,"Bearish":0,"Coverage":0}
    values=pd.to_numeric(scanner["Change %"],errors="coerce").dropna()
    bull=int((values>0).sum()); bear=int((values<0).sum())
    bias="Bullish" if bull>bear else "Bearish" if bear>bull else "Mixed"
    return {"Bias":bias,"Bullish":bull,"Bearish":bear,"Coverage":int(len(values))}

def fetch_fno_style_scanner(mapping=None):
    # Spot/underlying proxy scanner. Genuine futures % change requires futures contracts and broker quotes.
    mapping=mapping or major_stocks
    rows=[]
    for name,symbol in mapping.items():
        frame=yahoo(symbol,"5d","5m")
        if frame.empty: frame=yahoo(symbol,"5d","1d")
        if frame.empty:
            rows.append({"Instrument":name,"CMP":np.nan,"Change %":np.nan,"Intraday Trend":"Unavailable","Scanner Bias":"N/A"})
            continue
        cmp=safe(frame.close.iloc[-1]); prev=safe(frame.close.iloc[-2]) if len(frame)>1 else np.nan
        change=(cmp/prev-1)*100 if np.isfinite(cmp) and np.isfinite(prev) and prev else np.nan
        trend=get_intraday_trend_label(frame)
        rows.append({"Instrument":name,"CMP":cmp,"Change %":change,"Intraday Trend":trend,
                     "Scanner Bias":"BULLISH" if trend=="Bullish" and np.isfinite(change) and change>0 else "BEARISH" if trend=="Bearish" and np.isfinite(change) and change<0 else "WAIT"})
    return pd.DataFrame(rows).sort_values("Change %",ascending=False,na_position="last")

def get_option_chain_snapshot(index_name, spot_price, max_strikes=12):
    """Fetch verified Angel One NFO option quotes and polling-based OI changes.

    OI change is only the difference from the previous successful refresh in this
    app session; it is not historical exchange OI unless prior snapshots exist.
    """
    api = st.session_state.get('angel_api')
    if api is None:
        return pd.DataFrame(), 'Connect Angel One to request live option-chain quotes.'
    cache_key=f'option_chain_cache_{index_name}'
    cache_time=st.session_state.get(cache_key+'_time',0)
    cached=st.session_state.get(cache_key)
    if cached is not None and time.time()-cache_time<20:
        return cached, st.session_state.get(cache_key+'_note','Cached Angel One option-chain snapshot.')
    if index_name not in ('NIFTY 50', 'BANK NIFTY'):
        return pd.DataFrame(), 'Live index option chain is currently supported for NIFTY 50 and BANK NIFTY.'
    try:
        master = load_angel_master()
        required = {'symbol', 'name', 'exch_seg', 'instrumenttype', 'expiry', 'strike', 'token'}
        if not required.issubset(master.columns):
            return pd.DataFrame(), 'Instrument master does not contain all option-contract fields.'
        frame = master.copy()
        frame = frame[frame['exch_seg'].astype(str).str.upper().eq('NFO')]
        frame = frame[frame['instrumenttype'].astype(str).str.upper().str.contains('OPTIDX', na=False)]
        wanted = 'BANKNIFTY' if index_name == 'BANK NIFTY' else 'NIFTY'
        frame = frame[frame['name'].astype(str).str.upper().str.replace(' ', '', regex=False).eq(wanted)].copy()
        if frame.empty:
            return pd.DataFrame(), f'No {wanted} index-option contracts found in Angel One master.'
        frame['_expiry'] = pd.to_datetime(frame['expiry'], errors='coerce', dayfirst=True)
        frame = frame[frame['_expiry'] >= pd.Timestamp.now().normalize()]
        if frame.empty:
            return pd.DataFrame(), f'No unexpired {wanted} option contracts found.'
        expiry = frame['_expiry'].min()
        frame = frame[frame['_expiry'].eq(expiry)].copy()
        frame['_strike'] = pd.to_numeric(frame['strike'], errors='coerce')
        if frame['_strike'].max() > float(spot_price) * 20:
            frame['_strike'] = frame['_strike'] / 100.0
        sym = frame['symbol'].astype(str).str.upper()
        frame['_otype'] = np.where(sym.str.endswith('CE'), 'CE', np.where(sym.str.endswith('PE'), 'PE', ''))
        frame = frame[frame['_otype'].isin(['CE', 'PE'])].dropna(subset=['_strike'])
        strikes = sorted(frame['_strike'].drop_duplicates().tolist(), key=lambda x: abs(x-float(spot_price)))[:max_strikes]
        frame = frame[frame['_strike'].isin(strikes)].sort_values('_strike').head(50)
        if frame.empty:
            return pd.DataFrame(), 'No option strikes found around the current spot price.'
        tokens = frame['token'].astype(str).tolist()
        response = api.getMarketData('FULL', {'NFO': tokens})
        payload = (response or {}).get('data') or {}
        fetched = payload.get('fetched') or []
        if not fetched:
            return pd.DataFrame(), f"Angel One returned no option quotes: {(response or {}).get('message', 'No quote data')}"
        by_token = {str(q.get('symbolToken', q.get('token', ''))): q for q in fetched}
        rows=[]
        for _, row in frame.iterrows():
            q=by_token.get(str(row['token']), {})
            oi=q.get('opnInterest', q.get('openInterest', q.get('open_interest', np.nan)))
            rows.append({'Expiry':str(expiry.date()),'Strike':float(row['_strike']),'Type':row['_otype'],
                         'Symbol':str(row['symbol']),'Token':str(row['token']),
                         'LTP':safe_float(q.get('ltp', np.nan)), 'OI':safe_float(oi)})
        result=pd.DataFrame(rows)
        if result.empty or result['OI'].isna().all():
            return pd.DataFrame(), 'Quotes returned without OI. Check Angel One FULL market-data permissions.'
        key=f"option_oi_{wanted}"
        history=st.session_state.get(key, [])
        previous=history[-1]['quotes'] if history else {}
        result['Change in OI']=result.apply(lambda r: r['OI']-previous.get(r['Token'],r['OI']) if pd.notna(r['OI']) else np.nan,axis=1)
        quote_map={r['Token']:r['OI'] for _,r in result.iterrows() if pd.notna(r['OI'])}
        now=datetime.now()
        history.append({'time':now,'quotes':quote_map})
        history=[h for h in history if now-h['time']<=pd.Timedelta(minutes=30)]
        st.session_state[key]=history
        note=f"Angel One FULL option quotes · expiry {expiry.date()} · OI change versus previous successful refresh."
        st.session_state[cache_key]=result
        st.session_state[cache_key+'_time']=time.time()
        st.session_state[cache_key+'_note']=note
        return result, note
    except Exception as exc:
        return pd.DataFrame(), f'Option-chain request failed: {exc}'


def option_chain_metrics(chain):
    if chain is None or chain.empty or 'OI' not in chain:
        return {'PCR':np.nan,'Call OI':np.nan,'Put OI':np.nan,'Call OI Change':np.nan,'Put OI Change':np.nan,'Net OI Change':np.nan,'Bias':'Not assessed'}
    ce=chain[chain['Type'].eq('CE')]; pe=chain[chain['Type'].eq('PE')]
    call=ce['OI'].sum(min_count=1); put=pe['OI'].sum(min_count=1)
    cchg=ce['Change in OI'].sum(min_count=1); pchg=pe['Change in OI'].sum(min_count=1)
    pcr=put/call if pd.notna(call) and call>0 and pd.notna(put) else np.nan
    net=pchg-cchg if pd.notna(pchg) and pd.notna(cchg) else np.nan
    bias='Bullish' if pd.notna(pcr) and pd.notna(net) and pcr>=1.15 and net>0 else 'Bearish' if pd.notna(pcr) and pd.notna(net) and pcr<=0.85 and net<0 else 'Mixed' if pd.notna(pcr) else 'Not assessed'
    return {'PCR':pcr,'Call OI':call,'Put OI':put,'Call OI Change':cchg,'Put OI Change':pchg,'Net OI Change':net,'Bias':bias}


def render_option_chain_panel(chain, metrics, note):
    st.markdown('#### Angel One Live Option Chain / OI')
    st.caption(note)
    if chain is None or chain.empty:
        st.info('Live PCR/OI unavailable. No values are fabricated; manual context remains available in the sidebar.')
        return
    c=st.columns(4)
    c[0].metric('PCR (displayed strikes)',fmt(metrics['PCR']))
    c[1].metric('Call OI',fmt(metrics['Call OI'],0))
    c[2].metric('Put OI',fmt(metrics['Put OI'],0))
    c[3].metric('Net OI change (PE−CE)',fmt(metrics['Net OI Change'],0))
    st.dataframe(chain[['Strike','Type','LTP','OI','Change in OI']],use_container_width=True,hide_index=True)

def safe_float(value, default=np.nan):
    try:
        return float(value) if pd.notna(value) and np.isfinite(float(value)) else default
    except Exception:
        return default


def analyze_signal(d):
    """Preserved app.py signal-analysis contract, adapted to the base file's lower-case OHLC columns.

    The base get_signal() function remains the live signal engine used by the dashboard;
    this function retains the supplied app.py analysis API without replacing it.
    """
    if d is None or len(d) < 30:
        return {"signal": "WAIT", "score": 0, "confidence": 0,
                "reason": "Not enough candles for stable indicators."}
    a = d.iloc[-1]
    score, reasons = 0, []
    checks = [
        (safe(a.get("ema20")) > safe(a.get("ema50")), 1, "EMA20 above EMA50", "EMA20 below EMA50"),
        (safe(a.get("close")) > safe(a.get("vwap")), 1, "Price above VWAP", "Price below VWAP"),
        (safe(a.get("macd")) > safe(a.get("macd_signal")), 1, "MACD bullish", "MACD bearish"),
        (safe(a.get("rsi")) > 52, 1, "RSI bullish", "RSI not bullish"),
        (safe(a.get("di_plus")) > safe(a.get("di_minus")) and safe(a.get("adx")) >= 18, 1,
         "ADX/DI bullish trend", "ADX/DI not bullish"),
        (safe(a.get("close")) > safe(a.get("bb_mid")), 1, "Above Bollinger midline", "Below Bollinger midline"),
        (safe(a.get("vol_ratio")) >= 1.2, 1, "Volume above 1.2x average", "No strong volume expansion"),
    ]
    for condition, weight, bull_reason, bear_reason in checks:
        if pd.notna(condition) and condition:
            score += weight
            reasons.append(bull_reason)
        else:
            score -= weight
            reasons.append(bear_reason)
    current_patterns = patterns(d)
    if any("Bullish Engulfing" in x or "Hammer" in x for x in current_patterns): score += 1
    if any("Bearish Engulfing" in x or "Shooting Star" in x for x in current_patterns): score -= 1
    signal = "BUY" if score >= 3 else "SELL" if score <= -3 else "WAIT"
    confidence = int(min(95, abs(score) / 8 * 100))
    entry = safe(a.get("close")); atr_value = safe(a.get("atr"), entry * 0.005)
    if not np.isfinite(atr_value) or atr_value <= 0: atr_value = entry * 0.005
    if signal == "BUY": sl, t1, t2, t3 = entry-1.5*atr_value, entry+1.5*atr_value, entry+2*atr_value, entry+3*atr_value
    elif signal == "SELL": sl, t1, t2, t3 = entry+1.5*atr_value, entry-1.5*atr_value, entry-2*atr_value, entry-3*atr_value
    else: sl = t1 = t2 = t3 = np.nan
    return {"signal":signal,"score":score,"confidence":confidence,"reasons":reasons,
            "entry":entry,"sl":sl,"t1":t1,"t2":t2,"t3":t3,"atr":atr_value,
            "rr":"1:1 / 1:1.33 / 1:2 (ATR-based)","patterns":current_patterns}

def walk_forward_validation(frame, rr=2.0, atrmult=1.5, cost_bps=5.0, folds=4):
    if frame is None or len(frame)<240: return pd.DataFrame(), {"Status":"Insufficient history","Folds":0}
    d=norm(frame); chunks=[]; size=max(60,len(d)//(folds+1))
    for i in range(1,folds+1):
        train_end=size*i; test_end=min(len(d),train_end+size)
        if test_end-train_end<30: continue
        test=d.iloc[:test_end].copy()
        # Evaluate only the forward segment; no future rows are used to form its signals.
        trades,stats=backtest(test,rr,atrmult,cost_bps)
        if not trades.empty:
            start=str(d.index[train_end]); trades=trades[pd.to_datetime(trades["Entry Time"],errors="coerce")>=pd.Timestamp(d.index[train_end])]
        else: trades=pd.DataFrame()
        chunks.append({"Fold":i,"Train End":str(d.index[train_end-1]),"Test End":str(d.index[test_end-1]),
                       "Trades":len(trades),"Win Rate %":float((trades["Net Return %"]>0).mean()*100) if not trades.empty else 0.0,
                       "Net Return %":float(((1+trades["Net Return %"]/100).prod()-1)*100) if not trades.empty else 0.0})
    result=pd.DataFrame(chunks)
    return result,{"Status":"Completed" if not result.empty else "No test folds","Folds":len(result)}

def backtest(x,rr,atrmult,cost_bps):
    d=indicators(x); trades=[]; pos=None; fee=cost_bps/10000
    if len(d)<60:return pd.DataFrame(),{'Total Trades':0,'Wins':0,'Losses':0,'Win Rate %':0,'Net Return %':0,'Profit Factor':np.nan,'Max Drawdown %':0}
    for i in range(55,len(d)-1):
        r=d.iloc[i]; nxt=d.iloc[i+1]; sig,_,_,_=get_signal(r)
        if pos:
            stop_hit=(r.low<=pos['stop']) if pos['side']=='LONG' else (r.high>=pos['stop'])
            target_hit=(r.high>=pos['target']) if pos['side']=='LONG' else (r.low<=pos['target'])
            exitp=reason=None
            if stop_hit:exitp=pos['stop'];reason='Stop Loss'
            elif target_hit:exitp=pos['target'];reason='Target'
            elif (pos['side']=='LONG' and sig=='SELL') or (pos['side']=='SHORT' and sig=='BUY'):exitp=r.close;reason='Opposite signal'
            else:
                av=safe(r.atr)
                if np.isfinite(av) and av>0:
                    pos['stop']=max(pos['stop'],r.close-atrmult*av) if pos['side']=='LONG' else min(pos['stop'],r.close+atrmult*av)
            if exitp is not None:
                gross=(exitp-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-exitp)/pos['entry']; net=(gross-2*fee)*100
                trades.append({'Entry Time':pos['time'],'Exit Time':str(d.index[i]),'Side':pos['side'],'Entry':pos['entry'],'Exit':exitp,'Net Return %':net,'Result':'Win' if net>0 else 'Loss','Exit Reason':reason});pos=None
        if pos is None and sig in ('BUY','SELL'):
            av=safe(r.atr); ep=safe(nxt.open)
            if not np.isfinite(av) or av<=0 or not np.isfinite(ep) or ep<=0:continue
            side='LONG' if sig=='BUY' else 'SHORT'; risk=av*atrmult
            pos={'side':side,'entry':ep,'stop':ep-risk if side=='LONG' else ep+risk,'target':ep+risk*rr if side=='LONG' else ep-risk*rr,'time':str(d.index[i+1])}
    # Mark any remaining position to the final available close so the report
    # does not silently omit an open trade at the end of the test window.
    if pos is not None and len(d):
        final_price=safe(d.close.iloc[-1])
        if np.isfinite(final_price) and final_price>0:
            gross=(final_price-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-final_price)/pos['entry']
            net=(gross-2*fee)*100
            trades.append({'Entry Time':pos['time'],'Exit Time':str(d.index[-1]),'Side':pos['side'],'Entry':pos['entry'],'Exit':final_price,'Net Return %':net,'Result':'Win' if net>0 else 'Loss','Exit Reason':'End of data (mark-to-market)'})
    t=pd.DataFrame(trades)
    if t.empty:return t,{'Total Trades':0,'Wins':0,'Losses':0,'Win Rate %':0,'Net Return %':0,'Profit Factor':np.nan,'Max Drawdown %':0}
    wins=int((t['Net Return %']>0).sum()); losses=len(t)-wins; gp=t.loc[t['Net Return %']>0,'Net Return %'].sum(); gl=abs(t.loc[t['Net Return %']<=0,'Net Return %'].sum()); eq=(1+t['Net Return %']/100).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    return t,{'Total Trades':len(t),'Wins':wins,'Losses':losses,'Win Rate %':wins/len(t)*100,'Net Return %':(eq.iloc[-1]-1)*100,'Profit Factor':gp/gl if gl else np.inf,'Max Drawdown %':abs(dd)}

# Sidebar controls
st.sidebar.header('Market Controls')
market=st.sidebar.selectbox('Instrument',list(SYMBOLS)+['Custom Yahoo ticker'])
ticker=st.sidebar.text_input('Yahoo ticker',value='RELIANCE.NS') if market=='Custom Yahoo ticker' else SYMBOLS[market]
chart_tf=st.sidebar.selectbox('Chart timeframe',['1 minute','3 minutes','5 minutes','15 minutes','30 minutes','60 minutes','1 day'],index=2)
period_label=st.sidebar.selectbox('History',['1 day','5 days','1 month','3 months','6 months','1 year','2 years'],index=1)
periods={'1 day':'1d','5 days':'5d','1 month':'1mo','3 months':'3mo','6 months':'6mo','1 year':'1y','2 years':'2y'}
period=periods[period_label]; intervals={'1 minute':'1m','3 minutes':'1m','5 minutes':'5m','15 minutes':'15m','30 minutes':'30m','60 minutes':'60m','1 day':'1d'}; interval=intervals[chart_tf]
if chart_tf in ('1 minute','3 minutes') and period not in ('1d','5d'): period='5d'
# Independent providers: candle history, displayed live LTP, and supplementary market context.
candle_source=st.sidebar.radio('Candle / History Data Feed',['Yahoo Finance','Angel One API'],index=0,
    help='Controls OHLC candles, chart, indicators, backtest and daily support/resistance.')
live_price_feed=st.sidebar.radio('Live Price Feed',['REST Polling (Primary) — candle-feed price','Angel One WebSocket LTP (Optional)'],index=0,
    help='WebSocket supplies live LTP only. Candle OHLC and indicators continue to use the selected candle/history feed.')
other_feed_choice=st.sidebar.radio('Other Data Feed (MTF / PCR / OI)',['Same as Candle Feed','Yahoo Finance','Angel One API'],index=0,
    help='Controls supplementary MTF and supported option-chain/PCR-OI feed. Global/sector boards remain Yahoo Finance; FII/DII and news/event context are manual.')
other_source=candle_source if other_feed_choice=='Same as Candle Feed' else other_feed_choice
# Compatibility alias for captions and downstream UI.
data_source=candle_source
feed_mode='REST polling (Primary)' if live_price_feed.startswith('REST Polling') else 'WebSocket live ticks (Optional)'
needs_angel=(candle_source=='Angel One API' or other_source=='Angel One API' or live_price_feed.startswith('Angel One WebSocket'))
angel_interval={'1 minute':'ONE_MINUTE','3 minutes':'THREE_MINUTE','5 minutes':'FIVE_MINUTE','15 minutes':'FIFTEEN_MINUTE','30 minutes':'THIRTY_MINUTE','60 minutes':'ONE_HOUR','1 day':'ONE_DAY'}[chart_tf]
angel_search = ''
lookback_days = st.sidebar.selectbox('Angel One history days',[1,3,5,10,20,30],index=2) if needs_angel else 5
if needs_angel:
    st.sidebar.success('Symbol token and exchange will be looked up automatically from Angel One instrument master.')
    if market == 'Custom Yahoo ticker':
        angel_search = st.sidebar.text_input('Angel One instrument search (symbol/name)', value='RELIANCE')
    st.sidebar.warning('Only enter your own Angel One API credentials. Do not share API key, Client Code, PIN or TOTP secret. Credentials remain in this app session.')
    with st.sidebar.form('angel_connect_form'):
        akey=st.text_input('Angel One API Key',type='password')
        client=st.text_input('Client Code')
        apin=st.text_input('Trading PIN',type='password')
        totp_secret=st.text_input('TOTP Secret',type='password')
        connect_angel=st.form_submit_button('Connect Angel One')
    if connect_angel:
        try:
            import pyotp
            from SmartApi import SmartConnect
            if not all([akey.strip(), client.strip(), apin.strip(), totp_secret.strip()]):
                raise RuntimeError('Fill API Key, Client Code, Trading PIN and TOTP Secret.')
            api=SmartConnect(api_key=akey.strip())
            response=api.generateSession(client.strip(), apin.strip(), pyotp.TOTP(totp_secret.strip()).now())
            if not response or response.get('status') is False:
                raise RuntimeError((response or {}).get('message') or 'Login failed. Check credentials and TOTP.')
            auth_data=(response or {}).get('data') or {}
            auth_token=auth_data.get('jwtToken') or auth_data.get('jwt_token')
            feed_token=api.getfeedToken()
            if not auth_token or not feed_token:
                raise RuntimeError('Angel One login succeeded but JWT/feed token was not returned; WebSocket cannot start.')
            st.session_state.angel_api=api
            st.session_state.angel_auth_token=auth_token
            st.session_state.angel_feed_token=feed_token
            st.session_state.angel_api_key=akey.strip()
            st.session_state.angel_connected=True
            st.session_state.angel_client_code=client.strip()
            st.session_state.angel_candles=None
            st.session_state.angel_candle_key=None
            st.sidebar.success('Angel One connected. Resolving symbol and token automatically…')
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f'Angel One connection failed: {exc}')
    if st.sidebar.button('Refresh Angel One instrument master'):
        load_angel_master.clear()
        st.rerun()
auto=st.sidebar.toggle('Auto refresh ON/OFF',value=True,help='ON by default so primary REST polling refreshes automatically. Turn OFF to pause automatic polling.'); secs=st.sidebar.selectbox('Refresh interval (seconds)',[5,10,15,30,60],index=2)
if auto or (live_price_feed.startswith('Angel One WebSocket')):
    if st_autorefresh:st_autorefresh(interval=secs*1000,key='refresh')
    else:st.sidebar.error('Install streamlit-autorefresh to enable automatic refresh and display incoming WebSocket ticks.')
if st.sidebar.button('Refresh now',type='primary'):
    st.cache_data.clear()
    st.session_state.force_angel_refresh=True
    st.rerun()
rr=st.sidebar.selectbox('Target risk : reward',[1.0,1.5,2.0,3.0],index=2); atr_mult=st.sidebar.slider('ATR stop multiplier',0.5,4.0,1.5,0.25); risk_pct=st.sidebar.slider('Paper risk per trade (%)',0.25,2.0,1.0,0.25); capital=st.sidebar.number_input('Paper capital (₹)',min_value=1000.0,value=100000.0,step=5000.0)
st.sidebar.divider()
st.sidebar.subheader('Manual context data (if unavailable from feed)')
pcr=st.sidebar.number_input('PCR',0.0,10.0,1.0,.05)
pcr_delta=st.sidebar.number_input('PCR change',value=0.0,step=.01)
pcr_bias=st.sidebar.selectbox('PCR interpretation',['Not assessed','Bullish','Bearish','Mixed'])
call_oi=st.sidebar.number_input('Call OI change (%)',value=0.0,step=1.0)
put_oi=st.sidebar.number_input('Put OI change (%)',value=0.0,step=1.0)
fii=st.sidebar.number_input('FII/FPI net (₹ crore)',value=0.0,step=100.0)
dii=st.sidebar.number_input('DII net (₹ crore)',value=0.0,step=100.0)
oi_bias=st.sidebar.selectbox('OI interpretation',['Not assessed','Bullish','Bearish','Mixed'])
news=st.sidebar.selectbox('Verified news bias',['Neutral/unknown','Positive','Negative','Mixed'])
event=st.sidebar.selectbox('Government/geopolitical event risk',['Unknown/not checked','Low','Moderate','High'])

# Main data fetch: each selector controls only its own data domain.
selected_yahoo_ticker=ticker
live_tick=None
instrument=None
if candle_source=='Yahoo Finance':
    with st.spinner('Fetching Yahoo Finance candles...'):
        raw=yahoo(ticker,period,interval)
        data=resample(raw,'3min') if chart_tf=='3 minutes' else raw
else:
    data=st.session_state.get('angel_candles')

if needs_angel:
    if not st.session_state.get('angel_connected') or st.session_state.get('angel_api') is None:
        st.info('An Angel One feed is selected. Connect Angel One in the sidebar; the symbol/token will be resolved automatically.')
        st.stop()
    try:
        master=load_angel_master()
        instrument=resolve_angel_instrument(master, market, angel_search)
        st.session_state.angel_resolved_instrument=instrument
        st.sidebar.caption(f"Angel instrument: {instrument['symbol']} • {instrument['exchange']} • token auto-loaded")
        if candle_source=='Angel One API':
            candle_key=(instrument['exchange'], instrument['symbol'], str(instrument['token']), angel_interval, int(lookback_days))
            now_ts=time.time()
            old_key=st.session_state.get('angel_candle_key')
            last_fetch=st.session_state.get('angel_last_fetch_ts',0)
            force_refresh=st.session_state.pop('force_angel_refresh', False)
            should_fetch=(data is None or data.empty or old_key != candle_key or force_refresh or (auto and now_ts-last_fetch>=secs))
            if should_fetch:
                with st.spinner(f"Polling Angel One candles for {instrument['symbol']}…"):
                    data=fetch_angel_candles(st.session_state.angel_api, instrument, angel_interval, lookback_days)
                    st.session_state.angel_candles=data
                    st.session_state.angel_candle_key=candle_key
                    st.session_state.angel_last_fetch_ts=now_ts
            else:
                data=st.session_state.angel_candles
            ticker=instrument['symbol']

        # WebSocket is an independent optional LTP feed; it does not replace candle history.
        if live_price_feed.startswith('Angel One WebSocket'):
            ws_key=(st.session_state.get('angel_client_code'), instrument['exchange'], str(instrument['token']))
            old_ws_key=st.session_state.get('angel_ws_key')
            ws_feed=st.session_state.get('angel_ws_feed')
            if old_ws_key != ws_key or ws_feed is None:
                if ws_feed is not None: ws_feed.stop()
                ws_feed=AngelWebSocketFeed(
                    st.session_state.get('angel_api_key',''), st.session_state.get('angel_client_code',''),
                    st.session_state.get('angel_auth_token',''), st.session_state.get('angel_feed_token',''), instrument)
                st.session_state.angel_ws_feed=ws_feed
                st.session_state.angel_ws_key=ws_key
                ws_feed.start()
            else:
                ws_feed.start()
            live_tick=ws_feed.latest_tick()
            if live_tick and np.isfinite(live_tick.get('ltp', np.nan)) and live_tick['ltp']>0:
                st.session_state.angel_live_tick=live_tick
            else:
                live_tick=st.session_state.get('angel_live_tick')
            if ws_feed.error: st.sidebar.warning(f'WebSocket status: {ws_feed.error}')
            elif ws_feed.connected: st.sidebar.success('Angel One WebSocket connected • live LTP')
            else: st.sidebar.info('Angel One WebSocket connecting… Auto refresh displays incoming ticks.')
        else:
            old_feed=st.session_state.get('angel_ws_feed')
            if old_feed is not None: old_feed.stop()
            st.session_state.angel_ws_feed=None
            st.session_state.angel_ws_key=None
            live_tick=None
    except Exception as exc:
        st.error(f'Angel One feed error: {exc}')
        st.stop()
else:
    old_feed=st.session_state.get('angel_ws_feed')
    if old_feed is not None: old_feed.stop()
    st.session_state.angel_ws_feed=None
    st.session_state.angel_ws_key=None

if data.empty:
    st.error(f'No data returned for {ticker}. Verify ticker/network/interval or broker symbol token.');st.stop()
data=indicators(data); last=data.iloc[-1]; candle_close=safe(last.close); price=candle_close
if live_price_feed.startswith('Angel One WebSocket'):
    _tick=st.session_state.get('angel_live_tick')
    if _tick and np.isfinite(safe(_tick.get('ltp'))) and safe(_tick.get('ltp'))>0:
        price=safe(_tick.get('ltp'))
if not np.isfinite(price):st.error('Latest price unavailable.');st.stop()
analysis_yahoo_ticker=SYMBOLS.get(market,ticker) if candle_source=='Angel One API' else ticker
if candle_source=='Angel One API':
    daily,levels=daily_info_angel(st.session_state.angel_api,instrument)
else:
    daily,levels=daily_info(ticker)
if other_source=='Angel One API':
    mtf_table=mtf_angel(st.session_state.angel_api,instrument)
else:
    other_yahoo_ticker=SYMBOLS.get(market, selected_yahoo_ticker)
    mtf_table=mtf(other_yahoo_ticker)
pats=patterns(data)
# If Angel One is connected and an index is selected, use real option quotes when OI is present.
option_chain=pd.DataFrame(); option_note='Option chain not requested for this data-source/instrument selection.'
if other_source=='Angel One API' and market in ('NIFTY 50','BANK NIFTY'):
    option_chain,option_note=get_option_chain_snapshot(market,price)
option_metrics=option_chain_metrics(option_chain)
if pd.notna(option_metrics['PCR']):
    pcr=float(option_metrics['PCR'])
    pcr_bias=option_metrics['Bias']
    oi_bias=option_metrics['Bias']
# Manual values are used only when live option-chain data is unavailable.
mtf_bull_count=int((mtf_table.get('Trend',pd.Series(dtype=str))=='Bullish').sum()) if not mtf_table.empty else 0
mtf_bear_count=int((mtf_table.get('Trend',pd.Series(dtype=str))=='Bearish').sum()) if not mtf_table.empty else 0
mtf_aligned_count=max(mtf_bull_count,mtf_bear_count)
signal,bull,bear,reasons=get_signal(last)
# Transparent dashboard quality metrics (scores are diagnostics, not win probabilities).
validation_score=int(min(100, max(0, round((bull+bear)/14*100))))
trade_quality_score=int(min(100,max(0,round(abs(bull-bear)/max(1,bull+bear)*100))))
health_score=int(min(100,max(0,round((50 if safe(last.adx)>=MANDATORY_ADX else 25)+(25 if safe(last.ema200) and price>safe(last.ema200) else 10)+(25 if safe(last.supertrend_direction)>0 else 10)))))
strategy_grade='A' if validation_score>=80 and trade_quality_score>=70 else 'B' if validation_score>=60 else 'C'
# Current signal only: apply manually supplied options/institutional/news context.
# Historical backtest intentionally does not reuse today's manual context for past candles.
if pcr_bias=='Bullish': bull+=1; reasons.append('Manual PCR context: bullish')
elif pcr_bias=='Bearish': bear+=1; reasons.append('Manual PCR context: bearish')
if oi_bias=='Bullish': bull+=1; reasons.append('Manual OI context: bullish')
elif oi_bias=='Bearish': bear+=1; reasons.append('Manual OI context: bearish')
if news=='Positive': bull+=1; reasons.append('Manual news context: positive')
elif news=='Negative': bear+=1; reasons.append('Manual news context: negative')
if event=='High':
    signal='WAIT'; reasons.append('High event risk: confirmation withheld')
else:
    signal='BUY' if bull>=4 and bull>=bear+2 else 'SELL' if bear>=4 and bear>=bull+2 else 'WAIT'
# Production confirmation gates: ADX threshold and at least two aligned MTF frames.
if signal in ('BUY','SELL') and safe(last.adx)<MANDATORY_ADX:
    signal='WAIT'; reasons.append(f'ADX gate: ADX below {MANDATORY_ADX}')
if signal=='BUY' and (mtf_bull_count < MANDATORY_MTF_ALIGN_COUNT):
    signal='WAIT'; reasons.append(f'MTF gate: bullish alignment {mtf_bull_count}/{MANDATORY_MTF_ALIGN_COUNT} required')
if signal=='SELL' and (mtf_bear_count < MANDATORY_MTF_ALIGN_COUNT):
    signal='WAIT'; reasons.append(f'MTF gate: bearish alignment {mtf_bear_count}/{MANDATORY_MTF_ALIGN_COUNT} required')
if signal=='BUY' and DISABLE_BUY_BIAS_TRADING:
    signal='WAIT'; reasons.append('BUY bias disabled by production setting')
if signal=='SELL' and DISABLE_SELL_BIAS_TRADING:
    signal='WAIT'; reasons.append('SELL bias disabled by production setting')
if STRICT_TRADE_ONLY_STRONG and strategy_grade!='A':
    signal='WAIT'; reasons.append('Strict mode: only Grade A setups are allowed')
audit_entry={'Time':str(data.index[-1]),'Ticker':ticker,'Signal':signal,'Bullish Score':bull,'Bearish Score':bear,'Reasons':' | '.join(reasons)}
if (not st.session_state.signal_audit or
    any(st.session_state.signal_audit[-1].get(k) != audit_entry.get(k) for k in ('Time', 'Ticker', 'Signal'))):
    st.session_state.signal_audit.append(audit_entry)
st.session_state.signal_audit=st.session_state.signal_audit[-500:]
atr=safe(last.atr)
if not np.isfinite(atr) or atr<=0:atr=price*.005
entry=price; stop=price-atr_mult*atr if signal=='BUY' else price+atr_mult*atr if signal=='SELL' else np.nan; target=price+atr_mult*atr*rr if signal=='BUY' else price-atr_mult*atr*rr if signal=='SELL' else np.nan

st.subheader('1. Current Signal')
c=st.columns(5); c[0].metric('Instrument',market);c[1].metric('Last Price',f'{price:,.2f}');c[2].metric('Signal',signal);c[3].metric('Bullish Score',bull);c[4].metric('Bearish Score',bear)
st.caption(f'Candle/history: {candle_source} • Live price: {live_price_feed} • Other/MTF: {other_source} • Candle close: {candle_close:,.2f} • Indicators use candle data ({data.index[-1]}); displayed price may be a newer WebSocket LTP. Scores are rule alignment, not a calibrated winning probability.')
if live_price_feed.startswith('Angel One WebSocket'):
    _tick=st.session_state.get('angel_live_tick')
    if _tick:
        st.caption(f"WebSocket LTP timestamp: {_tick.get('timestamp')} • Historical candles/indicators refresh separately through REST.")
    else:
        st.warning('Waiting for first WebSocket tick. Last Price temporarily shows the latest REST candle close.')
st.write('Signal notes: '+' • '.join(reasons) if reasons else 'Signal notes: mixed/insufficient confirmation.')
quality_cols=st.columns(4)
quality_cols[0].metric('Validation Score',f'{validation_score}/100')
quality_cols[1].metric('Strategy Grade',strategy_grade)
quality_cols[2].metric('Trade Quality Score',f'{trade_quality_score}/100')
quality_cols[3].metric('Health Score',f'{health_score}/100')
st.caption('Quality scores are rule-based diagnostics. They are not calibrated win probabilities or guarantees.')
trend_now='Bullish' if last.ema20>last.ema50 else 'Bearish' if last.ema20<last.ema50 else 'Sideways'
ctx=st.columns(6);ctx[0].metric('Trend',trend_now);ctx[1].metric('Volume ratio',fmt(last.vol_ratio));ctx[2].metric('PCR (manual)',f'{pcr:.2f}');ctx[3].metric('Call OI change',f'{call_oi:+.1f}%');ctx[4].metric('Put OI change',f'{put_oi:+.1f}%');ctx[5].metric('ADX',fmt(last.adx))

st.subheader('2. Entry, Exit, Target, Stop Loss & Trailing Stop')
c=st.columns(5);c[0].metric('Entry',fmt(entry) if signal!='WAIT' else 'WAIT');c[1].metric('Stop Loss',fmt(stop));c[2].metric('Target',fmt(target));c[3].metric('ATR',fmt(atr));c[4].metric('Risk : Reward',f'1 : {rr:g}' if signal!='WAIT' else 'N/A')
trail=price-atr_mult*atr if signal=='BUY' else price+atr_mult*atr if signal=='SELL' else np.nan;st.metric('Indicative Trailing Stop',fmt(trail));st.caption('Trailing stop is an indicative calculation; it is not sent to any broker.')

st.subheader('3. Japanese Candlestick & Heikin-Ashi Chart')
st.caption('Japanese OHLC candlesticks and Heikin-Ashi are both available. Raw Japanese candles are used for price levels; Heikin-Ashi is a smoothed view and should not replace executable prices.')
chart_mode=st.radio('Candle view',['Japanese Candlestick','Heikin-Ashi','Both'],horizontal=True,index=0)
choices=st.multiselect('Overlay indicators',['EMA20','EMA50','EMA200','VWAP','Bollinger Upper','Bollinger Mid','Bollinger Lower','SMA20','SMA50','SMA100','SMA200'],default=['EMA20','EMA50','EMA200','VWAP'])
cmap={'EMA20':'ema20','EMA50':'ema50','EMA200':'ema200','VWAP':'vwap','Bollinger Upper':'bb_upper','Bollinger Mid':'bb_mid','Bollinger Lower':'bb_lower','SMA20':'sma20','SMA50':'sma50','SMA100':'sma100','SMA200':'sma200'}
chart_data=data.tail(300)
def make_candle_figure(frame, heikin=False):
    fig=go.Figure()
    if heikin:
        opn,hig,low,cls='ha_open','ha_high','ha_low','ha_close'
        title='Heikin-Ashi Candles'
    else:
        opn,hig,low,cls='open','high','low','close'
        title='Japanese Candlestick (OHLC)'
    fig.add_trace(go.Candlestick(x=frame.index,open=frame[opn],high=frame[hig],low=frame[low],close=frame[cls],name=title))
    for label in choices:
        col=cmap[label]
        if col in frame:
            fig.add_trace(go.Scatter(x=frame.index,y=frame[col],mode='lines',name=label,line={'width':1}))
    fig.update_layout(title=title,height=560,xaxis_rangeslider_visible=False,legend_orientation='h',margin={'l':10,'r':10,'t':45,'b':10})
    return fig
if chart_mode in ('Japanese Candlestick','Both'):
    st.plotly_chart(make_candle_figure(chart_data,False),use_container_width=True)
if chart_mode in ('Heikin-Ashi','Both'):
    st.plotly_chart(make_candle_figure(chart_data,True),use_container_width=True)
st.markdown('**MACD (12, 26, 9)**')
st.line_chart(data[['macd','macd_signal']].tail(300),use_container_width=True)
st.bar_chart(data[['macd_hist']].tail(100),use_container_width=True)
c=st.columns(8);c[0].metric('RSI',fmt(last.rsi));c[1].metric('ADX',fmt(last.adx));c[2].metric('DI+',fmt(last.di_plus));c[3].metric('DI−',fmt(last.di_minus));c[4].metric('Stochastic %K',fmt(last.stoch_k));c[5].metric('Volume / Avg',fmt(last.vol_ratio));c[6].metric('MACD',fmt(last.macd));c[7].metric('MACD Signal',fmt(last.macd_signal))
bbcols=st.columns(3);bbcols[0].metric('Bollinger Upper',fmt(last.bb_upper));bbcols[1].metric('Bollinger Mid',fmt(last.bb_mid));bbcols[2].metric('Bollinger Lower',fmt(last.bb_lower))

st.subheader('4. Candlestick Patterns')
st.write('Latest detected: '+' • '.join(pats));st.caption('Pattern rules are simplified. Confirm with context; a candle pattern alone is not a trade recommendation.')

st.subheader('5. Multi-Timeframe Analysis (1m / 3m / 5m / 15m)')
st.dataframe(mtf_table,use_container_width=True,hide_index=True)
st.caption(f'MTF alignment: bullish {mtf_bull_count}/4 • bearish {mtf_bear_count}/4. BUY/SELL requires at least {MANDATORY_MTF_ALIGN_COUNT} aligned frames. MTF source: {other_source} (independent supplementary feed).')

st.subheader('6. Previous Close, Support / Resistance & DMA')
if levels:
    rows=[]
    for k,v in levels.items():rows.append({'Level':k,'Price':v,'Type':'Support candidate' if np.isfinite(v) and v<price else 'Resistance candidate' if np.isfinite(v) and v>price else 'Near price / N/A'})
    if len(daily)>1:
        p=(daily.high.iloc[-2]+daily.low.iloc[-2]+daily.close.iloc[-2])/3;rows += [{'Level':'Classic Pivot','Price':p,'Type':'Pivot'},{'Level':'Pivot R1','Price':2*p-daily.low.iloc[-2],'Type':'Resistance candidate'},{'Level':'Pivot S1','Price':2*p-daily.high.iloc[-2],'Type':'Support candidate'}]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    adr=(daily.high-daily.low).tail(14).mean() if len(daily)>1 else np.nan; adr_pct=adr/daily.close.iloc[-1]*100 if len(daily)>1 and daily.close.iloc[-1] else np.nan
    c=st.columns(3);c[0].metric('ADR (14-day average range)',fmt(adr));c[1].metric('ADR %',fmt(adr_pct)+'%');c[2].metric('Previous Close',fmt(levels.get('Previous Close')))
else:st.info('Daily levels unavailable.')

st.subheader('7. PCR, Option Chain, OI Change, FII / DII and Event Context')
if not option_chain.empty:
    render_option_chain_panel(option_chain,option_metrics,option_note)
else:
    st.info('Live PCR/OI is unavailable for the current provider/instrument. Sidebar PCR/OI values are manual context and are not treated as live exchange data.')
c=st.columns(6)
c[0].metric('PCR',fmt(pcr))
c[1].metric('PCR change (manual)',f'{pcr_delta:+.2f}')
if not option_chain.empty and pd.notna(option_metrics.get('Call OI Change')):
    c[2].metric('Call OI change (captured)',fmt(option_metrics['Call OI Change'],0))
    c[3].metric('Put OI change (captured)',fmt(option_metrics['Put OI Change'],0))
else:
    c[2].metric('Call OI change (manual %)',f'{call_oi:+.1f}%')
    c[3].metric('Put OI change (manual %)',f'{put_oi:+.1f}%')
c[4].metric('FII/FPI net',f'₹{fii:,.0f} cr')
c[5].metric('DII net',f'₹{dii:,.0f} cr')
st.write(f'PCR interpretation: {pcr_bias} • OI interpretation: {oi_bias} • Verified news bias: {news} • Government/geopolitical event risk: {event}')

st.subheader('8. Global Market Dashboard')
global_snapshot=fetch_market_snapshot(global_indices)
if not global_snapshot.empty:
    st.dataframe(global_snapshot.style.format({'CMP':'{:,.2f}','Change %':'{:+.2f}%'},na_rep='N/A'),use_container_width=True,hide_index=True)

st.subheader('9. Indian Market Dashboard')
india_snapshot=fetch_market_snapshot(indian_indices)
if not india_snapshot.empty:
    st.dataframe(india_snapshot.style.format({'CMP':'{:,.2f}','Change %':'{:+.2f}%'},na_rep='N/A'),use_container_width=True,hide_index=True)

st.subheader('10. Top 20 Scanner')
top20_snapshot=fetch_market_snapshot(major_stocks)
if not top20_snapshot.empty:
    st.dataframe(top20_snapshot.style.format({'CMP':'{:,.2f}','Change %':'{:+.2f}%'},na_rep='N/A'),use_container_width=True,hide_index=True)

st.subheader('11. F&O Style Scanner')
fno_scanner=fetch_fno_style_scanner()
fno_summary=summarize_fno_bias(fno_scanner)
f1,f2,f3,f4=st.columns(4)
f1.metric('Underlying Bias Proxy',fno_summary['Bias'])
f2.metric('Bullish Counters',fno_summary['Bullish'])
f3.metric('Bearish Counters',fno_summary['Bearish'])
f4.metric('Coverage',fno_summary['Coverage'])
if not fno_scanner.empty:
    st.dataframe(fno_scanner.style.format({'CMP':'{:,.2f}','Change %':'{:+.2f}%'},na_rep='N/A'),use_container_width=True,hide_index=True)
st.caption('F&O Style Scanner is an underlying-stock momentum proxy. It is not a live futures/options contract scanner and does not fabricate futures prices or OI.')

st.subheader('12. Historical Backtest')
st.caption('Simplified rule-based OHLC backtest: closed-bar signal, next-bar-open entry, ATR stop/target, one position at a time. Costs are an estimate; taxes, latency and exact fills are not fully modelled.')
cost=st.slider('Estimated one-way cost + slippage (basis points)',0.0,50.0,5.0,1.0)
if st.button('Run Backtest',type='primary'):
    trades,stats=backtest(data,float(rr),float(atr_mult),float(cost));st.session_state.bt_trades=trades;st.session_state.bt_stats=stats
if 'bt_stats' in st.session_state:
    q=st.session_state.bt_stats;c=st.columns(6);c[0].metric('Total Trades',q['Total Trades']);c[1].metric('Wins',q['Wins']);c[2].metric('Losses',q['Losses']);c[3].metric('Win Rate',f"{q['Win Rate %']:.2f}%");c[4].metric('Net Return',f"{q['Net Return %']:.2f}%");pf=q['Profit Factor'];c[5].metric('Profit Factor','∞' if pf==np.inf else fmt(pf));st.metric('Maximum Drawdown',f"{q['Max Drawdown %']:.2f}%")
    trades=st.session_state.bt_trades
    if not trades.empty:
        st.dataframe(trades.tail(200),use_container_width=True,hide_index=True);st.download_button('Download backtest CSV',trades.to_csv(index=False),'backtest_trades.csv','text/csv')
    else:st.info('No completed trades for this period/settings. Try a longer period or different ticker.')
    if not trades.empty:
        returns=pd.to_numeric(trades['Net Return %'],errors='coerce').fillna(0)/100
        equity=(1+returns).cumprod()
        st.line_chart(pd.DataFrame({'Equity Curve':equity.to_numpy()},index=trades['Exit Time'].astype(str)))
        sharpe=(returns.mean()/returns.std(ddof=1)*np.sqrt(252)) if returns.std(ddof=1)>0 else np.nan
        st.metric('Trade-return Sharpe (annualisation is approximate)',fmt(sharpe,3))
    wf,wf_stats=walk_forward_validation(data,float(rr),float(atr_mult),float(cost))
    st.markdown('#### Walk Forward Validation')
    st.caption(f"Status: {wf_stats['Status']} • Folds: {wf_stats['Folds']}")
    if not wf.empty: st.dataframe(wf,use_container_width=True,hide_index=True)

st.subheader('13. Paper Trading & P/L')
st.caption('Paper trades only. No real orders are placed.')
paper_history=pd.DataFrame(st.session_state.paper_trades)
weekly_limit=capital*0.05; monthly_limit=capital*0.10
weekly_pnl=monthly_pnl=0.0
if not paper_history.empty and 'Closed At' in paper_history:
    dates=pd.to_datetime(paper_history['Closed At'],errors='coerce')
    now=pd.Timestamp.now()
    weekly_pnl=float(pd.to_numeric(paper_history.loc[dates>=now-pd.Timedelta(days=7),'P&L ₹'],errors='coerce').fillna(0).sum())
    monthly_pnl=float(pd.to_numeric(paper_history.loc[dates>=now-pd.Timedelta(days=30),'P&L ₹'],errors='coerce').fillna(0).sum())
weekly_locked=weekly_pnl<=-weekly_limit; monthly_locked=monthly_pnl<=-monthly_limit
lock_cols=st.columns(4)
lock_cols[0].metric('Weekly Paper P&L',f'₹{weekly_pnl:,.2f}')
lock_cols[1].metric('Weekly Lock','ACTIVE' if weekly_locked else 'OFF')
lock_cols[2].metric('Monthly Paper P&L',f'₹{monthly_pnl:,.2f}')
lock_cols[3].metric('Monthly Lock','ACTIVE' if monthly_locked else 'OFF')
if weekly_locked or monthly_locked: st.error('Capital protection lock active: new paper entries are disabled until the rolling loss window recovers.')
if st.toggle('Show paper trading panel',value=True):
    auto_exit_enabled = st.toggle(
        'Enable Automatic Exit (Paper Trading Only)',
        value=st.session_state.get('paper_auto_exit_enabled', True),
        key='paper_auto_exit_enabled',
        help='When ON, the paper position is automatically closed in the app simulation when its stop-loss or target is touched. This is not a broker order.'
    )
    st.caption('Auto exit runs when the dashboard refreshes. REST polling must be ON for periodic checks; WebSocket mode also needs dashboard reruns to process updates.')
    pos=st.session_state.paper_position
    if pos is None:
        with st.form('paper_open'):
            side=st.selectbox('Direction',['LONG','SHORT']); pe=st.number_input('Entry price',min_value=.01,value=float(price)); qty=st.number_input('Quantity',min_value=1,value=1,step=1)
            default_sl=float(max(.01,price-atr_mult*atr)) if side=='LONG' else float(price+atr_mult*atr)
            default_tg=float(price+atr_mult*atr*rr) if side=='LONG' else float(max(.01,price-atr_mult*atr*rr))
            sl=st.number_input('Stop loss',min_value=.01,value=default_sl);tg=st.number_input('Target',min_value=.01,value=default_tg);openp=st.form_submit_button('Open Paper Position')
        if openp:
            if weekly_locked or monthly_locked:
                st.error('Weekly/monthly capital protection lock is active; no new paper position opened.')
                st.stop()
            valid=(sl<pe<tg) if side=='LONG' else (tg<pe<sl)
            if not valid:st.error('Stop/entry/target levels do not match selected direction.')
            else:
                cashrisk=abs(pe-sl)*qty
                if cashrisk>capital*risk_pct/100:
                    st.error(f'Capital protection blocked this entry: paper risk ₹{cashrisk:,.2f} exceeds budget ₹{capital*risk_pct/100:,.2f}.')
                else:
                    st.session_state.paper_position={'Symbol':ticker,'Side':side,'Entry':float(pe),'Quantity':int(qty),'Stop Loss':float(sl),'Target':float(tg),'Opened At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    st.rerun()
    else:
        side=pos['Side'];qty=pos['Quantity'];pnl=(price-pos['Entry'])*qty if side=='LONG' else (pos['Entry']-price)*qty;trail=price-atr_mult*atr if side=='LONG' else price+atr_mult*atr
        c=st.columns(4);c[0].metric('Open position',f"{side} {qty} {pos['Symbol']}");c[1].metric('Entry',fmt(pos['Entry']));c[2].metric('Unrealised P&L',f'₹{pnl:,.2f}');c[3].metric('Indicative trailing SL',fmt(trail));st.write(f"Initial/current paper SL: {pos['Stop Loss']:.2f} • Target: {pos['Target']:.2f}")
        # Only evaluate the open position against data for its own symbol.
        if ticker != pos.get('Symbol'):
            st.warning(f"Open paper position belongs to {pos.get('Symbol')}. Select that symbol to monitor its automatic exit.")
        elif auto_exit_enabled:
            # Use the latest OHLC bar to detect a level touch. If both levels are
            # touched in one bar, assume stop-loss first (conservative simulation).
            bar = data.iloc[-1] if isinstance(data, pd.DataFrame) and not data.empty else None
            bar_high = safe(bar.get('high', price)) if bar is not None else price
            bar_low = safe(bar.get('low', price)) if bar is not None else price
            if side == 'LONG':
                stop_trigger = bar_low <= pos['Stop Loss']
                target_trigger = bar_high >= pos['Target']
            else:
                stop_trigger = bar_high >= pos['Stop Loss']
                target_trigger = bar_low <= pos['Target']
            exit_reason = None
            exit_price = None
            if stop_trigger:
                exit_reason, exit_price = 'Auto Stop Loss', float(pos['Stop Loss'])
            elif target_trigger:
                exit_reason, exit_price = 'Auto Target', float(pos['Target'])
            if exit_price is not None:
                realised = (exit_price-pos['Entry'])*qty if side=='LONG' else (pos['Entry']-exit_price)*qty
                st.session_state.paper_trades.append({'Symbol':pos['Symbol'],'Side':side,'Entry':pos['Entry'],'Exit':exit_price,'Quantity':qty,'P&L ₹':realised,'Opened At':pos['Opened At'],'Closed At':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Result':'Win' if realised>0 else 'Loss' if realised<0 else 'Breakeven','Exit Reason':exit_reason})
                st.session_state.paper_position=None
                st.success(f'Paper position auto-closed: {exit_reason} at simulated price {exit_price:.2f}.')
                st.rerun()
        else:
            st.info('Automatic paper exit is OFF. Monitor the position and use Close Paper Position manually.')
        c1,c2=st.columns(2)
        if c1.button('Close Paper Position'):
            st.session_state.paper_trades.append({'Symbol':ticker,'Side':side,'Entry':pos['Entry'],'Exit':price,'Quantity':qty,'P&L ₹':pnl,'Opened At':pos['Opened At'],'Closed At':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Result':'Win' if pnl>0 else 'Loss' if pnl<0 else 'Breakeven'});st.session_state.paper_position=None;st.rerun()
        if c2.button('Move stop to trailing level'):
            improve=trail>pos['Stop Loss'] if side=='LONG' else trail<pos['Stop Loss']
            safe_side=(trail<price) if side=='LONG' else (trail>price)
            if improve and safe_side:pos['Stop Loss']=float(trail);st.session_state.paper_position=pos;st.rerun()
            else:st.info('Trailing stop does not improve current stop.')
    if st.session_state.paper_trades:
        pt=pd.DataFrame(st.session_state.paper_trades);wins=int((pt['P&L ₹']>0).sum());losses=int((pt['P&L ₹']<0).sum());net=pt['P&L ₹'].sum();c=st.columns(4);c[0].metric('Closed trades',len(pt));c[1].metric('Wins',wins);c[2].metric('Losses',losses);c[3].metric('Realised P&L',f'₹{net:,.2f}');st.dataframe(pt,use_container_width=True,hide_index=True);st.download_button('Download paper trades CSV',pt.to_csv(index=False),'paper_trades.csv','text/csv')

st.subheader('Signal Audit & Trend Validation')
audit_rows=[
    {'Check':'EMA200','Reading':fmt(last.ema200),'Result':'Bullish' if price>safe(last.ema200) else 'Bearish' if price<safe(last.ema200) else 'N/A'},
    {'Check':'SuperTrend','Reading':fmt(last.supertrend),'Result':'Bullish' if safe(last.supertrend_direction)>0 else 'Bearish'},
    {'Check':'ADX gate','Reading':fmt(last.adx),'Result':'Pass' if safe(last.adx)>=MANDATORY_ADX else 'Below threshold'},
    {'Check':'PCR context','Reading':f'{pcr:.2f}','Result':pcr_bias},
    {'Check':'OI context','Reading':f'CE {call_oi:+.1f}% / PE {put_oi:+.1f}%','Result':oi_bias},
    {'Check':'News/event filter','Reading':news,'Result':event},
]
st.dataframe(pd.DataFrame(audit_rows),use_container_width=True,hide_index=True)
ha_cols=st.columns(4)
ha_cols[0].metric('Heikin-Ashi Open',fmt(last.ha_open))
ha_cols[1].metric('Heikin-Ashi Close',fmt(last.ha_close))
ha_cols[2].metric('Heikin-Ashi Candle','Bullish' if safe(last.ha_close)>safe(last.ha_open) else 'Bearish')
ha_cols[3].metric('SuperTrend Direction','Bullish' if safe(last.supertrend_direction)>0 else 'Bearish')

st.subheader('14. Combined Indicator Confirmation')
macd_state='Bullish' if safe(last.macd)>safe(last.macd_signal) else 'Bearish' if safe(last.macd)<safe(last.macd_signal) else 'Neutral'
bb_state='Above midline' if safe(last.close)>safe(last.bb_mid) else 'Below midline' if safe(last.close)<safe(last.bb_mid) else 'At midline'
ema_state='Bullish' if safe(last.ema20)>safe(last.ema50) else 'Bearish' if safe(last.ema20)<safe(last.ema50) else 'Neutral'
cc=st.columns(4);cc[0].metric('EMA20 / EMA50',ema_state);cc[1].metric('MACD Direction',macd_state);cc[2].metric('Bollinger Position',bb_state);cc[3].metric('Final Combined Signal',signal)
st.caption('Final signal is a rule-based confluence of technical indicators plus the manually entered PCR/OI/news context. A BUY/SELL signal is not a guarantee; validate on historical data and paper trading.')

st.subheader('Signal Audit Log')
if st.session_state.signal_audit:
    audit_df=pd.DataFrame(st.session_state.signal_audit)
    st.dataframe(audit_df.tail(100),use_container_width=True,hide_index=True)
    st.download_button('Download Signal Audit CSV',audit_df.to_csv(index=False).encode('utf-8'),'signal_audit.csv','text/csv')

st.subheader('15. Current Data / Signal Caveats')
st.write('• This merge uses the actual functions present in the uploaded files. The supplied app.py did not contain the exact named market-snapshot/F&O helper functions, so compatible equivalents are implemented here; the F&O-style scanner is a spot-momentum proxy, not a real futures/options chain. The supplied app(4).py did not contain the full named advanced feature set as separate engines; additions here are transparent approximations and should not be treated as preserved original engines.')
st.write('• Candle/history, live LTP and supplementary MTF/options feeds have independent selectors. REST polling remains the default candle refresh method. Angel One WebSocket is optional and supplies LTP only; it does not replace OHLC candles. Angel One instruments are resolved automatically from the instrument master. Global/sector performance boards remain Yahoo Finance because this build does not map every global instrument to Angel One. Live PCR/OI is attempted only when the Other Market Data Feed is Angel One, for supported NIFTY/BANK NIFTY contracts, and depends on Angel One FULL market-data permissions. FII/DII, verified news, government announcements and geopolitical events remain manual context fields.')
st.write('• “Win chance” is not inferred from score. The backtest win rate is historical for the selected ticker, period and rules and may not predict future results.')
st.write('• Yahoo ticker coverage varies; some index symbols or intraday intervals may be unavailable. Always verify latest timestamps and use paper trading first.')
st.caption('For public deployment, change the default login and use secure secrets. This app is not financial advice.')
