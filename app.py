import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.title("📈 Trading Dashboard")

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

selected = st.selectbox(
    "Select Stock / Index",
    list(stocks.keys())
)

st.write("Selected:", selected)

if st.button("Fetch Data"):

    ticker = stocks[selected]

    data = yf.download(
        ticker,
        period="1mo",
        progress=False
    )

    if data.empty:
        st.error("No data found")
        st.stop()

    volume = data["Volume"]

    vwap = (
        (data["Close"] * volume).cumsum()
        / volume.cumsum()
    )

    last_vwap = vwap.iloc[-1].item()
    close_price = data["Close"].iloc[-1].item()

    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()

    ema20 = data["EMA20"].iloc[-1].item()
    ema50 = data["EMA50"].iloc[-1].item()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Current Price", round(close_price, 2))
        st.metric("EMA20", round(ema20, 2))

    with col2:
        st.metric("VWAP", round(last_vwap, 2))
        st.metric("EMA50", round(ema50, 2))

    trend = "🟢 BULLISH 📈" if ema20 > ema50 else "🔴 BEARISH 📉"
    st.write("Trend:", trend)

    confidence = 50

    if ema20 > ema50:
        confidence += 25

    if close_price > ema20:
        confidence += 25

    st.write("Confidence Score:", f"{confidence}%")
    # RSI Calculation
delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

avg_loss = avg_loss.replace(0, 1e-10)

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

latest_rsi = data["RSI"].iloc[-1].item()

st.metric("RSI (14)", round(latest_rsi, 2))

if latest_rsi > 60:
    st.success("RSI Bullish ✅")
elif latest_rsi < 40:
    st.error("RSI Bearish ❌")
else:
    st.warning("RSI Neutral ⏸️")
    # MACD Calculation
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()

data["MACD"] = exp1 - exp2
data["SignalLine"] = data["MACD"].ewm(span=9, adjust=False).mean()

latest_macd = data["MACD"].iloc[-1].item()
latest_signal = data["SignalLine"].iloc[-1].item()

col1, col2 = st.columns(2)

with col1:
    st.metric("MACD", round(latest_macd, 2))

with col2:
    st.metric("Signal Line", round(latest_signal, 2))

if latest_macd > latest_signal:
    st.success("MACD Bullish ✅")
else:
    st.error("MACD Bearish ❌")

    st.write(
        "Last Updated:",
        datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

    st.subheader("Price Chart")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(data["Close"], label="Close")
    ax.plot(data["EMA20"], label="EMA20")
    ax.plot(data["EMA50"], label="EMA50")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
