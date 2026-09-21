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
