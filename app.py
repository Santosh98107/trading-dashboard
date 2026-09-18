import matplotlib.pyplot as plt

import streamlit as st


from datetime import datetime
st.title("📈 Trading Dashboard")

stocks = {
"NIFTY": "^NSEI",
"BANKNIFTY": "^NSEBANK",
"FINNIFTY": "^NSEFIN",
"SENSEX": "^BSESN",
"RELIANCE": "RELIANCE.NS",
"TCS": "TCS.NS",
"INFY": "INFY.NS",
"HDFCBANK": "HDFCBANK.NS",
 
"SBIN": "SBIN.NS",
"ICICIBANK": "ICICIBANK.NS",
"ITC": "ITC.NS",
"LT": "LT.NS",
"AXISBANK": "AXISBANK.NS",
"HINDUNILVR": "HINDUNILVR.NS",
"BHARTIARTL": "BHARTIARTL.NS"
}

selected = st.selectbox(
    "Select Stock / Index",
    stocks
)

st.write("Selected:", selected)

import yfinance as yf

if st.button("Fetch Data"):
 
    ticker = stocks[selected]
 
    data = yf.download(ticker, period="5d")
 
    if data.empty:
        st.error("No data found for selected symbol")
        st.stop()
 
    volume = data["Volume"]
 
    vwap = (
        (data["Close"] * volume).cumsum()
        / volume.cumsum()
    )
 
    last_vwap = vwap.iloc[-1].item()
 
    close_price = data["Close"].iloc[-1].item()
 
    data["EMA20"] = data["Close"].ewm(span=20).mean()
 
    ema20 = data["EMA20"].iloc[-1].item()
 
    data["EMA50"] = data["Close"].ewm(span=50).mean()
 
    ema50 = data["EMA50"].iloc[-1].item()












data["EMA50"] = data["Close"].ewm(span=50).mean()

ema50 = data["EMA50"].iloc[-1]
col1, col2 = st.columns(2)
 
with col1:
    st.metric(
        "Current Price",
        round(close_price, 2)
    )
 
    st.metric(
        "EMA20",
        round(ema20, 2)
    )
 
with col2:
    st.metric(
        "VWAP",
        round(last_vwap, 2)
    )
 
    st.metric(
        "EMA50",
        round(ema50, 2)
    )
 
    st.write("PCR:", "Coming Soon")
 
    st.write("PCR Trend:", "Coming Soon")

st.write(
    "Last Updated:",
    datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

if ema20 > ema50:
    trend = "🟢 BULLISH 📈"
else:
    trend = "🔴 BEARISH 📉"

if "BULLISH" in trend:
    st.success(trend)
else:
    st.error(trend)

call_signal = (ema20 > ema50) and (float(close_price) > float(ema20))

put_signal = (ema20 < ema50) and (float(close_price) < float(ema20))

confidence = 50
 
if call_signal:
    signal = "✅ CALL BUY"
 
elif put_signal:
    signal = "✅ PUT BUY"
 
else:
    signal = "⏸️ NO TRADE"
 
if ema20 > ema50:
    confidence += 25
 
if float(close_price) > float(ema20):
    confidence += 25
 
st.write(
    "Confidence Score:",
    f"{confidence}%"
)

if signal == "✅ CALL BUY":
    st.success(signal)

elif signal == "✅ PUT BUY":
    st.error(signal)

else:
    st.warning(signal)
if signal == "✅ CALL BUY":
 
    entry = float(close_price)
 
    sl = entry - 100
    target = entry + 200
 
    st.write("Entry:", round(entry, 2))
    st.write("Stop Loss:", round(sl, 2))
    st.write("Target:", round(target, 2))
 
    risk = abs(entry - sl)
    reward = abs(target - entry)
 
    st.write(
        "Risk Reward Ratio:",
        f"1:{round(reward / risk, 2)}"
    )
 
elif signal == "✅ PUT BUY":
 
    entry = float(close_price)
 
    sl = entry + 100
    target = entry - 200
 
    st.write("Entry:", round(entry, 2))
    st.write("Stop Loss:", round(sl, 2))
    st.write("Target:", round(target, 2))
 
    risk = abs(entry - sl)
    reward = abs(target - entry)
 
    st.write(
        "Risk Reward Ratio:",
        f"1:{round(reward / risk, 2)}"
    )



st.subheader("Price Chart")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(data["Close"], label="Close")
ax.plot(data["EMA20"], label="EMA20")
ax.plot(data["EMA50"], label="EMA50")

ax.legend()
ax.grid(True)

st.pyplot(fig)
