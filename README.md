# Elite Trading Dashboard Ultimate

A Streamlit-based trading dashboard for live market analysis, paper trading, signal generation, journaling, and performance tracking.

## Features

### Live Signal Engine
- Buy / Sell / No Trade signal generation
- Confidence %
- Win Chance %
- Signal strength
- Higher timeframe confirmation
- Support / Resistance detection
- Option strike suggestion

### Technical Indicators
- EMA20
- EMA50
- VWAP
- RSI
- MACD
- Signal Line
- MACD Histogram
- ATR
- Bollinger Bands
- Stochastic K / D
- ADX
- PlusDI / MinusDI
- Volume MA20
- Volume Spike
- RVOL

### Candlestick / Chart Pattern Detection
- Bullish Marubozu
- Bearish Marubozu
- Bullish Engulfing
- Bearish Engulfing
- Doji
- Hammer
- Shooting Star

### Advanced Context Inputs
- PCR OI
- PCR Volume
- Crude Oil
- Brent Crude
- USD/INR
- Top companies breadth
- Bank performance
- Volume bias
- Regime detection
- Final fused recommendation

### Backtesting
- Historical trade outcome simulation
- Win / Loss count
- Win rate calculation
- Backtest quality reference

### Paper Trading
- Paper trade entry
- Entry / Exit tracking
- Balance and used margin
- Journal integration
- Realized PnL
- Trade closure support

### Trade Journal
- Manual signal save
- Paper trade journal save
- Entry / Exit history
- Notes field
- Download CSV

### Alerts
- Telegram alert support
- Auto-send on new signal

### Reports / Downloads
- Market data CSV export
- Trade journal export
- Daily summary export

---

## Files Used by App

### Main App File
- `your_app_file.py`  
  Rename this to your preferred file name, for example:
  - `app.py`
  - `elite_dashboard.py`

### Generated / Persistent Files
- `trade_journal.csv`
- `last_signal_state.txt`
- `paper_trade_state.csv`
- `users.csv`
- `daily_summary.csv`

---

## Requirements

Create a `requirements.txt` file with:
