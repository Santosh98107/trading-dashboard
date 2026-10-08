# Elite Trading Dashboard Ultimate+

A Streamlit-based trading dashboard for market tracking, technical analysis, candlestick pattern detection, paper trading, and journaling.

## Features

- Secure login system with default admin credentials
- Live price dashboard for Indian stocks and major market indices
- Candlestick chart with EMA trend overlays
- Technical indicators:
  - RSI
  - MACD
  - ADX
  - EMA 20 / 50
  - VWAP
  - ATR
  - Bollinger Bands
  - Stochastic
  - Volume analysis
- Candlestick pattern detection
- Trade journal persistence using CSV files
- Paper trading balance and open positions tracking
- Simple and lightweight setup for local use

## Default Login

- Username: admin
- Password: admin121

## Requirements

Install the app dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Project Files

- `app.py` — main dashboard application
- `requirements.txt` — Python dependencies
- `trade_journal.csv` — trade journal data
- `paper_trade_state.csv` — paper trading balance state
- `users.csv` — login credentials storage
- `daily_summary.csv` — summary data

## Notes

- The app creates its required CSV files automatically on first run.
- The default admin account is created if `users.csv` does not already exist.
- This is designed for local trading dashboard use and not for production deployment.
