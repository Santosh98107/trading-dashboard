# Elite Trading Dashboard — Upstox + Angel One + Yahoo

Build: `2026-11-08-UPSTOX-INSTITUTIONAL-AI-TRAINER-02`

## Files
- `app.py` — full dashboard, based on the latest recovered `app (62).py`.
- `institutional_upgrades.py` — institutional diagnostics module, automatically imported by the app.
- `requirements.txt` — Python dependencies.

Keep all three files in the same folder. The institutional module is required; do not remove it.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data providers
- **Yahoo Finance** — existing selectable source and fallback.
- **Angel One** — existing SmartAPI candles; WebSocket remains primary for LTP with REST polling fallback.
- **Upstox** — OAuth 2.0 authorization-code exchange, automatic daily instrument-master lookup, Historical Candle V3 OHLCV and Market Quote V3 LTP snapshot.

Upstox setup: create an API app in the Upstox developer portal, register the exact redirect URI, enter Client ID/Secret/Redirect URI in the dashboard, open the secure Upstox login link, then paste the returned one-time authorization code and connect. The access token is kept in Streamlit session state only; reconnect when it expires (typically 3:30 AM IST). Never commit API secrets, access tokens, Angel PINs or TOTP secrets.

Upstox is implemented as a REST market-data source in this build. It is not an Upstox WebSocket streaming implementation. Angel One's existing WebSocket-primary behavior is retained. No live broker order placement is implemented; paper-trading controls remain simulated.

## Signal integration correction
Institutional market structure, liquidity sweep, order blocks, FVG, volume profile and relative-strength diagnostics are now aggregated **before** directional candidate eligibility and BUY/SELL selection. The capped institutional contribution is applied to the bullish/bearish totals, so it can affect directional bias, candidate qualification, validation and signal quality. Unavailable diagnostics add zero and do not fabricate evidence. Existing scoring components are retained.

## Important limitations
- Syntax and mocked provider parsing tests do not prove live broker connectivity. Upstox/Angel credentials, API permissions, network access, instrument availability and market hours must be verified in the deployed session.
- Yahoo and broker APIs may be delayed, rate-limited, or unavailable. The app does not invent candles or quotes.
- Signal scores are heuristic analytical scores, not calibrated win probabilities or guarantees. Backtests and paper trading are simulations, not predictions of future results.
- The app does not send real orders.


## AI Backtest Trainer & Strategy Optimizer
- Run **AI Trainer Analysis** from Backtest Analytics to generate evidence-based alerts from completed historical trades.
- Trainer reports BUY/SELL underperformance and RSI/ADX expectancy buckets where sample sizes permit.
- With at least 30 completed trades, it evaluates candidate ADX and side-specific RSI filters. Candidate selection uses the first 70% of trades chronologically; the final 30% is held out for evaluation.
- It compares win rate, profit factor, expectancy and trade count. It does not optimize on win rate alone.
- A user must explicitly approve a candidate. Approval applies a **paper-eligibility filter only**; it does not rewrite existing bullish/bearish scores, score weights, signal formulas or broker order behavior. The approved filter is held in Streamlit session state and may need re-approval after a new session.
- Disable the filter from the trainer panel at any time. Re-run walk-forward validation and paper-test before relying on any candidate.
- The trainer is an explainable rules-based optimizer, not a guarantee of higher future win rate. Small samples, changing regimes and selection bias remain risks.
