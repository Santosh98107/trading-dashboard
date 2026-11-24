"""Futures open-interest analysis (Upstox only).

fetch -> align -> classify:
  * pick_future(): nearest-expiry FUT contract from the Upstox instrument master rows.
  * fetch_future_frame(): candles via the app's own Upstox V3 fetcher (its OpenInterest column is the 7th candle field).
  * align_oi(): puts the futures OI on the dashboard's index candles with a backward fill (no future OI is ever used).
  * analyze_df(): price change vs OI change over the last few candles -> Long/Short Buildup, Short Covering, Long Unwinding.
Everything fails soft: when OI is missing (Yahoo mode, no token, bad contract) it returns available=False, 0 points, no blocks.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
BSE_UNDERLYING = {"SENSEX", "BANKEX"}
POINTS = {"Long Buildup": (2.0, 0.0), "Short Buildup": (0.0, 2.0), "Short Covering": (0.5, 0.0), "Long Unwinding": (0.0, 0.5)}


def _expiry(v):
    try:
        x = float(v)
        if x > 1e11:
            x /= 1000.0
        return datetime.fromtimestamp(x, IST).date()
    except Exception:
        return None


def pick_future(rows, underlying, today=None):
    u = str(underlying).strip().upper()
    seg = "BSE_FO" if u in BSE_UNDERLYING else "NSE_FO"
    today = today or datetime.now(IST).date()
    best = None
    for r in rows or []:
        try:
            if str(r.get("segment", "")).upper() != seg or str(r.get("instrument_type", "")).upper() != "FUT":
                continue
            if str(r.get("underlying_symbol", r.get("name", ""))).strip().upper() != u or not r.get("instrument_key"):
                continue
            e = _expiry(r.get("expiry"))
            if e is None or e < today:
                continue
            if best is None or e < best[0]:
                best = (e, r)
        except Exception:
            continue
    if not best:
        return None
    e, r = best
    return {"instrument_key": str(r["instrument_key"]), "expiry": e, "trading_symbol": str(r.get("trading_symbol", "")), "label": f"{u} {e.strftime('%d%b%y').upper()} FUT"}


def fetch_future_frame(rows, underlying, tf, fetch_candles_fn):
    """-> (frame or None, contract or None, message). fetch_candles_fn(instrument_key, tf) returns an OHLCV(+OpenInterest) frame."""
    try:
        c = pick_future(rows, underlying)
        if not c:
            return None, None, f"No nearest-expiry futures contract found for {underlying}"
        fr = fetch_candles_fn(c["instrument_key"], tf)
        if fr is None or fr.empty or "OpenInterest" not in fr.columns:
            return None, c, f"No futures candles / OI returned for {c['label']}"
        return fr, c, "ok"
    except Exception as exc:
        return None, None, f"Futures OI fetch failed: {type(exc).__name__}: {exc}"


def align_oi(data, fut, max_price_gap_pct=3.0):
    """Return (data_copy, ok, message). data['OpenInterest'] = futures OI as of each candle (backward fill only)."""
    try:
        if data is None or data.empty or fut is None or fut.empty or "OpenInterest" not in fut.columns:
            return data, False, "No futures OI to align"
        oi = pd.to_numeric(fut["OpenInterest"], errors="coerce")
        oi = oi[~oi.index.duplicated(keep="last")].sort_index()
        oi = oi.where(oi > 0)
        if oi.dropna().empty:
            return data, False, "Futures OI is empty / zero"
        if "Close" in fut.columns and "Close" in data.columns:
            fc, dc = float(fut["Close"].dropna().iloc[-1]), float(data["Close"].dropna().iloc[-1])
            if dc > 0 and abs(fc - dc) / dc * 100 > max_price_gap_pct:
                return data, False, f"Futures price {fc:,.2f} is {abs(fc - dc) / dc * 100:.1f}% away from the chart price {dc:,.2f}: wrong contract, not used"
        out = data.copy()
        out["OpenInterest"] = oi.reindex(out.index, method="ffill")
        return out, True, "ok"
    except Exception as exc:
        return data, False, f"Futures OI alignment failed: {type(exc).__name__}: {exc}"


def classify(price_pct, oi_pct, min_price_pct=0.03, min_oi_pct=0.2):
    if price_pct is None or oi_pct is None or abs(price_pct) < min_price_pct or abs(oi_pct) < min_oi_pct:
        return "Neutral"
    if price_pct > 0:
        return "Long Buildup" if oi_pct > 0 else "Short Covering"
    return "Short Buildup" if oi_pct > 0 else "Long Unwinding"


def analyze_df(df, window=5, min_price_pct=0.03, min_oi_pct=0.2):
    out = {"available": False, "state": "Unavailable", "bullish": 0.0, "bearish": 0.0, "block_buy_breakout": False,
           "block_sell_breakdown": False, "price_pct": None, "oi_pct": None, "bars": 0, "reason": "Futures OI unavailable"}
    try:
        if df is None or len(df) < 4 or "OpenInterest" not in df.columns or "Close" not in df.columns:
            return out
        d = df[["Close", "OpenInterest"]].apply(pd.to_numeric, errors="coerce")
        d = d[d["OpenInterest"] > 0].dropna()
        if len(d) < 3 or d["OpenInterest"].nunique() < 2:
            return out
        idx = d.index
        intraday = len(idx) > 2 and (idx[1:] - idx[:-1]).to_series().median() < pd.Timedelta(hours=20)
        if intraday:
            d = d[d.index.normalize() == d.index[-1].normalize()]      # never compare across the overnight gap
        w = min(int(window), len(d) - 1)
        if w < 2:
            out["reason"] = "Too early in the session for an OI read"
            return out
        p0, p1 = float(d["Close"].iloc[-1 - w]), float(d["Close"].iloc[-1])
        o0, o1 = float(d["OpenInterest"].iloc[-1 - w]), float(d["OpenInterest"].iloc[-1])
        if p0 <= 0 or o0 <= 0:
            return out
        price_pct, oi_pct = (p1 / p0 - 1) * 100, (o1 / o0 - 1) * 100
        state = classify(price_pct, oi_pct, min_price_pct, min_oi_pct)
        b, r = POINTS.get(state, (0.0, 0.0))
        out.update(available=True, state=state, bullish=b, bearish=r, price_pct=round(price_pct, 3), oi_pct=round(oi_pct, 3), bars=w,
                   block_buy_breakout=(state == "Short Covering"), block_sell_breakdown=(state == "Long Unwinding"),
                   reason=f"{state}: price {price_pct:+.2f}% with OI {oi_pct:+.2f}% over {w} candles")
        return out
    except Exception as exc:
        out["reason"] = f"Futures OI analysis failed: {type(exc).__name__}: {exc}"
        return out
