"""price_feed.py - tiny, dependency-light last-price lookup used by the background exit worker.

Uses Yahoo's chart endpoint (same source the app already uses). It is NOT exchange LTP: Yahoo can lag by
a few seconds to minutes. Every quote carries its own timestamp so the caller can refuse a stale one.
"""
import time
from urllib.parse import quote

import requests

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*"}


def parse_chart_payload(payload):
    """Pure parser (unit-tested). Returns {"price","ts","age"} or None."""
    try:
        res = (payload.get("chart", {}).get("result") or [None])[0]
        if not res:
            return None
        meta = res.get("meta", {}) or {}
        price = meta.get("regularMarketPrice")
        ts = meta.get("regularMarketTime")
        if price is None:       # fall back to the last non-null 1m close
            closes = (((res.get("indicators", {}) or {}).get("quote") or [{}])[0].get("close")) or []
            stamps = res.get("timestamp") or []
            for c, t in zip(reversed(closes), reversed(stamps)):
                if c is not None:
                    price, ts = c, t
                    break
        price = float(price)
        if not price > 0:
            return None
        ts = float(ts) if ts else None
        return {"price": price, "ts": ts, "age": (time.time() - ts) if ts else None}
    except Exception:
        return None


def fetch_last_price(ticker, timeout=(3, 6)):
    """Latest Yahoo price for a ticker, or None. Never raises."""
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            r = requests.get(f"https://{host}/v8/finance/chart/{quote(str(ticker), safe='^.-')}",
                             params={"range": "1d", "interval": "1m", "includePrePost": "false"},
                             headers=_HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            q = parse_chart_payload(r.json())
            if q:
                return q
        except Exception:
            continue
    return None


def parse_candles(payload):
    """Pure parser (unit-tested): Yahoo chart payload -> DataFrame[Open, High, Low, Close, Volume] or None."""
    try:
        import pandas as pd
        res = (payload.get("chart", {}).get("result") or [None])[0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"],
                           "Volume": q.get("volume") or [0] * len(q["close"])},
                          index=pd.to_datetime(res["timestamp"], unit="s"))
        df = df.dropna(subset=["High", "Low", "Close"])
        return df if len(df) else None
    except Exception:
        return None


_CANDLE_CACHE = {}


def fetch_candles(ticker, tf_minutes, max_age=45, timeout=(3, 8)):
    """1-minute candles (last 5 days) resampled to tf_minutes. Cached for `max_age` seconds. None on failure."""
    key = (str(ticker), int(tf_minutes))
    hit = _CANDLE_CACHE.get(key)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            r = requests.get(f"https://{host}/v8/finance/chart/{quote(str(ticker), safe='^.-')}",
                             params={"range": "5d", "interval": "1m", "includePrePost": "false"}, headers=_HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            df = parse_candles(r.json())
            if df is None:
                continue
            if int(tf_minutes) > 1:
                df = df.resample(f"{int(tf_minutes)}min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                                             "Volume": "sum"}).dropna(subset=["Close"])
            _CANDLE_CACHE[key] = (time.time(), df)
            return df
        except Exception:
            continue
    return None
