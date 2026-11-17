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
