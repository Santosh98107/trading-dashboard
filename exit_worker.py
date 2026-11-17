"""exit_worker.py - background Auto-Exit executor for PAPER trades.

One daemon thread per Streamlit server process. Every `interval` seconds during NSE hours it looks at the
open journal rows that have Auto Exit enabled, fetches each symbol's latest price and runs the same
`risk_manager.auto_paper_trade_exit` the page uses (stop, trailing stop, T1/T2/T3 scale-outs, optional
profit ladder). It does not need any browser tab to be open or refreshing - only the server process alive.

Limits (be honest in the UI): the process sleeps when Streamlit Cloud idles it; Yahoo prices can lag;
exits happen at the price the worker observed, not at the exact stop level; paper trades only.
"""
import threading
import time
from collections import deque

import pandas as pd

import risk_manager as rm
from database import ist_now, load_journal, log_app_error

try:                                    # optional: keeps the Supabase copy current after a worker exit
    import durable_store
except Exception:                       # pragma: no cover
    durable_store = None

DEFAULT_INTERVAL = 15
MAX_QUOTE_AGE_SEC = 600                 # refuse a quote older than 10 minutes (holiday / frozen feed)
MARKET_OPEN = (9, 10)                   # IST, small buffer around 09:15
MARKET_CLOSE = (15, 35)                 # IST, small buffer around 15:30

_LOCK = threading.Lock()
_CFG = {"enabled": True, "interval": DEFAULT_INTERVAL, "symbol_map": {}, "settings": {}, "ignore_hours": False}
STATUS = {"running": False, "started": None, "last_tick": None, "last_note": "not started",
          "ticks": 0, "exits": 0, "partials": 0, "last_error": ""}
EVENTS = deque(maxlen=60)
_THREAD = None
_WAKE = threading.Event()


def market_open(now=None):
    now = now or ist_now()
    if now.weekday() >= 5:
        return False
    hm = (now.hour, now.minute)
    return MARKET_OPEN <= hm <= MARKET_CLOSE


def update_config(enabled=True, interval=DEFAULT_INTERVAL, symbol_map=None, settings=None, ignore_hours=False):
    """Called by the page on every run so the worker always has the latest Settings (the thread cannot see session_state)."""
    with _LOCK:
        _CFG["enabled"] = bool(enabled)
        _CFG["interval"] = max(5, int(interval or DEFAULT_INTERVAL))
        if symbol_map is not None:
            _CFG["symbol_map"] = dict(symbol_map)
        if settings is not None:
            _CFG["settings"] = dict(settings)
        _CFG["ignore_hours"] = bool(ignore_hours)


def _cfg():
    with _LOCK:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _CFG.items()}


def _event(text):
    EVENTS.appendleft(f"{ist_now().strftime('%H:%M:%S')}  {text}")


def _open_auto_symbols():
    j = load_journal()
    if j is None or j.empty or "Status" not in j.columns:
        return []
    m = j["Status"].astype(str).str.upper().eq("OPEN")
    if "AutoExitEnabled" in j.columns:
        m &= j["AutoExitEnabled"].map(lambda v: rm._paper_bool(v, False))
    else:
        return []
    return sorted(set(j.loc[m, "Symbol"].astype(str)))


def _open_auto_keys():
    """{symbol: {(side, timeframe), ...}} for open Auto Exit rows."""
    j = load_journal()
    out = {}
    if j is None or j.empty or "AutoExitEnabled" not in j.columns:
        return out
    m = j["Status"].astype(str).str.upper().eq("OPEN") & j["AutoExitEnabled"].map(lambda v: rm._paper_bool(v, False))
    for _, r in j[m].iterrows():
        side = str(r.get("Side", r.get("Signal", "BUY"))).upper()
        out.setdefault(str(r.get("Symbol", "")), set()).add((side, str(r.get("Timeframe", ""))))
    return out


def build_health(ticker, keys, settings, candle_fn=None):
    """Trend-health readings for the open (side, timeframe) pairs of one symbol. Only used when Trend-ride mode is ON."""
    out = {}
    try:
        import trade_health as th
        if candle_fn is None:
            from price_feed import fetch_candles as candle_fn
        cfg = {"hold_max": float(settings.get("th_hold_max", 40.0)), "exit_min": float(settings.get("th_exit_min", 70.0))}
        for side, tf in keys:
            mins = rm.TF_MINUTES.get(tf)
            if not mins:
                continue
            df = candle_fn(ticker, mins)
            if df is None:
                continue
            h = th.assess(df, side, cfg=cfg, history=5)
            if h.get("ok"):
                h["ts"] = time.time()
                out[f"{side}|{tf}"] = h
    except Exception as exc:
        log_app_error("exit_worker_health", exc)
    return out


def tick(price_fn=None, now=None, force=False, candle_fn=None):
    """One pass. price_fn(ticker) -> {"price", "age"} or None. Returns a small summary dict (also used by tests)."""
    cfg = _cfg()
    out = {"checked": 0, "closed": 0, "partial": 0, "skipped": "", "symbols": []}
    STATUS["last_tick"] = (now or ist_now()).strftime("%d-%m-%Y %H:%M:%S")
    if not cfg["enabled"] and not force:
        out["skipped"] = "switched off"
        STATUS["last_note"] = "switched off in Settings"
        return out
    if not (force or cfg["ignore_hours"] or market_open(now)):
        out["skipped"] = "market closed"
        STATUS["last_note"] = "market closed - waiting"
        return out
    symbols = _open_auto_symbols()
    out["symbols"] = symbols
    if not symbols:
        out["skipped"] = "no open Auto Exit trades"
        STATUS["last_note"] = "no open Auto Exit trades"
        return out
    if price_fn is None:
        from price_feed import fetch_last_price as price_fn
    rm.set_thread_cfg(cfg["settings"])
    _keys = _open_auto_keys() if cfg["settings"].get("tr_ride_on") else {}
    try:
        for sym in symbols:
            ticker = cfg["symbol_map"].get(sym)
            if not ticker:
                _event(f"{sym}: no price source mapped - skipped")
                continue
            q = price_fn(ticker)
            if not q or not q.get("price"):
                _event(f"{sym}: price unavailable - will retry")
                continue
            age = q.get("age")
            if age is not None and age > MAX_QUOTE_AGE_SEC and not force:
                _event(f"{sym}: price is {int(age)}s old - not used")
                continue
            out["checked"] += 1
            _health = None
            if cfg["settings"].get("tr_ride_on"):
                _health = build_health(ticker, _keys.get(sym, set()), cfg["settings"], candle_fn)
            res = rm.auto_paper_trade_exit(q["price"], symbol=sym, atr=None, health=_health)
            c, p = int(res.get("closed", 0)), int(res.get("partial", 0))
            out["closed"] += c
            out["partial"] += p
            if c:
                STATUS["exits"] += c
                _event(f"{sym}: position CLOSED by Auto Exit at ~{q['price']:.2f}")
            if p:
                STATUS["partials"] += p
                _event(f"{sym}: target scale-out executed (price ~{q['price']:.2f})")
    finally:
        rm.set_thread_cfg(None)
    STATUS["ticks"] += 1
    STATUS["last_note"] = f"checked {out['checked']} symbol(s)"
    if (out["closed"] or out["partial"]) and durable_store is not None:
        try:
            durable_store.maybe_push()
        except Exception:
            pass
    return out


def _loop():
    STATUS["running"] = True
    STATUS["started"] = ist_now().strftime("%d-%m-%Y %H:%M:%S")
    while True:
        try:
            tick()
            STATUS["last_error"] = ""
        except Exception as exc:                      # never let the thread die
            STATUS["last_error"] = str(exc)[:200]
            try:
                log_app_error("exit_worker", exc)
            except Exception:
                pass
        _WAKE.wait(timeout=_cfg()["interval"])
        _WAKE.clear()


def ensure_started():
    """Start the daemon thread once per process; safe to call on every Streamlit run."""
    global _THREAD
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return False
        _THREAD = threading.Thread(target=_loop, name="ec-exit-worker", daemon=True)
        _THREAD.start()
        return True


def is_alive():
    return bool(_THREAD is not None and _THREAD.is_alive())


def heartbeat_age():
    """Seconds since the last tick, or None."""
    try:
        t = pd.to_datetime(STATUS["last_tick"], format="%d-%m-%Y %H:%M:%S")
        return (ist_now().replace(tzinfo=None) - t.to_pydatetime()).total_seconds()
    except Exception:
        return None
