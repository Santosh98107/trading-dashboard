"""runtime_state.py - state that must survive Streamlit reruns.

Streamlit re-executes app.py in a FRESH namespace on every rerun, so any variable defined at the top of
app.py (circuit breakers, "last good candles", Yahoo diagnostics, the WebSocket state, ...) was silently
reset to its starting value on every refresh. That made the retry pauses and the stale-data fallback
useless: each rerun started from scratch, waited out the full network timeouts again and showed a blank
page. An imported module stays loaded for the life of the server process, so the shared state lives here.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

# ---- circuit breakers (epoch seconds until which a source is skipped) ----
YAHOO_NET_DOWN_UNTIL = 0.0
YF_NET_DOWN_UNTIL = 0.0
NSE_DOWN_UNTIL = 0.0
FIIDII_DOWN_UNTIL = 0.0

# ---- Yahoo bookkeeping ----
YAHOO_LAST_ERRORS = {}          # ticker -> last error text
LAST_GOOD_FRAMES = {}           # (ticker, interval, period) -> last good candles (shown as stale when Yahoo fails)
YAHOO_DIAG_BY_KEY = {}
YAHOO_FETCH_DIAGNOSTICS = {"status": "not_requested", "source": None, "rows": 0, "error": None}
BATCH_FAIL_UNTIL = {}

# ---- background fetching: a slow network call never blocks the page for more than a few seconds ----
FETCH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ec-fetch")
FETCH_PENDING = {}              # key -> Future still running from an earlier rerun
LAST_FETCH_SECONDS = {}         # key -> how long the last completed fetch took

MACRO_BG = {"ctx": None, "ts": 0.0, "running": False, "geo": True}   # macro context built in a background thread
MACRO_BG_LOCK = threading.Lock()

# ---- Angel WebSocket shared state (the socket thread keeps writing into this one object) ----
WS_STATE_LOCK = threading.Lock()
WS_SHARED_STATE = {
    "running": False,
    "connected": False,
    "error": "",
    "last_tick": None,
    "live_price": None,
    "retries": 0,
    "should_run": False,
    "thread_started": False,
    "symbol": None,
    "socket": None,
    "last_data_time": None,
    "last_tick_epoch": None,
    "generation": 0,
    "retry_exhausted": False,
    "log": [],
}


def clear_all():
    """Manual 'Clear cache' button: forget breakers, stale frames and pending fetches."""
    global YAHOO_NET_DOWN_UNTIL, YF_NET_DOWN_UNTIL, NSE_DOWN_UNTIL, FIIDII_DOWN_UNTIL
    YAHOO_NET_DOWN_UNTIL = YF_NET_DOWN_UNTIL = NSE_DOWN_UNTIL = FIIDII_DOWN_UNTIL = 0.0
    MACRO_BG.update(ctx=None, ts=0.0, running=False)
    for d in (YAHOO_LAST_ERRORS, LAST_GOOD_FRAMES, YAHOO_DIAG_BY_KEY, BATCH_FAIL_UNTIL, FETCH_PENDING, LAST_FETCH_SECONDS):
        d.clear()
    YAHOO_FETCH_DIAGNOSTICS.clear()
    YAHOO_FETCH_DIAGNOSTICS.update({"status": "not_requested", "source": None, "rows": 0, "error": None})
