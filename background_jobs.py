"""Background jobs for the Elite Trading Dashboard.

Heavy, slow work (NIFTY Top 30 scanner, Top Opportunities ranking) runs here in worker threads, away from the page.
The page only READS the latest finished result, so it never waits, and a timer rerun can no longer cancel the work.
This module lives in sys.modules, so results survive Streamlit reruns (and are shared by every browser tab).
No streamlit calls in here: arguments are prepared by the page, the worker gets plain data.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ec-bg")
_LOCK = threading.Lock()
_JOBS = {}          # key -> {"value", "done_at", "error", "future", "started_at", "seconds"}


def _run(key, fn, args, kwargs):
    t0 = time.time()
    try:
        val = fn(*args, **kwargs)
        err = None
    except Exception as exc:                      # keep the old value, remember the error
        val, err = None, f"{type(exc).__name__}: {exc}"
    with _LOCK:
        ent = _JOBS.setdefault(key, {})
        if err is None:
            ent["value"] = val
            ent["done_at"] = time.time()
        ent["error"] = err
        ent["seconds"] = time.time() - t0
        ent["future"] = None


def request(key, fn, args=(), kwargs=None, max_age=180, retry_after=60, force=False):
    """Make sure `key` is fresh. Starts a background run when the stored result is older than max_age
    (or missing) and nothing is running. Never blocks. Returns the current state dict (copy)."""
    kwargs = kwargs or {}
    now = time.time()
    with _LOCK:
        ent = _JOBS.setdefault(key, {"value": None, "done_at": 0.0, "error": None, "future": None,
                                     "started_at": 0.0, "seconds": 0.0})
        running = ent.get("future") is not None
        age = now - ent.get("done_at", 0.0) if ent.get("done_at") else None
        stale = age is None or age > max_age or force
        recently_failed = ent.get("error") and (now - ent.get("started_at", 0.0) < retry_after) and not force
        if stale and not running and not recently_failed:
            ent["started_at"] = now
            ent["future"] = _POOL.submit(_run, key, fn, args, kwargs)
            running = True
        return {"value": ent.get("value"), "age": (now - ent["done_at"]) if ent.get("done_at") else None,
                "running": running, "error": ent.get("error"), "seconds": ent.get("seconds", 0.0)}


def peek(key):
    now = time.time()
    with _LOCK:
        ent = _JOBS.get(key)
        if not ent:
            return {"value": None, "age": None, "running": False, "error": None, "seconds": 0.0}
        return {"value": ent.get("value"), "age": (now - ent["done_at"]) if ent.get("done_at") else None,
                "running": ent.get("future") is not None, "error": ent.get("error"), "seconds": ent.get("seconds", 0.0)}


def clear(key=None):
    with _LOCK:
        if key is None:
            _JOBS.clear()
        else:
            _JOBS.pop(key, None)
