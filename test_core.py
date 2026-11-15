"""Safety-net tests for the Elite Trading Dashboard. Run from the repo root:   python -m pytest tests -q
They do not start Streamlit or touch the network: functions are cut out of app.py by name and run on small fake data,
so a change that breaks cost handling, the exposure guard, data validation or the background worker is caught before deploy."""
import os, sys, time, types
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
    sys.modules.setdefault(_m, MagicMock())
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _cut(start, end):
    return APP[APP.index(start):APP.index(end)]


def test_app_compiles():
    compile(APP, "app.py", "exec")


def test_background_jobs_never_blocks_and_does_not_duplicate():
    import background_jobs as b
    calls = []
    def slow(x):
        calls.append(x); time.sleep(0.2); return x * 2
    b.clear()
    t = time.time(); s = b.request("t", slow, (3,))
    assert time.time() - t < 0.1 and s["running"] and s["value"] is None
    b.request("t", slow, (3,)); assert len(calls) == 1
    time.sleep(0.4); assert b.peek("t")["value"] == 6
    assert b.request("t", slow, (3,), max_age=100)["value"] == 6 and len(calls) == 1


def test_background_jobs_keeps_old_value_on_error():
    import background_jobs as b
    b.clear(); b.request("e", lambda: 5); time.sleep(0.1)
    def bad(): raise ValueError("x")
    b.request("e", bad, max_age=0); time.sleep(0.1)
    s = b.peek("e"); assert s["value"] == 5 and s["error"]


def test_validate_candles_flags_bad_data():
    ns = dict(pd=pd, np=np, ist_now=lambda: pd.Timestamp("2026-11-16 11:00"))
    exec(_cut("def _validate_candles", "def _ec_render_system_health"), ns)
    idx = pd.date_range("2026-11-16 09:15", periods=20, freq="5min")
    c = 100 + np.cumsum(np.random.randn(20) * .1)
    df = pd.DataFrame({"Open": c, "High": c + .2, "Low": c - .2, "Close": c, "Volume": 1000}, index=idx)
    assert all(r[1] == "PASS" for r in ns["_validate_candles"](df, "5m"))
    bad = df.copy(); bad.iloc[3, bad.columns.get_loc("High")] = bad.iloc[3]["Low"] - 1
    assert any(r[1] != "PASS" for r in ns["_validate_candles"](bad, "5m"))
    assert ns["_validate_candles"](pd.DataFrame())[0][1] == "FAIL"


def test_exposure_guard_groups_and_direction():
    import ai_engine
    class St: session_state = {}
    ns = dict(st=St(), ai_engine=ai_engine, log_app_error=lambda *a: None)
    exec(_cut("def _exposure_group", "def _timing_guard("), ns)
    rows = lambda r: pd.DataFrame([{"Status": "OPEN", "Symbol": s, "Side": d, "Signal": d, "EntryPrice": 100, "LastPrice": 100,
                                    "Quantity": 1, "RemainingQty": 1, "StopLoss": 99, "TrailingStop": 0} for s, d in r])
    g = ns["_exposure_guard"]
    banks = rows([("HDFCBANK", "BUY"), ("ICICIBANK", "BUY")])
    assert g(banks, "SBIN", "BUY")["warning"] and not g(banks, "SBIN", "BUY")["block"]
    assert not g(banks, "SBIN", "SELL")["warning"] and not g(banks, "TCS", "BUY")["warning"]
    assert g(rows([("NIFTY", "BUY"), ("BANKNIFTY", "BUY")]), "SENSEX", "BUY")["warning"]
    St.session_state["exposure_guard_block"] = True
    assert g(banks, "SBIN", "BUY")["block"]


def test_wilson_interval_sane():
    ns = {}
    start = APP.index("def _wilson_interval")
    exec(APP[start:APP.index("\ndef ", start + 10)], ns)
    lo, hi = ns["_wilson_interval"](60, 100)
    assert 50 < lo < 60 < hi < 70
    lo2, hi2 = ns["_wilson_interval"](60, 1000)
    assert (hi2 - lo2) < (hi - lo)            # more trades -> tighter range
    assert ns["_wilson_interval"](60, 0) is None
