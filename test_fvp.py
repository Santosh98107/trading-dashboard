"""Futures-volume Volume Profile (optional, default OFF). Run: python -m pytest tests/test_fvp.py -q"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import _appns
import futures_oi as foi


def _frame(ns, seed=5, drift=4.0, spot_volume=0.0):
    d = _appns.sample_frame(300, seed, drift=drift)
    d["Volume"] = spot_volume                      # an index feed has no volume of its own
    d = ns["add_indicators"](d)
    d["FutVolume"] = (np.random.default_rng(1).uniform(1000, 9000, len(d)) * (1 + (np.arange(len(d)) % 7 == 0))).astype(float)
    return d


def _run(ns, d, switches):
    ns["st"].session_state.clear(); ns["st"].session_state.update(switches)
    return ns["analyze_signal"](d, timeframe_label="5m")


def _vp_row(a):
    rows = [c for c in a["institutional_components"] if c["name"] == "Volume Profile"]
    assert len(rows) == 1
    return rows[0]


def test_align_volume_is_exact_match_only_and_rejects_bad_input():
    idx = pd.date_range("2026-11-24 09:15", periods=12, freq="5min")
    data = pd.DataFrame({"Close": 100.0}, index=idx)
    fut = pd.DataFrame({"Close": 100.2, "Volume": np.arange(1, 11) * 100.0}, index=idx[:10])        # last two candles missing
    out, ok, _ = foi.align_volume(data, fut)
    assert ok and out["FutVolume"].iloc[:10].tolist() == fut["Volume"].tolist() and out["FutVolume"].iloc[10:].isna().all()   # no forward fill
    assert "FutVolume" not in data.columns                                                           # input untouched
    assert foi.align_volume(data, fut.assign(Close=130.0))[1] is False                              # wrong contract
    assert foi.align_volume(data, fut.assign(Volume=0.0))[1] is False
    assert foi.align_volume(data, fut.drop(columns="Volume"))[1] is False
    assert foi.align_volume(data, None)[1] is False


def test_futures_volume_profile_builds_when_spot_has_no_volume_and_fails_soft():
    ns = _appns.load()
    d = _frame(ns)
    vp = foi.futures_volume_profile(d.tail(250), ns["inst"].calculate_volume_profile)
    assert vp and vp["available"] and vp["poc"] and vp["val"] <= vp["poc"] <= vp["vah"]
    assert ns["inst"].calculate_volume_profile(d.tail(250))["available"] is False                    # the spot (index) profile stays unavailable
    assert foi.futures_volume_profile(d.drop(columns="FutVolume"), ns["inst"].calculate_volume_profile) is None
    assert foi.futures_volume_profile(None, ns["inst"].calculate_volume_profile) is None


def test_switch_off_changes_nothing_even_if_futures_volume_is_present():
    ns = _appns.load()
    d = _frame(ns)
    a = _run(ns, d, {"fvp_on": False})
    b = _run(ns, d.drop(columns="FutVolume"), {"fvp_on": False})
    assert a["net_score"] == b["net_score"] and a["score_audit"] == b["score_audit"] and a["reasons"] == b["reasons"]
    assert a["futures_volume_profile_used"] is False
    assert _vp_row(a)["bullish"] == _vp_row(a)["bearish"] == 0 and "futures" not in str(_vp_row(a)["reason"])


def test_switch_on_uses_futures_volume_for_an_index():
    ns = _appns.load()
    hit = 0
    for seed, drift in ((5, 4.0), (8, -4.0), (13, 3.0), (6, 4.0)):
        d = _frame(ns, seed, drift)
        off, on = _run(ns, d, {"fvp_on": False}), _run(ns, d, {"fvp_on": True})
        if on["futures_volume_profile_used"]:
            hit += 1
            row = _vp_row(on)
            assert "futures volume" in str(row["reason"]) and row["name"] == "Volume Profile"
            assert row["bullish"] + row["bearish"] <= 1.0 + 1e-9                                 # weight 0.2 x raw 5
            assert abs(on["net_score"] - off["net_score"]) <= 1.0 + 1e-9
        else:
            assert on["net_score"] == off["net_score"]                                           # nothing changed when the stage was not reached
    assert hit >= 1, "futures volume profile never reached the score on the trending samples"


def test_stocks_with_their_own_volume_are_not_changed_by_the_switch():
    ns = _appns.load()
    d = _frame(ns, spot_volume=5000.0)
    off, on = _run(ns, d, {"fvp_on": False}), _run(ns, d, {"fvp_on": True})
    assert on["futures_volume_profile_used"] is False
    assert on["net_score"] == off["net_score"] and on["score_audit"] == off["score_audit"]


def test_no_look_ahead_future_futures_volume_cannot_change_an_earlier_decision():
    ns = _appns.load()
    d = _frame(ns)
    cut = 260
    base = _run(ns, d.iloc[:cut], {"fvp_on": True})
    d2 = d.copy(); d2.iloc[cut:, d2.columns.get_loc("FutVolume")] = 1e9                              # later candles change wildly
    again = _run(ns, d2.iloc[:cut], {"fvp_on": True})
    assert base["net_score"] == again["net_score"] and base["score_audit"] == again["score_audit"]
