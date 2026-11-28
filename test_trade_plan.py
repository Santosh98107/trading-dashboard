"""Trade plan, chase check, open-trade alert and the optional anti-chase guard. Run: python -m pytest tests/test_trade_plan.py -q"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import _appns
import trade_plan as tp

NS = _appns.load()


def _frame(seed=5, drift=3.0, n=300):
    return NS["add_indicators"](_appns.sample_frame(n, seed, drift=drift))


def _two_days(n_prev=75, n_today=40, start_today=25500.0):
    """Yesterday 09:15-15:30 style candles + today's first candles, 5m."""
    rng = np.random.default_rng(2)
    i1 = pd.date_range("2026-11-25 09:15", periods=n_prev, freq="5min")
    i2 = pd.date_range("2026-11-26 09:15", periods=n_today, freq="5min")
    idx = i1.append(i2)
    c = np.r_[25400 + np.cumsum(rng.normal(0, 4, n_prev)), start_today + np.cumsum(rng.normal(0, 4, n_today))]
    d = pd.DataFrame({"Open": c, "High": c + 5, "Low": c - 5, "Close": c, "Volume": 1000.0}, index=idx)
    return NS["add_indicators"](d)


# ---------- levels ----------
def test_session_levels_prev_day_and_opening_range():
    d = _two_days()
    lv = tp.session_levels(d, "5m", 15)
    prev = d[d.index.normalize() == pd.Timestamp("2026-11-25")]
    assert lv["Prev-day high"] == prev["High"].max() and lv["Prev-day low"] == prev["Low"].min()
    today = d[d.index.normalize() == pd.Timestamp("2026-11-26")]
    first3 = today.iloc[:3]                                                    # 09:15, 09:20, 09:25 (< 15 min after the open)
    assert lv["Opening-range high"] == first3["High"].max() and lv["Opening-range low"] == first3["Low"].min()


def test_opening_range_is_not_reported_before_it_is_complete_and_daily_has_no_session_levels():
    d = _two_days(n_today=2)                                                   # only 09:15 and 09:20 today
    assert "Opening-range high" not in tp.session_levels(d, "5m", 15)
    assert tp.session_levels(_two_days(), "1d", 15) == {}


# ---------- plan maths ----------
def test_plan_buy_and_sell_geometry_matches_the_apps_own_2r_3r_4r():
    d = _frame()
    plan = tp.build_plan(d, "3m", resistance=float(d["High"].tail(21).iloc[:-1].max()), support=float(d["Low"].tail(21).iloc[:-1].min()))
    assert plan["available"]
    atr = plan["atr"]
    for key, s in (("buy", 1), ("sell", -1)):
        p = plan[key]
        if p.get("entry") is None:
            continue
        risk = abs(p["entry"] - p["stop"])
        assert 0.8 * atr - 1e-6 <= risk <= 2.0 * atr + 1e-6                    # stop clamp
        assert s * (p["entry"] - p["stop"]) > 0                                 # stop on the right side
        assert np.allclose([(t - p["entry"]) / (s * risk) for t in p["targets"]], [2.0, 3.0, 4.0])
        assert abs(p["risk_pct"] - risk / p["entry"] * 100) < 1e-9


def test_waiting_when_a_level_is_ahead_and_trigger_is_just_beyond_it():
    d = _frame(seed=9, drift=0.0)
    price = float(d["Close"].iloc[-1]); atr = float(d["ATR"].iloc[-1])
    plan = tp.build_plan(d, "3m", resistance=price + 1.5 * atr, support=price - 1.5 * atr, ext_atr=50)
    b, s = plan["buy"], plan["sell"]
    assert b["status"] == "WAITING" and abs(b["trigger"] - (price + 1.5 * atr + 0.1 * atr)) < 1e-6 and b["entry"] == b["trigger"]
    assert s["status"] == "WAITING" and abs(s["trigger"] - (price - 1.5 * atr - 0.1 * atr)) < 1e-6


def test_extended_status_and_pullback_zone_when_price_is_stretched():
    d = _frame(seed=5, drift=6.0)
    plan = tp.build_plan(d, "3m", resistance=None, support=None, ext_atr=0.2)
    b = plan["buy"]
    assert b["status"] == "EXTENDED"
    if b.get("zone"):
        assert b["zone"]["high"] < float(d["Close"].iloc[-1]) and b["entry_kind"] == "pullback"
        assert "do not chase" in tp.plan_text(b, plan["atr"], float(d["Close"].iloc[-1])).lower()


def test_level_in_the_way_is_reported_before_t1():
    d = _frame(seed=9, drift=0.0)
    price = float(d["Close"].iloc[-1]); atr = float(d["ATR"].iloc[-1])
    plan = tp.build_plan(d, "3m", resistance=price + 1.0 * atr, support=price - 9 * atr, ext_atr=50)
    # force a second, nearer resistance-type level above the trigger via the swing-high of a custom frame
    b = plan["buy"]
    assert b["entry"] is not None
    for n, v in b["levels_in_the_way"]:
        assert b["entry"] < v < b["targets"][0]


def test_level_in_the_way_is_found_with_a_custom_frame():
    idx = pd.date_range("2026-11-26 09:15", periods=40, freq="3min")
    c = np.full(40, 100.0)
    d = pd.DataFrame({"Open": c, "High": c + 1.0, "Low": c - 1.0, "Close": c, "ATR": 1.0, "EMA20": 99.5, "VWAP": 99.0}, index=idx)
    plan = tp.build_plan(d, "3m", resistance=102.0, support=90.0, ext_atr=50)       # swing high = 101 (nearest) -> trigger 101.1; 102 sits before T1
    b = plan["buy"]
    assert b["trigger_level"] == "Swing high" and abs(b["trigger"] - 101.1) < 1e-9
    assert [n for n, _ in b["levels_in_the_way"]] == ["Resistance"] and "only" in b["note"]


def test_build_plan_never_raises_on_bad_input():
    assert tp.build_plan(None)["available"] is False
    assert tp.build_plan(pd.DataFrame())["available"] is False
    assert tp.build_plan(_frame().tail(10))["available"] is False
    assert tp.plan_text(None, 1.0, 1.0) == ""


# ---------- chase ----------
def test_run_and_extension():
    idx = pd.date_range("2026-11-26 09:15", periods=40, freq="3min")
    c = np.r_[np.full(36, 100.0), 101.0, 102.0, 103.0, 104.0]
    d = pd.DataFrame({"Open": c, "High": c + 0.2, "Low": c - 0.2, "Close": c, "ATR": 1.0, "EMA20": 100.0}, index=idx)
    r = tp.run_info(d, "BUY")
    assert r["candles"] == 4 and abs(r["move"] - 4.0) < 1e-9 and abs(r["move_atr"] - 4.0) < 1e-9
    assert tp.run_info(d, "SELL")["candles"] == 0
    assert abs(tp.extension_atr(d, "BUY") - 4.0) < 1e-9 and abs(tp.extension_atr(d, "SELL") + 4.0) < 1e-9
    assert tp.chase_info(d, "BUY", 2.0)["late"] is True
    assert tp.chase_info(d, "SELL", 2.0)["late"] is False


def test_track_signal_resets_on_change_and_when_cleared():
    st = {}
    assert tp.track_signal(st, "k", "NO TRADE", 100, 1000) == (None, None)
    assert tp.track_signal(st, "k", "BUY", 100, 1000) == (0.0, 100.0)
    age, p0 = tp.track_signal(st, "k", "STRONG BUY", 101, 1045)
    assert age == 45.0 and p0 == 100.0                                           # BUY -> STRONG BUY is the same side: clock keeps running
    assert tp.track_signal(st, "k", "SELL", 99, 1050) == (0.0, 99.0)             # flipped side: new clock
    assert tp.track_signal(st, "k", "NO TRADE", 99, 1060) == (None, None)


# ---------- open trade alert ----------
def test_flip_alerts_index_and_option_rows():
    rows = [{"Symbol": "NIFTY", "Side": "BUY", "Notes": ""},
            {"Symbol": "NIFTY 25400 PE 25NOV", "Side": "BUY", "Notes": "Option paper BUY PE | [ikey=NSE_FO|1] [exp=2026-11-25]"},
            {"Symbol": "BANKNIFTY", "Side": "SELL", "Notes": ""}]
    assert tp.open_trade_direction(rows[1]) == ("NIFTY", "SELL")                  # long put = bearish
    a = tp.flip_alerts(rows, "NIFTY", "NO TRADE")
    assert len(a) == 2 and all(x["level"] == "warning" for x in a)
    a = tp.flip_alerts(rows, "NIFTY", "STRONG BUY")                               # index BUY still agrees; the put is now against the signal
    assert len(a) == 1 and a[0]["level"] == "error" and "PE" in a[0]["text"]
    assert tp.flip_alerts(rows, "NIFTY", "SELL")[0]["level"] == "error"           # index BUY against SELL
    assert tp.flip_alerts([], "NIFTY", "BUY") == [] and tp.flip_alerts(None, "NIFTY", "BUY") == []
    assert tp.flip_alerts(rows, "SENSEX", "NO TRADE") == []                       # other symbols are ignored


# ---------- optional anti-chase guard inside analyze_signal ----------
def _signal(d, **sw):
    NS["st"].session_state.clear(); NS["st"].session_state.update(sw)
    return NS["analyze_signal"](d, timeframe_label="5m")


def _buy_frame():
    for seed in range(1, 40):
        for drift in (3.0, 4.0, 5.0):
            d = _frame(seed, drift)
            if _signal(d)["signal_type"] == "buy":
                return d
    raise AssertionError("no BUY sample found")


def test_guard_off_changes_nothing_and_on_blocks_only_stretched_signals():
    d = _buy_frame()
    base = _signal(d)
    off = _signal(d, chase_guard_on=False, chase_atr=0.1)
    assert off["signal_text"] == base["signal_text"] and off["net_score"] == base["net_score"] and off["reasons"] == base["reasons"]
    ext = tp.extension_atr(d, "BUY")
    on_tight = _signal(d, chase_guard_on=True, chase_atr=max(0.01, ext - 0.05))    # limit below the real stretch -> blocked
    assert on_tight["signal_type"] == "neutral" and any("Anti-chase" in str(r) for r in on_tight["reasons"])
    on_loose = _signal(d, chase_guard_on=True, chase_atr=ext + 0.5)                 # limit above the real stretch -> untouched
    assert on_loose["signal_text"] == base["signal_text"] and on_loose["net_score"] == base["net_score"]


def test_dashboard_card_module_renders_and_skips_safely():
    import trade_plan_ui as tpu
    from unittest.mock import MagicMock
    st = MagicMock(); st.session_state = {}
    st.columns = lambda n: [MagicMock() for _ in range(n)]
    d = _frame(); NS["st"].session_state.clear()
    a = NS["analyze_signal"](d, timeframe_label="3m")
    errs = []
    journal = pd.DataFrame([{"Symbol": "NIFTY", "Side": "BUY", "Status": "OPEN", "Notes": ""}])
    tpu.render(st, "NIFTY", "3m", a, d, load_journal=lambda: journal, log_error=lambda *x: errs.append(x), clock=lambda: 1000.0)
    assert errs == [] and st.expander.called
    assert st.warning.called or st.error.called                                    # the open BUY vs this sample's NO TRADE / SELL signal
    st2 = MagicMock(); st2.session_state = {"tp_on": False}
    tpu.render(st2, "NIFTY", "3m", a, d)                                           # switched off: nothing drawn
    assert not st2.expander.called
    tpu.render(MagicMock(session_state={}), "NIFTY", "3m", None, None)             # bad input: silent


def test_sell_text_says_below_and_low_volatility_is_flagged():
    idx = pd.date_range("2026-11-26 09:15", periods=40, freq="3min")
    c = np.full(40, 100.0)
    d = pd.DataFrame({"Open": c, "High": c + 1.0, "Low": c - 1.0, "Close": c, "ATR": 1.0, "EMA20": 99.5, "VWAP": 99.0}, index=idx)
    plan = tp.build_plan(d, "3m", resistance=104.0, support=96.0, ext_atr=50)
    t_buy, t_sell = tp.plan_text(plan["buy"], plan["atr"], 100.0), tp.plan_text(plan["sell"], plan["atr"], 100.0)
    assert t_buy.startswith("BUY above") and "+ buffer" in t_buy
    assert t_sell.startswith("SELL below") and "- buffer" in t_sell and "SELL above" not in t_sell
    assert plan["low_vol"] is False
    flat = d.assign(ATR=0.005)                                                      # 0.005% of price
    assert tp.build_plan(flat, "3m")["low_vol"] is True


def test_stop_cap_is_reported_when_the_swing_is_farther_than_the_cap():
    idx = pd.date_range("2026-11-26 09:15", periods=40, freq="3min")
    c = np.full(40, 100.0)
    lows = c - 1.0; lows[-5] = 90.0                                                 # a deep swing low 10 points away, ATR is 1
    d = pd.DataFrame({"Open": c, "High": c + 1.0, "Low": lows, "Close": c, "ATR": 1.0, "EMA20": 99.5, "VWAP": 99.0}, index=idx)
    b = tp.build_plan(d, "3m", resistance=104.0, support=96.0, ext_atr=50)["buy"]
    assert b["stop_capped"] is True and abs(b["risk"] - 2.0) < 1e-9 and b["structure_stop"] < b["stop"] and "capped" in b["note"]
    shallow = d.assign(Low=c - 0.5)                                                 # swing 1.7 pts from the 101.1 entry: inside the 2 ATR cap
    b2 = tp.build_plan(shallow, "3m", resistance=104.0, support=96.0, ext_atr=50)["buy"]
    assert b2["stop_capped"] is False and "capped" not in b2["note"]
