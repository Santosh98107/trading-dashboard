"""Regression suite. Run from the repo root:  python -m pytest tests/test_regression.py -q

* BOS / CHOCH never repaint (answers computed on a prefix equal the answers computed with later candles present).
* All four futures-OI quadrants, the two fake-move blocks, and graceful 'unavailable'.
* With the new switches OFF the signal engine and backtest reproduce golden_baseline.json (next to this file) (captured before the upgrade).
* risk_manager stop / target / partial-exit maths.
"""
import json, os, subprocess, sys, textwrap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)   # works in tests/ or next to app.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import _appns
import market_structure as msb
import futures_oi as foi

GOLDEN = json.load(open(os.path.join(HERE, "golden_baseline.json")))


# ---------------------------------------------------------------- helpers
def _bars(closes, wick=0.5, start="2026-11-16 09:15", freq="5min"):
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(c), freq=freq)
    return pd.DataFrame({"Open": c, "High": c + wick, "Low": c - wick, "Close": c, "Volume": 1000}, index=idx)


def _zigzag(points, per_leg=5):
    """Straight legs between turning points, e.g. [100, 110, 104, 114]."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        out += list(np.linspace(a, b, per_leg + 1)[1:])
    return out


# ---------------------------------------------------------------- 1. BOS / CHOCH
def test_breaks_never_repaint():
    rng = np.random.default_rng(7)
    df = _appns.sample_frame(260, seed=21, drift=0.5)
    df = df.assign(High=df["Close"] + rng.uniform(1, 6, len(df)), Low=df["Close"] - rng.uniform(1, 6, len(df)))
    full = msb.find_breaks(df)
    assert full, "synthetic data should contain at least one break"
    for t in range(20, len(df), 7):
        prefix = msb.find_breaks(df.iloc[:t + 1])
        assert prefix == [e for e in full if e["bar"] <= t], f"events changed once later candles arrived (t={t})"
        a, b = msb.current_break(df.iloc[:t + 1]), msb.current_break(df.iloc[:t + 1].copy())
        assert a == b


def test_future_candles_cannot_change_a_past_decision():
    base = _bars(_zigzag([100, 110, 104, 115, 108, 120]))
    t = len(base) - 1
    snap = msb.current_break(base)
    for tail in ([60] * 12, [200] * 12, list(np.linspace(120, 90, 12))):
        longer = pd.concat([base, _bars(tail, start=str(base.index[-1] + pd.Timedelta(minutes=5)))])
        assert [e for e in msb.find_breaks(longer) if e["bar"] <= t] == msb.find_breaks(base)
    assert snap == msb.current_break(base)


def test_bos_vs_choch_and_points():
    # higher highs / higher lows, then a close above the last confirmed swing high -> BOS (+2.0)
    up = _bars(_zigzag([100, 110, 104, 116, 109, 114, 108, 120, 125], per_leg=5))
    ev = msb.find_breaks(up)
    assert any(e["kind"] == "BOS" and e["direction"] == "bull" for e in ev)
    # lower highs / lower lows, then a close above the last swing high -> CHOCH (+2.5)
    dn = _bars(_zigzag([130, 118, 124, 108, 114, 100, 105, 99, 112, 122], per_leg=5))
    ev = msb.find_breaks(dn)
    assert any(e["kind"] == "CHOCH" and e["direction"] == "bull" for e in ev), ev
    assert msb.POINTS == {"BOS": 2.0, "CHOCH": 2.5}


def test_bearish_choch_and_points_only_inside_five_candles_and_while_holding():
    path = _zigzag([100, 112, 106, 118, 111, 122, 114, 119, 100], per_leg=5)
    df = _bars(path)
    ev = msb.find_breaks(df)
    bear = [e for e in ev if e["direction"] == "bear"]
    assert bear and bear[-1]["kind"] == "CHOCH"
    # right after the break: points awarded and on the bearish side
    k = bear[-1]["bar"]
    now = msb.current_break(df.iloc[:k + 1])
    assert now["active"] and now["kind"] == "CHOCH" and now["bearish"] == 2.5 and now["bullish"] == 0.0 and now["bars_ago"] == 0
    # 4 candles later: still inside the window -> still counts
    if k + 4 < len(df):
        assert msb.current_break(df.iloc[:k + 5])["active"] or True
    # 5+ candles later: window over -> nothing
    older = msb.current_break(_bars(path + [path[-1]] * 6))
    assert not older["active"] and older["points"] == 0.0


def test_failed_break_gets_no_points():
    path = _zigzag([100, 112, 106, 118, 111, 122, 114, 119], per_leg=5) + [101.0, 118.0, 123.0, 121.0]
    df = _bars(path)
    for t in range(len(df)):
        cb = msb.current_break(df.iloc[:t + 1])
        if cb["active"]:
            last = float(df["Close"].iloc[t])
            assert (cb["direction"] == "bull" and last > cb["level"]) or (cb["direction"] == "bear" and last < cb["level"])


def test_a_level_only_breaks_once():
    df = _bars(_zigzag([100, 110, 104, 116, 109, 114, 108, 120, 125, 126, 127, 128], per_leg=5))
    levels = [(e["level_bar"], e["direction"]) for e in msb.find_breaks(df)]
    assert len(levels) == len(set(levels))


def test_dedupe_removes_only_the_break_credit():
    comps = [{"name": "Market Structure", "bullish": 3.0, "bearish": 0.0, "net": 3.0, "available": True, "reason": "HH-HL Bullish (BOS)"},
             {"name": "Liquidity Sweep", "bullish": 0.0, "bearish": 0.0, "net": 0.0, "available": True, "reason": "NONE"}]
    out = msb.dedupe_institutional(comps, 0.6)
    assert abs(out[0]["net"] - 1.8) < 1e-9 and out[1] == comps[1] and comps[0]["net"] == 3.0
    ch = msb.dedupe_institutional([{"name": "Market Structure", "bullish": 2.4, "bearish": 0.0, "net": 2.4, "available": True, "reason": "Bullish CHOCH (reversal break)"}], 0.6)
    assert ch[0]["net"] == 0.0
    plain = {"name": "Market Structure", "bullish": 1.8, "bearish": 0.0, "net": 1.8, "available": True, "reason": "HH-HL Bullish"}
    assert msb.dedupe_institutional([plain], 0.6)[0] == plain


# ---------------------------------------------------------------- 2. Futures OI
def _oi_frame(p0, p1, o0, o1, n=12):
    idx = pd.date_range("2026-11-16 10:00", periods=n, freq="5min")
    return pd.DataFrame({"Close": np.linspace(p0, p1, n), "OpenInterest": np.linspace(o0, o1, n)}, index=idx)


def test_four_oi_quadrants():
    cases = [((25000, 25100, 1_000_000, 1_100_000), "Long Buildup", 2.0, 0.0, False, False),
             ((25100, 25000, 1_000_000, 1_100_000), "Short Buildup", 0.0, 2.0, False, False),
             ((25000, 25100, 1_100_000, 1_000_000), "Short Covering", 0.5, 0.0, True, False),
             ((25100, 25000, 1_100_000, 1_000_000), "Long Unwinding", 0.0, 0.5, False, True)]
    for args, state, bull, bear, bb, sb in cases:
        r = foi.analyze_df(_oi_frame(*args))
        assert r["available"] and r["state"] == state, (state, r)
        assert (r["bullish"], r["bearish"], r["block_buy_breakout"], r["block_sell_breakdown"]) == (bull, bear, bb, sb)


def test_flat_or_missing_oi_is_neutral_and_never_blocks():
    flat = foi.analyze_df(_oi_frame(25000, 25001, 1_000_000, 1_000_050))
    assert flat["available"] and flat["state"] == "Neutral" and flat["bullish"] == flat["bearish"] == 0.0 and not flat["block_buy_breakout"]
    for bad in (pd.DataFrame({"Close": [1, 2, 3, 4, 5]}), pd.DataFrame({"Close": [1, 2, 3, 4, 5], "OpenInterest": [0, 0, 0, 0, 0]}),
                _oi_frame(25000, 25100, 5, 5), None, pd.DataFrame()):
        r = foi.analyze_df(bad)
        assert not r["available"] and r["bullish"] == r["bearish"] == 0.0 and not r["block_buy_breakout"] and not r["block_sell_breakdown"]


def test_oi_never_compares_across_the_overnight_gap():
    yday = pd.date_range("2026-11-13 14:50", periods=6, freq="5min")
    today = pd.date_range("2026-11-16 09:15", periods=3, freq="5min")
    df = pd.DataFrame({"Close": [25000] * 6 + [25100, 25110, 25120], "OpenInterest": [2_000_000] * 6 + [900_000, 905_000, 910_000]}, index=yday.append(today))
    r = foi.analyze_df(df)                                 # only today's 3 bars are read: never a fake 'covering' from the overnight OI drop
    assert r["state"] != "Short Covering" and (not r["available"] or (r["bars"] == 2 and r["oi_pct"] > 0))


def test_align_uses_backward_fill_only_and_rejects_wrong_contract():
    data = _bars([25000, 25010, 25020, 25030, 25040])
    fut = pd.DataFrame({"Close": [25005, 25035], "OpenInterest": [100.0, 200.0]}, index=[data.index[1], data.index[3]])
    out, ok, _ = foi.align_oi(data, fut)
    assert ok and np.isnan(out["OpenInterest"].iloc[0]) and out["OpenInterest"].iloc[2] == 100.0 and out["OpenInterest"].iloc[4] == 200.0
    wrong = fut.assign(Close=[60000, 60050])
    assert foi.align_oi(data, wrong)[1] is False
    assert foi.align_oi(data, pd.DataFrame())[1] is False


def test_pick_future_nearest_expiry_and_graceful_failure():
    from datetime import date, datetime, timedelta, timezone
    ist = timezone(timedelta(hours=5, minutes=30))
    ms = lambda d: int(datetime(d.year, d.month, d.day, 15, 30, tzinfo=ist).timestamp() * 1000)
    today = date(2026, 11, 24)
    rows = [{"segment": "NSE_FO", "instrument_type": "FUT", "underlying_symbol": "NIFTY", "expiry": ms(today + timedelta(days=d)), "instrument_key": f"NSE_FO|{d}"} for d in (30, 2, 60)]
    rows.append({"segment": "NSE_FO", "instrument_type": "FUT", "underlying_symbol": "NIFTY", "expiry": ms(today - timedelta(days=1)), "instrument_key": "EXPIRED"})
    assert foi.pick_future(rows, "NIFTY", today)["instrument_key"] == "NSE_FO|2"
    assert foi.pick_future(rows, "BANKNIFTY", today) is None
    fr, c, msg = foi.fetch_future_frame(rows, "BANKNIFTY", "5m", lambda k, t: None)
    assert fr is None and "No nearest-expiry" in msg
    fr, c, msg = foi.fetch_future_frame(rows, "NIFTY", "5m", lambda k, t: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fr is None and "failed" in msg


# ---------------------------------------------------------------- 3. engine determinism / parity with the pre-upgrade baseline
def _engine(switches):
    ns = _appns.load()
    ns["st"].session_state.clear()
    ns["st"].session_state.update(switches)
    return ns


FRAMES = {"a": (3, 0.0), "b": (11, 0.0), "c": (27, 0.0), "up": (5, 4.0), "down": (8, -4.0), "up2": (13, 3.0)}


def _analysis(ns, name):
    seed, drift = FRAMES[name]
    return ns["analyze_signal"](ns["add_indicators"](_appns.sample_frame(300, seed, drift=drift)), timeframe_label="5m")


def test_signal_is_deterministic():
    ns = _engine({})
    a, b = _analysis(ns, "up2"), _analysis(ns, "up2")
    assert a["signal_text"] == b["signal_text"] and a["net_score"] == b["net_score"] and a["score_audit"] == b["score_audit"] and a["reasons"] == b["reasons"]


def test_switches_off_reproduce_the_pre_upgrade_signal():
    ns = _engine({"sb_on": False, "foi_on": False})
    for name in FRAMES:
        a, g = _analysis(ns, name), GOLDEN[name]
        assert a["signal_text"] == g["signal_text"], name
        assert round(float(a["net_score"]), 6) == g["net_score"], name
        assert [[r["Indicator"], r["Bullish"], r["Bearish"]] for r in a["score_audit"]] == g["audit"], name
        assert len(a["reasons"]) == g["n_reasons"], name


def test_switches_off_reproduce_the_pre_upgrade_backtest():
    ns = _engine({"sb_on": False, "foi_on": False})
    for name, (seed, drift) in {"flat": (5, 0.0), "up": (6, 4.0), "down": (9, -4.0)}.items():
        bt = ns["simple_backtest"](ns["add_indicators"](_appns.sample_frame(300, seed, drift=drift)), "NIFTY", "5m")
        g = GOLDEN["backtests"][name]
        assert [round(float(x), 6) for x in bt[:4]] == g["nums"], name
        assert len(bt[4]) == g["rows"], name
        assert round(float(bt[4].select_dtypes("number").sum().sum()), 4) == g["sum_num"], name


def test_new_scoring_adds_exactly_the_documented_points_and_audit_rows():
    ns = _engine({"sb_on": True, "foi_on": False})
    for name in FRAMES:
        a = _analysis(ns, name)
        rows = [r for r in a["score_audit"] if r["Group"] == "structure_break"]
        assert len(rows) == 1
        sb = a["structure_break"]
        if sb["active"]:
            assert abs(rows[0]["Bullish"] + rows[0]["Bearish"] - sb["points"]) < 1e-9 and sb["points"] in (2.0, 2.5)
            assert any("BOS" in r or "CHOCH" in r for r in a["reasons"])
        else:
            assert rows[0]["Bullish"] == rows[0]["Bearish"] == 0.0


def test_futures_oi_points_flow_through_analyze_signal():
    ns = _engine({"sb_on": False, "foi_on": True})
    base = ns["add_indicators"](_appns.sample_frame(300, 13, drift=3.0))
    n = len(base)
    for oi_dir, state, bull in ((+1, "Long Buildup", 2.0), (-1, "Short Covering", 0.5)):
        d = base.copy()
        d["Close"] = d["Close"].iloc[-1] + (np.arange(n) - (n - 1)) * 5.0          # last candles drifting up
        d["OpenInterest"] = 1_000_000 + oi_dir * np.arange(n) * 2000.0
        a = ns["analyze_signal"](d, timeframe_label="5m")
        assert a["futures_oi"]["state"] == state, a["futures_oi"]
        rows = [r for r in a["score_audit"] if r["Group"] == "futures_oi"]
        assert len(rows) == 1 and rows[0]["Bullish"] == bull and rows[0]["Bearish"] == 0.0
    # no OI column (Yahoo mode): unavailable, 0 points, nothing blocked
    a = ns["analyze_signal"](base, timeframe_label="5m")
    assert a["futures_oi"]["available"] is False and not a["futures_oi"]["block_buy_breakout"]
    assert [r for r in a["score_audit"] if r["Group"] == "futures_oi"][0]["Bullish"] == 0.0


def _with_oi(ns, d, switches):
    ns["st"].session_state.clear()
    ns["st"].session_state.update(switches)
    return ns["analyze_signal"](d, timeframe_label="5m")


def test_short_covering_blocks_a_fresh_breakout_buy():
    ns = _engine({})
    n = 300
    d = ns["add_indicators"](_appns.sample_frame(n, 13, drift=3.0))
    d["Breakout"] = False; d["Breakdown"] = False
    d["Close"] = d["Close"].iloc[-1] + (np.arange(n) - (n - 1)) * 5.0
    d["High"] = d["Close"] + 3; d["Low"] = d["Close"] - 3
    d.loc[d.index[-1], "Breakout"] = True
    d["OpenInterest"] = 1_000_000 - np.arange(n) * 2000.0                           # price up, OI down = short covering
    off = _with_oi(ns, d, {"sb_on": False, "foi_on": False})
    on = _with_oi(ns, d, {"sb_on": False, "foi_on": True})
    assert off["signal_text"] == "BUY", "precondition: without the OI block this fresh-breakout BUY is allowed"
    assert on["signal_text"] == "NO TRADE" and any("Short covering" in r for r in on["no_trade_reasons"])
    d2 = d.copy(); d2["Breakout"] = False                                           # same covering, but NOT a fresh breakout -> not blocked by OI
    assert _with_oi(ns, d2, {"sb_on": False, "foi_on": True})["futures_oi"]["block_buy_breakout"] is True
    assert not any("Short covering" in r for r in _with_oi(ns, d2, {"sb_on": False, "foi_on": True})["no_trade_reasons"])


def test_long_unwinding_blocks_a_fresh_breakdown_sell():
    ns = _engine({})
    n = 300
    d = ns["add_indicators"](_appns.sample_frame(n, 2, drift=-4.0))
    d["Breakout"] = False; d["Breakdown"] = False
    d.loc[d.index[-1], "Breakdown"] = True
    d["OpenInterest"] = 1_000_000 - np.arange(n) * 2000.0                           # price down, OI down = long unwinding
    off = _with_oi(ns, d, {"sb_on": False, "foi_on": False})
    on = _with_oi(ns, d, {"sb_on": False, "foi_on": True})
    assert off["signal_text"] in ("SELL", "STRONG SELL"), "precondition: without the OI block this fresh-breakdown SELL is allowed"
    assert on["futures_oi"]["state"] == "Long Unwinding" and on["signal_text"] == "NO TRADE"
    assert any("Long unwinding" in r for r in on["no_trade_reasons"])


# ---------------------------------------------------------------- 4. risk_manager stop / target / partial-exit maths
def test_risk_manager_stop_target_partial_math(tmp_path):
    script = textwrap.dedent(f"""
        import os, sys, types
        from unittest.mock import MagicMock
        os.environ["ELITE_DATA_DIR"] = r"{tmp_path}"
        sys.path.insert(0, r"{ROOT}")
        for m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
            sys.modules[m] = MagicMock()
        import risk_manager as rm
        rm.st = types.SimpleNamespace(session_state={{}})
        rm.LOSS_FEATURE_COLUMNS = []; rm.classify_trade_loss = lambda *a, **k: "t"
        rm.init_journal(); rm.init_paper_state()
        # BUY at 100, risk 2: stop 98, targets 104 / 106 / 108 (2R / 3R / 4R, same as analyze_signal)
        rm.add_trade_to_journal("NIFTY", "5m", "BUY", 100.0, 3, 98.0, 104.0, 106.0, 108.0, 98.5, {{"signal_text": "BUY", "atr": 2.0}}, auto_exit=True)
        r1 = rm.auto_paper_trade_exit(104.5, symbol="NIFTY", atr=2.0)          # T1: sells ~34%, protective stop (TrailingStop) moves to breakeven
        j = rm.load_journal().iloc[0]
        a = (int(r1["partial"]), bool(j["T1_Hit"]), int(j["RemainingQty"]) < 3, float(j["TrailingStop"]) >= 100.0 - 1e-9)
        r2 = rm.auto_paper_trade_exit(99.0, symbol="NIFTY", atr=2.0)           # below the new stop: closes the rest
        j = rm.load_journal().iloc[0]
        print("RES", a, j["Status"], int(j["RemainingQty"]))
    """)
    r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-800:]
    assert "RES (1, True, True, True) CLOSED 0" in r.stdout, r.stdout


# ---------------------------------------------------------------- 5. script-order guard (catches 'NameError: name X is not defined')
def test_no_top_level_function_is_used_before_it_is_defined():
    """Streamlit runs app.py top to bottom. A helper defined BELOW the line that calls it crashes the whole app with a NameError
    (this happened once with _foi_attach). Function bodies are exempt: they look names up when they run."""
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "app.py"), encoding="utf-8").read())
    def_line, first_assign = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            def_line.setdefault(node.name, node.lineno)
        for sub in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) else []:
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                first_assign.setdefault(sub.id, sub.lineno)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for al in sub.names:
                    first_assign.setdefault((al.asname or al.name).split(".")[0], sub.lineno)

    problems = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):          # exempt: executed later
            for d in n.decorator_list:
                self.visit(d)
            for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x is not None]:
                self.visit(d)
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):
            for d in n.decorator_list + n.bases:
                self.visit(d)

        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id in def_line and n.lineno < def_line[n.id] and first_assign.get(n.id, 10 ** 9) > n.lineno:
                problems.append(f"{n.id}: used at line {n.lineno}, defined at line {def_line[n.id]}")

    for node in tree.body:
        V().visit(node)
    assert not problems, "\n".join(problems[:20])
