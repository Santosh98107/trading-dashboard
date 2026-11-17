"""Profit ladder: pure stop maths plus a full run through the real journal files (isolated temp folder)."""
import os, subprocess, sys, textwrap
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pure():
    sys.path.insert(0, ROOT)
    for m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
        sys.modules.setdefault(m, MagicMock())
    import risk_manager as rm
    return rm


CFG = dict(profit_lock_on=True, pl_be_arm_r=1.25, pl_be_buffer_r=0.05, pl_gb_arm_r=1.5, pl_gb_low=0.5, pl_gb_mid=0.4, pl_gb_high=0.3,
           pl_lock2_r=0.75, pl_opp_exit_on=False, pl_time_stop_on=False, pl_weak_r=0.5)


def test_no_lock_below_arm_level():
    rm = _load_pure()
    assert rm.ladder_stop_price(False, 25249.6, 75.75, 25249.6 - 1.0 * 75.75, 10.0, CFG)[0] is None     # peak +1.0R: nothing yet


def test_breakeven_includes_charges_and_buffer_sell_and_buy():
    rm = _load_pure()
    sell, stage = rm.ladder_stop_price(False, 25249.6, 75.75, 25249.6 - 1.3 * 75.75, 10.0, CFG)
    assert stage == "BE" and abs(sell - (25249.6 - (10.0 + 0.05 * 75.75))) < 1e-6 and sell < 25249.6      # SELL: stop below entry, locks a small profit
    buy, _ = rm.ladder_stop_price(True, 100.0, 1.0, 101.3, 0.1, CFG)
    assert abs(buy - (100.0 + 0.1 + 0.05)) < 1e-9


def test_giveback_tiers_and_only_tightens():
    rm = _load_pure()
    p1, s1 = rm.ladder_stop_price(True, 100.0, 1.0, 101.7, 0.0, CFG)       # +1.7R, 50% give-back -> 0.85R
    assert abs(p1 - 100.85) < 1e-9 and "GB50" in s1
    p2, s2 = rm.ladder_stop_price(True, 100.0, 1.0, 102.5, 0.0, CFG)       # +2.5R, 40% -> 1.5R, and lock 0.75R is lower
    assert abs(p2 - 101.5) < 1e-9 and "LOCK" in s2 and "GB40" in s2
    p3, _ = rm.ladder_stop_price(True, 100.0, 1.0, 104.0, 0.0, CFG)        # +4R, 30% -> 2.8R
    assert abs(p3 - 102.8) < 1e-9


def test_full_flow_profit_does_not_turn_into_loss(tmp_path):
    """SELL trade runs to +1.7R, then price reverses past the original stop. Ladder OFF loses ~1R; ladder ON exits in profit."""
    script = textwrap.dedent(f"""
        import os, sys, types
        from unittest.mock import MagicMock
        os.environ["ELITE_DATA_DIR"] = r"{tmp_path}"
        sys.path.insert(0, r"{ROOT}")
        for m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
            sys.modules[m] = MagicMock()
        import pandas as pd
        import risk_manager as rm
        rm.st = types.SimpleNamespace(session_state={{"paper_cost_bps": 2.0, "profit_lock_on": __ON__}})
        rm.LOSS_FEATURE_COLUMNS = []                      # supplied by app.py in the real app
        rm.classify_trade_loss = lambda *a, **k: "test"
        rm.init_journal(); rm.init_paper_state()
        entry = 25249.6; risk = 75.7488
        ok = rm.add_trade_to_journal("NIFTY", "5m", "SELL", entry, 10, entry + risk, entry - 2*risk, entry - 3*risk, entry - 4*risk, entry + risk,
                                     {{"signal_text": "SELL", "atr": 20.0}}, auto_exit=True)
        for px in (25230, 25180, 25120, 25140, 25200, 25260, 25330):       # peak 25120 = +1.7R, then back up through the stop
            rm.auto_paper_trade_exit(px, symbol="NIFTY", atr=20.0)
        j = rm.load_journal()
        r = j.iloc[0]
        print(r["Status"], round(float(r["RealizedPnL"]), 2), str(r.get("LockStage", "")), str(r["Notes"])[-60:])
    """)
    out = {}
    for flag in ("False", "True"):
        res = subprocess.run([sys.executable, "-I", "-c", script.replace("__ON__", flag)], capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, res.stderr[-800:]
        out[flag] = res.stdout.strip().splitlines()[-1]
        for f in os.listdir(tmp_path):
            if f.endswith((".db", ".csv", ".json")):
                try: os.remove(os.path.join(tmp_path, f))
                except Exception: pass
    off, on = out["False"].split(), out["True"].split()
    assert off[0] == "CLOSED" and float(off[1]) < 0            # today: the profit is lost and the trade ends in a loss
    assert on[0] == "CLOSED" and float(on[1]) > 0              # ladder: the same path ends in a profit
    assert "Profit Lock" in out["True"]


def test_auto_exit_mode_survives_widget_state_loss(tmp_path):
    """Streamlit drops a widget's value on a run where it is not drawn. The mode must not fall back to Manual."""
    script = textwrap.dedent(f"""
        import os, sys, types
        from unittest.mock import MagicMock
        os.environ["ELITE_DATA_DIR"] = r"{tmp_path}"
        sys.path.insert(0, r"{ROOT}")
        for m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
            sys.modules[m] = MagicMock()
        import risk_manager as rm
        ss = {{}}
        class St:
            session_state = ss
            def radio(self, label, options=None, key=None, **k): return ss[key]
            def success(self, *a, **k): pass
            def info(self, *a, **k): pass
        rm.st = St()
        rm.LOSS_FEATURE_COLUMNS = []; rm.classify_trade_loss = lambda *a, **k: "t"
        rm.init_journal(); rm.init_paper_state()
        print("fresh", rm.render_paper_exit_mode_selector())                       # new session, nothing open -> Manual
        ss["paper_exit_mode"] = "Auto Exit (Enabled)"                                # user picks Auto
        print("picked", rm.render_paper_exit_mode_selector())
        del ss["paper_exit_mode"]                                                    # Streamlit forgets the widget on a run without it
        print("after_loss", rm.render_paper_exit_mode_selector())
        entry = 25249.6; risk = 75.7488
        rm.add_trade_to_journal("NIFTY", "5m", "SELL", entry, 10, entry + risk, entry - 2*risk, entry - 3*risk, entry - 4*risk, entry + risk,
                                {{"signal_text": "SELL", "atr": 20.0}}, auto_exit=True)
        print("active", rm.paper_auto_exit_active())
        ss.clear()                                                                   # brand-new browser session
        print("new_session", rm.render_paper_exit_mode_selector())                  # open trade carries AutoExitEnabled=True -> Auto
        ss.clear(); ss["paper_exit_mode"] = "Manual Exit"; rm.render_paper_exit_mode_selector()
        print("manual_wins", rm.paper_auto_exit_active())
    """)
    res = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-800:]
    out = dict(l.split(" ", 1) for l in res.stdout.strip().splitlines())
    assert out["fresh"] == "False" and out["picked"] == "True"
    assert out["after_loss"] == "True"            # the bug: this used to be False (reset to Manual)
    assert out["active"] == "True" and out["new_session"] == "True" and out["manual_wins"] == "False"
