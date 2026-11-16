"""Tests for the signal-audit forward test, path capture, exit fitting and the meta-label model (pure ai_engine, no Streamlit)."""
import os, sys
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _m in ("plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
    sys.modules.setdefault(_m, MagicMock())
import ai_engine as ae


def _bars(highs, lows, closes, start="2026-11-16 09:15"):
    idx = pd.date_range(start, periods=len(highs), freq="5min")
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": 1000}, index=idx)


def _audit(rows):
    base = {"Symbol": "NIFTY", "Timeframe": "5m", "SignalType": "buy", "Price": 100.0, "StopLoss": 99.0, "Target1": 102.0,
            "Outcome": "", "OutcomeR": np.nan, "BarsToOutcome": np.nan, "ResolvedAt": ""}
    out = []
    for r in rows:
        d = dict(base); d.update(r); out.append(d)
    return pd.DataFrame(out)


def test_path_capture_after_target_hit():
    # signal at bar 0; bar1 reaches T1 (high 102.5); later bars drift up to 103.5 and dip to 99.5
    n = 14
    hi = [100.2, 102.5] + [103.5] * 12
    lo = [99.8, 100.0, 99.5] + [100.5] * 11
    cl = [100.0, 102.0] + [103.0] * 12
    data = _bars(hi, lo, cl)
    a = _audit([{"DateTime": data.index[0].strftime("%d-%m-%Y %H:%M:%S")}])
    a, done = ae.ai_lab_resolve_audit(a, data, "NIFTY", "5m", horizon=12, cost_bps=0.0)
    assert done == 1 and a.at[0, "Outcome"] == "T1"
    assert abs(a.at[0, "MFE_R"] - 3.5) < 1e-6           # best move 103.5 vs entry 100, risk 1
    assert abs(a.at[0, "MAE_R"] - (-0.5)) < 1e-6        # worst 99.5
    assert abs(a.at[0, "CloseR"] - 3.0) < 1e-6          # last close of the 12-bar horizon


def test_path_not_filled_until_horizon_exists():
    data = _bars([100.2, 102.5, 103], [99.8, 100.0, 100.5], [100, 102, 102.5])     # only 2 bars after the signal
    a = _audit([{"DateTime": data.index[0].strftime("%d-%m-%Y %H:%M:%S")}])
    a, done = ae.ai_lab_resolve_audit(a, data, "NIFTY", "5m", horizon=12)
    assert done == 1 and a.at[0, "Outcome"] == "T1" and pd.isna(a.at[0, "MFE_R"])
    longer = _bars([100.2, 102.5] + [103] * 12, [99.8, 100.0] + [100.5] * 12, [100, 102] + [102.5] * 12)
    a, _ = ae.ai_lab_resolve_audit(a, longer, "NIFTY", "5m", horizon=12)
    assert a.at[0, "Outcome"] == "T1" and np.isfinite(a.at[0, "MFE_R"])             # outcome untouched, path now filled


def test_sell_side_path_sign():
    hi = [100.2, 100.8] + [100.5] * 12
    lo = [99.8, 98.0] + [98.5] * 12
    cl = [100.0, 98.5] + [98.5] * 12
    data = _bars(hi, lo, cl)
    a = _audit([{"DateTime": data.index[0].strftime("%d-%m-%Y %H:%M:%S"), "SignalType": "sell", "StopLoss": 101.0, "Target1": 98.0}])
    a, _ = ae.ai_lab_resolve_audit(a, data, "NIFTY", "5m", horizon=12)
    assert a.at[0, "MFE_R"] > 1.9 and a.at[0, "MAE_R"] < 0           # price fell in favour of the short, rose a bit against it


def _path_audit(n, mfe, mae, close, start="2026-11-10 09:15"):
    dts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"Symbol": "NIFTY", "Timeframe": "5m", "SignalType": "buy", "Price": 100.0, "StopLoss": 99.0, "Target1": 102.0,
                         "DateTime": [d.strftime("%d-%m-%Y %H:%M:%S") for d in dts], "Outcome": "TIMEOUT", "OutcomeR": 0.0,
                         "MFE_R": mfe, "MAE_R": mae, "CloseR": close})


def test_exit_fit_needs_enough_signals():
    rows = ae.ai_lab_fit_exits(_path_audit(10, 1.0, -0.3, 0.5), min_n=50)
    assert rows and rows[0]["Status"].startswith("Need") and np.isnan(rows[0]["Best stop (x)"])


def test_exit_fit_finds_wider_target_when_moves_run():
    # every signal runs +3R without ever going below -0.3R: a 3R target and a tight-ish stop must beat the default 1R/1R shape
    rows = ae.ai_lab_fit_exits(_path_audit(120, 3.0, -0.3, 2.0), min_n=50)
    r = rows[0]
    assert r["Status"] == "Fitted" and r["Best target (x)"] >= 2.0 and r["Newer 40% (out of sample)"] > 0
    assert "worth a paper test" in r["Verdict"]


def test_exit_fit_rarely_calls_noise_an_edge():
    # pure random-walk paths (no edge; fine-grained so stops fill at their level like real intrabar highs/lows): the fit may find a "best" pair on older signals, but the verdict must almost never say it holds
    false_alarms = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = 300
        path = np.cumsum(rng.normal(0, 0.35 / np.sqrt(30), (n, 12 * 30)), axis=1)   # fine steps: each bar's high/low is the extreme of 30 sub-steps
        a = _path_audit(n, 0.0, 0.0, 0.0)
        a["MFE_R"] = np.maximum(path.max(axis=1), 0); a["MAE_R"] = np.minimum(path.min(axis=1), 0); a["CloseR"] = path[:, -1]
        r = ae.ai_lab_fit_exits(a, cost_bps=2.0, min_n=50)[0]
        false_alarms += "worth a paper test" in r["Verdict"]
    assert false_alarms <= 4, false_alarms          # <= 10% of 40 random datasets


def _meta_audit(n, signal, seed=0):
    rng = np.random.default_rng(seed)
    adx = rng.uniform(10, 45, n); tq = rng.uniform(30, 95, n); rsi = rng.uniform(35, 75, n)
    z = (adx - 27) / 8.0 if signal else rng.normal(0, 1, n)
    win = rng.random(n) < 1.0 / (1.0 + np.exp(-1.6 * z))
    R = np.where(win, rng.uniform(0.8, 2.0, n), -1.0)
    dts = pd.date_range("2026-10-01 09:30", periods=n, freq="45min")
    return pd.DataFrame({"DateTime": [d.strftime("%d-%m-%Y %H:%M:%S") for d in dts], "Symbol": "NIFTY", "Timeframe": "5m",
                         "SignalType": rng.choice(["buy", "sell"], n), "ADX": adx, "TradeQualityScore": tq, "RSI": rsi,
                         "ValidationScore": rng.uniform(40, 90, n), "VWAPDistPct": rng.normal(0, .3, n), "TimeBucket": "10:00",
                         "Outcome": np.where(win, "T1", "STOP"), "OutcomeR": R})


def test_meta_waits_for_enough_signals():
    m = ae.ai_lab_train_meta(_meta_audit(60, True))
    assert m["status"] == "learning" and m["need"] == ae.META_MIN_SIGNALS


def test_meta_learns_real_signal_and_predicts_direction():
    a = _meta_audit(600, True)
    m = ae.ai_lab_train_meta(a)
    assert m["status"] == "active" and m["preq"]["auc"] > 0.6
    hi = ae.ai_lab_meta_predict(m, {"SignalType": "buy", "ADX": 42, "TradeQualityScore": 70, "RSI": 55, "ValidationScore": 60, "VWAPDistPct": 0.1, "TimeBucket": "10:00"})
    lo = ae.ai_lab_meta_predict(m, {"SignalType": "buy", "ADX": 12, "TradeQualityScore": 70, "RSI": 55, "ValidationScore": 60, "VWAPDistPct": 0.1, "TimeBucket": "10:00"})
    assert hi > lo + 0.2


def test_meta_refuses_noise():
    active = sum(ae.ai_lab_train_meta(_meta_audit(400, False, seed=s))["status"] == "active" for s in range(20))
    assert active <= 3                    # at most ~15% of pure-noise datasets slip through the walk-forward test


def test_meta_predict_none_when_not_active():
    assert ae.ai_lab_meta_predict({"status": "learning"}, {"SignalType": "buy"}) is None
