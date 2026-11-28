"""Candle-age helper (display only). Run: python -m pytest tests/test_age.py -q"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import data_freshness
from unittest.mock import MagicMock
age = data_freshness.candle_age


def _df(ts, tz=None):
    return pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([ts], tz=tz))


def test_fresh_candle_is_not_late():
    a, late, txt = age(_df("2026-11-26 10:00"), "5m", now="2026-11-26 10:04")
    assert round(a) == 4 and late is False and "timeframe 5m" in txt


def test_old_candle_in_market_hours_is_late():
    a, late, _ = age(_df("2026-11-26 10:00"), "5m", now="2026-11-26 10:20")
    assert round(a) == 20 and late is True
    assert age(_df("2026-11-26 10:00"), "1m", now="2026-11-26 10:04")[1] is True        # 4 min > 2*1+1
    assert age(_df("2026-11-26 10:00"), "1m", now="2026-11-26 10:02")[1] is False


def test_never_late_outside_market_hours_or_weekend():
    assert age(_df("2026-11-26 15:25"), "5m", now="2026-11-26 18:00")[1] is False       # after close
    assert age(_df("2026-11-26 09:00"), "5m", now="2026-11-26 09:10")[1] is False       # before open
    assert age(_df("2026-11-27 15:25"), "5m", now="2026-11-28 11:00")[1] is False       # Saturday


def test_timezone_aware_index_is_converted_to_ist():
    a, late, _ = age(_df("2026-11-26 04:30", tz="UTC"), "5m", now="2026-11-26 10:04")     # 04:30 UTC = 10:00 IST
    assert round(a) == 4 and late is False


def test_bad_input_never_raises():
    assert age(None, "5m")[0] is None and age(pd.DataFrame(), "5m")[2] == "no candles"
    assert age(_df("2026-11-26 10:00"), "weird", now="2026-11-26 10:30")[1] is False      # unknown timeframe never warns


def test_render_shows_a_warning_when_late_and_a_caption_otherwise_and_never_raises():
    st = MagicMock()
    data_freshness.render(st, _df("2026-11-26 10:00"), "5m", "Yahoo", now="2026-11-26 10:30")
    assert st.warning.called and "LATE" in st.warning.call_args[0][0] and "Yahoo" in st.warning.call_args[0][0]
    st = MagicMock()
    data_freshness.render(st, _df("2026-11-26 10:00"), "5m", "Upstox", now="2026-11-26 10:03")
    assert st.caption.called and not st.warning.called
    data_freshness.render(MagicMock(), None, "5m")                                                # bad input: silent
