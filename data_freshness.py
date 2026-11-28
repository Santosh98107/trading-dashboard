"""Data freshness: how old is the newest candle? Display only; never changes a signal.

A candle is stamped with its OPEN time, so a healthy feed is at most one timeframe old. 'Late' = older than 2 timeframes + 1 minute while the
NSE cash market is open (Mon-Fri 09:15-15:30 IST). Exchange holidays are not known, so on a holiday this can show a false 'late'.
"""
import pandas as pd


def _ist_now():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def candle_age(data, timeframe, now=None):
    """(minutes_behind, is_late, text)."""
    try:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None, False, "no candles"
        last = pd.Timestamp(data.index[-1])
        if last.tzinfo is not None:
            last = last.tz_convert("Asia/Kolkata").tz_localize(None)
        now = pd.Timestamp(now) if now is not None else _ist_now()
        if now.tzinfo is not None:
            now = now.tz_convert("Asia/Kolkata").tz_localize(None)
        t = str(timeframe).strip().lower()
        tf_min = float(t[:-1]) if t.endswith("m") and t[:-1].isdigit() else (60.0 if t.endswith("h") else 0.0)
        age = max(0.0, (now - last).total_seconds() / 60.0)
        hm = (now.hour, now.minute)
        open_now = now.weekday() < 5 and (9, 15) <= hm <= (15, 30)
        late = bool(tf_min and open_now and age > 2 * tf_min + 1)
        return age, late, f"newest candle {last:%d %b %H:%M} is {age:.0f} min old (timeframe {timeframe})"
    except Exception as exc:
        return None, False, f"candle age unavailable ({type(exc).__name__})"


def render(st, data, timeframe, source="n/a", now=None):
    """Show a warning when late, otherwise a one-line caption. Never raises."""
    try:
        age, late, text = candle_age(data, timeframe, now)
        if late:
            st.warning(f"Signals may be LATE: {text}. The signal is calculated from the newest candle, so a late candle means a late signal. "
                       f"Source now: {source} (Yahoo can run behind the broker feed).")
        elif age is not None:
            st.caption("Data freshness: " + text)
    except Exception:
        pass
