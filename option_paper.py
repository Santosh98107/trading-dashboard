"""Option-premium paper trading helpers (BUY CE / BUY PE only - no option writing).

Pure functions, no network. Upstox field names follow the public instrument master
(complete.json.gz): segment, instrument_type, underlying_symbol, strike_price, expiry (epoch ms),
lot_size, instrument_key, trading_symbol. Contract picking is tested with sample rows only; the real
file/LTP calls were NOT run against Upstox here.
"""
import re
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
SEGMENT = {"NIFTY": "NSE_FO", "BANKNIFTY": "NSE_FO", "FINNIFTY": "NSE_FO", "SENSEX": "BSE_FO"}
UNDERLYING_NAMES = {"NIFTY": {"NIFTY"}, "BANKNIFTY": {"BANKNIFTY"}, "FINNIFTY": {"FINNIFTY"}, "SENSEX": {"SENSEX"}}
TAG_RE = re.compile(r"\[ikey=([^\]\s]+)\]")
EXP_RE = re.compile(r"\[exp=(\d{4}-\d{2}-\d{2})\]")


def _expiry_date(v):
    try:
        x = float(v)
        if x > 1e11:
            x /= 1000.0
        return datetime.fromtimestamp(x, IST).date()
    except Exception:
        return None


def pick_contract(rows, underlying, spot, signal, today=None):
    """BUY signal -> nearest-expiry ATM CE, SELL signal -> nearest-expiry ATM PE. None if nothing safe is found."""
    u = str(underlying).upper()
    if u not in SEGMENT or not spot or spot <= 0 or signal not in ("BUY", "SELL"):
        return None
    want = "CE" if signal == "BUY" else "PE"
    today = today or datetime.now(IST).date()
    cands = []
    for r in rows or []:
        try:
            if str(r.get("segment", "")).upper() != SEGMENT[u]:
                continue
            if str(r.get("instrument_type", "")).upper() != want:
                continue
            if str(r.get("underlying_symbol", r.get("name", ""))).strip().upper() not in UNDERLYING_NAMES[u]:
                continue
            exp = _expiry_date(r.get("expiry"))
            strike = float(r.get("strike_price"))
            lot = int(float(r.get("lot_size")))
            if exp is None or exp < today or lot <= 0 or strike <= 0 or not r.get("instrument_key"):
                continue
            cands.append((exp, abs(strike - spot), strike, lot, r))
        except Exception:
            continue
    if not cands:
        return None
    first_exp = min(c[0] for c in cands)
    best = min((c for c in cands if c[0] == first_exp), key=lambda c: c[1])
    exp, dist, strike, lot, r = best
    if dist > spot * 0.02:          # nearest strike more than 2% from spot = wrong / thin data
        return None
    return {"instrument_key": str(r["instrument_key"]), "trading_symbol": str(r.get("trading_symbol", "")),
            "strike": strike, "expiry": exp, "lot_size": lot, "type": want, "underlying": u,
            "label": f"{u} {int(strike) if strike == int(strike) else strike} {want} {exp.strftime('%d%b%y').upper()}"}


def premium_sane(premium, spot):
    """A real ATM weekly/monthly premium is a small fraction of spot. Anything else = wrong instrument / bad quote."""
    try:
        return premium is not None and 0.5 <= float(premium) <= float(spot) * 0.10
    except Exception:
        return False


def premium_levels(entry, sl_pct=30.0, t1_pct=25.0, t2_pct=50.0, t3_pct=80.0):
    entry = float(entry)
    stop = round(entry * (1 - sl_pct / 100.0), 2)
    return {"stop_loss": stop, "target1": round(entry * (1 + t1_pct / 100.0), 2), "target2": round(entry * (1 + t2_pct / 100.0), 2),
            "target3": round(entry * (1 + t3_pct / 100.0), 2), "trailing_stop": stop,
            "atr": round(max(entry * sl_pct / 100.0 / 1.5, 0.05), 2)}


def make_notes(contract, extra=""):
    return f"Option paper BUY {contract['type']} | [ikey={contract['instrument_key']}] [exp={contract['expiry']}] {extra}".strip()


def key_from_notes(notes):
    m = TAG_RE.search(str(notes or ""))
    return m.group(1) if m else None


def expiry_from_notes(notes):
    m = EXP_RE.search(str(notes or ""))
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None
    except Exception:
        return None


def is_option_row(notes):
    return key_from_notes(notes) is not None
