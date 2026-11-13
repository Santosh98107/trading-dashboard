"""ai_engine.py - the whole AI Lab: weakness / strength analysis, strategy builder + evolution, self-learning model,
weekly report, calibration, prediction accuracy, copilot, trade replay, opportunity ranking and portfolio risk,
plus render_ai_lab() for the pages.  Split out of app.py; behaviour is unchanged."""
import os
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import *   # noqa: F401,F403

stocks = {}   # filled by bind()


def bind(**deps):
    """app.py passes in names that still live in the main file: add_indicators, fetch_yahoo_batch, stocks."""
    globals().update(deps)


# =========================================================
# ===================== AI LAB BEGIN ======================
# Weakness / Strength analyzers, Strategy Builder + Evolution, Self-Learning,
# Weekly Report, Confidence Calibration, Prediction Accuracy, Copilot, Trade Replay,
# Opportunity Ranking and Portfolio Risk.
# Everything is computed from YOUR journal, signal log, backtest and candles. When a sample is too
# small the feature says so; no number is invented.
# =========================================================
AI_LAB_EVOLUTION_FILE = os.path.join(DATA_DIR, "ai_lab_evolution.csv")
AI_LAB_MODEL_FILE = os.path.join(DATA_DIR, "ai_lab_model.json")
AI_LAB_REPORT_FILE = os.path.join(DATA_DIR, "ai_lab_weekly_reports.csv")
AI_LAB_DT_FMT = "%d-%m-%Y %H:%M:%S"
AI_LAB_MIN_MODEL_TRADES = 25

AL_SECTOR_MAP = {
    "NIFTY": "Index (broad)", "SENSEX": "Index (broad)", "BANKNIFTY": "Banking (index)",
    "RELIANCE": "Energy / Conglomerate", "ONGC": "Energy / Conglomerate",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "AXISBANK": "Banking", "KOTAKBANK": "Banking",
    "BAJFINANCE": "Financials", "LICI": "Financials",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "TMPV": "Auto", "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "LT": "Infrastructure", "ADANIPORTS": "Infrastructure",
    "BHARTIARTL": "Telecom", "SUNPHARMA": "Pharma", "TITAN": "Consumer", "TRENT": "Consumer",
    "NTPC": "Power", "POWERGRID": "Power", "BEL": "Defence", "ULTRACEMCO": "Cement",
    "JSWSTEEL": "Metals", "TATASTEEL": "Metals",
}

AL_FIX_TEXT = {
    "FOMO": "Skip entries when price is already >1% from EMA20 or RSI (in trade direction) is above 72. Wait for the pullback.",
    "Early Entry": "Wait for confirmation: require ADX >= 20 and a confirmed (not 'early alert') signal before entering.",
    "Late Entry": "You are entering after the move has run. Prefer an EMA20/EMA21 retest instead of chasing extended candles.",
    "Counter-Trend": "Trade only in the direction of EMA20 vs EMA50 and the higher-timeframe trend.",
    "SL Shift": "Losses exceeded the planned risk. Do not widen or remove the stop-loss; let the auto-exit work.",
    "Revenge Trading": "Take a cooling-off break after a loss. Do not re-enter within the revenge window.",
    "Overtrading": "Cap the number of trades per day; extra trades late in the day have lower quality.",
}


# ---------------------------------------------------------
# generic helpers
# ---------------------------------------------------------
def _al_num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _al_wilson_lb(wins, n, z=1.645):
    """Lower confidence bound of a win rate (penalises small samples)."""
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _al_perf(pnl):
    s = pd.Series(pnl).dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return {"n": 0, "wins": 0, "wr": np.nan, "pnl": 0.0, "avg": np.nan, "pf": np.nan}
    wins = int((s > 0).sum())
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else np.nan)
    return {"n": n, "wins": wins, "wr": wins / n * 100.0, "pnl": float(s.sum()), "avg": float(s.mean()), "pf": min(pf, 99.0) if pd.notna(pf) else np.nan}


def _al_week_key(dt=None):
    dt = dt or ist_now()
    iso = dt.isocalendar()
    return f"{int(iso[0])}-W{int(iso[1]):02d}"


def _al_memo(key, sig, fn):
    store = st.session_state.setdefault("al_memo", {})
    ent = store.get(key)
    if ent is not None and ent[0] == sig:
        return ent[1]
    val = fn()
    store[key] = (sig, val)
    return val


def _al_journal_sig(journal):
    """Changes only when a trade opens/closes or a closed trade changes, not on every live price tick."""
    try:
        if journal is None or journal.empty or "Status" not in journal.columns:
            return (0, 0, 0.0)
        is_closed = journal["Status"].astype(str).str.upper() == "CLOSED"
        pnl = pd.to_numeric(journal.loc[is_closed, "PnL"], errors="coerce").fillna(0.0).sum()
        return (int(len(journal)), int(is_closed.sum()), round(float(pnl), 2))
    except Exception:
        return (len(journal) if journal is not None else 0, 0, 0.0)


def _al_parse_dt(series):
    s = pd.Series(series).astype(str).replace({"nan": "", "None": "", "NaT": ""})
    out = pd.to_datetime(s, format=AI_LAB_DT_FMT, errors="coerce")
    if out.isna().any():
        alt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        out = out.fillna(alt)
    return out


# ---------------------------------------------------------
# journal preparation
# ---------------------------------------------------------
def _al_classify_setup(r):
    """Name the setup of one journal trade from the indicators stored at entry."""
    reg = str(r.get("MarketRegime", "")).lower()
    aligned = r.get("Aligned", np.nan)
    d20 = abs(_al_num(r.get("DistanceEMA20Pct")))
    dv = abs(_al_num(r.get("DistanceVWAPPct")))
    adx = _al_num(r.get("ADX"))
    rv = _al_num(r.get("RelativeVolume"))
    side = str(r.get("Side", "BUY")).upper()
    lvl = _al_num(r.get("ResistanceDistancePct") if side == "BUY" else r.get("SupportDistancePct"))
    if pd.notna(aligned) and float(aligned) == 0.0:
        return "Counter-Trend"
    if "breakout" in reg or "breakdown" in reg or (pd.notna(rv) and rv >= 1.5 and pd.notna(lvl) and abs(lvl) <= 0.3):
        return "Breakout"
    if pd.notna(d20) and d20 <= 0.35:
        return "EMA Pullback"
    if pd.notna(dv) and dv <= 0.25:
        return "VWAP Hold"
    if pd.notna(adx) and adx >= 25:
        return "Trend Continuation"
    if "range" in reg or "sideways" in reg:
        return "Range Trade"
    return "Other"


def _al_closed_trades(journal):
    if journal is None or journal.empty or "Status" not in journal.columns:
        return pd.DataFrame()
    df = journal[journal["Status"].astype(str).str.upper() == "CLOSED"].copy()
    if df.empty:
        return df
    num_cols = ["EntryPrice", "ExitPrice", "Quantity", "StopLoss", "PnL", "RealizedPnL", "EMA20", "EMA50", "RSI", "ADX",
                "VWAP", "RelativeVolume", "TrendSlope", "SupportDistancePct", "ResistanceDistancePct",
                "DistanceEMA20Pct", "DistanceVWAPPct", "RiskRewardRatio", "ConfidenceAtEntry", "HistoricalWinRateAtEntry"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in ["Symbol", "Timeframe", "Signal", "Side", "MarketRegime", "LossReason", "SignalStage", "MTFTrendStatus"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    df["PnLv"] = df["PnL"].where(df["PnL"].notna(), df["RealizedPnL"])
    df = df[df["PnLv"].notna()].copy()
    if df.empty:
        return df
    if "DateTime" in df.columns:
        df["Entry_DT"] = _al_parse_dt(df["DateTime"])
    else:
        df["Entry_DT"] = pd.NaT
    if df["Entry_DT"].isna().all() and "TradeDate" in df.columns:
        df["Entry_DT"] = _al_parse_dt(df["TradeDate"].astype(str) + " " + df.get("TradeTime", "").astype(str))
    df["Exit_DT"] = _al_parse_dt(df["ExitTime"]) if "ExitTime" in df.columns else pd.NaT
    side = df["Side"].str.upper()
    side = np.where(side.isin(["BUY", "SELL"]), side, np.where(df["Signal"].str.upper().str.contains("SELL"), "SELL", "BUY"))
    df["Side"] = side
    df["sgn"] = np.where(df["Side"] == "BUY", 1.0, -1.0)
    e20, e50 = df["EMA20"].to_numpy(float), df["EMA50"].to_numpy(float)
    aligned = np.where(df["sgn"] > 0, e20 > e50, e20 < e50).astype(float)
    df["Aligned"] = np.where(np.isnan(e20) | np.isnan(e50), np.nan, aligned)
    df["RSI_dir"] = np.where(df["sgn"] > 0, df["RSI"], 100.0 - df["RSI"])
    df["PlannedRisk"] = (df["EntryPrice"] - df["StopLoss"]).abs() * df["Quantity"]
    df["R_multiple"] = np.where(df["PlannedRisk"] > 0, df["PnLv"] / df["PlannedRisk"].replace(0, np.nan), np.nan)
    df["Win"] = df["PnLv"] > 0
    df = df.sort_values("Entry_DT", kind="stable").reset_index(drop=True)
    df["Setup"] = df.apply(_al_classify_setup, axis=1)
    df["Hour"] = df["Entry_DT"].dt.hour
    return df


# ---------------------------------------------------------
# #4 WEAKNESS ANALYZER
# ---------------------------------------------------------
def _al_flag_frame(df, revenge_minutes=20, max_per_day=5):
    idx = df.index
    d20 = df["DistanceEMA20Pct"].abs()
    rv, adx, rsid = df["RelativeVolume"], df["ADX"], df["RSI_dir"]
    stage = df["SignalStage"].str.upper()
    lossr = df["LossReason"].str.upper()
    slope_dir = df["TrendSlope"] * df["sgn"]
    flags = pd.DataFrame(index=idx)
    flags["FOMO"] = ((d20 > 1.0) | (rsid > 72) | ((rv >= 2.5) & (d20 > 0.7))).fillna(False)
    flags["Early Entry"] = ((adx < 20) | stage.str.contains("EARLY")).fillna(False)
    flags["Late Entry"] = (lossr.str.contains("LATE|EXHAUST") | (d20 > 1.5) | (rsid > 78)).fillna(False)
    flags["Counter-Trend"] = ((df["Aligned"] == 0) | ((slope_dir < 0) & (df["TrendSlope"].abs() > 0))).fillna(False)
    flags["SL Shift"] = ((df["PnLv"] < 0) & (df["PlannedRisk"] > 0) & (-df["PnLv"] > 1.25 * df["PlannedRisk"])).fillna(False)
    rev = np.zeros(len(df), dtype=bool)
    prev_loss, prev_exit, prev_entry, prev_qty = False, pd.NaT, pd.NaT, np.nan
    for i in range(len(df)):
        ent = df["Entry_DT"].iloc[i]
        if prev_loss and pd.notna(ent):
            ref = prev_exit if pd.notna(prev_exit) else prev_entry
            if pd.notna(ref) and 0 <= (ent - ref).total_seconds() / 60.0 <= revenge_minutes:
                rev[i] = True
            q = df["Quantity"].iloc[i]
            if pd.notna(q) and pd.notna(prev_qty) and prev_qty > 0 and q >= 1.5 * prev_qty:
                rev[i] = True
        prev_loss = bool(df["PnLv"].iloc[i] < 0)
        prev_exit, prev_entry, prev_qty = df["Exit_DT"].iloc[i], ent, df["Quantity"].iloc[i]
    flags["Revenge Trading"] = rev
    day = df["Entry_DT"].dt.date
    flags["Overtrading"] = (df.groupby(day).cumcount() >= max_per_day).fillna(False).to_numpy()
    return flags.astype(bool)


def ai_lab_weakness(closed, revenge_minutes=20, max_per_day=5):
    """Return (summary_table, per_trade_table, largest_mistake_dict_or_None)."""
    if closed is None or closed.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    df = closed.reset_index(drop=True)
    flags = _al_flag_frame(df, revenge_minutes, max_per_day)
    total = len(df)
    rows = []
    for name in flags.columns:
        m = flags[name].to_numpy()
        if not m.any():
            continue
        p_f, p_n = _al_perf(df.loc[m, "PnLv"]), _al_perf(df.loc[~m, "PnLv"])
        if name == "SL Shift":
            # cost = the loss beyond what the stop was supposed to allow
            sub = df.loc[m]
            cost = -float(np.maximum(0.0, -sub["PnLv"] - sub["PlannedRisk"].fillna(0.0)).sum())
        elif p_n["n"]:
            cost = p_f["n"] * (p_f["avg"] - p_n["avg"])     # what these trades earned versus the same number of normal trades
        else:
            cost = p_f["pnl"]
        rows.append({
            "Behaviour": name, "Trades": p_f["n"], "% of trades": round(p_f["n"] / total * 100, 1),
            "Win rate %": round(p_f["wr"], 1), "Total PnL": round(p_f["pnl"], 2), "Avg PnL": round(p_f["avg"], 2),
            "Avg PnL when NOT flagged": round(p_n["avg"], 2) if p_n["n"] else np.nan,
            "Estimated cost": round(cost, 2), "Fix": AL_FIX_TEXT.get(name, ""),
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("Estimated cost").reset_index(drop=True)
    per = pd.DataFrame({
        "Entry time": df["Entry_DT"], "Symbol": df["Symbol"], "Side": df["Side"], "Setup": df["Setup"],
        "PnL": df["PnLv"].round(2),
        "Flags": [", ".join([c for c in flags.columns if flags.at[i, c]]) or "-" for i in flags.index],
    })
    worst = None
    if not summary.empty:
        cand = summary[(summary["Estimated cost"] < 0) & (summary["Trades"] >= 2)]
        if cand.empty:
            cand = summary[summary["Estimated cost"] < 0]
        if not cand.empty:
            r = cand.iloc[0]
            worst = {"name": r["Behaviour"], "pct": float(r["% of trades"]), "pnl": float(r["Total PnL"]), "cost": float(r["Estimated cost"]),
                     "trades": int(r["Trades"]), "fix": r["Fix"]}
    return summary, per, worst


# ---------------------------------------------------------
# #5 STRENGTH ANALYZER
# ---------------------------------------------------------
def ai_lab_strength(closed, min_n=5):
    """Return ({dimension: table}, {dimension: best_row_dict_or_None})."""
    tables, best = {}, {}
    if closed is None or closed.empty:
        return tables, best
    dims = {"Setup": "Setup", "Market regime": "MarketRegime", "Symbol": "Symbol", "Timeframe": "Timeframe",
            "Side": "Side", "Entry hour": "Hour"}
    for label, col in dims.items():
        if col not in closed.columns:
            continue
        rows = []
        for k, g in closed.groupby(col):
            if str(k).strip() in ("", "nan", "Unknown"):
                continue
            p = _al_perf(g["PnLv"])
            rows.append({label: k, "Trades": p["n"], "Win rate %": round(p["wr"], 1), "Profit factor": round(p["pf"], 2) if pd.notna(p["pf"]) else np.nan,
                         "Total PnL": round(p["pnl"], 2), "Avg PnL": round(p["avg"], 2),
                         "Sample": "OK" if p["n"] >= min_n else f"Low (<{min_n})"})
        if rows:
            t = pd.DataFrame(rows).sort_values(["Avg PnL", "Win rate %"], ascending=False).reset_index(drop=True)
            tables[label] = t
            ok = t[(t["Trades"] >= min_n) & (t["Avg PnL"] > 0)]
            best[label] = ok.iloc[0].to_dict() if not ok.empty else None
    return tables, best


# ---------------------------------------------------------
# backtest feature frame, rule mining (#11) and evolution (#12)
# ---------------------------------------------------------
def _al_bt_features(bt):
    if bt is None or not isinstance(bt, pd.DataFrame) or bt.empty or "PnL" not in bt.columns:
        return pd.DataFrame()
    df = bt.copy()
    sgn = np.where(df.get("Signal", pd.Series("BUY", index=df.index)).astype(str).str.upper().str.contains("SELL"), -1.0, 1.0)

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    def flag(name):
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[name].map(lambda v: 1.0 if str(v).lower() in ("true", "1", "1.0") else 0.0)

    out = pd.DataFrame({
        "Side": np.where(sgn > 0, "BUY", "SELL"), "PnL": col("PnL"),
        "RSI_dir": np.where(sgn > 0, col("RSI"), 100.0 - col("RSI")), "ADX": col("ADX"),
        "EMA": flag("EMA_Aligned"), "VOL": flag("Volume_Spike"),
        "MACD": np.where(col("MACD_Hist").isna(), np.nan, (col("MACD_Hist") * sgn > 0).astype(float)),
    }, index=df.index)
    if {"Date", "Time"}.issubset(df.columns):
        out["DT"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), format="%d-%m-%Y %H:%M:%S", errors="coerce")
        out = out.sort_values("DT", kind="stable")
    out = out.dropna(subset=["PnL"]).reset_index(drop=True)
    return out


def _al_journal_features(closed):
    if closed is None or closed.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Side": closed["Side"], "PnL": closed["PnLv"], "RSI_dir": closed["RSI_dir"], "ADX": closed["ADX"],
        "EMA": closed["Aligned"], "VOL": np.where(closed["RelativeVolume"].isna(), np.nan, (closed["RelativeVolume"] >= 1.5).astype(float)),
        "MACD": np.nan,
    }).reset_index(drop=True)


def _al_rule_text(rule):
    parts = []
    if rule.get("EMA"):
        parts.append("EMA20 aligned with trend (vs EMA50)")
    if rule.get("RSI") is not None:
        parts.append(f"RSI (trade direction) > {rule['RSI']}")
    if rule.get("ADX") is not None:
        parts.append(f"ADX > {rule['ADX']}")
    if rule.get("VOL"):
        parts.append("Volume spike")
    if rule.get("MACD"):
        parts.append("MACD histogram agrees")
    return " AND ".join(parts) if parts else "No filter (all signals)"


def _al_rule_mask(feat, rule, na_ok=False):
    m = np.ones(len(feat), dtype=bool)

    def apply(colname, cond):
        nonlocal m
        c = cond.to_numpy() if hasattr(cond, "to_numpy") else np.asarray(cond)
        if na_ok:
            c = c | feat[colname].isna().to_numpy()
        m &= c

    if rule.get("RSI") is not None:
        apply("RSI_dir", feat["RSI_dir"] > rule["RSI"])
    if rule.get("ADX") is not None:
        apply("ADX", feat["ADX"] > rule["ADX"])
    if rule.get("EMA"):
        apply("EMA", feat["EMA"] == 1.0)
    if rule.get("VOL"):
        apply("VOL", feat["VOL"] == 1.0)
    if rule.get("MACD"):
        apply("MACD", feat["MACD"] == 1.0)
    return m


def ai_lab_mine_rules(feat, min_train=8, top_k=5, train_frac=0.7, min_total=30):
    """Search 240 simple filter combinations on the older 70% of trades and re-test the best on the newest 30%."""
    meta = {"combos": 0, "reason": ""}
    if feat is None or feat.empty or len(feat) < min_total:
        meta["reason"] = f"Need at least {min_total} simulated/closed trades (have {0 if feat is None else len(feat)})."
        return pd.DataFrame(), meta
    if feat["RSI_dir"].isna().all() and feat["ADX"].isna().all():
        meta["reason"] = "Trade rows carry no RSI/ADX features (refresh the backtest)."
        return pd.DataFrame(), meta
    feat = feat.reset_index(drop=True)
    n = len(feat)
    cut = int(n * train_frac)
    pnl = feat["PnL"].to_numpy(float)
    tr = np.arange(n) < cut
    te = ~tr
    meta.update({"n": n, "train_n": int(tr.sum()), "test_n": int(te.sum()),
                 "base_train_wr": float((pnl[tr] > 0).mean() * 100), "base_test_wr": float((pnl[te] > 0).mean() * 100) if te.any() else np.nan})
    combos = []
    for rsi in (None, 50, 55, 58, 60, 65):
        for adx in (None, 18, 20, 25, 30):
            for ema in (None, True):
                for vol in (None, True):
                    for macd in (None, True):
                        rule = {"RSI": rsi, "ADX": adx, "EMA": ema, "VOL": vol, "MACD": macd}
                        k = sum(v is not None for v in rule.values())
                        if k:
                            combos.append((k, rule))
    combos.sort(key=lambda t: t[0])
    meta["combos"] = len(combos)
    seen, rows = set(), []
    for _, rule in combos:
        m = _al_rule_mask(feat, rule)
        sig = m.tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        mt, me = m & tr, m & te
        ntr = int(mt.sum())
        if ntr < min_train:
            continue
        ptr = pnl[mt]
        if ptr.mean() <= 0:
            continue
        wtr = int((ptr > 0).sum())
        gl = -ptr[ptr < 0].sum()
        pf_tr = min((ptr[ptr > 0].sum() / gl) if gl > 0 else 99.0, 99.0)
        pe = pnl[me]
        nte = int(me.sum())
        wte = int((pe > 0).sum())
        gle = -pe[pe < 0].sum() if nte else 0
        pf_te = min((pe[pe > 0].sum() / gle) if gle > 0 else (99.0 if nte and pe[pe > 0].sum() > 0 else np.nan), 99.0) if nte else np.nan
        rows.append({
            "Rule": _al_rule_text(rule), "Train trades": ntr, "Train WR %": round(wtr / ntr * 100, 1), "Train PF": round(pf_tr, 2),
            "Test trades": nte, "Test WR %": round(wte / nte * 100, 1) if nte else np.nan,
            "Test PF": round(pf_te, 2) if pd.notna(pf_te) else np.nan,
            "Test avg PnL": round(float(pe.mean()), 2) if nte else np.nan,
            "Validated": bool(nte >= 5 and pe.mean() > 0),
            "RuleJSON": json.dumps(rule), "_score": _al_wilson_lb(wtr, ntr),
        })
    if not rows:
        meta["reason"] = "No filter combination had a positive average result with enough trades."
        return pd.DataFrame(), meta
    res = pd.DataFrame(rows).sort_values(["_score", "Train trades"], ascending=False).head(top_k).drop(columns="_score").reset_index(drop=True)
    res.insert(0, "Rank", np.arange(1, len(res) + 1))
    return res, meta


def ai_lab_evolution(feat, symbol, timeframe, force=False):
    """Weekly re-optimisation. Stores one ranked snapshot per (week, symbol, timeframe)."""
    cols = ["WeekKey", "RunAt", "Symbol", "Timeframe", "Rank", "Rule", "Train trades", "Train WR %", "Train PF",
            "Test trades", "Test WR %", "Test PF", "Test avg PnL", "Validated", "RuleJSON"]
    wk = _al_week_key()
    hist = safe_read_csv(AI_LAB_EVOLUTION_FILE, pd.DataFrame())
    if hist is None or hist.empty or "WeekKey" not in hist.columns:
        hist = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in hist.columns:
            hist[c] = np.nan
    sel = (hist["WeekKey"] == wk) & (hist["Symbol"] == symbol) & (hist["Timeframe"] == timeframe)
    ran, meta = False, {}
    if force or not sel.any():
        ranked, meta = ai_lab_mine_rules(feat)
        if not ranked.empty:
            new = ranked.copy()
            new.insert(0, "WeekKey", wk)
            new.insert(1, "RunAt", now_str())
            new.insert(2, "Symbol", symbol)
            new.insert(3, "Timeframe", timeframe)
            hist = pd.concat([hist[~sel], new[cols]], ignore_index=True)
            safe_write_csv(hist, AI_LAB_EVOLUTION_FILE)
            ran = True
            sel = (hist["WeekKey"] == wk) & (hist["Symbol"] == symbol) & (hist["Timeframe"] == timeframe)
    mine = hist[(hist["Symbol"] == symbol) & (hist["Timeframe"] == timeframe)].copy()
    cur = mine[mine["WeekKey"] == wk].sort_values("Rank")
    prev_weeks = sorted([w for w in mine["WeekKey"].dropna().unique() if w < wk])
    notes = []
    if not cur.empty and prev_weeks:
        prev = mine[mine["WeekKey"] == prev_weeks[-1]].sort_values("Rank")
        try:
            a, b = json.loads(prev.iloc[0]["RuleJSON"]), json.loads(cur.iloc[0]["RuleJSON"])
            labels = {"RSI": "RSI threshold", "ADX": "ADX threshold", "EMA": "EMA alignment", "VOL": "Volume spike", "MACD": "MACD agreement"}
            for k, lab in labels.items():
                if a.get(k) != b.get(k):
                    notes.append(f"{lab}: {a.get(k) if a.get(k) is not None else 'off'} -> {b.get(k) if b.get(k) is not None else 'off'}")
            if not notes:
                notes.append("Top rule unchanged versus last week.")
        except Exception:
            pass
        prev_rank = {r["Rule"]: int(r["Rank"]) for _, r in prev.iterrows()}
        for _, r in cur.iterrows():
            if r["Rule"] in prev_rank and prev_rank[r["Rule"]] != int(r["Rank"]):
                notes.append(f"'{r['Rule']}' moved rank {prev_rank[r['Rule']]} -> {int(r['Rank'])}")
    elif not cur.empty:
        notes.append("First snapshot for this symbol/timeframe: nothing to compare with yet.")
    return cur, mine, notes, ran, meta, wk


# ---------------------------------------------------------
# #18 SELF-LEARNING ENGINE (logistic regression refitted whenever a trade closes)
# ---------------------------------------------------------
AL_MODEL_FEATURES = ["RSI_dir", "ADX", "RelVol", "AbsD20", "AbsDVWAP", "RR", "Confidence", "Aligned", "SideBuy",
                     "Reg_trend", "Reg_range", "Reg_vol", "Reg_break", "Beh_fomo", "Beh_late", "Beh_counter", "Beh_early", "Beh_revenge"]


def _al_model_matrix(df):
    X = pd.DataFrame(index=df.index)
    X["RSI_dir"] = df["RSI_dir"]
    X["ADX"] = df["ADX"]
    X["RelVol"] = df["RelativeVolume"].clip(0, 5)
    X["AbsD20"] = df["DistanceEMA20Pct"].abs().clip(0, 3)
    X["AbsDVWAP"] = df["DistanceVWAPPct"].abs().clip(0, 3)
    X["RR"] = df["RiskRewardRatio"].clip(0, 5)
    conf = df["ConfidenceAtEntry"].copy()
    if conf.notna().any() and conf.max() <= 1.0:
        conf = conf * 100.0
    X["Confidence"] = conf
    X["Aligned"] = df["Aligned"]
    X["SideBuy"] = (df["Side"] == "BUY").astype(float)
    reg = df["MarketRegime"].astype(str).str.lower()
    X["Reg_trend"] = reg.str.contains("trend").astype(float)
    X["Reg_range"] = reg.str.contains("range|sideways").astype(float)
    X["Reg_vol"] = reg.str.contains("volatil").astype(float)
    X["Reg_break"] = reg.str.contains("break").astype(float)
    fl = _al_flag_frame(df.reset_index(drop=True)).set_index(df.index)
    X["Beh_fomo"], X["Beh_late"] = fl["FOMO"].astype(float), fl["Late Entry"].astype(float)
    X["Beh_counter"], X["Beh_early"] = fl["Counter-Trend"].astype(float), fl["Early Entry"].astype(float)
    X["Beh_revenge"] = fl["Revenge Trading"].astype(float)
    return X[AL_MODEL_FEATURES]


def _al_fit_logit(Z, y, l2=3.0, iters=300, lr=0.3):
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        w -= lr * (Z.T @ (p - y) / n + l2 * w / n)
        b -= lr * float((p - y).mean())
    return w, b


def _al_standardise(X_train):
    med = X_train.median().fillna(0.0)
    Xf = X_train.fillna(med)
    mu = Xf.mean()
    sd = Xf.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return med, mu, sd


def ai_lab_train_model(closed):
    if closed is None or closed.empty:
        return {"status": "learning", "n": 0, "need": AI_LAB_MIN_MODEL_TRADES}
    df = closed.dropna(subset=["PnLv"]).tail(600).reset_index(drop=True)
    n = len(df)
    if n < AI_LAB_MIN_MODEL_TRADES:
        return {"status": "learning", "n": n, "need": AI_LAB_MIN_MODEL_TRADES}
    X = _al_model_matrix(df)
    y = (df["PnLv"] > 0).astype(float).to_numpy()
    med, mu, sd = _al_standardise(X)
    Z = ((X.fillna(med) - mu) / sd).to_numpy(float)
    w, b = _al_fit_logit(Z, y)
    # progressive (prequential) validation: each trade is predicted by a model that had not seen it
    preds, actual, base = [], [], []
    start = max(AI_LAB_MIN_MODEL_TRADES - 5, n - 40)
    for k in range(start, n):
        m_k, mu_k, sd_k = _al_standardise(X.iloc[:k])
        Zk = ((X.iloc[:k].fillna(m_k) - mu_k) / sd_k).to_numpy(float)
        wk, bk = _al_fit_logit(Zk, y[:k], iters=150)
        zi = ((X.iloc[[k]].fillna(m_k) - mu_k) / sd_k).to_numpy(float)
        preds.append(float(1.0 / (1.0 + np.exp(-np.clip(zi @ wk + bk, -30, 30)))[0]))
        actual.append(y[k])
        base.append(float(y[:k].mean()))
    preds, actual, base = np.array(preds), np.array(actual), np.array(base)
    preq = {
        "n": int(len(preds)),
        "acc": float(((preds >= 0.5) == (actual == 1)).mean() * 100),
        "base_acc": float(((base >= 0.5) == (actual == 1)).mean() * 100),
        "brier": float(((preds - actual) ** 2).mean()),
        "base_brier": float(((base - actual) ** 2).mean()),
    }
    reg_rows = []
    for name, col in (("Trending", "Reg_trend"), ("Range", "Reg_range"), ("Volatile", "Reg_vol"), ("Breakout", "Reg_break")):
        mk = X[col] == 1.0
        if mk.any():
            reg_rows.append({"Regime": name, "Trades": int(mk.sum()), "Win rate %": round(float(y[mk.to_numpy()].mean() * 100), 1)})
    beh_rows = []
    for name, col in (("FOMO", "Beh_fomo"), ("Late entry", "Beh_late"), ("Counter-trend", "Beh_counter"), ("Early entry", "Beh_early"), ("Revenge", "Beh_revenge")):
        mk = (X[col] == 1.0).to_numpy()
        if mk.any():
            beh_rows.append({"Behaviour": name, "Trades": int(mk.sum()), "Win rate when flagged %": round(float(y[mk].mean() * 100), 1),
                             "Win rate otherwise %": round(float(y[~mk].mean() * 100), 1) if (~mk).any() else np.nan})
    model = {
        "status": "active", "n": int(n), "features": AL_MODEL_FEATURES, "w": [float(v) for v in w], "b": float(b),
        "med": {k: float(v) for k, v in med.items()}, "mu": {k: float(v) for k, v in mu.items()}, "sd": {k: float(v) for k, v in sd.items()},
        "base_rate": float(y.mean()), "preq": preq, "regimes": reg_rows, "behaviours": beh_rows, "updated_at": now_str(),
    }
    try:
        with open(AI_LAB_MODEL_FILE, "w") as f:
            json.dump(model, f)
    except Exception as exc:
        log_app_error("ai_lab_save_model", exc)
    return model


def ai_lab_get_model(journal):
    closed = _al_closed_trades(journal)
    sig = _al_journal_sig(journal)
    return _al_memo("model", sig, lambda: ai_lab_train_model(closed))


def ai_lab_predict(model, feats):
    if not isinstance(model, dict) or model.get("status") != "active":
        return None
    z = []
    for name in model["features"]:
        v = _al_num(feats.get(name))
        if not np.isfinite(v):
            v = model["med"][name]
        z.append(max(-3.0, min(3.0, (v - model["mu"][name]) / (model["sd"][name] or 1.0))))
    s = float(np.dot(np.array(z), np.array(model["w"])) + model["b"])
    return float(1.0 / (1.0 + math.exp(-max(-30, min(30, s)))))


def _al_current_features(analysis):
    """Entry-time features for the live signal, in the same units as the journal."""
    if not isinstance(analysis, dict):
        return {}
    buy = analysis.get("signal_type") != "sell"
    close = _al_num(analysis.get("close_price"))
    e20, e50, rsi = _al_num(analysis.get("ema20")), _al_num(analysis.get("ema50")), _al_num(analysis.get("rsi"))
    reg = str(analysis.get("market_regime", "")).lower()
    f = {"RSI_dir": rsi if buy else 100 - rsi, "ADX": _al_num(analysis.get("adx")),
         "RR": _al_num(analysis.get("rr_ratio")), "Confidence": _al_num(analysis.get("confidence")),
         "SideBuy": 1.0 if buy else 0.0,
         "Reg_trend": 1.0 if "trend" in reg else 0.0, "Reg_range": 1.0 if ("range" in reg or "sideways" in reg) else 0.0,
         "Reg_vol": 1.0 if "volatil" in reg else 0.0, "Reg_break": 1.0 if "break" in reg else 0.0,
         "Beh_fomo": 0.0, "Beh_late": 0.0, "Beh_counter": 0.0, "Beh_early": 0.0, "Beh_revenge": 0.0}
    if np.isfinite(close) and close > 0:
        if np.isfinite(e20):
            f["AbsD20"] = abs(close - e20) / close * 100
        vw = _al_num(analysis.get("vwap"))
        if np.isfinite(vw):
            f["AbsDVWAP"] = abs(close - vw) / close * 100
    if np.isfinite(e20) and np.isfinite(e50):
        f["Aligned"] = 1.0 if ((e20 > e50) if buy else (e20 < e50)) else 0.0
    return f


# ---------------------------------------------------------
# #25 PREDICTION ACCURACY and #21 CONFIDENCE CALIBRATION
# ---------------------------------------------------------
def ai_lab_eval_predictions(audit, data, symbol, timeframe, horizon=10, last_n=30):
    """Compare each logged BUY/SELL signal with what the candles did in the next `horizon` bars."""
    if audit is None or audit.empty or data is None or data.empty:
        return pd.DataFrame()
    a = audit.copy()
    for c in ("Symbol", "Timeframe", "SignalType"):
        if c not in a.columns:
            return pd.DataFrame()
    a = a[(a["Symbol"].astype(str) == str(symbol)) & (a["Timeframe"].astype(str) == str(timeframe)) &
          (a["SignalType"].astype(str).str.lower().isin(["buy", "sell"]))].copy()
    if a.empty:
        return pd.DataFrame()
    a["DT"] = _al_parse_dt(a["DateTime"])
    a = a.dropna(subset=["DT"]).sort_values("DT").tail(last_n)
    idx = pd.DatetimeIndex(pd.to_datetime(data.index))
    rows = []
    for _, r in a.iterrows():
        sgn = 1.0 if str(r["SignalType"]).lower() == "buy" else -1.0
        entry = _al_num(r.get("Price"))
        pos = int(idx.searchsorted(r["DT"], side="right"))
        bars = data.iloc[pos:pos + horizon]
        atr = _al_num(r.get("ATR"))
        if not np.isfinite(atr) and "ATR" in data.columns and pos > 0:
            atr = _al_num(data["ATR"].iloc[pos - 1])
        t1 = _al_num(r.get("Target1"))
        expected = abs(t1 - entry) if np.isfinite(t1) and np.isfinite(entry) else atr
        row = {"Signal time": r["DT"], "Signal": r.get("Signal", ""), "Entry": entry,
               "Confidence": _al_num(r.get("Confidence")), "Expected move": round(expected, 2) if np.isfinite(expected) else np.nan}
        if len(bars) < horizon or not np.isfinite(entry):
            row.update({"Status": f"Pending ({len(bars)}/{horizon} bars)", "Move": np.nan, "Move %": np.nan, "Best move (MFE)": np.nan,
                        "Worst move (MAE)": np.nan, "Direction OK": np.nan, "Magnitude OK": np.nan, "Abs error %": np.nan})
        else:
            close_h = float(bars["Close"].iloc[-1])
            move = sgn * (close_h - entry)
            mfe = (float(bars["High"].max()) - entry) if sgn > 0 else (entry - float(bars["Low"].min()))
            mae = (entry - float(bars["Low"].min())) if sgn > 0 else (float(bars["High"].max()) - entry)
            row.update({"Status": "Resolved", "Move": round(move, 2), "Move %": round(move / entry * 100, 3),
                        "Best move (MFE)": round(mfe, 2), "Worst move (MAE)": round(mae, 2),
                        "Direction OK": bool(move > 0),
                        "Magnitude OK": bool(mfe >= expected) if np.isfinite(expected) else np.nan,
                        "Abs error %": round(abs(move - expected) / entry * 100, 3) if np.isfinite(expected) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def ai_lab_prediction_summary(ev):
    if ev is None or ev.empty or "Status" not in ev.columns:
        return None
    r = ev[ev["Status"] == "Resolved"]
    if r.empty:
        return {"resolved": 0, "pending": int(len(ev))}
    mag = r["Magnitude OK"].dropna()
    return {"resolved": int(len(r)), "pending": int(len(ev) - len(r)),
            "direction": float(r["Direction OK"].astype(float).mean() * 100),
            "magnitude": float(mag.astype(float).mean() * 100) if len(mag) else np.nan,
            "avg_error": float(r["Abs error %"].mean()) if r["Abs error %"].notna().any() else np.nan,
            "avg_move": float(r["Move %"].mean())}


def ai_lab_calibration(pairs, edges=(0, 50, 60, 70, 80, 101), shrink_k=10):
    """pairs: DataFrame[confidence (0-100), outcome (0/1)]. Returns (table, summary)."""
    if pairs is None or pairs.empty:
        return pd.DataFrame(), None
    p = pairs.dropna().copy()
    if p.empty:
        return pd.DataFrame(), None
    if p["confidence"].max() <= 1.0:
        p["confidence"] = p["confidence"] * 100.0
    base = float(p["outcome"].mean())
    labels = [f"{edges[i]}-{min(edges[i + 1], 100)}" for i in range(len(edges) - 1)]
    p["bucket"] = pd.cut(p["confidence"], bins=list(edges), labels=labels, right=False)
    rows = []
    for lab in labels:
        g = p[p["bucket"] == lab]
        if g.empty:
            continue
        n, wins = len(g), float(g["outcome"].sum())
        pred, act = float(g["confidence"].mean()), float(g["outcome"].mean() * 100)
        rows.append({"Confidence bucket": lab, "Signals": n, "Avg predicted %": round(pred, 1), "Actual success %": round(act, 1),
                     "Gap (pred - actual)": round(pred - act, 1),
                     "Calibrated estimate %": round((wins + shrink_k * base) / (n + shrink_k) * 100, 1)})
    tab = pd.DataFrame(rows)
    N = len(p)
    brier = float(((p["confidence"] / 100.0 - p["outcome"]) ** 2).mean())
    ece = float(sum(r["Signals"] / N * abs(r["Gap (pred - actual)"]) for r in rows))
    avg_gap = float(p["confidence"].mean() - p["outcome"].mean() * 100)
    verdict = ("Over-confident: the model claims more than it delivers." if avg_gap > 5 else
               "Under-confident: results are better than the stated confidence." if avg_gap < -5 else "Reasonably calibrated.")
    return tab, {"n": N, "brier": brier, "ece": ece, "avg_gap": avg_gap, "base": base * 100, "verdict": verdict}


def ai_lab_calibrated_value(table, conf):
    if table is None or table.empty or not np.isfinite(_al_num(conf)):
        return None
    c = float(conf) * (100.0 if float(conf) <= 1.0 else 1.0)
    for _, r in table.iterrows():
        lo, hi = r["Confidence bucket"].split("-")
        if float(lo) <= c < (float(hi) if float(hi) < 100 else 101):
            return float(r["Calibrated estimate %"]), int(r["Signals"])
    return None


# ---------------------------------------------------------
# #19 WEEKLY AI REPORT
# ---------------------------------------------------------
def ai_lab_weekly_report(closed, week_key, calib_summary=None):
    if closed is None or closed.empty:
        return None
    df = closed.copy()
    ref = df["Exit_DT"].where(df["Exit_DT"].notna(), df["Entry_DT"])
    df["WeekKey"] = ref.map(lambda d: _al_week_key(d) if pd.notna(d) else "")
    wk = df[df["WeekKey"] == week_key].reset_index(drop=True)
    if wk.empty:
        return None
    p = _al_perf(wk["PnLv"])
    tables, best = ai_lab_strength(wk, min_n=2)
    summary, _, worst = ai_lab_weakness(wk)
    setups = tables.get("Setup", pd.DataFrame())
    best_setup = setups.iloc[0].to_dict() if not setups.empty and setups.iloc[0]["Avg PnL"] > 0 else None
    worst_setup = setups.iloc[-1].to_dict() if not setups.empty and setups.iloc[-1]["Avg PnL"] < 0 else None
    wins = wk.loc[wk["PnLv"] > 0, "PnLv"]
    losses = wk.loc[wk["PnLv"] < 0, "PnLv"]
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan
    plan = []
    if worst:
        plan.append(f"Biggest mistake - {worst['name']} ({worst['trades']} trades, estimated cost {worst['cost']:+.2f}): {worst['fix']}")
    if pd.notna(payoff) and payoff < 1.0:
        plan.append(f"Average win is only {payoff:.2f}x the average loss. Hold winners to Target 1/2 or tighten entries.")
    if worst_setup:
        plan.append(f"Pause the '{worst_setup['Setup']}' setup until it recovers (avg {worst_setup['Avg PnL']:+.2f} over {worst_setup['Trades']} trades).")
    reg = tables.get("Market regime", pd.DataFrame())
    if not reg.empty and reg.iloc[-1]["Avg PnL"] < 0:
        plan.append(f"Avoid trading in '{reg.iloc[-1]['Market regime']}' conditions (avg {reg.iloc[-1]['Avg PnL']:+.2f}).")
    per_day = wk.groupby(wk["Entry_DT"].dt.date).size()
    if len(per_day) and per_day.max() > 5:
        plan.append(f"You took up to {int(per_day.max())} trades in a day. Cap it at 5.")
    if calib_summary and calib_summary.get("avg_gap", 0) > 10:
        plan.append(f"Confidence was overstated by about {calib_summary['avg_gap']:.0f} points. Demand a higher score before entering.")
    if not plan:
        plan.append("No clear weakness this week. Keep the same rules and keep logging trades.")
    rep = {"week": week_key, "trades": p["n"], "win_rate": p["wr"], "pnl": p["pnl"], "profit_factor": p["pf"], "payoff": payoff,
           "best_setup": best_setup, "worst_setup": worst_setup, "biggest_mistake": worst, "plan": plan}
    rep["text"] = ai_lab_report_text(rep)
    return rep


def ai_lab_report_text(rep):
    def setup_txt(s):
        return f"{s['Setup']} (win rate {s['Win rate %']}%, {s['Trades']} trades, avg {s['Avg PnL']:+.2f})" if s else "None identified"
    m = rep["biggest_mistake"]
    lines = [f"WEEKLY AI REPORT - {rep['week']}", "=" * 40,
             f"Trades: {rep['trades']}   Win rate: {rep['win_rate']:.1f}%   Net PnL: {rep['pnl']:+.2f}   Profit factor: {format(rep['profit_factor'], '.2f') if pd.notna(rep['profit_factor']) else 'n/a'}",
             f"Best setup : {setup_txt(rep['best_setup'])}", f"Worst setup: {setup_txt(rep['worst_setup'])}",
             f"Biggest mistake: {m['name'] + ' - ' + str(m['trades']) + ' trades, estimated cost ' + format(m['cost'], '+.2f') if m else 'None detected'}",
             "", "Improvement plan:"] + [f"  {i + 1}. {t}" for i, t in enumerate(rep["plan"])]
    return "\n".join(lines)


def ai_lab_save_report(rep):
    cols = ["WeekKey", "GeneratedAt", "Trades", "WinRate", "NetPnL", "BestSetup", "WorstSetup", "BiggestMistake", "Report"]
    old = safe_read_csv(AI_LAB_REPORT_FILE, pd.DataFrame())
    if old is None or old.empty or "WeekKey" not in old.columns:
        old = pd.DataFrame(columns=cols)
    row = {"WeekKey": rep["week"], "GeneratedAt": now_str(), "Trades": rep["trades"], "WinRate": round(rep["win_rate"], 1),
           "NetPnL": round(rep["pnl"], 2), "BestSetup": rep["best_setup"]["Setup"] if rep["best_setup"] else "",
           "WorstSetup": rep["worst_setup"]["Setup"] if rep["worst_setup"] else "",
           "BiggestMistake": rep["biggest_mistake"]["name"] if rep["biggest_mistake"] else "", "Report": rep["text"]}
    out = pd.concat([old[old["WeekKey"] != rep["week"]], pd.DataFrame([row])[cols]], ignore_index=True)
    return safe_write_csv(out, AI_LAB_REPORT_FILE)


# ---------------------------------------------------------
# #23 TRADE REPLAY
# ---------------------------------------------------------
def ai_lab_replay_metrics(candles, side, entry_time, exit_time, entry, exit_price, sl, t1, pnl, lookback=12):
    sgn = 1.0 if str(side).upper() == "BUY" else -1.0
    out = {"mistakes": [], "good": [], "mfe": np.nan, "mae": np.nan, "capture": np.nan}
    if candles is None or candles.empty or pd.isna(entry_time):
        return out
    idx = pd.DatetimeIndex(pd.to_datetime(candles.index))
    c = candles.copy()
    c.index = idx
    end = exit_time if pd.notna(exit_time) else idx[-1]
    after = c.loc[entry_time:end]
    before = c.loc[:entry_time].tail(lookback)
    if after.empty:
        return out
    mfe = (after["High"].max() - entry) if sgn > 0 else (entry - after["Low"].min())
    mae = (entry - after["Low"].min()) if sgn > 0 else (after["High"].max() - entry)
    out["mfe"], out["mae"] = float(mfe), float(mae)
    realised = sgn * (exit_price - entry) if np.isfinite(exit_price) else np.nan
    if mfe > 0 and np.isfinite(realised):
        out["capture"] = float(realised / mfe)
    atr = _al_num(c["ATR"].loc[:entry_time].iloc[-1]) if "ATR" in c.columns and len(c.loc[:entry_time]) else np.nan
    if not before.empty:
        best_px = float(before["Low"].min()) if sgn > 0 else float(before["High"].max())
        saved = (entry - best_px) if sgn > 0 else (best_px - entry)
        if saved > 0:
            when = before["Low"].idxmin() if sgn > 0 else before["High"].idxmax()
            out["better_entry"] = (best_px, when, saved)
            if np.isfinite(atr) and atr > 0 and saved > 0.5 * atr:
                out["mistakes"].append(f"Late/chased entry: a better price ({best_px:.2f} at {when:%H:%M}) existed in the {lookback} bars before entry, {saved:.2f} points better.")
        if "EMA20" in before.columns:
            touch = before[(before["Low"] <= before["EMA20"] * 1.002) & (before["High"] >= before["EMA20"] * 0.998)]
            if not touch.empty:
                out["ema_retest"] = (float(touch["Close"].iloc[-1]), touch.index[-1])
    risk = abs(entry - sl) if np.isfinite(sl) else np.nan
    reached_t1 = np.isfinite(t1) and mfe >= abs(t1 - entry)
    if np.isfinite(pnl) and pnl < 0 and reached_t1:
        out["mistakes"].append("Price reached Target 1 before the trade ended in a loss. Book a partial at T1 and move the stop to breakeven.")
    if np.isfinite(pnl) and pnl > 0 and np.isfinite(out["capture"]) and out["capture"] < 0.5:
        out["mistakes"].append(f"Exited early: captured only {out['capture'] * 100:.0f}% of the available move (best move {mfe:.2f}, realised {realised:.2f}).")
    if np.isfinite(risk) and risk > 0 and mae > 1.5 * risk:
        out["mistakes"].append(f"Stop was not respected or was widened: price moved {mae:.2f} against the entry vs planned risk {risk:.2f}.")
    if np.isfinite(pnl) and pnl > 0 and not out["mistakes"]:
        out["good"].append("Entry and exit were reasonable: no obvious mistake was found from the candles.")
    if np.isfinite(out["capture"]) and out["capture"] >= 0.7 and pnl > 0:
        out["good"].append(f"Good exit: captured {out['capture'] * 100:.0f}% of the available move.")
    return out


# ---------------------------------------------------------
# #24 OPPORTUNITY RANKING
# ---------------------------------------------------------
def _al_grade(score):
    return "A+" if score >= 85 else "A" if score >= 75 else "B" if score >= 65 else "C" if score >= 50 else "D"


def _al_quick_setup(symbol, frame, model, hist_wr):
    df = add_indicators(frame.copy()).replace([np.inf, -np.inf], np.nan).ffill()
    need = ["EMA20", "EMA50", "RSI", "ADX", "ATR", "VWAP", "SuperTrendDirection", "MACD", "SignalLine"]
    if len(df) < 60 or any(c not in df.columns for c in need):
        return None
    last = df.iloc[-1]
    close, atr = float(last["Close"]), float(last["ATR"])
    if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
        return None
    ema9, ema21 = float(df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]), float(df["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
    ema50, ema20 = float(last["EMA50"]), float(last["EMA20"])
    vwap, rsi, adx = float(last["VWAP"]), float(last["RSI"]), float(last["ADX"])
    vol_mean = df["Volume"].rolling(20).mean().iloc[-1]
    rv = float(last["Volume"] / vol_mean) if pd.notna(vol_mean) and vol_mean > 0 else np.nan
    bull = [ema9 > ema21 > ema50, close > vwap, last["SuperTrendDirection"] == 1, last["MACD"] > last["SignalLine"], rsi > 52]
    bear = [ema9 < ema21 < ema50, close < vwap, last["SuperTrendDirection"] == -1, last["MACD"] < last["SignalLine"], rsi < 48]
    diff = sum(bull) - sum(bear)
    direction = "BUY" if diff >= 2 else "SELL" if diff <= -2 else "NEUTRAL"
    buy = direction != "SELL"
    stack = (ema9 > ema21 > ema50) if buy else (ema9 < ema21 < ema50)
    rsid = rsi if buy else 100 - rsi
    q = 25 * bool(stack) + min(adx, 40) / 40 * 20 + (min(rv, 2) / 2 * 15 if np.isfinite(rv) else 7.5)
    q += 15 if 50 <= rsid <= 68 else 8 if 68 < rsid <= 75 else 0
    q += 10 * bool((close > vwap) if buy else (close < vwap)) + 10 * bool(last["SuperTrendDirection"] == (1 if buy else -1))
    q += 5 * bool((last["MACD"] > last["SignalLine"]) if buy else (last["MACD"] < last["SignalLine"]))
    if abs(close - ema20) > 1.5 * atr:
        q -= 10
    q = float(max(0.0, min(100.0, q)))
    recent = df.tail(60)
    room = (float(recent["High"].max()) - close) if buy else (close - float(recent["Low"].min()))
    basis = "nearest swing"
    if room < 0.5 * atr:
        room, basis = 2.0 * atr, "2xATR (price at swing extreme)"
    rr = min(room / atr, 4.0)
    feats = {"RSI_dir": rsid, "ADX": adx, "AbsD20": abs(close - ema20) / close * 100, "AbsDVWAP": abs(close - vwap) / close * 100,
             "RR": rr, "Aligned": 1.0 if (ema20 > ema50) == buy else 0.0, "SideBuy": 1.0 if buy else 0.0, "RelVol": rv if np.isfinite(rv) else np.nan,
             "Reg_trend": 1.0 if adx >= 25 else 0.0, "Reg_range": 1.0 if adx < 18 else 0.0}
    prob, psrc = ai_lab_predict(model, feats), "self-learned model"
    if prob is None:
        h = hist_wr.get(symbol)
        prob, psrc = (h / 100.0, "your history") if h is not None else (None, "no history")
    return {"Symbol": symbol, "Direction": direction, "Quality": round(q, 1), "Grade": _al_grade(q), "RR": round(rr, 2), "RR basis": basis,
            "Probability %": round(prob * 100, 1) if prob is not None else np.nan, "Prob. source": psrc, "Price": round(close, 2),
            "ADX": round(adx, 1), "RSI": round(rsi, 1)}


def ai_lab_rank_opportunities(model, closed, universe):
    hist_wr = {}
    if closed is not None and not closed.empty:
        base = float((closed["PnLv"] > 0).mean())
        for sym, g in closed.groupby("Symbol"):
            if len(g) >= 5:
                hist_wr[sym] = ((g["PnLv"] > 0).sum() + 10 * base) / (len(g) + 10) * 100
    tickers = tuple(universe.values())
    frames = fetch_yahoo_batch(tickers, "5m", "5d")
    rows, missing = [], []
    for sym, tk in universe.items():
        fr = frames.get(tk)
        if fr is None or fr.empty:
            missing.append(sym)
            continue
        try:
            r = _al_quick_setup(sym, fr, model, hist_wr)
        except Exception as exc:
            log_app_error("ai_lab_quick_setup", f"{sym}: {type(exc).__name__}: {exc}")
            r = None
        if r is None:
            missing.append(sym)
        else:
            rows.append(r)
    if not rows:
        return pd.DataFrame(), missing
    df = pd.DataFrame(rows)
    rr_score = df["RR"].clip(0, 3) / 3 * 100
    has_p = df["Probability %"].notna()
    df["Score"] = np.where(has_p, 0.5 * df["Quality"] + 0.25 * rr_score + 0.25 * df["Probability %"].fillna(0),
                           (0.67 * df["Quality"] + 0.33 * rr_score))
    df.loc[df["Direction"] == "NEUTRAL", "Score"] *= 0.5
    df["Score"] = df["Score"].round(1)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df, missing


# ---------------------------------------------------------
# #27 PORTFOLIO RISK ANALYZER
# ---------------------------------------------------------
def ai_lab_open_positions(journal):
    if journal is None or journal.empty or "Status" not in journal.columns:
        return pd.DataFrame()
    df = journal[journal["Status"].astype(str).str.upper() == "OPEN"].copy()
    if df.empty:
        return df
    for c in ("EntryPrice", "LastPrice", "Quantity", "RemainingQty", "StopLoss", "TrailingStop"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["Qty"] = df["RemainingQty"].where(df["RemainingQty"].notna() & (df["RemainingQty"] > 0), df["Quantity"])
    df["Price"] = df["LastPrice"].where(df["LastPrice"].notna() & (df["LastPrice"] > 0), df["EntryPrice"])
    side = df["Side"].astype(str).str.upper() if "Side" in df.columns else pd.Series("BUY", index=df.index)
    df["Side"] = np.where(side.isin(["BUY", "SELL"]), side, np.where(df["Signal"].astype(str).str.upper().str.contains("SELL"), "SELL", "BUY"))
    df["sgn"] = np.where(df["Side"] == "BUY", 1.0, -1.0)
    # Same rule as the exit engine: the effective stop is the TIGHTER of stop-loss and trailing stop
    # (higher for BUY, lower for SELL).
    _both = pd.concat([df["StopLoss"].where(df["StopLoss"] > 0), df["TrailingStop"].where(df["TrailingStop"] > 0)], axis=1)
    df["Stop"] = np.where(df["sgn"] > 0, _both.max(axis=1), _both.min(axis=1))
    df["Notional"] = (df["Price"] * df["Qty"]).abs()
    df["OpenPnL"] = df["sgn"] * (df["Price"] - df["EntryPrice"]) * df["Qty"]
    risk = df["sgn"] * (df["Price"] - df["Stop"]) * df["Qty"]
    df["RiskToStop"] = risk.clip(lower=0).fillna(np.nan)
    df["LockedProfit"] = (-risk).clip(lower=0).fillna(0.0)
    df["Sector"] = df["Symbol"].map(lambda s: AL_SECTOR_MAP.get(str(s), "Unmapped"))
    return df.reset_index(drop=True)


def ai_lab_portfolio_corr(positions, universe):
    """Return-correlation matrix of the open symbols (5m, 5 days) or (None, message)."""
    syms = [s for s in positions["Symbol"].unique() if universe.get(s)]
    if len(syms) < 2:
        return None, "Need at least two open symbols with a known Yahoo ticker."
    frames = fetch_yahoo_batch(tuple(universe[s] for s in syms), "5m", "5d")
    closes = {}
    for s in syms:
        f = frames.get(universe[s])
        if f is not None and not f.empty:
            closes[s] = f["Close"].astype(float)
    if len(closes) < 2:
        return None, "Price history for the open symbols could not be fetched."
    ret = pd.DataFrame(closes).pct_change().dropna(how="any")
    if len(ret) < 30:
        return None, "Not enough overlapping candles to measure correlation."
    return ret.corr(), ""


def ai_lab_effective_bets(positions, corr):
    pos = (positions["sgn"] * positions["Notional"]).groupby(positions["Symbol"]).sum()
    pos = pos[pos.index.isin(corr.index)]
    if len(pos) < 2 or pos.abs().sum() == 0:
        return np.nan
    w = (pos / pos.abs().sum()).to_numpy()
    C = corr.loc[pos.index, pos.index].to_numpy()
    var = float(w @ C @ w)
    if var <= 1e-9:
        return float(len(pos))
    return float(min(len(pos), 1.0 / var))


# ---------------------------------------------------------
# #22 AI COPILOT (rule-based explanations built from the app's own numbers; no external LLM)
# ---------------------------------------------------------
def _al_fmt(v, nd=2):
    v = _al_num(v)
    return f"{v:,.{nd}f}" if np.isfinite(v) else "n/a"


def _al_copilot_why_side(a, side):
    bs, ss = _al_num(a.get("bullish_score"), 0), _al_num(a.get("bearish_score"), 0)
    cur = str(a.get("signal_type", "neutral")).lower()
    own, opp = (bs, ss) if side == "buy" else (ss, bs)
    out = []
    if cur == side:
        out.append(f"**The app currently signals {a.get('signal_text')}** (setup confidence {_al_fmt(a.get('confidence'), 0)}, grade {a.get('strategy_grade')}, quality {_al_fmt(a.get('trade_quality_score'), 0)}).")
        out.append(f"Score: {side} {own:.1f} vs opposite {opp:.1f}; trend votes {a.get('buy_trend_votes') if side == 'buy' else a.get('sell_trend_votes')}.")
        reasons = [str(r) for r in (a.get("reasons") or [])][:8]
        if reasons:
            out.append("Reasons the engine recorded:\n" + "\n".join(f"- {r}" for r in reasons))
        mtf = a.get("mtf_bull_ok") if side == "buy" else a.get("mtf_bear_ok")
        out.append(f"Higher-timeframe alignment: {'confirmed' if mtf else 'not confirmed'}. Regime: {a.get('market_regime')}. ADX {_al_fmt(a.get('adx'), 1)}, RSI {_al_fmt(a.get('rsi'), 1)}.")
    else:
        out.append(f"**The app is NOT signalling a {side.upper()} right now** (current: {a.get('signal_text')}).")
        out.append(f"{side.upper()} score {own:.1f} vs opposite {opp:.1f}.")
        blockers = a.get("no_trade_reasons") or ([a.get("no_trade_reason")] if a.get("no_trade_reason") else [])
        if blockers:
            out.append("What is blocking an entry:\n" + "\n".join(f"- {b}" for b in blockers[:8]))
        else:
            out.append("The score has not cleared the entry threshold, or the other side is stronger.")
    return "\n\n".join(out)


def _al_copilot_why_sl(a):
    sl, close, atr = _al_num(a.get("stop_loss")), _al_num(a.get("close_price")), _al_num(a.get("atr"))
    if not np.isfinite(sl):
        return "There is no active BUY/SELL signal, so no stop-loss is calculated. The app sets the stop only when it approves an entry."
    risk = abs(close - sl)
    side = "BUY" if sl < close else "SELL"
    out = [f"**Stop-loss {_al_fmt(sl)}** for a {side} at {_al_fmt(close)}: risk per unit {_al_fmt(risk)} = {_al_fmt(risk / atr, 2) if np.isfinite(atr) and atr > 0 else 'n/a'} x ATR ({_al_fmt(atr)}).",
           "The engine uses one ATR (minimum 0.1% of price) so normal candle noise does not stop you out, while the loss stays small and fixed."]
    sup, res = _al_num(a.get("support")), _al_num(a.get("resistance"))
    lvl, name = (sup, "support") if side == "BUY" else (res, "resistance")
    if np.isfinite(lvl):
        inside = (sl >= lvl) if side == "BUY" else (sl <= lvl)
        out.append(f"Nearest {name} is {_al_fmt(lvl)}: the stop sits {'on the protected side of it' if inside else 'beyond it, so a break of the level would not stop you earlier'}.")
    t1 = _al_num(a.get("target1"))
    if np.isfinite(t1):
        out.append(f"Targets: T1 {_al_fmt(t1)}, T2 {_al_fmt(a.get('target2'))}, T3 {_al_fmt(a.get('target3'))}; RR to T1 = {_al_fmt(a.get('rr_ratio'))}. Trailing stop starts at {_al_fmt(a.get('trailing_stop'))}.")
    return "\n\n".join(out)


def _al_copilot_why_exit(a, journal, selected):
    pos = ai_lab_open_positions(journal)
    pos = pos[pos["Symbol"].astype(str) == str(selected)] if not pos.empty else pos
    sig = str(a.get("signal_type", "neutral")).lower()
    if pos.empty:
        msg = ["You have no open paper position in this symbol."]
        msg.append(f"For reference, the app's current view is {a.get('signal_text')}; an exit would normally be triggered by the stop, a target, or the signal flipping.")
        return "\n\n".join(msg)
    out = []
    for _, p in pos.iterrows():
        side = p["Side"]
        reasons, action = [], "HOLD"
        price, stop = p["Price"], p["Stop"]
        if np.isfinite(stop) and ((p["sgn"] > 0 and price <= stop) or (p["sgn"] < 0 and price >= stop)):
            reasons.append(f"Price {_al_fmt(price)} is at/through the stop {_al_fmt(stop)}."); action = "FULL EXIT"
        opposite = (sig == "sell" and side == "BUY") or (sig == "buy" and side == "SELL")
        if opposite:
            reasons.append(f"The app now signals {a.get('signal_text')}, against your {side}."); action = "FULL EXIT" if action == "FULL EXIT" else "PARTIAL / FULL EXIT"
        rsi = _al_num(a.get("rsi"))
        rsid = rsi if side == "BUY" else 100 - rsi
        if np.isfinite(rsid) and rsid > 75:
            reasons.append(f"RSI in your direction is stretched ({rsid:.0f}). Consider a partial exit or a tighter trail."); action = "PARTIAL EXIT" if action == "HOLD" else action
        t1 = _al_num(p.get("Target1"))
        if np.isfinite(t1) and ((p["sgn"] > 0 and price >= t1) or (p["sgn"] < 0 and price <= t1)):
            reasons.append(f"Target 1 ({_al_fmt(t1)}) has been reached. Book a partial and trail the rest."); action = "PARTIAL EXIT" if action == "HOLD" else action
        if not reasons:
            reasons.append("No exit trigger is active: stop not threatened, no opposite signal, momentum not stretched.")
        out.append(f"**{selected} {side}** entry {_al_fmt(p['EntryPrice'])}, price {_al_fmt(price)}, open PnL {_al_fmt(p['OpenPnL'])}, stop {_al_fmt(stop)}.\n\nSuggested action: **{action}**\n" + "\n".join(f"- {r}" for r in reasons))
    return "\n\n".join(out)


def ai_lab_copilot_answer(question, ctx):
    q = str(question or "").lower().strip()
    a = ctx.get("analysis")
    journal = ctx.get("journal")
    if not q:
        return "Ask me why the app says Buy/Sell, why the stop is where it is, or when to exit."
    wants_trade = any(k in q for k in ("buy", "sell", "stop", " sl", "sl ", "target", "exit", "why", "wait", "no trade", "risk", "regime", "confidence", "signal"))
    if a is None and wants_trade and not any(k in q for k in ("weak", "mistake", "strength", "best", "report", "calibr")):
        return "Live candles are not available right now, so I cannot explain the current signal. Journal-based questions (weakness, strengths, weekly report) still work."
    if "exit" in q or "close" in q or "book" in q:
        return _al_copilot_why_exit(a, journal, ctx.get("selected"))
    if "stop" in q or "sl" in q.split() or "sl?" in q or "stoploss" in q or "stop-loss" in q:
        return _al_copilot_why_sl(a)
    if "target" in q:
        return _al_copilot_why_sl(a)
    if "sell" in q and "buy" not in q:
        return _al_copilot_why_side(a, "sell")
    if "buy" in q:
        return _al_copilot_why_side(a, "buy")
    if any(k in q for k in ("no trade", "wait", "neutral", "why not")):
        blockers = a.get("no_trade_reasons") or ([a.get("no_trade_reason")] if a.get("no_trade_reason") else [])
        return (f"Current signal: **{a.get('signal_text')}**.\n\n" + ("Blockers:\n" + "\n".join(f"- {b}" for b in blockers[:8]) if blockers else "No explicit blocker is recorded; the score simply has not cleared the entry threshold."))
    if "regime" in q or "market condition" in q:
        return f"Regime: **{a.get('market_regime')}** (ADX {_al_fmt(a.get('adx'), 1)}, ATR {_al_fmt(a.get('atr_pct'), 2)}% of price). Trending: {a.get('trending_market')}, sideways: {a.get('sideways_market')}, high volatility: {a.get('high_volatility_market')}."
    if "confidence" in q or "probab" in q or "win" in q:
        cal = ctx.get("calibrated")
        txt = f"Setup confidence is **{_al_fmt(a.get('confidence'), 0)}** ({a.get('setup_confidence_label')}). Backtest win chance: {_al_fmt(a.get('win_chance'), 0)}% from {a.get('win_chance_sample_size')} trades. Confidence is a ranking score, not a guaranteed probability."
        if cal:
            txt += f"\n\nCalibrated against past outcomes, similar confidence has actually succeeded about **{cal[0]:.0f}%** of the time ({cal[1]} signals)."
        if ctx.get("model_prob") is not None:
            txt += f"\n\nYour self-learned model gives this setup about **{ctx['model_prob'] * 100:.0f}%**."
        return txt
    if "risk" in q:
        return _al_copilot_why_sl(a)
    if "weak" in q or "mistake" in q:
        w = ctx.get("worst")
        return (f"Your largest mistake is **{w['name']}**: {w['trades']} trades ({w['pct']}% of all trades), estimated cost {w['cost']:+.2f}.\n\nFix: {w['fix']}" if w else "No weakness with a negative result was detected yet (or there are too few closed trades).")
    if "strength" in q or "best" in q:
        b = (ctx.get("best") or {}).get("Setup")
        return (f"Your best setup is **{b['Setup']}**: win rate {b['Win rate %']}% over {b['Trades']} trades, average {b['Avg PnL']:+.2f}." if b else "No setup has a positive result over enough trades yet.")
    if "report" in q:
        rep = ctx.get("report")
        return rep["text"] if rep else "There are no closed trades this week to report on."
    return ("I can explain: **Why Buy? / Why Sell? / Why no trade? / Why SL here? / Why exit? / What is the regime? / How reliable is the confidence? / What is my weakness? / What is my best setup? / Weekly report.**")
# ===================== AI LAB ENGINE END ================



# =========================================================
# AI LAB - USER INTERFACE
# =========================================================
AL_ALL_TABS = ["Weakness", "Strengths", "Strategy Builder", "Evolution", "Self-Learning", "Weekly Report",
               "Calibration", "Prediction Accuracy", "Copilot", "Trade Replay", "Opportunities", "Portfolio Risk"]
# Which AI Lab tabs appear on which page of the app.
AL_VIEWS = {
    "command": ["Copilot", "Weakness", "Strengths", "Self-Learning", "Calibration", "Prediction Accuracy"],
    "scanner": ["Opportunities"],
    "research": ["Strategy Builder", "Evolution"],
    "replay": ["Trade Replay"],
    "reports": ["Weekly Report"],
    "portfolio": ["Portfolio Risk"],
}
_AL_LABEL_TAB = {"weakness": "Weakness", "strengths": "Strengths", "strategy_builder": "Strategy Builder", "evolution": "Evolution",
                 "self_learning": "Self-Learning", "weekly_report": "Weekly Report", "calibration": "Calibration",
                 "prediction_accuracy": "Prediction Accuracy", "copilot": "Copilot", "trade_replay": "Trade Replay",
                 "opportunities": "Opportunities", "portfolio_risk": "Portfolio Risk"}
_AL_ACTIVE = set(AL_ALL_TABS)


def _al_safe(label, fn):
    """Run one tab body; a failure in one feature never breaks the page or the other tabs."""
    if _AL_LABEL_TAB.get(label) not in _AL_ACTIVE:
        return                       # this tab is not shown on the current page
    try:
        fn()
    except Exception as exc:
        log_app_error(f"ai_lab_{label}", f"{type(exc).__name__}: {exc}")
        st.error(f"AI Lab - {label} could not complete: {type(exc).__name__}: {exc}")


def _al_money(v):
    v = _al_num(v)
    return f"{v:,.2f}" if np.isfinite(v) else "n/a"


def render_ai_lab(selected, timeframe, data, analysis, backtest_trades_df, view="all", show_header=True):
    """view = 'all' renders the 12 tabs together; otherwise only the tabs that belong on that page."""
    global _AL_ACTIVE
    from contextlib import nullcontext
    names = AL_ALL_TABS if view == "all" else AL_VIEWS[view]
    _AL_ACTIVE = set(names)
    if show_header:
        st.divider()
        st.header("AI Lab - Trader Intelligence")
        st.caption("Everything here is computed from your own journal, signal log, backtest and candles. Small samples are labelled; nothing is invented. "
                   "Learning features are research aids: they do not change the live signal engine.")
    journal = load_journal()
    closed = _al_memo("closed", _al_journal_sig(journal), lambda: _al_closed_trades(journal))
    n_closed = 0 if closed is None or closed.empty else len(closed)
    have_data = isinstance(data, pd.DataFrame) and not data.empty
    bt_feat = _al_bt_features(backtest_trades_df) if isinstance(backtest_trades_df, pd.DataFrame) else pd.DataFrame()
    model = ai_lab_get_model(journal) if view in ("all", "command", "scanner") else {"status": "learning", "n": 0, "need": AI_LAB_MIN_MODEL_TRADES}
    if view in ("all", "command"):
        st.caption(f"Closed journal trades available: {n_closed} | Simulated backtest trades: {len(bt_feat)} | "
                   f"Self-learning model: {'active' if model.get('status') == 'active' else 'learning'}")
    if len(names) == 1:
        _tab_objs = {names[0]: st.container()}
    else:
        _tab_objs = dict(zip(names, st.tabs(names)))

    def _T(name):
        return _tab_objs.get(name) or nullcontext()

    # ---------- shared (cached) values ----------
    def get_eval(h):
        if not have_data:
            return pd.DataFrame()
        audit = load_signal_audit()
        sig = (selected, timeframe, h, len(audit), len(data), str(data.index[-1]))
        return _al_memo("pred_eval", sig, lambda: ai_lab_eval_predictions(audit, data, selected, timeframe, horizon=h))

    def calib_pairs(source):
        if source == "Paper trades (real outcomes)":
            if closed is None or closed.empty:
                return pd.DataFrame()
            return pd.DataFrame({"confidence": closed["ConfidenceAtEntry"], "outcome": closed["Win"].astype(float)})
        ev = get_eval(int(st.session_state.get("al_horizon", 10)))
        if ev is None or ev.empty:
            return pd.DataFrame()
        r = ev[ev["Status"] == "Resolved"]
        return pd.DataFrame({"confidence": r["Confidence"], "outcome": r["Direction OK"].astype(float)})

    _need_coach = view in ("all", "command") and n_closed > 0
    weak_summary, weak_per, worst = ai_lab_weakness(closed) if _need_coach else (pd.DataFrame(), pd.DataFrame(), None)
    str_tables, str_best = ai_lab_strength(closed) if _need_coach else ({}, {})

    # ---------- 1. WEAKNESS ----------
    with _T("Weakness"):
        def body():
            st.subheader("Personal Weakness Analyzer")
            if not n_closed:
                st.info("No closed trades in the journal yet. Close some paper trades and this analyser will start working.")
                return
            c1, c2 = st.columns(2)
            rev_min = c1.number_input("Revenge-trade window (minutes after a loss)", 5, 120, 20, key="al_rev_min")
            max_day = c2.number_input("Overtrading: more than N trades per day", 2, 30, 5, key="al_max_day")
            summ, per, w = ai_lab_weakness(closed, int(rev_min), int(max_day))
            if w:
                st.error(f"Largest mistake: **{w['name']}** - {w['pct']}% of trades ({w['trades']} trades), estimated cost {w['cost']:+,.2f}.\n\nFix: {w['fix']}")
            elif not summ.empty:
                st.warning("Behaviour patterns were found, but none has a clearly negative total yet.")
            else:
                st.success("No FOMO / early / late / counter-trend / SL-shift / revenge behaviour detected in the closed trades.")
            if not summ.empty:
                st.dataframe(summ, use_container_width=True, hide_index=True)
            with st.expander("Per-trade flags"):
                st.dataframe(per.sort_values("Entry time", ascending=False), use_container_width=True, hide_index=True)
            st.caption("Rules: FOMO = >1% from EMA20, RSI beyond 72 (trade direction) or volume-chase; Early = ADX<20 or early alert; Late = >1.5% from EMA20 or RSI>78; "
                       "Counter-trend = EMA20 vs EMA50 against the side; SL shift = loss >1.25x planned risk; Revenge = re-entry within the window after a loss or 1.5x size. "
                       "Estimated cost = how much worse those trades did than the same number of normal trades (for SL shift: the loss beyond the planned risk).")
        _al_safe("weakness", body)

    # ---------- 2. STRENGTH ----------
    with _T("Strengths"):
        def body():
            st.subheader("Personal Strength Analyzer")
            if n_closed:
                b = str_best.get("Setup")
                if b:
                    st.success(f"Best setup: **{b['Setup']}** - win rate {b['Win rate %']}% ({b['Trades']} trades), average {b['Avg PnL']:+,.2f}, profit factor {b['Profit factor']}")
                else:
                    st.info("No setup has a positive average with at least 5 trades yet.")
                cols = st.columns(3)
                for i, dim in enumerate(["Market regime", "Symbol", "Timeframe", "Side", "Entry hour"]):
                    bb = str_best.get(dim)
                    cols[i % 3].metric(f"Best {dim.lower()}", str(bb[dim]) if bb else "n/a", f"{bb['Win rate %']}% WR, {bb['Trades']} trades" if bb else "need >=5 trades")
                for dim, tab in str_tables.items():
                    with st.expander(f"{dim} breakdown", expanded=(dim == "Setup")):
                        st.dataframe(tab, use_container_width=True, hide_index=True)
            else:
                st.info("No closed journal trades yet. Showing what the simulated backtest says instead.")
            if len(bt_feat) >= 20:
                st.markdown("**Backtest conditions (simulated, same symbol/timeframe)**")
                rows = []
                conds = {"EMA aligned": bt_feat["EMA"] == 1.0, "EMA not aligned": bt_feat["EMA"] == 0.0, "Volume spike": bt_feat["VOL"] == 1.0,
                         "ADX > 25": bt_feat["ADX"] > 25, "ADX < 18": bt_feat["ADX"] < 18, "RSI(dir) > 58": bt_feat["RSI_dir"] > 58,
                         "BUY trades": bt_feat["Side"] == "BUY", "SELL trades": bt_feat["Side"] == "SELL"}
                for name, m in conds.items():
                    p = _al_perf(bt_feat.loc[m, "PnL"])
                    if p["n"]:
                        rows.append({"Condition": name, "Trades": p["n"], "Win rate %": round(p["wr"], 1), "Profit factor": round(p["pf"], 2) if pd.notna(p["pf"]) else np.nan, "Avg PnL": round(p["avg"], 2)})
                st.dataframe(pd.DataFrame(rows).sort_values("Avg PnL", ascending=False), use_container_width=True, hide_index=True)
        _al_safe("strengths", body)

    # ---------- 3. STRATEGY BUILDER ----------
    with _T("Strategy Builder"):
        def body():
            st.subheader("Auto Strategy Builder")
            st.caption("Searches 240 simple filter combinations (RSI, ADX, EMA alignment, volume spike, MACD) on the older 70% of simulated trades, "
                       "then re-tests the best ones on the newest 30% the search never saw. Quote the OUT-OF-SAMPLE win rate, not the training one.")
            ranked, meta = ai_lab_mine_rules(bt_feat)
            if ranked.empty:
                st.info(meta.get("reason") or "Not enough data.")
                return
            m1, m2, m3 = st.columns(3)
            m1.metric("Trades analysed", meta["n"]); m2.metric("Baseline win rate (train / test)", f"{meta['base_train_wr']:.0f}% / {meta['base_test_wr']:.0f}%")
            m3.metric("Combinations tried", meta["combos"])
            top = ranked.iloc[0]
            if bool(top["Validated"]):
                st.success(f"Best validated rule: **{top['Rule']}**\n\nExpected win rate (out-of-sample): **{top['Test WR %']}%** over {int(top['Test trades'])} unseen trades "
                           f"(training: {top['Train WR %']}% over {int(top['Train trades'])}).")
            else:
                st.warning(f"Best candidate: **{top['Rule']}** (training win rate {top['Train WR %']}%), but it did NOT hold up on unseen trades "
                           f"({int(top['Test trades'])} test trades). Treat it as a hypothesis, not an edge.")
            st.dataframe(ranked.drop(columns=["RuleJSON"]), use_container_width=True, hide_index=True)
            if n_closed:
                try:
                    jf = _al_journal_features(closed)
                    rule = json.loads(top["RuleJSON"])
                    mk = _al_rule_mask(jf, rule, na_ok=True)
                    p = _al_perf(jf.loc[mk, "PnL"])
                    st.caption(f"Check against YOUR real paper trades (unavailable features ignored): {p['n']} matching trades, win rate {p['wr']:.0f}%." if p["n"] else "Check against your real trades: no matching trades.")
                except Exception:
                    pass
            st.warning(f"With {meta['combos']} combinations tested, some will look good by luck. Prefer rules that are validated, simple, and also confirmed by your live paper trades.")
        _al_safe("strategy_builder", body)

    # ---------- 4. EVOLUTION ----------
    with _T("Evolution"):
        def body():
            st.subheader("Strategy Evolution (weekly)")
            st.caption("Once per ISO week (and per symbol/timeframe) the builder re-optimises, re-ranks and stores a snapshot, then shows what changed since last week.")
            force = st.button("Re-optimise now", key="al_evo_now")
            cur, hist, notes, ran, meta, wk = ai_lab_evolution(bt_feat, selected, timeframe, force=force)
            if cur.empty:
                st.info(meta.get("reason") or f"No snapshot for {wk} yet: {len(bt_feat)} trades are available, 30 are needed.")
            else:
                st.success(f"Snapshot for {wk}{' (just generated)' if ran else ''} - {len(cur)} ranked rules for {selected} / {timeframe}.")
                st.dataframe(cur.drop(columns=["RuleJSON", "WeekKey", "Symbol", "Timeframe"], errors="ignore"), use_container_width=True, hide_index=True)
                if notes:
                    st.markdown("**Changes vs last week**")
                    for n_ in notes:
                        st.markdown(f"- {n_}")
            champ = hist[hist["Rank"] == 1].sort_values("WeekKey") if not hist.empty else hist
            if not champ.empty:
                with st.expander("History of the #1 rule"):
                    st.dataframe(champ[["WeekKey", "Rule", "Train WR %", "Test trades", "Test WR %", "Validated"]], use_container_width=True, hide_index=True)
        _al_safe("evolution", body)

    # ---------- 5. SELF LEARNING ----------
    with _T("Self-Learning"):
        def body():
            st.subheader("Self-Learning Engine")
            if model.get("status") != "active":
                st.info(f"Learning: {model.get('n', 0)} of {model.get('need', AI_LAB_MIN_MODEL_TRADES)} closed trades needed before the model is trained. "
                        "It re-trains automatically every time a new trade closes.")
                return
            pq = model["preq"]
            st.success(f"Model trained on {model['n']} closed trades (updated {model['updated_at']}). Base win rate {model['base_rate'] * 100:.0f}%.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Progressive accuracy", f"{pq['acc']:.0f}%", f"{pq['acc'] - pq['base_acc']:+.0f} pts vs always-guess-majority")
            c2.metric("Brier score (lower=better)", f"{pq['brier']:.3f}", f"{pq['brier'] - pq['base_brier']:+.3f} vs base rate", delta_color="inverse")
            c3.metric("Tested on", f"{pq['n']} later trades")
            if pq["acc"] <= pq["base_acc"] or pq["brier"] >= pq["base_brier"]:
                st.warning("The model has NOT yet shown it predicts better than the plain base rate. Do not rely on its probabilities.")
            else:
                st.info("On trades it had not seen, the model beat the base rate. Still keep position sizes unchanged until the lead persists.")
            wdf = pd.DataFrame({"Feature": model["features"], "Effect": model["w"]}).sort_values("Effect")
            fig = go.Figure(go.Bar(x=wdf["Effect"], y=wdf["Feature"], orientation="h"))
            fig.update_layout(height=420, title="What the model learned (negative = hurts win chance, positive = helps)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key="al_model_weights")
            cA, cB = st.columns(2)
            if model["regimes"]:
                cA.markdown("**By market regime**"); cA.dataframe(pd.DataFrame(model["regimes"]), use_container_width=True, hide_index=True)
            if model["behaviours"]:
                cB.markdown("**Your behaviour**"); cB.dataframe(pd.DataFrame(model["behaviours"]), use_container_width=True, hide_index=True)
            f = _al_current_features(analysis) if analysis else {}
            p = ai_lab_predict(model, f) if f else None
            if p is not None:
                st.metric("Learned win probability for the current signal", f"{p * 100:.0f}%")
        _al_safe("self_learning", body)

    # ---------- 6. WEEKLY REPORT ----------
    with _T("Weekly Report"):
        def body():
            st.subheader("Weekly AI Report")
            if not n_closed:
                st.info("No closed trades yet.")
                return
            ref = closed["Exit_DT"].where(closed["Exit_DT"].notna(), closed["Entry_DT"]).dropna()
            weeks = sorted({_al_week_key(d) for d in ref}, reverse=True)
            if not weeks:
                st.info("Closed trades have no usable dates.")
                return
            wk = st.selectbox("Week", weeks, key="al_report_week")
            tab, csum = ai_lab_calibration(calib_pairs("Paper trades (real outcomes)"))
            rep = ai_lab_weekly_report(closed, wk, csum)
            if rep is None:
                st.info("No closed trades in that week.")
                return
            c = st.columns(4)
            c[0].metric("Trades", rep["trades"]); c[1].metric("Win rate", f"{rep['win_rate']:.0f}%"); c[2].metric("Net PnL", f"{rep['pnl']:+,.2f}")
            c[3].metric("Profit factor", f"{rep['profit_factor']:.2f}" if pd.notna(rep["profit_factor"]) else "n/a")
            bs, ws, bm = rep["best_setup"], rep["worst_setup"], rep["biggest_mistake"]
            st.markdown(f"**Best setup:** {bs['Setup'] + ' (' + str(bs['Win rate %']) + '% WR, ' + str(bs['Trades']) + ' trades)' if bs else 'none identified'}")
            st.markdown(f"**Worst setup:** {ws['Setup'] + ' (' + str(ws['Win rate %']) + '% WR, ' + str(ws['Trades']) + ' trades)' if ws else 'none identified'}")
            st.markdown(f"**Biggest mistake:** {bm['name'] + ' - ' + str(bm['trades']) + ' trades, estimated cost ' + format(bm['cost'], '+,.2f') if bm else 'none detected'}")
            st.markdown("**Improvement plan**")
            for t in rep["plan"]:
                st.markdown(f"- {t}")
            cA, cB = st.columns(2)
            if cA.button("Save this report", key="al_report_save"):
                st.success("Saved to the report archive.") if ai_lab_save_report(rep) else st.error("Could not save the report.")
            cB.download_button("Download report (.txt)", rep["text"], file_name=f"weekly_ai_report_{wk}.txt", key="al_report_dl")
            # auto-archive: store the last completed week once, without any click
            try:
                last_done = _al_week_key(ist_now() - timedelta(days=7))
                saved = safe_read_csv(AI_LAB_REPORT_FILE, pd.DataFrame())
                if wk == last_done and (saved.empty or "WeekKey" not in saved.columns or wk not in set(saved["WeekKey"])):
                    ai_lab_save_report(rep)
            except Exception:
                pass
            saved = safe_read_csv(AI_LAB_REPORT_FILE, pd.DataFrame())
            if saved is not None and not saved.empty:
                with st.expander("Saved weekly reports"):
                    st.dataframe(saved.drop(columns=["Report"], errors="ignore").sort_values("WeekKey", ascending=False), use_container_width=True, hide_index=True)
        _al_safe("weekly_report", body)

    # ---------- 7. CALIBRATION ----------
    with _T("Calibration"):
        def body():
            st.subheader("Confidence Calibration")
            st.caption("Compares the confidence the app printed with how often those signals actually worked.")
            src = st.radio("Outcome source", ["Paper trades (real outcomes)", "Signal follow-through (price)"], horizontal=True, key="al_calib_src")
            pairs = calib_pairs(src)
            tab, s = ai_lab_calibration(pairs)
            if s is None or tab.empty:
                st.info("No confidence/outcome pairs yet. " + ("Close paper trades that were entered with a confidence value." if src.startswith("Paper") else "Signals need to be logged and then have enough later candles."))
                return
            c = st.columns(4)
            c[0].metric("Samples", s["n"]); c[1].metric("Avg confidence - actual", f"{s['avg_gap']:+.1f} pts")
            c[2].metric("Brier score", f"{s['brier']:.3f}"); c[3].metric("Calibration error (ECE)", f"{s['ece']:.1f} pts")
            (st.error if abs(s["avg_gap"]) > 10 else st.warning if abs(s["avg_gap"]) > 5 else st.success)(s["verdict"])
            st.dataframe(tab, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=tab["Avg predicted %"], y=tab["Actual success %"], mode="lines+markers", name="Actual",
                                     text=tab["Signals"], hovertemplate="Predicted %{x}%<br>Actual %{y}%<br>n=%{text}<extra></extra>"))
            fig.update_layout(height=360, xaxis_title="Predicted confidence %", yaxis_title="Actual success %", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True, key="al_calib_chart")
            if analysis and analysis.get("confidence") is not None:
                cv = ai_lab_calibrated_value(tab, analysis.get("confidence"))
                if cv:
                    st.info(f"Current signal confidence {_al_num(analysis.get('confidence')):.0f} -> calibrated estimate **{cv[0]:.0f}%** (from {cv[1]} similar past signals, shrunk toward the overall rate).")
            st.caption("Calibrated estimate = bucket success rate shrunk toward the overall rate, so tiny buckets are not over-trusted. It is shown for guidance and does not alter the signal.")
        _al_safe("calibration", body)

    # ---------- 8. PREDICTION ACCURACY ----------
    with _T("Prediction Accuracy"):
        def body():
            st.subheader("Prediction Accuracy Dashboard")
            if not have_data:
                st.info("Needs live candles to judge what happened after each signal.")
                return
            h = st.slider("Judge each signal after N candles", 3, 30, 10, key="al_horizon")
            ev = get_eval(int(h))
            if ev.empty:
                st.info(f"No BUY/SELL signals logged yet for {selected} / {timeframe}. Signals are logged automatically as they appear; change symbol/timeframe to see others.")
                return
            s = ai_lab_prediction_summary(ev)
            if not s or not s.get("resolved"):
                st.info(f"{s['pending'] if s else 0} signals are still pending (need {h} candles after the signal).")
            else:
                c = st.columns(4)
                c[0].metric("Direction accuracy", f"{s['direction']:.0f}%", f"{s['resolved']} resolved signals")
                c[1].metric("Magnitude accuracy", f"{s['magnitude']:.0f}%" if pd.notna(s["magnitude"]) else "n/a")
                c[2].metric("Average error", f"{s['avg_error']:.2f}%" if pd.notna(s["avg_error"]) else "n/a")
                c[3].metric("Average move", f"{s['avg_move']:+.2f}%")
                if s["resolved"] < 20:
                    st.warning("Fewer than 20 resolved signals: treat these percentages as indicative only.")
            st.dataframe(ev.sort_values("Signal time", ascending=False), use_container_width=True, hide_index=True)
            st.caption("Direction OK = price closed on the predicted side after N candles. Magnitude OK = the best move reached the expected distance "
                       "(Target 1 if logged, else 1 ATR). Average error = |actual move - expected move| as % of price. Last 30 signals of this symbol/timeframe.")
        _al_safe("prediction_accuracy", body)

    # ---------- 9. COPILOT ----------
    with _T("Copilot"):
        def body():
            st.subheader("AI Copilot")
            st.caption("Answers come from the app's own scores, levels and your journal (rule-based, no external AI, so it cannot invent facts).")
            hist_ = st.session_state.setdefault("al_chat", [])
            qcols = st.columns(5)
            quick = {"Why Buy?": "Why buy?", "Why Sell?": "Why sell?", "Why SL here?": "Why is the stop-loss here?", "Why Exit?": "Why exit?", "Why no trade?": "Why no trade?"}
            asked = None
            for col, (lab, q) in zip(qcols, quick.items()):
                if col.button(lab, key=f"al_q_{lab}"):
                    asked = q
            with st.form("al_copilot_form", clear_on_submit=True):
                typed = st.text_input("Ask anything about the current signal, your risk, weaknesses or best setup", key="al_copilot_text")
                sent = st.form_submit_button("Ask")
            if sent and typed.strip():
                asked = typed.strip()
            if asked:
                tab_c, _ = ai_lab_calibration(calib_pairs("Paper trades (real outcomes)"))
                ctx = {"analysis": analysis, "journal": journal, "selected": selected, "worst": worst, "best": str_best,
                       "report": ai_lab_weekly_report(closed, _al_week_key()) if n_closed else None,
                       "calibrated": ai_lab_calibrated_value(tab_c, analysis.get("confidence")) if analysis and analysis.get("confidence") is not None else None,
                       "model_prob": ai_lab_predict(model, _al_current_features(analysis)) if analysis else None}
                hist_.append(("user", asked))
                hist_.append(("assistant", ai_lab_copilot_answer(asked, ctx)))
                del hist_[:-20]
            for role, msg in reversed(hist_[-8:]):
                try:
                    with st.chat_message(role):
                        st.markdown(msg)
                except Exception:
                    st.markdown(f"**{'You' if role == 'user' else 'Copilot'}:** {msg}")
        _al_safe("copilot", body)

    # ---------- 10. TRADE REPLAY ----------
    with _T("Trade Replay"):
        def body():
            st.subheader("Trade Replay")
            if not have_data:
                st.info("Replay needs candles for the selected symbol/timeframe.")
                return
            idx = pd.DatetimeIndex(pd.to_datetime(data.index))
            sources = []
            jt = closed[(closed["Symbol"] == selected) & (closed["Timeframe"] == timeframe)] if n_closed else pd.DataFrame()
            if not jt.empty:
                sources.append("My journal trades")
            if isinstance(backtest_trades_df, pd.DataFrame) and not backtest_trades_df.empty and "Date" in backtest_trades_df.columns:
                sources.append("Backtest trades")
            if not sources:
                st.info(f"No closed trades for {selected} / {timeframe} and no backtest trades to replay. Select the symbol/timeframe the trade was taken on.")
                return
            src = st.radio("Trades", sources, horizontal=True, key="al_replay_src")
            trades = []
            if src == "My journal trades":
                for _, r in jt.iterrows():
                    if pd.notna(r["Entry_DT"]) and idx[0] <= r["Entry_DT"] <= idx[-1]:
                        trades.append({"label": f"{r['Entry_DT']:%d %b %H:%M} {r['Side']} PnL {r['PnLv']:+,.2f}", "side": r["Side"], "entry_time": r["Entry_DT"],
                                       "exit_time": r["Exit_DT"], "entry": r["EntryPrice"], "exit": r["ExitPrice"], "sl": r["StopLoss"],
                                       "t1": _al_num(r.get("Target1")), "t2": _al_num(r.get("Target2")), "t3": _al_num(r.get("Target3")), "pnl": r["PnLv"]})
            else:
                bt = backtest_trades_df.copy()
                for _, r in bt.iterrows():
                    et = pd.to_datetime(f"{r['Date']} {r['Time']}", format="%d-%m-%Y %H:%M:%S", errors="coerce")
                    xt = pd.to_datetime(r.get("ExitTime"), format="%d-%m-%Y %H:%M:%S", errors="coerce") if "ExitTime" in bt.columns else pd.NaT
                    if pd.notna(et):
                        trades.append({"label": f"{et:%d %b %H:%M} {r['Signal']} PnL {r['PnL']:+,.2f}", "side": "BUY" if "BUY" in str(r["Signal"]).upper() else "SELL",
                                       "entry_time": et, "exit_time": xt, "entry": _al_num(r["Entry"]), "exit": _al_num(r["Exit"]), "sl": _al_num(r["SL"]),
                                       "t1": _al_num(r["T1"]), "t2": _al_num(r["T2"]), "t3": _al_num(r["T3"]), "pnl": _al_num(r["PnL"])})
            if not trades:
                st.info("None of these trades fall inside the candles currently loaded (older candles are not kept). Try a higher timeframe or a recent trade.")
                return
            trades = trades[::-1]
            pick = st.selectbox("Trade", range(len(trades)), format_func=lambda i: trades[i]["label"], key="al_replay_pick")
            t = trades[pick]
            m = ai_lab_replay_metrics(data, t["side"], t["entry_time"], t["exit_time"], t["entry"], t["exit"], t["sl"], t["t1"], t["pnl"])
            end_t = t["exit_time"] if pd.notna(t["exit_time"]) else t["entry_time"] + pd.Timedelta(minutes=30)
            lo = idx.searchsorted(t["entry_time"]) - 25
            hi = idx.searchsorted(end_t, side="right") + 10
            win = data.iloc[max(0, lo):min(len(data), hi)]
            show = st.slider("Replay: show candles up to", 1, len(win), len(win), key="al_replay_bar")
            part = win.iloc[:show]
            fig = go.Figure(go.Candlestick(x=part.index, open=part["Open"], high=part["High"], low=part["Low"], close=part["Close"], name="Price"))
            if "EMA20" in part.columns:
                fig.add_trace(go.Scatter(x=part.index, y=part["EMA20"], name="EMA20", line=dict(width=1)))
            shown_to = part.index[-1]
            for nm, lv in (("SL", t["sl"]), ("T1", t["t1"]), ("T2", t["t2"]), ("T3", t["t3"])):
                if np.isfinite(_al_num(lv)) and shown_to >= t["entry_time"]:
                    fig.add_hline(y=lv, line_dash="dot", annotation_text=nm, line_width=1)
            if shown_to >= t["entry_time"]:
                fig.add_trace(go.Scatter(x=[t["entry_time"]], y=[t["entry"]], mode="markers+text", text=["ENTRY"], textposition="top center",
                                         marker=dict(size=13, symbol="triangle-up" if t["side"] == "BUY" else "triangle-down"), name="Entry"))
            if pd.notna(t["exit_time"]) and shown_to >= t["exit_time"]:
                fig.add_trace(go.Scatter(x=[t["exit_time"]], y=[t["exit"]], mode="markers+text", text=["EXIT"], textposition="bottom center",
                                         marker=dict(size=12, symbol="x"), name="Exit"))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True, key="al_replay_chart")
            c = st.columns(3)
            c[0].metric("Result", f"{t['pnl']:+,.2f}")
            c[1].metric("Best move available (MFE)", _al_money(m["mfe"]))
            c[2].metric("Move captured", f"{m['capture'] * 100:.0f}%" if pd.notna(m["capture"]) else "n/a")
            if "better_entry" in m:
                px, when, saved = m["better_entry"]
                st.info(f"Correct entry (hindsight): the best price in the 12 bars before entry was {px:.2f} at {when:%H:%M}, {saved:.2f} points better.")
            if "ema_retest" in m:
                st.info(f"An EMA20 retest was available near {m['ema_retest'][0]:.2f} at {m['ema_retest'][1]:%H:%M}.")
            if pd.notna(m["mfe"]) and pd.notna(t["exit"]):
                st.info(f"Correct exit (hindsight): the best price reached after entry was {t['entry'] + (m['mfe'] if t['side'] == 'BUY' else -m['mfe']):.2f}; the trade exited at {t['exit']:.2f}.")
            for x in m["mistakes"]:
                st.warning("Mistake: " + x)
            for x in m["good"]:
                st.success(x)
            st.caption("Replay uses the candles currently loaded. This is hindsight analysis to learn from, not a promise that the 'better' prices were obtainable.")
        _al_safe("trade_replay", body)

    # ---------- 11. OPPORTUNITIES ----------
    with _T("Opportunities"):
        def body():
            st.subheader("Opportunity Ranking - Top Trades Today")
            st.caption("Scores NIFTY, BANKNIFTY, SENSEX and the stock list on 5-minute candles: Quality (trend stack, ADX, volume, RSI zone, VWAP, SuperTrend), "
                       "RR (room to the nearest swing vs 1 ATR risk) and Probability (your self-learned model, else your own history for that symbol). Runs on demand (about 10-20 s).")
            if st.button("Rank opportunities now", key="al_opp_run"):
                with st.spinner("Scoring the watchlist..."):
                    df_, missing = ai_lab_rank_opportunities(model, closed, dict(stocks))
                st.session_state["al_opps"] = {"df": df_, "missing": missing, "at": now_str()}
            res = st.session_state.get("al_opps")
            if not res:
                st.info("Press the button to rank the watchlist.")
                return
            df_ = res["df"]
            if df_.empty:
                st.error("No symbol could be scored (price feed unavailable). Check the Scanner data diagnostics panel.")
                return
            st.caption(f"Ranked at {res['at']}")
            top = df_[df_["Direction"] != "NEUTRAL"].head(5)
            cols = st.columns(max(1, len(top)))
            for col, (_, r) in zip(cols, top.iterrows()):
                col.metric(f"#{int(r['Rank'])} {r['Symbol']} - {r['Direction']}", f"Quality {r['Quality']:.0f} ({r['Grade']})",
                           f"RR {r['RR']:.1f}" + (f" | P {r['Probability %']:.0f}%" if pd.notna(r["Probability %"]) else " | P n/a"))
            st.dataframe(df_, use_container_width=True, hide_index=True)
            if res["missing"]:
                st.caption("No usable data for: " + ", ".join(res["missing"]))
            st.caption("Probability shows n/a until you have a trained model or at least 5 closed trades in that symbol. Ranking is a screening aid, not a trade instruction.")
        _al_safe("opportunities", body)

    # ---------- 12. PORTFOLIO RISK ----------
    with _T("Portfolio Risk"):
        def body():
            st.subheader("Portfolio Risk Analyzer")
            pos = ai_lab_open_positions(journal)
            ps = load_paper_state()
            bal = _al_num(ps["Balance"].iloc[0], DEFAULT_INITIAL_BALANCE) if ps is not None and not ps.empty else DEFAULT_INITIAL_BALANCE
            used = _al_num(ps["UsedMargin"].iloc[0], 0.0) if ps is not None and not ps.empty and "UsedMargin" in ps.columns else 0.0
            if pos.empty:
                st.info("No open paper positions, so there is no portfolio risk right now.")
                return
            equity = bal + used + float(pos["OpenPnL"].fillna(0).sum())
            c1, c2, c3, c4 = st.columns(4)
            lim_risk = c1.number_input("Max open risk % of equity", 0.5, 20.0, 3.0, key="al_lim_risk")
            lim_pos = c2.number_input("Max single position %", 5, 100, 40, key="al_lim_pos")
            lim_sec = c3.number_input("Max sector share %", 10, 100, 50, key="al_lim_sec")
            lim_cor = c4.number_input("Correlation alert level", 0.5, 0.99, 0.80, 0.05, key="al_lim_cor")
            gross, net = float(pos["Notional"].sum()), float((pos["sgn"] * pos["Notional"]).sum())
            open_risk = float(pos["RiskToStop"].fillna(0).sum())
            m = st.columns(5)
            m[0].metric("Paper equity (approx.)", _al_money(equity)); m[1].metric("Gross exposure", _al_money(gross), f"{gross / equity * 100:.0f}% of equity")
            m[2].metric("Net exposure", _al_money(net), "net long" if net > 0 else "net short" if net < 0 else "flat")
            m[3].metric("Open risk to stops", _al_money(open_risk), f"{open_risk / equity * 100:.2f}% of equity")
            m[4].metric("Open PnL", _al_money(pos["OpenPnL"].sum()))
            show = pos[["Symbol", "Side", "Qty", "EntryPrice", "Price", "Stop", "Notional", "OpenPnL", "RiskToStop", "LockedProfit", "Sector"]].copy()
            show["% of equity"] = (show["Notional"] / equity * 100).round(1)
            st.dataframe(show.round(2), use_container_width=True, hide_index=True)
            alerts = []
            if open_risk / equity * 100 > lim_risk:
                alerts.append(f"Open risk {open_risk / equity * 100:.2f}% exceeds your {lim_risk}% limit.")
            big = show[show["% of equity"] > lim_pos]
            for _, r in big.iterrows():
                alerts.append(f"{r['Symbol']} is {r['% of equity']}% of equity (limit {lim_pos}%).")
            sec = pos.groupby("Sector")["Notional"].sum().sort_values(ascending=False)
            sec_df = pd.DataFrame({"Sector": sec.index, "Exposure": sec.values.round(2), "Share of gross %": (sec.values / gross * 100).round(1)})
            for _, r in sec_df.iterrows():
                if r["Share of gross %"] > lim_sec and len(sec_df) > 1:
                    alerts.append(f"Sector concentration: {r['Sector']} is {r['Share of gross %']}% of gross exposure (limit {lim_sec}%).")
            if len(sec_df) == 1 and len(pos) > 1:
                alerts.append(f"All {len(pos)} positions are in one sector ({sec_df.iloc[0]['Sector']}).")
            st.markdown("**Sector risk**")
            st.dataframe(sec_df, use_container_width=True, hide_index=True)
            st.markdown("**Correlation risk**")
            if pos["Symbol"].nunique() < 2:
                st.caption("Only one symbol is open, so there is no correlation between positions.")
            elif st.button("Measure correlation now", key="al_corr_run"):
                corr, msg = ai_lab_portfolio_corr(pos, dict(stocks))
                st.session_state["al_corr"] = {"corr": corr, "msg": msg, "at": now_str(), "syms": sorted(pos["Symbol"].unique())}
            cr = st.session_state.get("al_corr")
            if cr and cr["syms"] == sorted(pos["Symbol"].unique()) and pos["Symbol"].nunique() >= 2:
                if cr["corr"] is None:
                    st.warning(cr["msg"])
                else:
                    corr = cr["corr"]
                    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}"))
                    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True, key="al_corr_chart")
                    side_of = pos.groupby("Symbol")["sgn"].sum().apply(np.sign)
                    syms = list(corr.index)
                    for i in range(len(syms)):
                        for j in range(i + 1, len(syms)):
                            rho = float(corr.iloc[i, j])
                            same = side_of.get(syms[i], 0) * side_of.get(syms[j], 0) > 0
                            if rho >= lim_cor and same:
                                alerts.append(f"{syms[i]} and {syms[j]} move together (correlation {rho:.2f}) and you hold both in the same direction: this is really one bigger bet.")
                    eb = ai_lab_effective_bets(pos, corr)
                    if pd.notna(eb):
                        st.metric("Effective independent positions", f"{eb:.1f} of {pos['Symbol'].nunique()}")
                    st.caption(f"Measured {cr['at']} on 5-minute returns over ~5 days.")
            if alerts:
                for a_ in alerts:
                    st.warning(a_)
            else:
                st.success("All checked limits are within bounds.")
        _al_safe("portfolio_risk", body)
# ===================== AI LAB END =======================
