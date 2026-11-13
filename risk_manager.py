"""risk_manager.py - daily/weekly/monthly risk locks, position sizing, paper-trade entry/exit, scale-out
targets, trailing stops and the auto-exit engine.  Split out of app.py; behaviour is unchanged."""
import os
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from database import *   # noqa: F401,F403  (paths, load_/save_ helpers, ist_now, now_str, log_app_error ...)


def bind(**deps):
    """app.py passes in the few names that still live in the main file (classify_trade_loss ...)."""
    globals().update(deps)


# =========================================================
# RISK / PAPER TRADING
# =========================================================
def _closed_with_exit_dt(journal):
    """Closed trades with an `ExitDT` column (exit time; falls back to entry time for old rows)."""
    if journal is None or journal.empty or "Status" not in journal.columns:
        return pd.DataFrame()
    closed = journal[journal["Status"].astype(str).str.upper() == "CLOSED"].copy()
    if closed.empty:
        return closed
    fmt = "%d-%m-%Y %H:%M:%S"
    entry_dt = pd.to_datetime(closed["DateTime"], format=fmt, errors="coerce")
    if "ExitTime" in closed.columns:
        exit_dt = pd.to_datetime(closed["ExitTime"], format=fmt, errors="coerce")
        closed["ExitDT"] = exit_dt.fillna(entry_dt)
    else:
        closed["ExitDT"] = entry_dt
    return closed


def get_today_closed_trades():
    closed = _closed_with_exit_dt(load_journal())
    if closed.empty:
        return closed
    return closed[closed["ExitDT"].dt.date == ist_now().date()].copy()


def get_week_closed_trades():
    closed = _closed_with_exit_dt(load_journal())
    if closed.empty:
        return closed
    now = ist_now()
    start_week = datetime(now.year, now.month, now.day) - timedelta(days=now.weekday())
    return closed[closed["ExitDT"] >= start_week].copy()


def get_month_closed_trades():
    closed = _closed_with_exit_dt(load_journal())
    if closed.empty:
        return closed
    now = ist_now()
    return closed[closed["ExitDT"] >= datetime(now.year, now.month, 1)].copy()


def get_today_opened_count():
    """Trades OPENED today (open + closed) so MaxTradesPerDay cannot be bypassed by leaving trades open."""
    journal = load_journal()
    if journal.empty or "DateTime" not in journal.columns:
        return 0
    entry_dt = pd.to_datetime(journal["DateTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    return int((entry_dt.dt.date == ist_now().date()).sum())


def get_open_position_count():
    journal = load_journal()
    if journal.empty or "Status" not in journal.columns:
        return 0
    return int((journal["Status"].astype(str).str.upper() == "OPEN").sum())


def update_risk_locks():
    try:
        daily_trades = get_today_closed_trades()
        weekly_trades = get_week_closed_trades()
        monthly_trades = get_month_closed_trades()

        risk_state = load_risk_state()
        weekly = load_weekly_risk()
        monthly = load_monthly_risk()
        paper = load_paper_state()

        daily_pnl = pd.to_numeric(daily_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        weekly_pnl = pd.to_numeric(weekly_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        monthly_pnl = pd.to_numeric(monthly_trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

        daily_limit = float(risk_state.loc[0, "DailyLossLimit"])
        weekly_limit = float(weekly.loc[0, "WeeklyLossLimit"])
        monthly_limit = float(monthly.loc[0, "MonthlyLossLimit"])
        capital_protection_pct = float(risk_state.loc[0, "CapitalProtectionPct"])

        # Equity = free cash + notional reserved by open positions. Using cash alone
        # would falsely trip capital protection the moment a large position is opened.
        balance = float(paper.loc[0, "Balance"]) + float(paper.loc[0, "UsedMargin"])
        initial_balance = float(paper.loc[0, "InitialBalance"])
        capital_floor = initial_balance * (1 - capital_protection_pct / 100)

        risk_state.loc[0, "DailyLock"] = bool(daily_pnl <= -abs(daily_limit) or balance <= capital_floor)
        weekly.loc[0, "WeeklyPnL"] = weekly_pnl
        weekly.loc[0, "WeeklyLock"] = bool(weekly_pnl <= -abs(weekly_limit))
        monthly.loc[0, "MonthlyPnL"] = monthly_pnl
        monthly.loc[0, "MonthlyLock"] = bool(monthly_pnl <= -abs(monthly_limit))

        save_risk_state(risk_state)
        save_weekly_risk(weekly)
        save_monthly_risk(monthly)
    except Exception as e:
        log_app_error("update_risk_locks", e)


def risk_check_before_trade():
    update_risk_locks()
    risk_state = load_risk_state()
    weekly_risk = load_weekly_risk()
    monthly_risk = load_monthly_risk()

    trading_enabled = bool(risk_state.loc[0, "TradingEnabled"])
    daily_lock = bool(risk_state.loc[0, "DailyLock"])
    weekly_lock = bool(weekly_risk.loc[0, "WeeklyLock"])
    monthly_lock = bool(monthly_risk.loc[0, "MonthlyLock"])
    max_trades_per_day = int(risk_state.loc[0, "MaxTradesPerDay"])
    trades_today = get_today_opened_count()
    max_open = int(risk_state.loc[0, "MaxOpenPositions"]) if "MaxOpenPositions" in risk_state.columns else 3

    if not trading_enabled:
        return False, "Trading disabled by risk management"
    if daily_lock:
        return False, "Daily lock active"
    if weekly_lock:
        return False, "Weekly lock active"
    if monthly_lock:
        return False, "Monthly lock active"
    if get_open_position_count() >= max_open:
        return False, f"Max open positions reached ({max_open})"
    if trades_today >= max_trades_per_day:
        return False, f"Max trades per day reached ({max_trades_per_day})"
    return True, "Risk checks passed"


def calculate_position_size(entry_price, stop_loss, paper_balance):
    risk_state = load_risk_state()
    risk_pct = float(risk_state.loc[0, "RiskPerTradePct"])
    if entry_price is None or stop_loss is None:
        return 1
    risk_amount = paper_balance * (risk_pct / 100.0)
    per_unit_risk = abs(float(entry_price) - float(stop_loss))
    if per_unit_risk <= 0:
        return 1
    qty = int(risk_amount // per_unit_risk)
    return max(1, qty)


def paper_trade_entry(price, quantity, side="BUY"):
    """Reserve virtual cash/margin for either a paper long or paper short."""
    try:
        price = float(price)
        quantity = int(quantity)
        side = str(side).upper()
        if not np.isfinite(price) or price <= 0 or quantity <= 0 or side not in ("BUY", "SELL"):
            return False
        state = load_paper_state()
        bal = float(state.loc[0, "Balance"])
        used = float(state.loc[0, "UsedMargin"])
        reserved = price * quantity
        # This simulator reserves full notional for both long and short positions.
        if bal + 1e-9 < reserved:
            return False
        state.loc[0, "Balance"] = bal - reserved
        state.loc[0, "UsedMargin"] = used + reserved
        state.loc[0, "OpenPositions"] = int(state.loc[0, "OpenPositions"]) + 1
        return save_paper_state(state)
    except Exception as exc:
        log_app_error("paper_trade_entry", exc)
        return False



def paper_trade_exit(entry_price, exit_price, quantity, signal, final_close=True):
    """Release reserved notional and credit/debit simulated P&L."""
    state = load_paper_state()
    bal = float(state.loc[0, "Balance"])
    used = float(state.loc[0, "UsedMargin"])
    entry_price = float(entry_price)
    exit_price = float(exit_price)
    quantity = float(quantity)
    reserved = entry_price * quantity
    is_long = "BUY" in str(signal).upper() or "CALL" in str(signal).upper()
    pnl = (exit_price - entry_price) * quantity if is_long else (entry_price - exit_price) * quantity
    state.loc[0, "Balance"] = bal + reserved + pnl
    state.loc[0, "UsedMargin"] = max(0.0, used - reserved)
    state.loc[0, "OpenPositions"] = max(0, int(state.loc[0, "OpenPositions"]) - (1 if final_close and quantity > 0 else 0))
    save_paper_state(state)
    return float(pnl)



def get_suggested_paper_levels(analysis, side, entry_price):
    """Return side-correct suggested levels; ATR fallback is explicitly a simulation plan."""
    side = str(side).upper()
    entry = float(entry_price)
    atr = float(analysis.get("atr") or 0.0)
    risk = min(max(atr if np.isfinite(atr) else 0.0, entry * 0.003, 0.01), entry * 0.20)
    same_signal = (side == "BUY" and analysis.get("signal_type") == "buy") or (side == "SELL" and analysis.get("signal_type") == "sell")
    if same_signal:
        sl = analysis.get("stop_loss")
        t1, t2, t3 = analysis.get("target1"), analysis.get("target2"), analysis.get("target3")
        try:
            sl, t1, t2, t3 = float(sl), float(t1), float(t2), float(t3)
            valid = (sl < entry < t1 < t2 < t3) if side == "BUY" else (t3 < t2 < t1 < entry < sl)
            if not valid:
                raise ValueError("Signal levels are not on the correct side")
        except Exception:
            sl = t1 = t2 = t3 = None
    else:
        sl = t1 = t2 = t3 = None
    if sl is None:
        if side == "BUY":
            sl, t1, t2, t3 = entry - risk, entry + 2 * risk, entry + 3 * risk, entry + 4 * risk
        else:
            sl, t1, t2, t3 = entry + risk, entry - 2 * risk, entry - 3 * risk, entry - 4 * risk
    trailing = sl
    return {
        "side": side, "entry": entry, "stop_loss": float(sl),
        "target1": float(t1), "target2": float(t2), "target3": float(t3),
        "trailing_stop": float(trailing), "atr": float(atr if atr > 0 else risk)
    }


def add_trade_to_journal(symbol, timeframe, signal, entry_price, quantity, stop_loss, target1, target2, target3,
                         trailing_stop, analysis, notes="", auto_exit=True, atr=None):
    journal = load_journal()
    side = "SELL" if "SELL" in str(signal).upper() else "BUY"
    new_row = {
        "DateTime": now_str(), "TradeDate": ist_now().strftime("%Y-%m-%d"),
        "TradeTime": ist_now().strftime("%H:%M:%S"), "Symbol": symbol, "Timeframe": timeframe,
        "Signal": side, "SignalType": str(analysis.get("signal_text", side)), "Side": side,
        "EntryPrice": float(entry_price), "ExitPrice": np.nan,
        "Quantity": int(quantity), "RemainingQty": int(quantity), "StopLoss": float(stop_loss),
        "Target1": float(target1), "Target2": float(target2), "Target3": float(target3),
        "TrailingStop": float(trailing_stop), "ATR": float(atr if atr is not None else analysis.get("atr", 0.0)),
        "AutoExitEnabled": bool(auto_exit), "Status": "OPEN", "PnL": 0.0, "RealizedPnL": 0.0,
        "PnLPercent": 0.0, "Notes": notes, "Result": "", "T1_Hit": False, "T2_Hit": False,
        "T3_Hit": False, "LastPrice": float(entry_price), "LastUpdateTime": now_str(),
        "TradeQualityScore": analysis.get("trade_quality_score", 0),
        "ValidationScore": analysis.get("validation_score", 0),
        "StrategyGrade": analysis.get("strategy_grade", "D"),
        **(analysis.get("feature_snapshot", {}) if isinstance(analysis.get("feature_snapshot"), dict) else {}),
        "LossReason": ""
    }
    journal = pd.concat([journal, pd.DataFrame([new_row])], ignore_index=True)
    save_journal(journal)
    return True



def close_trade_in_journal(index_id, exit_price, result=None, t1_hit=None, t2_hit=None, t3_hit=None):
    """Legacy compatibility closer; closes the position's currently remaining quantity."""
    journal = load_journal()
    if index_id not in journal.index:
        return False
    row = journal.loc[index_id]
    entry = float(row["EntryPrice"])
    qty = float(row.get("RemainingQty", row["Quantity"]))
    side = str(row.get("Side", row["Signal"])).upper()
    if qty <= 0:
        return False
    pnl = paper_trade_exit(entry, float(exit_price), qty, side)
    prior = float(row.get("RealizedPnL", 0.0) or 0.0)
    total = prior + pnl
    journal.at[index_id, "RemainingQty"] = 0
    journal.at[index_id, "RealizedPnL"] = total
    journal.at[index_id, "ExitPrice"] = float(exit_price)
    journal.at[index_id, "ExitTime"] = now_str()
    journal.at[index_id, "Status"] = "CLOSED"
    journal.at[index_id, "PnL"] = round(total, 2)
    journal.at[index_id, "PnLPercent"] = round((total / (entry * float(row["Quantity"])) * 100) if entry else 0, 2)
    journal.at[index_id, "Result"] = result or ("WIN" if total > 0 else "LOSS" if total < 0 else "BREAKEVEN")
    journal.at[index_id, "LastPrice"] = float(exit_price)
    journal.at[index_id, "LastUpdateTime"] = now_str()
    for col, value in (("T1_Hit", t1_hit), ("T2_Hit", t2_hit), ("T3_Hit", t3_hit)):
        if value is not None:
            journal.at[index_id, col] = bool(value)
    if str(journal.at[index_id, "Result"]).upper() == "LOSS":
        loss_snapshot = {k: journal.at[index_id, k] for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"] if k in journal.columns}
        journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total, side, float(exit_price), entry)
    save_journal(journal)
    return True



def _paper_bool(value, default=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    try:
        return bool(value)
    except Exception:
        return default



def render_paper_exit_mode_selector():
    """Offer an explicit Manual or Auto Exit mode; Manual is the safe default."""
    valid_modes = {"Manual Exit", "Auto Exit (Enabled)"}
    if st.session_state.get("paper_exit_mode") not in valid_modes:
        st.session_state["paper_exit_mode"] = "Manual Exit"
    mode = st.radio(
        "Exit Management Mode",
        options=["Manual Exit", "Auto Exit (Enabled)"],
        horizontal=True,
        key="paper_exit_mode",
        help=(
            "Manual Exit: positions remain open until you press CLOSE. "
            "Auto Exit: the selected symbol is monitored on Streamlit reruns/auto-refresh "
            "and may scale out at targets or close at stop/trailing stop."
        ),
    )
    enabled = mode == "Auto Exit (Enabled)"
    if enabled:
        st.success("AUTO EXIT ENABLED — keep Auto Refresh ON and keep the position's symbol selected.")
    else:
        st.info("MANUAL EXIT MODE — positions close only when you press CLOSE.")
    return enabled


def sync_open_position_auto_exit(enabled, journal=None):
    """Apply the selected Manual/Auto mode to every currently open paper journal row."""
    if journal is None:
        journal = load_journal()
    if journal.empty:
        return journal
    if "AutoExitEnabled" not in journal.columns:
        journal["AutoExitEnabled"] = False
    status = journal.get("Status", pd.Series("", index=journal.index)).astype(str).str.upper()
    open_mask = status.eq("OPEN")
    if not open_mask.any():
        return journal
    current_flags = journal.loc[open_mask, "AutoExitEnabled"].map(lambda value: _paper_bool(value, False))
    if not current_flags.eq(bool(enabled)).all():
        journal.loc[open_mask, "AutoExitEnabled"] = bool(enabled)
        save_journal(journal)
    return journal


def _paper_side_pnl(entry, exit_price, quantity, side):
    return (exit_price - entry) * quantity if str(side).upper() == "BUY" else (entry - exit_price) * quantity


def close_paper_position(index_id, exit_price, reason="Manual Close"):
    """Close all remaining units of one paper position at the supplied simulated price."""
    journal = load_journal()
    if journal.empty or index_id not in journal.index:
        return False, "Paper position was not found."
    row = journal.loc[index_id]
    if str(row.get("Status", "")).upper() != "OPEN":
        return False, "This paper position is already closed."
    try:
        entry = float(row["EntryPrice"])
        qty = int(float(row.get("RemainingQty", row["Quantity"])))
        exit_price = float(exit_price)
        side = str(row.get("Side", row["Signal"])).upper()
        if qty <= 0 or exit_price <= 0 or not np.isfinite(exit_price):
            return False, "Invalid quantity or exit price."
        pnl = paper_trade_exit(entry, exit_price, qty, side)
        prior = float(row.get("RealizedPnL", 0.0) or 0.0)
        total = prior + pnl
        journal.at[index_id, "RemainingQty"] = 0
        journal.at[index_id, "RealizedPnL"] = total
        journal.at[index_id, "ExitPrice"] = exit_price
        journal.at[index_id, "LastPrice"] = exit_price
        journal.at[index_id, "LastUpdateTime"] = now_str()
        journal.at[index_id, "ExitTime"] = now_str()
        journal.at[index_id, "Status"] = "CLOSED"
        journal.at[index_id, "PnL"] = round(total, 2)
        initial_notional = entry * float(row.get("Quantity", qty))
        journal.at[index_id, "PnLPercent"] = round((total / initial_notional * 100) if initial_notional else 0.0, 2)
        journal.at[index_id, "Result"] = "WIN" if total > 0 else "LOSS" if total < 0 else "BREAKEVEN"
        journal.at[index_id, "Notes"] = (str(row.get("Notes", "")) + f" | {reason}").strip(" |")
        if total < 0:
            loss_snapshot = {k: row.get(k) for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"]}
            journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total, side, exit_price, entry)
        save_journal(journal)
        return True, f"Paper position closed. Total realized P&L: ₹{total:,.2f}"
    except Exception as exc:
        log_app_error("close_paper_position", exc)
        return False, f"Could not close paper position: {exc}"


def _paper_partial_exit(index_id, exit_price, quantity, target_flag=None):
    """Execute a virtual partial exit and update remaining quantity/realized P&L."""
    journal = load_journal()
    if journal.empty or index_id not in journal.index:
        return False
    row = journal.loc[index_id]
    if str(row.get("Status", "")).upper() != "OPEN":
        return False
    entry = float(row["EntryPrice"])
    side = str(row.get("Side", row["Signal"])).upper()
    remaining = int(float(row.get("RemainingQty", row["Quantity"])))
    q = min(remaining, max(0, int(quantity)))
    if q <= 0:
        return False
    pnl = paper_trade_exit(entry, float(exit_price), q, side, final_close=(q >= remaining))
    total_realized = float(row.get("RealizedPnL", 0.0) or 0.0) + pnl
    new_remaining = remaining - q
    journal.at[index_id, "RemainingQty"] = new_remaining
    journal.at[index_id, "RealizedPnL"] = total_realized
    journal.at[index_id, "PnL"] = round(total_realized, 2)
    journal.at[index_id, "LastPrice"] = float(exit_price)
    journal.at[index_id, "LastUpdateTime"] = now_str()
    if target_flag:
        journal.at[index_id, target_flag] = True
    if new_remaining <= 0:
        journal.at[index_id, "RemainingQty"] = 0
        journal.at[index_id, "ExitPrice"] = float(exit_price)
        journal.at[index_id, "ExitTime"] = now_str()
        journal.at[index_id, "Status"] = "CLOSED"
        journal.at[index_id, "Result"] = "WIN" if total_realized > 0 else "LOSS" if total_realized < 0 else "BREAKEVEN"
        initial_notional = entry * float(row.get("Quantity", q))
        journal.at[index_id, "PnLPercent"] = round((total_realized / initial_notional * 100) if initial_notional else 0.0, 2)
        if total_realized < 0:
            loss_snapshot = {k: row.get(k) for k in LOSS_FEATURE_COLUMNS + ["MarketRegime", "MTFTrendStatus"]}
            journal.at[index_id, "LossReason"] = classify_trade_loss(loss_snapshot, total_realized, side, float(exit_price), entry)
    save_journal(journal)
    return True


def auto_paper_trade_exit(current_price, symbol=None, atr=None):
    """Auto-manage only the currently selected symbol; scale out at T1/T2 and close at T3/SL."""
    try:
        price = float(current_price)
        if not np.isfinite(price) or price <= 0:
            return {"closed": 0, "partial": 0, "updated": 0}
        journal = load_journal()
        if journal.empty:
            return {"closed": 0, "partial": 0, "updated": 0}
        open_rows = journal[journal["Status"].astype(str).str.upper().eq("OPEN")].copy()
        closed_count = partial_count = updated_count = 0
        for idx, original in open_rows.iterrows():
            if symbol and str(original.get("Symbol", "")) != str(symbol):
                continue
            if not _paper_bool(original.get("AutoExitEnabled", False), False):
                continue
            # Refresh the row after any partial operation.
            journal = load_journal()
            if idx not in journal.index or str(journal.at[idx, "Status"]).upper() != "OPEN":
                continue
            row = journal.loc[idx]
            side = str(row.get("Side", row.get("Signal", "BUY"))).upper()
            is_long = side == "BUY"
            entry = float(row["EntryPrice"])
            initial_qty = int(float(row.get("Quantity", 1)))
            rem_qty = int(float(row.get("RemainingQty", initial_qty)))
            if rem_qty <= 0:
                continue
            row_atr = row.get("ATR", np.nan)
            try:
                atr_value = float(row_atr) if pd.notna(row_atr) and float(row_atr) > 0 else float(atr or 0.0)
            except Exception:
                atr_value = float(atr or 0.0)
            trail_gap = max(1.0 * atr_value, entry * 0.0005, 0.01)
            old_trail = row.get("TrailingStop", row.get("StopLoss", np.nan))
            try:
                old_trail = float(old_trail) if pd.notna(old_trail) else float(row["StopLoss"])
            except Exception:
                old_trail = float(row["StopLoss"])
            # Trail only once T1 is banked; trailing from entry stops trades out on normal noise.
            if _paper_bool(row.get("T1_Hit", False), False):
                new_trail = max(old_trail, price - trail_gap) if is_long else min(old_trail, price + trail_gap)
            else:
                new_trail = old_trail
            journal.at[idx, "TrailingStop"] = new_trail
            journal.at[idx, "LastPrice"] = price
            journal.at[idx, "LastUpdateTime"] = now_str()
            save_journal(journal)

            stop_loss = float(row["StopLoss"])
            effective_stop = max(stop_loss, new_trail) if is_long else min(stop_loss, new_trail)
            stop_hit = price <= effective_stop if is_long else price >= effective_stop
            if stop_hit:
                ok, _ = close_paper_position(idx, price, "Auto Exit — Stop Loss / Trailing Stop")
                if ok:
                    closed_count += 1
                continue

            target_specs = [
                ("T1_Hit", "Target1", 0.34),
                ("T2_Hit", "Target2", 0.33),
                ("T3_Hit", "Target3", 1.00),
            ]
            for flag, target_col, fraction in target_specs:
                journal = load_journal()
                if idx not in journal.index or str(journal.at[idx, "Status"]).upper() != "OPEN":
                    break
                row_now = journal.loc[idx]
                if _paper_bool(row_now.get(flag, False), False):
                    continue
                target = row_now.get(target_col, np.nan)
                if pd.isna(target):
                    continue
                target = float(target)
                target_hit = price >= target if is_long else price <= target
                if not target_hit:
                    continue
                current_remaining = int(float(row_now.get("RemainingQty", row_now["Quantity"])))
                if fraction >= 1.0:
                    q_exit = current_remaining
                else:
                    q_exit = min(current_remaining, max(1, int(round(initial_qty * fraction))))
                if _paper_partial_exit(idx, target, q_exit, target_flag=flag):
                    partial_count += 1
                    # After T1 protect the remainder around breakeven; after T2 protect T1.
                    journal = load_journal()
                    if idx in journal.index and str(journal.at[idx, "Status"]).upper() == "OPEN":
                        protect_level = entry if flag == "T1_Hit" else float(row_now.get("Target1", entry))
                        existing = float(journal.at[idx, "TrailingStop"])
                        tightened = max(existing, protect_level) if is_long else min(existing, protect_level)
                        journal.at[idx, "TrailingStop"] = tightened
                        save_journal(journal)
            # Count a fully closed position once; partial_count tracks scale-outs.
            final_journal = load_journal()
            if idx in final_journal.index and str(final_journal.at[idx, "Status"]).upper() == "CLOSED":
                closed_count += 1
            updated_count += 1
        return {"closed": closed_count, "partial": partial_count, "updated": updated_count}
    except Exception as exc:
        log_app_error("auto_paper_trade_exit", exc)
        return {"closed": 0, "partial": 0, "updated": 0}
