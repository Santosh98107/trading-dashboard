"""Dashboard card for the trade plan: open-trade alert, chase check on the live signal, and the buy/sell plan prepared before the signal.
Display only. The maths is in trade_plan.py; the app passes in everything this file needs (no imports from app.py).
"""
import pandas as pd

import trade_plan as tpl


def render(st, selected, timeframe, analysis, data, load_journal=None, log_error=None, clock=None):
    """Never raises: any failure is passed to log_error and the card is skipped."""
    if not st.session_state.get("tp_on", True) or not isinstance(data, pd.DataFrame) or data.empty or not isinstance(analysis, dict):
        return
    try:
        import time as _time
        clock = clock or _time.time
        price = float(data["Close"].iloc[-1])
        sig_text = analysis.get("signal_text", "NO TRADE")
        if st.session_state.get("flip_on", True):
            rows = []
            try:
                j = load_journal() if load_journal else None
                if j is not None and not j.empty and "Status" in j.columns:
                    rows = j[j["Status"].astype(str).str.upper() == "OPEN"].to_dict("records")
            except Exception:
                rows = []
            for a in tpl.flip_alerts(rows, selected, sig_text):
                (st.error if a["level"] == "error" else st.warning)(a["text"])
        age, p0 = tpl.track_signal(st.session_state, f"_tp_seen_{selected}_{timeframe}", sig_text, price, clock())
        side = "BUY" if analysis.get("signal_type") == "buy" else "SELL" if analysis.get("signal_type") == "sell" else None
        limit = float(st.session_state.get("chase_atr", 2.0))
        if side:
            ci = tpl.chase_info(data, side, ext_limit=limit)
            (st.warning if ci["late"] else st.info)(f"{sig_text}: {ci['text']}")
            if age is not None and p0:
                moved = (price - p0) / p0 * 100.0 * (1 if side == "BUY" else -1)
                st.caption(f"This page first saw the {side} signal {age:.0f} s ago at {p0:,.2f}; price has moved {moved:+.2f}% in your favour since then.")
        plan = tpl.build_plan(data, timeframe, resistance=analysis.get("resistance"), support=analysis.get("support"),
                              or_minutes=int(st.session_state.get("tp_or_minutes", 15)), ext_atr=limit)
        with st.expander("🧭 Trade plan: levels prepared before the signal", expanded=True):
            if not plan.get("available"):
                st.caption(plan.get("reason", "Plan not available"))
                return
            atr = plan["atr"]
            if plan.get("low_vol"):
                st.warning(f"ATR is only {atr:,.2f} points ({plan['atr_pct']:.4f}% of price). This usually means the market is closed or the data is flat, "
                           "so the stop and targets below are far too tight to trade. Use this plan only while the market is live.")
            c1, c2 = st.columns(2)
            for col, key, title in ((c1, "buy", "BUY plan"), (c2, "sell", "SELL plan")):
                p = plan[key]
                with col:
                    st.markdown(f"**{title}**: {p['status'].replace('_', ' ').title()}")
                    st.write(tpl.plan_text(p, atr, price))
                    if p.get("note") and p.get("entry") is not None:
                        st.caption(p["note"])
                    if p.get("entry") is not None:
                        st.caption(f"Plan is void if price closes beyond the stop {p['stop']:,.2f}. Targets use the app's own 2R / 3R / 4R.")
            lv = sorted(plan["levels"].items(), key=lambda t: -t[1])
            st.dataframe(pd.DataFrame([{"Level": n, "Price": round(v, 2), "From price (ATR)": round((v - price) / atr, 2)} for n, v in lv]),
                         use_container_width=True, hide_index=True)
            st.caption("A plan is a preparation, not a prediction. Enter only when price reaches the trigger or the zone, and size the trade by the stop shown. "
                       "Levels come from the app's own support/resistance, last 20 candles, previous day, opening range, VWAP and EMA20.")
    except Exception as exc:
        if log_error:
            try:
                log_error("trade_plan_card", exc)
            except Exception:
                pass
