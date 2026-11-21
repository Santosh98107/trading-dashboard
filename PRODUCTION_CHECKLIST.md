# Production checklist - honest status (build 2026-11-21-FINAL)

| # | Item | Status |
|---|------|--------|
| 1 | 4-8 weeks of paper results to prove an edge | NOT code - needs real time. Use the app daily, review Journal/Audit. |
| 2 | Enough resolved signals to trust win-rate stats | NOT code - same as above. |
| 3 | Daily loss limit / max trades / max open positions | ALREADY existed in the app (risk state). Added now: per-symbol cooldown (Settings, default 0 = OFF). |
| 4 | Kill switch | DONE: "Kill switch" expander closes all open paper positions and halts trading (confirm tick needed). |
| 5 | Always-on running + alerts | PARTIAL: Telegram alerts on every worker exit; Docker files + DEPLOY_ALWAYS_ON.md. You must host it. |
| 6 | Better price source | PARTIAL: if a page is open and Angel/Upstox price is live, the worker prefers it (max 30 s old); otherwise Yahoo. Broker calls from background threads were NOT attempted. |
| 7 | Live broker orders | NOT built on purpose: untestable here, real-money risk. order_gateway.py has checks + DryRunBroker only. |
| 8 | Indicator parity (no look-ahead) | DONE for new indicators (advanced_features.py, tested for parity). Scoring OFF by default. |
| 9 | Staged rollout (paper -> tiny live -> scale) | YOUR decision. |
| 10 | SEBI / broker compliance check | YOUR job (algo/API rules, registration). |

Not verified here: live Streamlit run, Yahoo/NSE/broker connectivity, SMALLCAP Yahoo code (^CNXSC), real Telegram delivery.
