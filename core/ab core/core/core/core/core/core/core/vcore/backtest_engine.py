import numpy as np
import pandas as pd


class BacktestEngine:
    def __init__(self, config):
        self.cfg = config

    def run(self, df):
        df = df.copy()

        df["StrategyEquity"] = np.nan
        equity = self.cfg.initial_capital
        trades = []

        in_position = False
        direction = None
        entry_price = None
        stop_loss = None
        target = None
        entry_bar = None

        for i in range(len(df)):
            row = df.iloc[i]

            if not in_position:
                if bool(row.get("ExecuteBuyNextBar", False)):
                    atr = row.get("ATR", np.nan)
                    price = row["Open"] if self.cfg.entry_mode == "next_open" else row["Close"]

                    if pd.notna(atr) and pd.notna(price):
                        in_position = True
                        direction = "LONG"
                        entry_price = price
                        stop_loss = price - atr
                        target = price + self.cfg.risk_reward * atr
                        entry_bar = i

                elif bool(row.get("ExecuteSellNextBar", False)):
                    atr = row.get("ATR", np.nan)
                    price = row["Open"] if self.cfg.entry_mode == "next_open" else row["Close"]

                    if pd.notna(atr) and pd.notna(price):
                        in_position = True
                        direction = "SHORT"
                        entry_price = price
                        stop_loss = price + atr
                        target = price - self.cfg.risk_reward * atr
                        entry_bar = i

            else:
                high = row["High"]
                low = row["Low"]
                close = row["Close"]
                exit_price = None
                exit_reason = None

                if direction == "LONG":
                    if low <= stop_loss and high >= target:
                        exit_price = stop_loss
                        exit_reason = "SL_and_Target_same_bar_SL_priority"
                    elif low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "StopLoss"
                    elif high >= target:
                        exit_price = target
                        exit_reason = "Target"
                    elif (i - entry_bar) >= self.cfg.max_holding_bars:
                        exit_price = close
                        exit_reason = "TimeExit"

                    if exit_price is not None:
                        pnl = (exit_price - entry_price) * self.cfg.position_size
                        ret = ((exit_price - entry_price) / entry_price) * 100

                else:
                    if high >= stop_loss and low <= target:
                        exit_price = stop_loss
                        exit_reason = "SL_and_Target_same_bar_SL_priority"
                    elif high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "StopLoss"
                    elif low <= target:
                        exit_price = target
                        exit_reason = "Target"
                    elif (i - entry_bar) >= self.cfg.max_holding_bars:
                        exit_price = close
                        exit_reason = "TimeExit"

                    if exit_price is not None:
                        pnl = (entry_price - exit_price) * self.cfg.position_size
                        ret = ((entry_price - exit_price) / entry_price) * 100

                if exit_price is not None:
                    equity += pnl
                    trades.append({
                        "EntryBar": entry_bar,
                        "ExitBar": i,
                        "Direction": direction,
                        "EntryPrice": entry_price,
                        "ExitPrice": exit_price,
                        "PnL": pnl,
                        "ReturnPct": ret,
                        "ExitReason": exit_reason
                    })

                    in_position = False
                    direction = None
                    entry_price = None
                    stop_loss = None
                    target = None
                    entry_bar = None

            df.at[df.index[i], "StrategyEquity"] = equity

        trades_df = pd.DataFrame(trades)
        return df, trades_df, self.summarize(trades_df, equity)

    def summarize(self, trades_df, final_equity):
        if trades_df.empty:
            return {
                "TotalTrades": 0,
                "WinRatePct": 0,
                "LossRatePct": 0,
                "FinalEquity": self.cfg.initial_capital
            }

        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]

        return {
            "TotalTrades": len(trades_df),
            "WinRatePct": round((len(wins) / len(trades_df)) * 100, 2),
            "LossRatePct": round((len(losses) / len(trades_df)) * 100, 2),
            "FinalEquity": round(final_equity, 2),
            "TotalPnL": round(trades_df["PnL"].sum(), 2),
            "AvgPnL": round(trades_df["PnL"].mean(), 2)
        }
