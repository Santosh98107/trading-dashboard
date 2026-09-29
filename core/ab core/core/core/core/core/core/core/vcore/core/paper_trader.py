import numpy as np
import pandas as pd


class PaperTrader:
    def __init__(self, config):
        self.cfg = config
        self.reset()

    def reset(self):
        self.position_open = False
        self.direction = None
        self.entry_price = np.nan
        self.stop_loss = np.nan
        self.initial_stop_loss = np.nan
        self.target_price = np.nan
        self.trailing_stop_loss = np.nan
        self.entry_time = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_log = []
        self.highest_price_since_entry = np.nan
        self.lowest_price_since_entry = np.nan

    def _calculate_trailing_sl(self, row):
        if not self.cfg.enable_trailing_sl or not self.position_open:
            return self.stop_loss

        atr = row.get("ATR", np.nan)

        if self.direction == "LONG":
            if self.cfg.trailing_sl_mode == "atr" and pd.notna(atr):
                return max(self.stop_loss, self.highest_price_since_entry - self.cfg.trailing_atr_mult * atr)
            if self.cfg.trailing_sl_mode == "percent":
                return max(self.stop_loss, self.highest_price_since_entry * (1 - self.cfg.trailing_percent / 100))
            if self.cfg.trailing_sl_mode == "fixed":
                return max(self.stop_loss, self.highest_price_since_entry - self.cfg.trailing_fixed_points)

        if self.direction == "SHORT":
            if self.cfg.trailing_sl_mode == "atr" and pd.notna(atr):
                return min(self.stop_loss, self.lowest_price_since_entry + self.cfg.trailing_atr_mult * atr)
            if self.cfg.trailing_sl_mode == "percent":
                return min(self.stop_loss, self.lowest_price_since_entry * (1 + self.cfg.trailing_percent / 100))
            if self.cfg.trailing_sl_mode == "fixed":
                return min(self.stop_loss, self.lowest_price_since_entry + self.cfg.trailing_fixed_points)

        return self.stop_loss

    def on_new_closed_candle(self, df):
        row = df.iloc[-1]
        idx = df.index[-1]

        if not self.position_open:
            if bool(row.get("ExecuteBuyNextBar", False)):
                entry = row["Open"] if self.cfg.entry_mode == "next_open" else row["Close"]
                atr = row.get("ATR", np.nan)

                if pd.notna(entry) and pd.notna(atr):
                    self.position_open = True
                    self.direction = "LONG"
                    self.entry_price = float(entry)
                    self.stop_loss = float(entry - atr)
                    self.initial_stop_loss = self.stop_loss
                    self.target_price = float(entry + self.cfg.risk_reward * atr)
                    self.trailing_stop_loss = self.stop_loss
                    self.entry_time = idx
                    self.highest_price_since_entry = row["High"]
                    self.lowest_price_since_entry = row["Low"]
                    return {"action": "OPEN_LONG", "message": f"LONG opened @ {entry:.2f}"}

            if bool(row.get("ExecuteSellNextBar", False)):
                entry = row["Open"] if self.cfg.entry_mode == "next_open" else row["Close"]
                atr = row.get("ATR", np.nan)

                if pd.notna(entry) and pd.notna(atr):
                    self.position_open = True
                    self.direction = "SHORT"
                    self.entry_price = float(entry)
                    self.stop_loss = float(entry + atr)
                    self.initial_stop_loss = self.stop_loss
                    self.target_price = float(entry - self.cfg.risk_reward * atr)
                    self.trailing_stop_loss = self.stop_loss
                    self.entry_time = idx
                    self.highest_price_since_entry = row["High"]
                    self.lowest_price_since_entry = row["Low"]
                    return {"action": "OPEN_SHORT", "message": f"SHORT opened @ {entry:.2f}"}

            return {"action": "NONE", "message": ""}

        high = row["High"]
        low = row["Low"]
        close = row["Close"]

        self.highest_price_since_entry = max(self.highest_price_since_entry, high)
        self.lowest_price_since_entry = min(self.lowest_price_since_entry, low)
        self.stop_loss = self._calculate_trailing_sl(row)
        self.trailing_stop_loss = self.stop_loss

        if self.direction == "LONG":
            self.unrealized_pnl = (close - self.entry_price) * self.cfg.position_size

            if low <= self.stop_loss:
                exit_price = self.stop_loss
                pnl = (exit_price - self.entry_price) * self.cfg.position_size
                reason = "TrailingStopLoss" if self.stop_loss > self.initial_stop_loss else "StopLoss"
            elif high >= self.target_price:
                exit_price = self.target_price
                pnl = (exit_price - self.entry_price) * self.cfg.position_size
                reason = "Target"
            else:
                return {"action": "HOLD", "message": "LONG active"}

        else:
            self.unrealized_pnl = (self.entry_price - close) * self.cfg.position_size

            if high >= self.stop_loss:
                exit_price = self.stop_loss
                pnl = (self.entry_price - exit_price) * self.cfg.position_size
                reason = "TrailingStopLoss" if self.stop_loss < self.initial_stop_loss else "StopLoss"
            elif low <= self.target_price:
                exit_price = self.target_price
                pnl = (self.entry_price - exit_price) * self.cfg.position_size
                reason = "Target"
            else:
                return {"action": "HOLD", "message": "SHORT active"}

        self.realized_pnl += pnl
        self.trade_log.append({
            "EntryTime": self.entry_time,
            "ExitTime": idx,
            "Direction": self.direction,
            "EntryPrice": self.entry_price,
            "InitialStopLoss": self.initial_stop_loss,
            "FinalStopLoss": self.stop_loss,
            "TargetPrice": self.target_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "ExitReason": reason
        })

        self.position_open = False
        self.direction = None
        self.entry_price = np.nan
        self.stop_loss = np.nan
        self.initial_stop_loss = np.nan
        self.target_price = np.nan
        self.trailing_stop_loss = np.nan
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.highest_price_since_entry = np.nan
        self.lowest_price_since_entry = np.nan

        return {"action": "CLOSE", "message": f"Trade closed | {reason}"}

    def get_snapshot(self):
        return {
            "PositionOpen": self.position_open,
            "Direction": self.direction,
            "EntryPrice": None if pd.isna(self.entry_price) else float(self.entry_price),
            "InitialStopLoss": None if pd.isna(self.initial_stop_loss) else float(self.initial_stop_loss),
            "CurrentStopLoss": None if pd.isna(self.stop_loss) else float(self.stop_loss),
            "TargetPrice": None if pd.isna(self.target_price) else float(self.target_price),
            "TrailingStopLoss": None if pd.isna(self.trailing_stop_loss) else float(self.trailing_stop_loss),
            "RealizedPnL": float(self.realized_pnl),
            "UnrealizedPnL": float(self.unrealized_pnl),
        }

    def get_trade_log_df(self):
        return pd.DataFrame(self.trade_log)
