import pandas as pd
import numpy as np


class RegimeEngine:
    def get_regime_snapshot(self, features_df):
        if features_df is None or features_df.empty:
            return {
                "Regime": "UNKNOWN",
                "RegimeScore": 0
            }

        latest = features_df.iloc[-1]

        adx = latest.get("ADX", np.nan)
        bb_width = latest.get("BB_Width", np.nan)
        atr_pct = latest.get("ATR_Pct", np.nan)

        score = 0

        if pd.notna(adx):
            if adx >= 25:
                score += 2
            elif adx < 15:
                score -= 1

        if pd.notna(bb_width) and pd.notna(atr_pct):
            if atr_pct > 1.5:
                score += 1

        if score >= 2:
            regime = "TRENDING"
        elif score <= -1:
            regime = "SIDEWAYS"
        else:
            regime = "NEUTRAL"

        return {
            "Regime": regime,
            "RegimeScore": score
        }
