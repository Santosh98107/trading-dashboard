import pandas as pd
import numpy as np


class MacroEngine:
    def get_latest_macro_snapshot(self, df):
        if df is None or df.empty:
            return {
                "CrudeOil": None,
                "BrentCrude": None,
                "USDINR": None,
                "MacroScore": 0,
                "MacroBias": "NEUTRAL"
            }

        latest = df.iloc[-1]
        crude = latest.get("CrudeOil", np.nan)
        brent = latest.get("BrentCrude", np.nan)
        usdinr = latest.get("USDINR", np.nan)

        score = 0

        if len(df) > 1:
            prev = df.iloc[-2]

            if pd.notna(crude) and pd.notna(prev.get("CrudeOil", np.nan)):
                score += 1 if crude < prev["CrudeOil"] else -1

            if pd.notna(brent) and pd.notna(prev.get("BrentCrude", np.nan)):
                score += 1 if brent < prev["BrentCrude"] else -1

            if pd.notna(usdinr) and pd.notna(prev.get("USDINR", np.nan)):
                score += 1 if usdinr < prev["USDINR"] else -1

        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"

        return {
            "CrudeOil": None if pd.isna(crude) else float(crude),
            "BrentCrude": None if pd.isna(brent) else float(brent),
            "USDINR": None if pd.isna(usdinr) else float(usdinr),
            "MacroScore": score,
            "MacroBias": bias
        }
