import pandas as pd
import numpy as np


class OptionsSentimentEngine:
    def calculate_pcr_metrics(self, df):
        if df is None or df.empty:
            return {
                "PCR_OI": None,
                "PCR_Volume": None,
                "PCRBias": "NEUTRAL",
                "PCRScore": 0,
                "PutOI": None,
                "CallOI": None,
                "PutVolume": None,
                "CallVolume": None,
            }

        latest = df.iloc[-1]

        put_oi = latest.get("TotalPutOI", np.nan)
        call_oi = latest.get("TotalCallOI", np.nan)
        put_vol = latest.get("TotalPutVolume", np.nan)
        call_vol = latest.get("TotalCallVolume", np.nan)

        pcr_oi = put_oi / call_oi if pd.notna(put_oi) and pd.notna(call_oi) and call_oi != 0 else np.nan
        pcr_vol = put_vol / call_vol if pd.notna(put_vol) and pd.notna(call_vol) and call_vol != 0 else np.nan

        score = 0

        if pd.notna(pcr_oi):
            if pcr_oi > 1.2:
                score += 2
            elif pcr_oi < 0.8:
                score -= 2

        if pd.notna(pcr_vol):
            if pcr_vol > 1.2:
                score += 1
            elif pcr_vol < 0.8:
                score -= 1

        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"

        return {
            "PCR_OI": None if pd.isna(pcr_oi) else round(float(pcr_oi), 4),
            "PCR_Volume": None if pd.isna(pcr_vol) else round(float(pcr_vol), 4),
            "PCRBias": bias,
            "PCRScore": score,
            "PutOI": None if pd.isna(put_oi) else float(put_oi),
            "CallOI": None if pd.isna(call_oi) else float(call_oi),
            "PutVolume": None if pd.isna(put_vol) else float(put_vol),
            "CallVolume": None if pd.isna(call_vol) else float(call_vol),
        }
