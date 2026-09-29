import numpy as np
import pandas as pd


class RecommendationEngine:
    def __init__(self, config):
        self.cfg = config

    def get_recommendation(self, latest_row):
        bull = latest_row.get("BullishConfluenceScore", 0)
        bear = latest_row.get("BearishConfluenceScore", 0)
        adx = latest_row.get("ADX", np.nan)

        recommendation = "HOLD"
        confidence = 50.0
        signal_type = "NEUTRAL"

        if bull >= self.cfg.bullish_score_threshold and bull > bear:
            signal_type = "BUY"
            recommendation = "STRONG BUY" if bull >= self.cfg.bullish_score_threshold + 2 else "BUY"
            confidence = min(95, 50 + bull * 3)

            if pd.notna(adx) and adx > 25:
                confidence += 5

        elif bear >= self.cfg.bearish_score_threshold and bear > bull:
            signal_type = "SELL"
            recommendation = "STRONG SELL" if bear >= self.cfg.bearish_score_threshold + 2 else "SELL"
            confidence = min(95, 50 + bear * 3)

            if pd.notna(adx) and adx > 25:
                confidence += 5

        return {
            "Recommendation": recommendation,
            "SignalType": signal_type,
            "ConfidencePct": round(min(confidence, 99), 2)
        }
