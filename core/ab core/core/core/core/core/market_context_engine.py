import pandas as pd
import numpy as np


class MarketContextEngine:
    def get_breadth_snapshot(self, df):
        if df is None or df.empty:
            return {
                "BreadthScore": 0,
                "BreadthBias": "NEUTRAL",
                "BreadthPct": None,
                "TopCompaniesGreen": None,
                "TopCompaniesRed": None
            }

        latest = df.iloc[-1]
        green = latest.get("NiftyTopCompaniesGreen", np.nan)
        red = latest.get("NiftyTopCompaniesRed", np.nan)
        breadth = latest.get("NiftyBreadthPct", np.nan)

        score = 0

        if pd.notna(breadth):
            if breadth >= 60:
                score += 2
            elif breadth < 40:
                score -= 2

        if pd.notna(green) and pd.notna(red):
            if green > red:
                score += 1
            elif red > green:
                score -= 1

        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"

        return {
            "BreadthScore": score,
            "BreadthBias": bias,
            "BreadthPct": None if pd.isna(breadth) else float(breadth),
            "TopCompaniesGreen": None if pd.isna(green) else int(green),
            "TopCompaniesRed": None if pd.isna(red) else int(red)
        }

    def get_bank_snapshot(self, df):
        if df is None or df.empty:
            return {
                "BankScore": 0,
                "BankBias": "NEUTRAL",
                "PositiveBanks": 0,
                "NegativeBanks": 0
            }

        latest = df.iloc[-1]
        keys = ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]

        pos = 0
        neg = 0

        for k in keys:
            v = latest.get(k, np.nan)
            if pd.notna(v):
                if v > 0:
                    pos += 1
                elif v < 0:
                    neg += 1

        score = 0
        if pos >= 4:
            score += 2
        elif pos >= 3:
            score += 1

        if neg >= 4:
            score -= 2
        elif neg >= 3:
            score -= 1

        bias = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"

        return {
            "BankScore": score,
            "BankBias": bias,
            "PositiveBanks": pos,
            "NegativeBanks": neg
        }
