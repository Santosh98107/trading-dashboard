from fastapi import FastAPI
import pandas as pd

from config import StrategyConfig
from core.live_signal_engine import LiveSignalEngine

app = FastAPI(title="Trading Signal API")

cfg = StrategyConfig()
engine = LiveSignalEngine(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signal")
def generate_signal():
    price_df = pd.read_csv("data/price.csv")
    macro_df = pd.read_csv("data/macro.csv")
    breadth_df = pd.read_csv("data/breadth.csv")
    banks_df = pd.read_csv("data/banks.csv")
    options_df = pd.read_csv("data/options.csv")

    if "Datetime" in price_df.columns:
        price_df["Datetime"] = pd.to_datetime(price_df["Datetime"])
        price_df = price_df.sort_values("Datetime").set_index("Datetime")

    for d in [macro_df, breadth_df, banks_df, options_df]:
        if "Datetime" in d.columns:
            d["Datetime"] = pd.to_datetime(d["Datetime"])
            d.set_index("Datetime", inplace=True)

    result = engine.generate(price_df, macro_df, breadth_df, banks_df, options_df)

    return {
        "recommendation": result["fusion"]["FinalRecommendation"],
        "confidence_pct": result["fusion"]["AdjustedConfidencePct"],
        "win_chance_pct": result["fusion"]["CombinedWinChancePct"],
        "trade_quality": result["fusion"]["TradeQualityGrade"],
        "macro_bias": result["macro_snapshot"]["MacroBias"],
        "breadth_bias": result["breadth_snapshot"]["BreadthBias"],
        "bank_bias": result["bank_snapshot"]["BankBias"],
        "pcr_bias": result["pcr_snapshot"]["PCRBias"],
        "volume_bias": result["volume_snapshot"]["VolumeBias"],
        "regime": result["regime_snapshot"]["Regime"]
    }
