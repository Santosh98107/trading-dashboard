from core.feature_engine import FeatureEngine
from core.recommendation_engine import RecommendationEngine
from core.probability_engine import HistoricalProbabilityEngine
from core.macro_engine import MacroEngine
from core.market_context_engine import MarketContextEngine
from core.options_sentiment_engine import OptionsSentimentEngine
from core.volume_engine import VolumeDecisionEngine
from core.regime_engine import RegimeEngine
from core.signal_fusion_engine import SignalFusionEngine


class LiveSignalEngine:
    def __init__(self, config):
        self.cfg = config
        self.feature_engine = FeatureEngine(config)
        self.recommendation_engine = RecommendationEngine(config)
        self.probability_engine = HistoricalProbabilityEngine(config)
        self.macro_engine = MacroEngine()
        self.market_engine = MarketContextEngine()
        self.options_engine = OptionsSentimentEngine()
        self.volume_engine = VolumeDecisionEngine(config)
        self.regime_engine = RegimeEngine()
        self.signal_fusion_engine = SignalFusionEngine(config)

    def generate(self, price_df, macro_df=None, breadth_df=None, banks_df=None, options_df=None):
        features_df = self.feature_engine.transform(price_df)
        latest = features_df.iloc[-1]

        tech_rec = self.recommendation_engine.get_recommendation(latest)
        hist_prob = self.probability_engine.estimate_from_history(features_df, latest)

        macro_snapshot = (
            self.macro_engine.get_latest_macro_snapshot(macro_df)
            if macro_df is not None
            else {
                "MacroScore": 0,
                "MacroBias": "NEUTRAL",
                "CrudeOil": None,
                "BrentCrude": None,
                "USDINR": None
            }
        )

        breadth_snapshot = (
            self.market_engine.get_breadth_snapshot(breadth_df)
            if breadth_df is not None
            else {
                "BreadthScore": 0,
                "BreadthBias": "NEUTRAL",
                "BreadthPct": None,
                "TopCompaniesGreen": None,
                "TopCompaniesRed": None
            }
        )

        bank_snapshot = (
            self.market_engine.get_bank_snapshot(banks_df)
            if banks_df is not None
            else {
                "BankScore": 0,
                "BankBias": "NEUTRAL",
                "PositiveBanks": 0,
                "NegativeBanks": 0
            }
        )

        pcr_snapshot = (
            self.options_engine.calculate_pcr_metrics(options_df)
            if options_df is not None
            else {
                "PCRScore": 0,
                "PCRBias": "NEUTRAL",
                "PCR_OI": None,
                "PCR_Volume": None
            }
        )

        volume_snapshot = self.volume_engine.get_volume_snapshot(features_df)
        regime_snapshot = self.regime_engine.get_regime_snapshot(features_df)

        fusion = self.signal_fusion_engine.fuse(
            tech_rec,
            hist_prob,
            macro_snapshot,
            breadth_snapshot,
            bank_snapshot,
            pcr_snapshot,
            volume_snapshot,
            regime_snapshot
        )

        return {
            "latest_row": latest,
            "features_df": features_df,
            "tech_rec": tech_rec,
            "hist_prob": hist_prob,
            "macro_snapshot": macro_snapshot,
            "breadth_snapshot": breadth_snapshot,
            "bank_snapshot": bank_snapshot,
            "pcr_snapshot": pcr_snapshot,
            "volume_snapshot": volume_snapshot,
            "regime_snapshot": regime_snapshot,
            "fusion": fusion
        }
