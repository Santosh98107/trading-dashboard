from dataclasses import dataclass


@dataclass
class StrategyConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_long: int = 200
    sma_period: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_period: int = 14
    stoch_signal: int = 3
    adx_period: int = 14
    volume_ma_period: int = 20
    slope_window: int = 5
    volume_spike_mult: float = 1.5
    rvol_threshold: float = 1.2
    adx_threshold: float = 20.0
    bullish_score_threshold: int = 8
    bearish_score_threshold: int = 8
    min_score_gap: int = 2
    risk_reward: float = 2.0
    entry_mode: str = "next_open"
    max_holding_bars: int = 10
    position_size: int = 1
    use_talib: bool = True

    enable_trailing_sl: bool = True
    trailing_sl_mode: str = "atr"
    trailing_atr_mult: float = 1.0
    trailing_percent: float = 1.0
    trailing_fixed_points: float = 10.0

    symbol: str = "NIFTY"
    timeframe: str = "5m"
    initial_capital: float = 100000.0

    strict_trend_filter: bool = True
    use_session_filter: bool = False
    allowed_start_time: str = "09:20"
    allowed_end_time: str = "15:15"
    enable_htf_bias: bool = False
    cooldown_bars: int = 3


APP_USERNAME = "admin"
APP_PASSWORD = "admin123"
