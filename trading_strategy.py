"""
trading_strategy.py — Backward-compatibility facade.

This module re-exports components from the modular `advisor` package.
All functionality is now structured in `advisor.data`, `advisor.strategies`,
`advisor.sentiment`, and `advisor.backtesting`.
"""

# Re-export data ingestion, normalization, and constants
from advisor.data import (
    INTERVAL_MAX_DAYS,
    PERIOD_DAYS,
    VALID_INTERVALS,
    VALID_PERIODS,
    _INTERVAL_MAX_DAYS,
    _PERIOD_DAYS,
    download_data,
    downloadData,
    to_naive_s,
    validate_period_and_interval,
    validate_ticker,
)

# Re-export sentiment signal mapping
from advisor.sentiment import (
    news_sentiment_signal,
)

# Re-export strategies and OOP strategy classes
from advisor.strategies import (
    AtrTrailingStopStrategy,
    BaseStrategy,
    BollingerSqueezeStrategy,
    GapFillStrategy,
    MacdHistogramReversalStrategy,
    MaRsiStrategy,
    SignalType,
    StrategyRegistry,
    VwapStrategy,
    atr_trailing_stop,
    bollinger_squeeze_strategy,
    calculate_vwap,
    gap_fill_algorithm,
    macd_histogram_reversal_strategy,
    ma_rsi_strategy,
)

# Re-export backtesting engine and models
from advisor.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    backtest_strategy,
    calculate_metrics,
)

__all__ = [
    # Data & Normalization
    "VALID_PERIODS",
    "VALID_INTERVALS",
    "INTERVAL_MAX_DAYS",
    "PERIOD_DAYS",
    "_INTERVAL_MAX_DAYS",
    "_PERIOD_DAYS",
    "to_naive_s",
    "download_data",
    "downloadData",
    "validate_period_and_interval",
    "validate_ticker",
    # Sentiment
    "news_sentiment_signal",
    # Strategies
    "BaseStrategy",
    "SignalType",
    "StrategyRegistry",
    "AtrTrailingStopStrategy",
    "atr_trailing_stop",
    "MaRsiStrategy",
    "ma_rsi_strategy",
    "BollingerSqueezeStrategy",
    "bollinger_squeeze_strategy",
    "MacdHistogramReversalStrategy",
    "macd_histogram_reversal_strategy",
    "VwapStrategy",
    "calculate_vwap",
    "GapFillStrategy",
    "gap_fill_algorithm",
    # Backtesting
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "BacktestEngine",
    "calculate_metrics",
    "backtest_strategy",
]
