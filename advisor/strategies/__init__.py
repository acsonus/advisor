"""Trading strategies package implementing the Strategy pattern."""

from advisor.strategies.atr import AtrTrailingStopStrategy, atr_trailing_stop
from advisor.strategies.base import BaseStrategy, SignalType
from advisor.strategies.bollinger import BollingerSqueezeStrategy, bollinger_squeeze_strategy
from advisor.strategies.gap_fill import GapFillStrategy, gap_fill_algorithm
from advisor.strategies.macd import MacdHistogramReversalStrategy, macd_histogram_reversal_strategy
from advisor.strategies.ma_rsi import MaRsiStrategy, ma_rsi_strategy
from advisor.strategies.registry import StrategyRegistry
from advisor.strategies.vwap import VwapStrategy, calculate_vwap

__all__ = [
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
]
