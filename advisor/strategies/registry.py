"""Strategy Registry and Factory."""

from typing import Type
from advisor.strategies.atr import AtrTrailingStopStrategy
from advisor.strategies.base import BaseStrategy
from advisor.strategies.bollinger import BollingerSqueezeStrategy
from advisor.strategies.macd import MacdHistogramReversalStrategy
from advisor.strategies.ma_rsi import MaRsiStrategy
from advisor.strategies.vwap import VwapStrategy


class StrategyRegistry:
    """Registry pattern for strategy discovery and instantiation."""

    _strategies: dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[BaseStrategy]) -> None:
        """Register a strategy class under an identifier."""
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseStrategy:
        """Instantiate a registered strategy by name."""
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {cls.list_available()}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._strategies.keys())


# Register core strategies
StrategyRegistry.register("atr_trailing_stop", AtrTrailingStopStrategy)
StrategyRegistry.register("ma_rsi", MaRsiStrategy)
StrategyRegistry.register("bollinger_squeeze", BollingerSqueezeStrategy)
StrategyRegistry.register("macd_histogram_reversal", MacdHistogramReversalStrategy)
StrategyRegistry.register("vwap", VwapStrategy)
