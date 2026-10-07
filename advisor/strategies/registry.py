"""Strategy Registry and Factory."""

from typing import Type
from advisor.strategies.atr import AtrTrailingStopStrategy
from advisor.strategies.base import BaseStrategy
from advisor.strategies.bollinger import BollingerSqueezeStrategy
from advisor.strategies.macd import MacdHistogramReversalStrategy
from advisor.strategies.ma_rsi import MaRsiStrategy
from advisor.strategies.vwap import VwapStrategy


class StrategyRegistry:
    """Registry pattern for strategy discovery, registration, and dynamic instantiation."""

    _strategies: dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[BaseStrategy]) -> None:
        """
        Register a strategy class under a unique identifier.

        Goal:
        -----
        Provide an open-for-extension mechanism where new strategies can be dynamically
        registered without modifying core simulator or API routing code.

        Execution Principle:
        --------------------
        Stores the strategy class type `strategy_cls` into the internal class-level
        dictionary `_strategies` keyed by string `name`.

        Parameters:
        -----------
        name : str
            Unique strategy identifier (e.g. 'atr_trailing_stop').
        strategy_cls : Type[BaseStrategy]
            Subclass of `BaseStrategy`.
        """
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseStrategy:
        """
        Instantiate a registered strategy by identifier with custom hyperparameters.

        Goal:
        -----
        Act as a factory method, decoupling callers from concrete strategy constructors.

        Execution Principle:
        --------------------
        1. Check if `name` is present in `_strategies`. If not, raise `KeyError` listing
           all currently registered strategy names.
        2. Retrieve the class object from `_strategies[name]`.
        3. Instantiate the class passing any keyword arguments (`**kwargs`) to the constructor.
        4. Return the instantiated `BaseStrategy` object.

        Parameters:
        -----------
        name : str
            Identifier of the registered strategy.
        **kwargs : Any
            Hyperparameters forwarded to the strategy class constructor.

        Returns:
        --------
        BaseStrategy
            Instantiated strategy instance.

        Raises:
        -------
        KeyError
            If `name` is not registered.
        """
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {cls.list_available()}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all registered strategy identifiers alphabetically.

        Goal:
        -----
        Provide introspection for API endpoints and CLI runners to query supported strategies.

        Execution Principle:
        --------------------
        Extracts keys from `_strategies` dictionary and returns them sorted in lexicographical order.

        Returns:
        --------
        list[str]
            Sorted list of registered strategy names.
        """
        return sorted(cls._strategies.keys())


# Register core strategies
StrategyRegistry.register("atr_trailing_stop", AtrTrailingStopStrategy)
StrategyRegistry.register("ma_rsi", MaRsiStrategy)
StrategyRegistry.register("bollinger_squeeze", BollingerSqueezeStrategy)
StrategyRegistry.register("macd_histogram_reversal", MacdHistogramReversalStrategy)
StrategyRegistry.register("vwap", VwapStrategy)
