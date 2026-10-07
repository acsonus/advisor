"""Base strategy class and signal definitions."""

from abc import ABC, abstractmethod
from typing import ClassVar
import pandas as pd


class SignalType:
    """Canonical strategy signal constants."""
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class BaseStrategy(ABC):
    """
    Abstract base class establishing the contract for all trading strategies.

    Follows the Strategy design pattern, decoupling indicator calculation
    and signal generation logic from the backtesting and API consumers.
    """

    name: ClassVar[str] = "base_strategy"
    description: ClassVar[str] = "Base Trading Strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute strategy logic and generate directional trading signals on OHLCV data.

        Goal:
        -----
        Process raw time-series market price bars, compute strategy-specific technical indicators,
        and emit unambiguous directional signals ('Buy', 'Sell', 'Hold') per bar without mutating
        the caller's input DataFrame.

        Execution Principle:
        --------------------
        1. Subclasses must override this method.
        2. Create a clean internal copy of `data` to guarantee no input mutation.
        3. Compute indicators (e.g., EMAs, ATR, Bollinger Bands, MACD, VWAP).
        4. Apply entry/exit decision logic on historical bars up to the current bar,
           ensuring no look-ahead bias (using shifted signals where required).
        5. Populate a `'Signal'` column with `'Buy'`, `'Sell'`, or `'Hold'`.
        6. Return a DataFrame containing relevant indicator columns and `'Signal'`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame indexed by timestamp with columns `['Open', 'High', 'Low', 'Close', 'Volume']`.

        Returns:
        --------
        pd.DataFrame
            DataFrame with indicator columns and `'Signal'` column ('Buy', 'Sell', 'Hold').
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Provide callable syntax interface for strategy instances.

        Goal:
        -----
        Enable strategy instances to be invoked cleanly like functions: `strategy(df)`.

        Execution Principle:
        --------------------
        Forwards the input `data` argument directly to `self.generate_signals(data)`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV market price bars.

        Returns:
        --------
        pd.DataFrame
            Result of `generate_signals(data)`.
        """
        return self.generate_signals(data)
