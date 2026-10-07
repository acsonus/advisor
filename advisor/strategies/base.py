"""Base strategy class and signal definitions."""

from abc import ABC, abstractmethod
from typing import ClassVar
import pandas as pd


class SignalType:
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses implement `generate_signals` which accepts an OHLCV DataFrame
    and returns a DataFrame containing indicators and a 'Signal' column.
    """

    name: ClassVar[str] = "base_strategy"
    description: ClassVar[str] = "Base Trading Strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute strategy logic on OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data indexed by timestamp/date.

        Returns
        -------
        pd.DataFrame
            DataFrame with indicator columns and 'Signal' column ('Buy', 'Sell', 'Hold').
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Allow calling instance directly as a callable."""
        return self.generate_signals(data)
