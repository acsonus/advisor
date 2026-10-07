"""MACD Histogram Reversal Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class MacdHistogramReversalStrategy(BaseStrategy):
    """Moving Average Convergence Divergence (MACD) histogram zero-cross strategy."""

    name: ClassVar[str] = "macd_histogram_reversal"
    description: ClassVar[str] = "MACD histogram zero-crossover reversal"

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate the MACD line
        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']

        # Calculate the Signal line
        df['Signal_Line'] = df['MACD'].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate the MACD Histogram
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        # Generate Signals
        df['Signal'] = 'Hold'

        # Buy signal: Histogram turns positive from negative
        df.loc[(df['MACD_Histogram'].shift(1) < 0) & (df['MACD_Histogram'] >= 0), 'Signal'] = 'Buy'

        # Sell signal: Histogram turns negative from positive
        df.loc[(df['MACD_Histogram'].shift(1) > 0) & (df['MACD_Histogram'] <= 0), 'Signal'] = 'Sell'

        return df[['Close', 'MACD', 'Signal_Line', 'MACD_Histogram', 'Signal']]


def macd_histogram_reversal_strategy(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Backward-compatible function wrapper for MACD Histogram Reversal strategy."""
    return MacdHistogramReversalStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    ).generate_signals(data)
