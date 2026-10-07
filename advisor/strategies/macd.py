"""MACD Histogram Reversal Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class MacdHistogramReversalStrategy(BaseStrategy):
    """
    Moving Average Convergence Divergence (MACD) histogram zero-cross strategy.

    Measures momentum acceleration by tracking differences between the MACD
    oscillator and its signal line, triggering entries as the histogram traverses zero.
    """

    name: ClassVar[str] = "macd_histogram_reversal"
    description: ClassVar[str] = "MACD histogram zero-crossover reversal"

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute fast/slow EMAs, MACD line, signal line, and histogram zero-crossover signals.

        Goal:
        -----
        Identify early shifts in price momentum before moving averages themselves cross,
        detecting turning points where selling pressure yields to buyers (positive histogram flip)
        or buying momentum dissolves into distribution (negative histogram flip).

        Execution Principle:
        --------------------
        1. Input Copy:
           Creates a local copy `df = data.copy()`.
        2. MACD Construction:
           - Fast EMA: `span=fast_period` (12) on Close.
           - Slow EMA: `span=slow_period` (26) on Close.
           - MACD Line: `EMA_Fast - EMA_Slow`.
           - Signal Line: Exponential moving average of the MACD line with `span=signal_period` (9).
           - MACD Histogram: `MACD - Signal_Line`. Satisfies the mathematical identity
             `Histogram = MACD - Signal_Line` on every bar.
        3. Zero-Crossover Detection:
           - Default signal initialized to `'Hold'`.
           - Bullish histogram flip ('Buy'):
             `MACD_Histogram[i-1] < 0` AND `MACD_Histogram[i] >= 0`
             (Histogram turns from negative to non-negative).
           - Bearish histogram flip ('Sell'):
             `MACD_Histogram[i-1] > 0` AND `MACD_Histogram[i] <= 0`
             (Histogram turns from positive to non-positive).
        4. Output:
           Returns DataFrame containing `['Close', 'MACD', 'Signal_Line', 'MACD_Histogram', 'Signal']`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame with `'Close'` column.

        Returns:
        --------
        pd.DataFrame
            Calculated MACD components and trade signal column.
        """
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
    """
    Functional wrapper for the MACD Histogram Reversal strategy.

    Goal:
    -----
    Preserve functional API parity for external scripts and existing test suites.

    Execution Principle:
    --------------------
    Instantiates `MacdHistogramReversalStrategy` with the specified periods and calls
    `generate_signals(data)`.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV market price series.
    fast_period : int, default 12
        Lookback span for fast EMA.
    slow_period : int, default 26
        Lookback span for slow EMA.
    signal_period : int, default 9
        Lookback span for smoothing MACD into the signal line.

    Returns:
    --------
    pd.DataFrame
        Strategy output with MACD indicators and signals.
    """
    return MacdHistogramReversalStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    ).generate_signals(data)
