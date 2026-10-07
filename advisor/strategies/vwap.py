"""Volume Weighted Average Price (VWAP) Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class VwapStrategy(BaseStrategy):
    """
    Volume Weighted Average Price (VWAP) price crossover strategy.

    Measures average price benchmarked against total traded volume,
    identifying institutional value zones and trend direction changes.
    """

    name: ClassVar[str] = "vwap"
    description: ClassVar[str] = "VWAP price crossover strategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cumulative VWAP and emit price-crossover trade signals.

        Goal:
        -----
        Identify trend regime shifts where buyers push market price sustainably above
        the volume-weighted average price (bullish crossover) or sellers push price below
        fair value (bearish crossover).

        Execution Principle:
        --------------------
        1. Input Copy:
           Creates a local copy `df = data.copy()` to maintain caller immutability.
        2. Typical Price (TP) Calculation:
           `TP = (High + Low + Close) / 3.0`.
        3. Volume-Weighted Cumulation:
           - Per-bar dollar volume: `TP_Volume = TP * Volume`.
           - Cumulative volume: `Cumulative_Volume = Volume.cumsum()`.
           - Cumulative dollar volume: `Cumulative_TP_Volume = TP_Volume.cumsum()`.
        4. VWAP Computation:
           `VWAP = Cumulative_TP_Volume / Cumulative_Volume`.
        5. Crossover Signals:
           - Default signal initialized to `'Hold'`.
           - Upward Cross ('Buy'):
             `Close[i-1] <= VWAP[i-1]` AND `Close[i] > VWAP[i]`
             (Price crosses strictly from at/below VWAP to above VWAP).
           - Downward Cross ('Sell'):
             `Close[i-1] >= VWAP[i-1]` AND `Close[i] < VWAP[i]`
             (Price crosses strictly from at/above VWAP to below VWAP).
        6. Output:
           Returns DataFrame with `['Close', 'Volume', 'VWAP', 'Signal']`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame containing 'High', 'Low', 'Close', and 'Volume'.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing VWAP column and directional signals.
        """
        df = data.copy()

        # Calculate Typical Price (TP)
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3.0

        # Calculate Cumulative TP * Volume
        df['TP_Volume'] = df['TP'] * df['Volume']
        df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()

        # Calculate Cumulative Volume
        df['Cumulative_Volume'] = df['Volume'].cumsum()

        # Calculate VWAP
        df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']

        # Generate Signals
        df['Signal'] = 'Hold'
        # Buy signal: Close price crosses above VWAP
        df.loc[(df['Close'].shift(1) <= df['VWAP'].shift(1)) & (df['Close'] > df['VWAP']), 'Signal'] = 'Buy'
        # Sell signal: Close price crosses below VWAP
        df.loc[(df['Close'].shift(1) >= df['VWAP'].shift(1)) & (df['Close'] < df['VWAP']), 'Signal'] = 'Sell'

        return df[['Close', 'Volume', 'VWAP', 'Signal']]


def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Functional wrapper for VWAP calculation and crossover signal generation.

    Goal:
    -----
    Preserve backward compatibility with functional test suites and calling scripts.

    Execution Principle:
    --------------------
    Instantiates `VwapStrategy` and calls its `generate_signals(data)` method.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV market price series.

    Returns:
    --------
    pd.DataFrame
        Strategy output with VWAP indicators and signals.
    """
    return VwapStrategy().generate_signals(data)
