"""Moving Average + Relative Strength Index (MA-RSI) Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class MaRsiStrategy(BaseStrategy):
    """
    Dual EMA crossover with RSI momentum filter strategy.

    Combines trend-following exponential moving average (EMA) golden/death crosses
    with a Relative Strength Index (RSI) momentum gate to eliminate exhaustion entries.
    """

    name: ClassVar[str] = "ma_rsi"
    description: ClassVar[str] = "EMA crossover with RSI momentum filter"

    short_ema_period: int = 12
    long_ema_period: int = 26
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dual EMAs and Wilder-smoothed RSI to generate trend-following entries.

        Goal:
        -----
        Identify valid trend change crossovers (Short EMA crossing Long EMA) while preventing
        buying at market tops (overbought RSI) or selling at market bottoms (oversold RSI).

        Execution Principle:
        --------------------
        1. Input Copy:
           Creates a shallow copy `df = data.copy()` to maintain caller immutability.
        2. EMA Calculation:
           Computes Short EMA (`span=short_ema_period`, default 12) and Long EMA
           (`span=long_ema_period`, default 26) on the `'Close'` price series using
           `ewm(adjust=False).mean()`.
        3. RSI Calculation with J. Welles Wilder Smoothing:
           - Computes 1-bar price delta: `delta = Close.diff(1)`.
           - Separates positive gain (`gain = delta.where(delta > 0, 0)`) and
             negative loss (`loss = -delta.where(delta < 0, 0)`).
           - Smoothes gain and loss using Wilder's alpha = `1 / rsi_period`, which corresponds
             to pandas center of mass `com = rsi_period - 1`.
             (Standard span=rsi_period uses alpha=2/(n+1) which produces a non-standard fast RSI).
           - Relative Strength `rs = avg_gain / avg_loss`.
           - `RSI = 100 - (100 / (1 + rs))`.
        4. Signal Logic:
           - Default signal initialized to `'Hold'`.
           - Bullish crossover ('Buy'):
             `Short_EMA[i-1] < Long_EMA[i-1]` AND `Short_EMA[i] > Long_EMA[i]`
             AND `RSI[i] < rsi_overbought` (room left to run before becoming overbought).
           - Bearish crossover ('Sell'):
             `Short_EMA[i-1] > Long_EMA[i-1]` AND `Short_EMA[i] < Long_EMA[i]`
             AND `RSI[i] > rsi_oversold` (room left to fall before becoming oversold).
        5. Returns DataFrame with `['Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal']`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame with `'Close'` column.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing calculated indicators and signal column.
        """
        df = data.copy()

        # Calculate EMAs
        df['Short_EMA'] = df['Close'].ewm(span=self.short_ema_period, adjust=False).mean()
        df['Long_EMA'] = df['Close'].ewm(span=self.long_ema_period, adjust=False).mean()

        # Calculate RSI
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's smoothing: alpha = 1/rsi_period, equivalent to com = rsi_period - 1.
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Generate Signals
        df['Signal'] = 'Hold'

        # Buy signal: Short EMA crosses above Long EMA AND RSI < rsi_overbought
        df.loc[
            (df['Short_EMA'].shift(1) < df['Long_EMA'].shift(1))
            & (df['Short_EMA'] > df['Long_EMA'])
            & (df['RSI'] < self.rsi_overbought),
            'Signal',
        ] = 'Buy'

        # Sell signal: Short EMA crosses below Long EMA AND RSI > rsi_oversold
        df.loc[
            (df['Short_EMA'].shift(1) > df['Long_EMA'].shift(1))
            & (df['Short_EMA'] < df['Long_EMA'])
            & (df['RSI'] > self.rsi_oversold),
            'Signal',
        ] = 'Sell'

        return df[['Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal']]


def ma_rsi_strategy(
    data: pd.DataFrame,
    short_ema_period: int = 12,
    long_ema_period: int = 26,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
) -> pd.DataFrame:
    """
    Functional wrapper for the MA-RSI strategy.

    Goal:
    -----
    Maintain full backward compatibility with legacy scripts and existing test suites.

    Execution Principle:
    --------------------
    Instantiates `MaRsiStrategy` with the specified periods and thresholds and delegates
    to `generate_signals(data)`.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV price bars.
    short_ema_period : int, default 12
        Lookback span for fast EMA.
    long_ema_period : int, default 26
        Lookback span for slow EMA.
    rsi_period : int, default 14
        Lookback period for Wilder's smoothed RSI.
    rsi_oversold : float, default 30.0
        Lower gate for RSI momentum.
    rsi_overbought : float, default 70.0
        Upper gate for RSI momentum.

    Returns:
    --------
    pd.DataFrame
        Strategy output DataFrame with indicators and signals.
    """
    return MaRsiStrategy(
        short_ema_period=short_ema_period,
        long_ema_period=long_ema_period,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
    ).generate_signals(data)
