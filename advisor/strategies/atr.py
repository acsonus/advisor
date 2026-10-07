"""ATR Trailing Stop Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class AtrTrailingStopStrategy(BaseStrategy):
    """
    Average True Range (ATR) dynamic trailing stop strategy.

    Uses market volatility (ATR) to size adaptive trailing stops that follow
    trending price moves while locking in gains and managing drawdown.
    """

    name: ClassVar[str] = "atr_trailing_stop"
    description: ClassVar[str] = "Average True Range (ATR) dynamic trailing stop"

    atr_period: int = 14
    atr_multiplier: float = 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Average True Range and simulate dynamic volatility-based trailing stop exits.

        Goal:
        -----
        Identify volatility-expanded directional breakouts and protect long/short positions
        using adaptive trailing stop levels calculated from multiples of the ATR.

        Execution Principle:
        --------------------
        1. Copy Input:
           Creates a shallow copy `df = data.copy()` to preserve original caller data.
        2. Compute True Range (TR):
           Evaluates the maximum of three components across each bar:
           - High - Low (`H-L`)
           - Absolute value of High - Previous Close (`|H - Close[i-1]|`)
           - Absolute value of Low - Previous Close (`|L - Close[i-1]|`)
           `TR = max(H-L, H-PC, L-PC)`.
        3. Compute Average True Range (ATR):
           Applies exponential weighted moving average (`ewm`) with `span=atr_period` on TR.
        4. State Tracking Loop:
           Iterates through bars sequentially tracking active position (`long_position`, `short_position`):
           - In Flat state:
             * Triggers Buy if `Close[i] > Close[i-1]` and price surge exceeds `(ATR * atr_multiplier / 2)`.
               Initializes `Buy_Stop = Close - (ATR * atr_multiplier)`.
             * Triggers Sell if `Close[i] < Close[i-1]` and price drop exceeds `(ATR * atr_multiplier / 2)`.
               Initializes `Sell_Stop = Close + (ATR * atr_multiplier)`.
           - In Long position:
             * Ratchets `Buy_Stop` upward: `new_stop = Close - (ATR * multiplier)`.
               Guards against NaN on entry, ensuring stop only moves higher: `max(prev_stop, new_stop)`.
             * If `Close[i] < Buy_Stop[i]`: Stop hit -> exit long position with a `'Sell'` signal.
           - In Short position:
             * Ratchets `Sell_Stop` downward: `min(prev_stop, new_stop)`.
             * If `Close[i] > Sell_Stop[i]`: Stop hit -> exit short position with a `'Buy'` signal.
        5. Output:
           Returns DataFrame with columns `['Close', 'ATR', 'Buy_Stop', 'Sell_Stop', 'Signal']`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
        --------
        pd.DataFrame
            Indicator columns and generated signals.
        """
        df = data.copy()

        # Calculate True Range (TR)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # Calculate Average True Range (ATR)
        df['ATR'] = df['TR'].ewm(span=self.atr_period, adjust=False).mean()

        # Initialize Trailing Stops and Signals
        df['Buy_Stop'] = np.nan
        df['Sell_Stop'] = np.nan
        df['Signal'] = 'Hold'

        long_position = False
        short_position = False

        for i in range(1, len(df)):
            current_atr = df['ATR'].iloc[i]
            current_close = df['Close'].iloc[i]
            prev_close = df['Close'].iloc[i - 1]
            prev_buy_stop = df['Buy_Stop'].iloc[i - 1]
            prev_sell_stop = df['Sell_Stop'].iloc[i - 1]

            if not long_position and not short_position:
                # Potential Buy Signal
                if current_close > prev_close and (current_close - prev_close) > (current_atr * self.atr_multiplier / 2):
                    long_position = True
                    df.loc[df.index[i], 'Signal'] = 'Buy'
                    df.loc[df.index[i], 'Buy_Stop'] = current_close - (current_atr * self.atr_multiplier)
                # Potential Sell Signal
                elif current_close < prev_close and (prev_close - current_close) > (current_atr * self.atr_multiplier / 2):
                    short_position = True
                    df.loc[df.index[i], 'Signal'] = 'Sell'
                    df.loc[df.index[i], 'Sell_Stop'] = current_close + (current_atr * self.atr_multiplier)

            elif long_position:
                new_buy_stop = current_close - (current_atr * self.atr_multiplier)
                df.loc[df.index[i], 'Buy_Stop'] = (
                    new_buy_stop if np.isnan(prev_buy_stop) else max(prev_buy_stop, new_buy_stop)
                )
                if current_close < df['Buy_Stop'].iloc[i]:
                    long_position = False
                    df.loc[df.index[i], 'Signal'] = 'Sell'
                else:
                    df.loc[df.index[i], 'Signal'] = 'Hold'

            elif short_position:
                new_sell_stop = current_close + (current_atr * self.atr_multiplier)
                df.loc[df.index[i], 'Sell_Stop'] = (
                    new_sell_stop if np.isnan(prev_sell_stop) else min(prev_sell_stop, new_sell_stop)
                )
                if current_close > df['Sell_Stop'].iloc[i]:
                    short_position = False
                    df.loc[df.index[i], 'Signal'] = 'Buy'
                else:
                    df.loc[df.index[i], 'Signal'] = 'Hold'

        return df[['Close', 'ATR', 'Buy_Stop', 'Sell_Stop', 'Signal']]


def atr_trailing_stop(data: pd.DataFrame, atr_period: int = 14, atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Functional wrapper for the ATR Trailing Stop strategy.

    Goal:
    -----
    Maintain full backward compatibility with functional test suites and existing scripts.

    Execution Principle:
    --------------------
    Constructs an `AtrTrailingStopStrategy` dataclass instance with the provided parameters
    and invokes its `generate_signals` method.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV market price bars.
    atr_period : int, default 14
        Lookback span for exponential ATR smoothing.
    atr_multiplier : float, default 2.0
        Multiplier applied to ATR for distance of trailing stops.

    Returns:
    --------
    pd.DataFrame
        Strategy output with indicator columns and signals.
    """
    return AtrTrailingStopStrategy(atr_period=atr_period, atr_multiplier=atr_multiplier).generate_signals(data)
