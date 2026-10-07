"""ATR Trailing Stop Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class AtrTrailingStopStrategy(BaseStrategy):
    """Average True Range (ATR) dynamic trailing stop strategy."""

    name: ClassVar[str] = "atr_trailing_stop"
    description: ClassVar[str] = "Average True Range (ATR) dynamic trailing stop"

    atr_period: int = 14
    atr_multiplier: float = 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
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
    """Backward-compatible function wrapper for ATR trailing stop strategy."""
    return AtrTrailingStopStrategy(atr_period=atr_period, atr_multiplier=atr_multiplier).generate_signals(data)
