"""Moving Average + Relative Strength Index (MA-RSI) Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class MaRsiStrategy(BaseStrategy):
    """Dual EMA crossover with RSI momentum filter strategy."""

    name: ClassVar[str] = "ma_rsi"
    description: ClassVar[str] = "EMA crossover with RSI momentum filter"

    short_ema_period: int = 12
    long_ema_period: int = 26
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
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
    """Backward-compatible function wrapper for MA-RSI strategy."""
    return MaRsiStrategy(
        short_ema_period=short_ema_period,
        long_ema_period=long_ema_period,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
    ).generate_signals(data)
