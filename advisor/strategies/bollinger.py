"""Bollinger Band Squeeze Breakout Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class BollingerSqueezeStrategy(BaseStrategy):
    """Bollinger Band volatility squeeze and breakout strategy."""

    name: ClassVar[str] = "bollinger_squeeze"
    description: ClassVar[str] = "Bollinger Band volatility squeeze breakout"

    window: int = 20
    num_std_dev: float = 2.0
    squeeze_window: int = 100
    squeeze_threshold: float = 0.2

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Middle Band (Moving Average)
        df['Middle_Band'] = df['Close'].rolling(window=self.window).mean()

        # Calculate Standard Deviation
        df['Std_Dev'] = df['Close'].rolling(window=self.window).std()

        # Calculate Upper and Lower Bollinger Bands
        df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * self.num_std_dev)
        df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * self.num_std_dev)

        # Calculate Bollinger Band Width
        df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']

        # Identify Squeeze Condition
        df['Min_BB_Width'] = df['BB_Width'].rolling(window=self.squeeze_window).min()
        rolling_std = df['BB_Width'].rolling(window=self.squeeze_window).std().shift(1)
        df['Is_Squeeze'] = df['BB_Width'] < (df['Min_BB_Width'].shift(1) + rolling_std * self.squeeze_threshold)

        # Generate Signals
        df['Signal'] = 'Hold'

        # Buy signal: Price breaks above upper band after a squeeze
        df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] > df['Upper_Band'].shift(1)), 'Signal'] = 'Buy'

        # Sell signal: Price breaks below lower band after a squeeze
        df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] < df['Lower_Band'].shift(1)), 'Signal'] = 'Sell'

        return df[['Close', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Is_Squeeze', 'Signal']]


def bollinger_squeeze_strategy(
    data: pd.DataFrame,
    window: int = 20,
    num_std_dev: float = 2.0,
    squeeze_window: int = 100,
    squeeze_threshold: float = 0.2,
) -> pd.DataFrame:
    """Backward-compatible function wrapper for Bollinger Squeeze strategy."""
    return BollingerSqueezeStrategy(
        window=window,
        num_std_dev=num_std_dev,
        squeeze_window=squeeze_window,
        squeeze_threshold=squeeze_threshold,
    ).generate_signals(data)
