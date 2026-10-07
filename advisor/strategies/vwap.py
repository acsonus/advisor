"""Volume Weighted Average Price (VWAP) Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class VwapStrategy(BaseStrategy):
    """Volume Weighted Average Price (VWAP) price crossover strategy."""

    name: ClassVar[str] = "vwap"
    description: ClassVar[str] = "VWAP price crossover strategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
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
    """Backward-compatible function wrapper for VWAP calculation."""
    return VwapStrategy().generate_signals(data)
